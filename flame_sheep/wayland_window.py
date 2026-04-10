"""
WallpaperWindow — wlr-layer-shell surface with an EGL/OpenGL context.

Replaces the moderngl-window GLFW window for wallpaper mode.
Creates a BACKGROUND-layer surface that sits behind all windows,
spans the full output, and has no keyboard/pointer interactivity.

Usage:
    win = WallpaperWindow(width, height)
    ctx = win.ctx          # moderngl.Context, use exactly like normal
    while True:
        # ... render into ctx ...
        win.swap()         # eglSwapBuffers + wayland dispatch
        if win.should_close:
            break
    win.destroy()
"""

import ctypes
import ctypes.util
import os
import time
import signal
from collections import deque

import cffi
import moderngl
import moderngl.mgl as _mgl
from glcontext import egl as _glcontext_egl
import numpy as np

from pywayland.client import Display

_ffi = cffi.FFI()


def _cffi_to_void_p(cdata) -> ctypes.c_void_p:
    """Convert a cffi cdata pointer to a ctypes c_void_p."""
    return ctypes.c_void_p(int(_ffi.cast('uintptr_t', cdata)))


from pywayland.protocol.wayland import WlCompositor, WlOutput, WlSeat
from .protocol.wlr_layer_shell_unstable_v1.zwlr_layer_shell_v1 import ZwlrLayerShellV1
from .protocol.wlr_layer_shell_unstable_v1.zwlr_layer_surface_v1 import ZwlrLayerSurfaceV1

# ---------------------------------------------------------------------------
# EGL constants (enough to bootstrap a GLES3/GL context on a Wayland surface)
# ---------------------------------------------------------------------------
EGL_NONE                    = 0x3038
EGL_OPENGL_BIT              = 0x0008
EGL_OPENGL_ES3_BIT          = 0x0040
EGL_OPENGL_API              = 0x30A2
EGL_OPENGL_ES_API           = 0x30A0
EGL_SURFACE_TYPE            = 0x3033
EGL_WINDOW_BIT              = 0x0004
EGL_RENDERABLE_TYPE         = 0x3040
EGL_RED_SIZE                = 0x3024
EGL_GREEN_SIZE              = 0x3023
EGL_BLUE_SIZE               = 0x3022
EGL_ALPHA_SIZE              = 0x3021
EGL_DEPTH_SIZE              = 0x3025
EGL_COLOR_BUFFER_TYPE       = 0x303F
EGL_RGB_BUFFER              = 0x308E
EGL_CONTEXT_MAJOR_VERSION   = 0x3098
EGL_CONTEXT_MINOR_VERSION   = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_NO_DISPLAY              = ctypes.c_void_p(0)
EGL_NO_CONTEXT              = ctypes.c_void_p(0)
EGL_NO_SURFACE              = ctypes.c_void_p(0)
EGL_DEFAULT_DISPLAY         = ctypes.c_void_p(0)
EGL_PLATFORM_WAYLAND_KHR    = 0x31D8

# ---------------------------------------------------------------------------
# Load libraries
# ---------------------------------------------------------------------------
_libegl   = ctypes.CDLL('libEGL.so.1')
_libwlegl = ctypes.CDLL('libwayland-egl.so.1')

# eglGetPlatformDisplayEXT is an EGL extension — not always exported as a
# direct symbol.  Look it up via eglGetProcAddress instead.
_libegl.eglGetProcAddress.restype  = ctypes.c_void_p
_libegl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
_proc_addr = _libegl.eglGetProcAddress(b'eglGetPlatformDisplayEXT')
if _proc_addr:
    _eglGetPlatformDisplayEXT = ctypes.CFUNCTYPE(
        ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
    )(_proc_addr)
else:
    _eglGetPlatformDisplayEXT = None

# Regular EGL function signatures
_libegl.eglGetDisplay.restype             = ctypes.c_void_p
_libegl.eglGetDisplay.argtypes            = [ctypes.c_void_p]
_libegl.eglInitialize.restype             = ctypes.c_bool
_libegl.eglInitialize.argtypes            = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
_libegl.eglChooseConfig.restype           = ctypes.c_bool
_libegl.eglChooseConfig.argtypes          = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
_libegl.eglCreateWindowSurface.restype    = ctypes.c_void_p
_libegl.eglCreateWindowSurface.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglCreateContext.restype          = ctypes.c_void_p
_libegl.eglCreateContext.argtypes         = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglMakeCurrent.restype            = ctypes.c_bool
_libegl.eglMakeCurrent.argtypes           = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglSwapBuffers.restype            = ctypes.c_bool
_libegl.eglSwapBuffers.argtypes           = [ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglBindAPI.restype                = ctypes.c_bool
_libegl.eglBindAPI.argtypes               = [ctypes.c_uint32]
_libegl.eglGetError.restype               = ctypes.c_int
_libegl.eglSwapInterval.restype           = ctypes.c_bool
_libegl.eglSwapInterval.argtypes          = [ctypes.c_void_p, ctypes.c_int]
_libegl.eglDestroyContext.restype         = ctypes.c_bool
_libegl.eglDestroyContext.argtypes        = [ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglDestroySurface.restype         = ctypes.c_bool
_libegl.eglDestroySurface.argtypes        = [ctypes.c_void_p, ctypes.c_void_p]
_libegl.eglTerminate.restype              = ctypes.c_bool
_libegl.eglTerminate.argtypes             = [ctypes.c_void_p]

# wl_egl_window_create / destroy
_libwlegl.wl_egl_window_create.restype    = ctypes.c_void_p
_libwlegl.wl_egl_window_create.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libwlegl.wl_egl_window_destroy.restype   = None
_libwlegl.wl_egl_window_destroy.argtypes  = [ctypes.c_void_p]
_libwlegl.wl_egl_window_resize.restype    = None
_libwlegl.wl_egl_window_resize.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _egl_check(label: str):
    err = _libegl.eglGetError()
    if err != 0x3000:  # EGL_SUCCESS
        raise RuntimeError(f'EGL error after {label}: 0x{err:04x}')


class WallpaperWindow:
    """
    A wlr-layer-shell BACKGROUND surface with an OpenGL 4.3 core context.
    """

    @staticmethod
    def list_outputs() -> list[str]:
        """Return names of all active Wayland outputs (e.g. ['DP-2', 'DP-3'])."""
        names = []
        display = Display()
        display.connect()
        registry = display.get_registry()
        outputs  = []

        def _on_global(reg, name, interface, version):
            if interface == WlOutput.name:
                out = reg.bind(name, WlOutput, min(version, 4))
                outputs.append(out)

        registry.dispatcher['global'] = _on_global
        display.roundtrip()

        # Collect output names via WlOutput.name event
        output_names: dict = {}

        def make_name_handler(out):
            def _on_name(output, name):
                output_names[id(out)] = name
            return _on_name

        for out in outputs:
            out.dispatcher['name'] = make_name_handler(out)
        display.roundtrip()
        display.disconnect()

        return list(output_names.values())

    def __init__(self, width: int = 1920, height: int = 1080,
                 output_name: str | None = None):
        self.width        = width
        self.height       = height
        self.should_close = False
        self._configured  = False   # True once compositor sends configure
        self._target_output_name = output_name

        # State gathered during registry enumeration
        self._compositor  = None
        self._layer_shell = None
        self._output      = None    # matched output (or first if no name given)
        self._seat        = None
        self._outputs_by_name: dict[str, object] = {}

        # Wayland objects
        self._wl_display  = None
        self._wl_surface  = None
        self._layer_surface = None

        # EGL handles
        self._egl_display  = None
        self._egl_config   = None
        self._egl_context  = None
        self._egl_window   = None   # wl_egl_window*
        self._egl_surface  = None

        self._setup_wayland()
        self._setup_egl()
        self._setup_layer_surface()
        self._wait_for_configure()
        self._make_current()
        _libegl.eglSwapInterval(self._egl_display, 1)  # vsync on

        # Wrap the already-current EGL context into a moderngl Context.
        #
        # Strategy: glcontext.egl.create_context(mode='share') detects the
        # currently-bound EGL context (our Wayland surface context) and wraps
        # it. We then pass that glcontext object to mgl.create_context so that
        # the mgl C extension also uses EGL instead of defaulting to X11/GLX.
        #
        # This avoids the "glXGetCurrentContext: cannot detect" error that
        # occurs when mgl tries to use the x11 glcontext backend on Linux.
        _gc = _glcontext_egl.create_context(mode='share', glversion=430)
        self.ctx = moderngl.Context.__new__(moderngl.Context)
        self.ctx.mglo, self.ctx.version_code = _mgl.create_context(
            glversion=430, mode='share', context=_gc)
        self.ctx._info       = None
        self.ctx._extensions = None
        self.ctx.extra       = None
        self.ctx._gc_mode    = None
        self.ctx._objects    = deque()
        self.ctx._screen     = None
        self.ctx.fbo         = None
        print(f'[wallpaper] GL {self.ctx.version_code} renderer: {self.ctx.info["GL_RENDERER"]}')

        # Install SIGINT handler so Ctrl-C closes cleanly
        signal.signal(signal.SIGINT, lambda *_: self._signal_close())

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_wayland(self):
        self._wl_display = Display()
        self._wl_display.connect()

        registry = self._wl_display.get_registry()
        registry.dispatcher['global'] = self._on_global
        self._wl_display.roundtrip()  # get globals
        self._wl_display.roundtrip()  # get output name events

        if self._compositor is None:
            raise RuntimeError('No wl_compositor in registry')
        if self._layer_shell is None:
            raise RuntimeError('No zwlr_layer_shell_v1 in registry — is this sway/wlroots?')
        if self._output is None:
            raise RuntimeError(f'Output {self._target_output_name!r} not found. '
                               f'Available: {list(self._outputs_by_name.keys())}')

        self._wl_surface = self._compositor.create_surface()

    def _on_global(self, registry, name: int, interface: str, version: int):
        if interface == WlCompositor.name:
            self._compositor = registry.bind(name, WlCompositor, min(version, 5))
        elif interface == ZwlrLayerShellV1.name:
            self._layer_shell = registry.bind(name, ZwlrLayerShellV1, min(version, 4))
        elif interface == WlOutput.name:
            out = registry.bind(name, WlOutput, min(version, 4))
            # Wire up name event to match target output
            def _on_output_name(output, oname, _out=out):
                self._outputs_by_name[oname] = _out
                if self._target_output_name is None and self._output is None:
                    self._output = _out
                elif oname == self._target_output_name:
                    self._output = _out
            out.dispatcher['name'] = _on_output_name
            if self._output is None and self._target_output_name is None:
                self._output = out  # fallback: first output before name arrives

    def _setup_egl(self):
        # Get the raw wl_display* pointer from pywayland (cffi cdata -> ctypes)
        wl_display_ptr = _cffi_to_void_p(self._wl_display._ptr)

        if _eglGetPlatformDisplayEXT is not None:
            self._egl_display = _eglGetPlatformDisplayEXT(
                EGL_PLATFORM_WAYLAND_KHR, wl_display_ptr, None)
        else:
            self._egl_display = None

        if not self._egl_display:
            # fallback: eglGetDisplay (works on most Mesa Wayland setups)
            self._egl_display = _libegl.eglGetDisplay(wl_display_ptr)
        _egl_check('eglGetDisplay')

        major, minor = ctypes.c_int(), ctypes.c_int()
        _libegl.eglInitialize(self._egl_display, ctypes.byref(major), ctypes.byref(minor))
        _egl_check('eglInitialize')
        print(f'[wallpaper] EGL {major.value}.{minor.value}')

        _libegl.eglBindAPI(EGL_OPENGL_API)
        _egl_check('eglBindAPI')

        attribs = (ctypes.c_int * 17)(
            EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE,   8,
            EGL_GREEN_SIZE, 8,
            EGL_BLUE_SIZE,  8,
            EGL_ALPHA_SIZE, 0,
            EGL_DEPTH_SIZE, 0,
            EGL_NONE,
        )
        config    = ctypes.c_void_p()
        n_configs = ctypes.c_int()
        _libegl.eglChooseConfig(
            self._egl_display, attribs,
            ctypes.byref(config), 1, ctypes.byref(n_configs))
        _egl_check('eglChooseConfig')
        if n_configs.value == 0:
            raise RuntimeError('No suitable EGL config found')
        self._egl_config = config

        ctx_attribs = (ctypes.c_int * 7)(
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE,
        )
        self._egl_context = _libegl.eglCreateContext(
            self._egl_display, self._egl_config,
            EGL_NO_CONTEXT, ctx_attribs)
        _egl_check('eglCreateContext')

    def _setup_layer_surface(self):
        # Create the wl_egl_window and EGL surface BEFORE the layer surface,
        # so the compositor gets a buffer when we first commit.
        self._egl_window = _libwlegl.wl_egl_window_create(
            _cffi_to_void_p(self._wl_surface._ptr),
            self.width, self.height)
        if not self._egl_window:
            raise RuntimeError('wl_egl_window_create failed')

        self._egl_surface = _libegl.eglCreateWindowSurface(
            self._egl_display, self._egl_config,
            ctypes.c_void_p(self._egl_window), None)
        _egl_check('eglCreateWindowSurface')

        # Create layer-shell surface
        layer      = ZwlrLayerShellV1.layer.background.value
        namespace  = 'flame-sheep'

        self._layer_surface = self._layer_shell.get_layer_surface(
            self._wl_surface,
            self._output,
            layer,
            namespace,
        )

        # Anchor to all four edges so compositor gives us the full output
        anchor_all = (ZwlrLayerSurfaceV1.anchor.top.value    |
                      ZwlrLayerSurfaceV1.anchor.bottom.value |
                      ZwlrLayerSurfaceV1.anchor.left.value   |
                      ZwlrLayerSurfaceV1.anchor.right.value)
        self._layer_surface.set_anchor(anchor_all)
        self._layer_surface.set_size(0, 0)          # 0,0 = full output
        self._layer_surface.set_exclusive_zone(-1)  # don't push other windows
        self._layer_surface.set_keyboard_interactivity(0)

        self._layer_surface.dispatcher['configure'] = self._on_configure
        self._layer_surface.dispatcher['closed']    = self._on_closed

        # commit() triggers sway to send a configure event back.
        # roundtrip() flushes our request AND processes all incoming events
        # (including configure) before returning.
        self._wl_surface.commit()
        self._wl_display.roundtrip()

    def _wait_for_configure(self, timeout: float = 5.0):
        # roundtrip() in _setup_layer_surface should have already delivered
        # the configure event. This loop is a safety net for slow compositors.
        deadline = time.monotonic() + timeout
        while not self._configured:
            self._wl_display.roundtrip()
            if time.monotonic() > deadline:
                raise RuntimeError('Timed out waiting for layer-surface configure')
            time.sleep(0.005)

    def _on_configure(self, layer_surface, serial: int, width: int, height: int):
        if width > 0:
            self.width = width
        if height > 0:
            self.height = height
        layer_surface.ack_configure(serial)
        self._wl_surface.commit()
        self._configured = True
        print(f'[wallpaper] configured {self.width}x{self.height}')

        # Resize the EGL window to match
        if self._egl_window:
            _libwlegl.wl_egl_window_resize(
                ctypes.c_void_p(self._egl_window),
                self.width, self.height, 0, 0)

    def _on_closed(self, layer_surface):
        self.should_close = True

    def make_current(self):
        """Make this window's EGL context current on the calling thread."""
        _libegl.eglMakeCurrent(
            self._egl_display,
            ctypes.c_void_p(self._egl_surface),
            ctypes.c_void_p(self._egl_surface),
            ctypes.c_void_p(self._egl_context))
        _egl_check('eglMakeCurrent')

    def release_current(self):
        """Release this EGL context from the calling thread."""
        _libegl.eglMakeCurrent(
            self._egl_display,
            EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)

    # Keep _make_current as internal alias used during init
    _make_current = make_current

    # ------------------------------------------------------------------
    # Per-frame interface
    # ------------------------------------------------------------------

    def swap(self):
        """Present the rendered frame and dispatch wayland events."""
        _libegl.eglSwapBuffers(
            self._egl_display,
            ctypes.c_void_p(self._egl_surface))
        self._wl_display.dispatch(block=False)
        self._wl_display.flush()

    def dispatch(self):
        """Dispatch wayland events without swapping (call between frames if needed)."""
        self._wl_display.dispatch(block=False)
        self._wl_display.flush()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _signal_close(self):
        self.should_close = True

    def destroy(self):
        if self._layer_surface:
            self._layer_surface.destroy()
        if self._wl_surface:
            self._wl_surface.destroy()
        if self._egl_surface:
            _libegl.eglDestroySurface(self._egl_display,
                                      ctypes.c_void_p(self._egl_surface))
        if self._egl_window:
            _libwlegl.wl_egl_window_destroy(ctypes.c_void_p(self._egl_window))
        if self._egl_context:
            _libegl.eglDestroyContext(self._egl_display,
                                      ctypes.c_void_p(self._egl_context))
        if self._egl_display:
            _libegl.eglTerminate(self._egl_display)
        if self._wl_display:
            self._wl_display.disconnect()
