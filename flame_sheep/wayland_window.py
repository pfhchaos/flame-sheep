"""
WallpaperWindow — wlr-layer-shell surfaces with a shared EGL display/context.

For single-output mode, use WallpaperWindow directly.

For multi-output mode (continuous image across monitors), use WallpaperSession:
  session = WallpaperSession()
  wins = [session.add_output('DP-3'), session.add_output('DP-2'), ...]
  session.make_current(wins[0])   # bind context + first surface
  ctx = session.create_moderngl_context()  # one moderngl ctx for all
  while True:
      # compute pass ...
      for win in wins:
          session.make_current(win)   # switch surface, same context
          # tonemap pass for win ...
          session.swap(win)
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


import ctypes
import ctypes.util
import os
import select
import time
import signal
from collections import deque

from typing import TYPE_CHECKING, Any

import cffi
import moderngl
import moderngl.mgl as _mgl
from glcontext import egl as _glcontext_egl
import numpy as np

from pywayland.client import Display

_ffi = cffi.FFI()


def _cffi_to_void_p(cdata: Any) -> ctypes.c_void_p:
    """Convert a cffi cdata pointer to a ctypes c_void_p."""
    return ctypes.c_void_p(int(_ffi.cast('uintptr_t', cdata)))


from pywayland.protocol.wayland import WlCompositor, WlOutput, WlSeat
from .protocol.wlr_layer_shell_unstable_v1.zwlr_layer_shell_v1 import ZwlrLayerShellV1
from .protocol.wlr_layer_shell_unstable_v1.zwlr_layer_surface_v1 import ZwlrLayerSurfaceV1

# ---------------------------------------------------------------------------
# EGL constants
# ---------------------------------------------------------------------------
EGL_NONE                         = 0x3038
EGL_OPENGL_BIT                   = 0x0008
EGL_OPENGL_API                   = 0x30A2
EGL_SURFACE_TYPE                 = 0x3033
EGL_WINDOW_BIT                   = 0x0004
EGL_RENDERABLE_TYPE              = 0x3040
EGL_RED_SIZE                     = 0x3024
EGL_GREEN_SIZE                   = 0x3023
EGL_BLUE_SIZE                    = 0x3022
EGL_ALPHA_SIZE                   = 0x3021
EGL_DEPTH_SIZE                   = 0x3025
EGL_CONTEXT_MAJOR_VERSION        = 0x3098
EGL_CONTEXT_MINOR_VERSION        = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK  = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_NO_DISPLAY                   = ctypes.c_void_p(0)
EGL_NO_CONTEXT                   = ctypes.c_void_p(0)
EGL_NO_SURFACE                   = ctypes.c_void_p(0)
EGL_PLATFORM_WAYLAND_KHR         = 0x31D8

# ---------------------------------------------------------------------------
# Load libraries
# ---------------------------------------------------------------------------
_libegl   = ctypes.CDLL('libEGL.so.1')
_libwlegl = ctypes.CDLL('libwayland-egl.so.1')

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

_libwlegl.wl_egl_window_create.restype    = ctypes.c_void_p
_libwlegl.wl_egl_window_create.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_libwlegl.wl_egl_window_destroy.restype   = None
_libwlegl.wl_egl_window_destroy.argtypes  = [ctypes.c_void_p]
_libwlegl.wl_egl_window_resize.restype    = None
_libwlegl.wl_egl_window_resize.argtypes   = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]


def _egl_check(label: str) -> None:
    err = _libegl.eglGetError()
    if err != 0x3000:  # EGL_SUCCESS
        raise RuntimeError(f'EGL error after {label}: 0x{err:04x}')


# ---------------------------------------------------------------------------
# Output surface — one per monitor
# ---------------------------------------------------------------------------

class OutputSurface:
    """
    One wlr-layer-shell BACKGROUND surface for a single output.
    Does NOT own an EGL context — that lives in WallpaperSession.
    """
    def __init__(self) -> None:
        self.width: int        = 0
        self.height: int       = 0
        self.should_close: bool = False
        self._configured: bool  = False
        self._frame_pending: bool   = True   # start True so first frame renders
        self._frame_callback: Any   = None   # prevent GC of pending wl_callback
        self._wl_surface: Any       = None
        self._layer_surface: Any    = None
        self._egl_window: int | None      = None  # wl_egl_window*
        self.egl_surface: int | None      = None  # EGL surface handle (int)


# ---------------------------------------------------------------------------
# WallpaperSession — one Wayland connection + EGL display + context, N surfaces
# ---------------------------------------------------------------------------

class WallpaperSession:
    """
    Manages one wl_display, one EGL display, one GL context, and N output
    surfaces. All surfaces share the same EGL display so we can freely
    switch between them with eglMakeCurrent.

    Usage:
        session = WallpaperSession()
        win_dp3 = session.add_output('DP-3')
        win_dp2 = session.add_output('DP-2')
        session.make_current(win_dp3)
        ctx = session.create_moderngl_context()
        while True:
            # compute ...
            for win in [win_dp3, win_dp2]:
                session.make_current(win)
                # render win's slice ...
                session.swap(win)
        session.destroy()
    """

    @staticmethod
    def list_outputs() -> list[str]:
        """Return names of all active Wayland outputs."""
        display = Display()
        display.connect()
        registry = display.get_registry()
        outputs  = []

        def _on_global(reg: Any, name: int, interface: str, version: int) -> None:
            if interface == WlOutput.name:
                out = reg.bind(name, WlOutput, min(version, 4))
                outputs.append(out)

        registry.dispatcher['global'] = _on_global
        display.roundtrip()

        output_names: dict[int, str] = {}

        def make_name_handler(out: Any) -> Any:
            def _on_name(output: Any, name: str) -> None:
                output_names[id(out)] = name
            return _on_name

        for out in outputs:
            out.dispatcher['name'] = make_name_handler(out)
        display.roundtrip()
        display.disconnect()
        return list(output_names.values())

    def __init__(self) -> None:
        self.ctx: moderngl.Context | None = None
        self._egl_display: int | None  = None
        self._egl_config: Any   = None
        self._egl_context: int | None  = None

        # Wayland
        self._wl_display: Any   = Display()
        self._wl_display.connect()

        self._compositor   = None
        self._layer_shell  = None
        self._wl_outputs: dict[str, object] = {}  # name -> WlOutput proxy

        registry = self._wl_display.get_registry()
        registry.dispatcher['global'] = self._on_global
        self._wl_display.roundtrip()
        self._wl_display.roundtrip()  # collect output names

        if self._compositor is None:
            raise RuntimeError('No wl_compositor in Wayland registry')
        if self._layer_shell is None:
            raise RuntimeError('No zwlr_layer_shell_v1 — is this sway/wlroots?')

        # Initialise EGL on this wl_display
        self._init_egl()

        signal.signal(signal.SIGINT, lambda *_: self._signal_close())
        self._surfaces: list[OutputSurface] = []

    def _on_global(self, registry: Any, name: int, interface: str, version: int) -> None:
        if interface == WlCompositor.name:
            self._compositor = registry.bind(name, WlCompositor, min(version, 5))
        elif interface == ZwlrLayerShellV1.name:
            self._layer_shell = registry.bind(name, ZwlrLayerShellV1, min(version, 4))
        elif interface == WlOutput.name:
            out = registry.bind(name, WlOutput, min(version, 4))
            def _on_name(output, oname, _out=out):
                self._wl_outputs[oname] = _out
            out.dispatcher['name'] = _on_name

    def _init_egl(self) -> None:
        wl_ptr = _cffi_to_void_p(self._wl_display._ptr)

        if _eglGetPlatformDisplayEXT is not None:
            self._egl_display = _eglGetPlatformDisplayEXT(
                EGL_PLATFORM_WAYLAND_KHR, wl_ptr, None)

        if not self._egl_display:
            self._egl_display = _libegl.eglGetDisplay(wl_ptr)
        _egl_check('eglGetDisplay')

        major, minor = ctypes.c_int(), ctypes.c_int()
        _libegl.eglInitialize(self._egl_display,
                              ctypes.byref(major), ctypes.byref(minor))
        _egl_check('eglInitialize')
        log.info(f'EGL {major.value}.{minor.value}')

        _libegl.eglBindAPI(EGL_OPENGL_API)
        _egl_check('eglBindAPI')

        attribs = (ctypes.c_int * 15)(
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
        _libegl.eglChooseConfig(self._egl_display, attribs,
                                ctypes.byref(config), 1, ctypes.byref(n_configs))
        _egl_check('eglChooseConfig')
        if n_configs.value == 0:
            raise RuntimeError('No suitable EGL config')
        self._egl_config = config

        ctx_attribs = (ctypes.c_int * 7)(
            EGL_CONTEXT_MAJOR_VERSION, 4,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
            EGL_NONE,
        )
        self._egl_context = _libegl.eglCreateContext(
            self._egl_display, self._egl_config, EGL_NO_CONTEXT, ctx_attribs)
        _egl_check('eglCreateContext')

    def add_output(self, output_name: str) -> OutputSurface:
        """
        Create a BACKGROUND layer-shell surface on the named output.
        Returns an OutputSurface; call make_current(surf) + swap(surf) per frame.
        """
        if output_name not in self._wl_outputs:
            available = list(self._wl_outputs.keys())
            raise RuntimeError(
                f'Output {output_name!r} not found. Available: {available}')

        wl_out = self._wl_outputs[output_name]
        surf   = OutputSurface()

        # Wayland surface + EGL window (size 1x1 initially; configure resizes)
        surf._wl_surface = self._compositor.create_surface()
        surf._egl_window = _libwlegl.wl_egl_window_create(
            _cffi_to_void_p(surf._wl_surface._ptr), 1, 1)
        if not surf._egl_window:
            raise RuntimeError('wl_egl_window_create failed')

        surf.egl_surface = _libegl.eglCreateWindowSurface(
            self._egl_display, self._egl_config,
            ctypes.c_void_p(surf._egl_window), None)
        _egl_check('eglCreateWindowSurface')

        # Layer-shell surface
        layer = ZwlrLayerShellV1.layer.background.value
        surf._layer_surface = self._layer_shell.get_layer_surface(
            surf._wl_surface, wl_out, layer, 'flame-sheep')

        anchor_all = (ZwlrLayerSurfaceV1.anchor.top.value    |
                      ZwlrLayerSurfaceV1.anchor.bottom.value |
                      ZwlrLayerSurfaceV1.anchor.left.value   |
                      ZwlrLayerSurfaceV1.anchor.right.value)
        surf._layer_surface.set_anchor(anchor_all)
        surf._layer_surface.set_size(0, 0)
        surf._layer_surface.set_exclusive_zone(-1)
        surf._layer_surface.set_keyboard_interactivity(0)

        surf._layer_surface.dispatcher['configure'] = \
            lambda ls, serial, w, h: self._on_configure(surf, ls, serial, w, h)
        surf._layer_surface.dispatcher['closed'] = \
            lambda ls: setattr(surf, 'should_close', True)

        surf._wl_surface.commit()
        self._wl_display.roundtrip()

        # Wait for configure
        deadline = time.monotonic() + 5.0
        while not surf._configured:
            self._wl_display.roundtrip()
            if time.monotonic() > deadline:
                raise RuntimeError(f'Timed out waiting for configure on {output_name}')
            time.sleep(0.005)

        self._surfaces.append(surf)
        return surf

    def _on_configure(self, surf: OutputSurface, layer_surface: Any,
                      serial: int, width: int, height: int) -> None:
        if width  > 0: surf.width  = width
        if height > 0: surf.height = height
        layer_surface.ack_configure(serial)
        surf._wl_surface.commit()
        surf._configured = True
        log.debug(f'configured {surf.width}x{surf.height}')
        if surf._egl_window:
            _libwlegl.wl_egl_window_resize(
                ctypes.c_void_p(surf._egl_window),
                surf.width, surf.height, 0, 0)

    def make_current(self, surf: OutputSurface) -> bool:
        """Bind the shared GL context to surf's EGL surface.
        Returns False if the surface is dead (e.g. sway reloaded)."""
        if surf.should_close or not surf.egl_surface:
            return False
        ok = _libegl.eglMakeCurrent(
            self._egl_display,
            ctypes.c_void_p(surf.egl_surface),
            ctypes.c_void_p(surf.egl_surface),
            ctypes.c_void_p(self._egl_context))
        if not ok:
            err = _libegl.eglGetError()
            log.error(f'eglMakeCurrent failed (0x{err:04x}), marking surface dead')
            surf.should_close = True
            return False
        return True

    def release_current(self) -> None:
        _libegl.eglMakeCurrent(
            self._egl_display,
            EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)

    def create_moderngl_context(self) -> moderngl.Context:
        """
        Wrap the current EGL context into a moderngl Context.
        Must be called after make_current() with any surface.
        """
        _gc = _glcontext_egl.create_context(mode='share', glversion=430)
        ctx = moderngl.Context.__new__(moderngl.Context)
        ctx.mglo, ctx.version_code = _mgl.create_context(
            glversion=430, mode='share', context=_gc)
        ctx._info       = None
        ctx._extensions = None
        ctx.extra       = None
        ctx._gc_mode    = None
        ctx._objects    = deque()
        ctx._screen     = None
        ctx.fbo         = None
        log.info(f'GL {ctx.version_code} '
              f'renderer: {ctx.info["GL_RENDERER"]}')
        self.ctx = ctx
        _libegl.eglSwapInterval(self._egl_display, 0)  # non-blocking — frame callbacks pace us
        return ctx

    def swap(self, surf: OutputSurface) -> bool:
        """Present surf and dispatch Wayland events.
        Returns False if the connection is broken."""
        if surf.should_close:
            return False

        # Request frame callback BEFORE swap — eglSwapBuffers implicitly
        # commits the surface, so the callback request must be pending first.
        # Hold a reference on the surface to prevent GC before delivery.
        def _on_done(cb, ts, _s=surf):
            _s._frame_pending = True
            _s._frame_callback = None  # release after delivery
        surf._frame_callback = surf._wl_surface.frame()
        surf._frame_callback.dispatcher['done'] = _on_done
        surf._frame_pending = False

        ok = _libegl.eglSwapBuffers(
            self._egl_display,
            ctypes.c_void_p(surf.egl_surface))
        if not ok:
            err = _libegl.eglGetError()
            log.error(f'eglSwapBuffers failed (0x{err:04x}), marking surface dead')
            surf.should_close = True
            return False
        try:
            self._wl_display.flush()
        except Exception as e:
            log.error(f'flush failed: {e}')
            self._signal_close()
            return False
        return True

    def dispatch(self) -> bool:
        """Non-blocking read + dispatch of pending Wayland events.
        Returns False if the connection is broken."""
        try:
            self._wl_display.flush()
            fd = self._wl_display.get_fd()
            r, _, err = select.select([fd], [], [fd], 0)
            if err:
                log.error('wayland fd error')
                self._signal_close()
                return False
            if r:
                self._wl_display.read()
            self._wl_display.dispatch(block=False)
            return True
        except Exception as e:
            log.error(f'dispatch failed: {e}')
            self._signal_close()
            return False

    def wait_for_events(self, timeout: float = 0.016) -> bool:
        """Block until Wayland events arrive or timeout expires.
        Use when all surfaces are occluded to avoid busy-spinning.
        Returns False if the connection is broken."""
        try:
            self._wl_display.flush()
            fd = self._wl_display.get_fd()
            r, _, err = select.select([fd], [], [fd], timeout)
            if err:
                self._signal_close()
                return False
            if r:
                self._wl_display.read()
            self._wl_display.dispatch(block=False)
            return True
        except Exception as e:
            log.error(f'wait failed: {e}')
            self._signal_close()
            return False

    def _signal_close(self) -> None:
        for s in self._surfaces:
            s.should_close = True

    def destroy(self) -> None:
        self.release_current()
        for surf in self._surfaces:
            if surf._layer_surface:
                surf._layer_surface.destroy()
            if surf._wl_surface:
                surf._wl_surface.destroy()
            if surf.egl_surface:
                _libegl.eglDestroySurface(self._egl_display,
                                          ctypes.c_void_p(surf.egl_surface))
            if surf._egl_window:
                _libwlegl.wl_egl_window_destroy(
                    ctypes.c_void_p(surf._egl_window))
        if self._egl_context:
            _libegl.eglDestroyContext(self._egl_display,
                                      ctypes.c_void_p(self._egl_context))
        if self._egl_display:
            _libegl.eglTerminate(self._egl_display)
        if self._wl_display:
            self._wl_display.disconnect()


# ---------------------------------------------------------------------------
# WallpaperWindow — single-output convenience wrapper (kept for compat)
# ---------------------------------------------------------------------------

class WallpaperWindow:
    """Single-output wallpaper surface. Wraps WallpaperSession for one output."""

    @staticmethod
    def list_outputs() -> list[str]:
        return WallpaperSession.list_outputs()

    def __init__(self, output_name: str | None = None):
        self._session = WallpaperSession()
        names = self._session.list_outputs()
        name  = output_name or (names[0] if names else None)
        if name is None:
            raise RuntimeError('No outputs available')
        self._surf = self._session.add_output(name)
        self.width  = self._surf.width
        self.height = self._surf.height

        self._session.make_current(self._surf)
        self.ctx = self._session.create_moderngl_context()
        signal.signal(signal.SIGINT, lambda *_: self._session._signal_close())

    @property
    def should_close(self) -> bool:
        return self._surf.should_close

    def make_current(self) -> None:
        self._session.make_current(self._surf)

    def release_current(self) -> None:
        self._session.release_current()

    def swap(self) -> None:
        self._session.swap(self._surf)

    def destroy(self) -> None:
        self._session.destroy()
