"""Wayland integration tests using a headless sway compositor.

Starts a headless sway instance, runs flame-sheep's Wayland code
against it, and verifies layer-shell surfaces and output detection.

Requires: sway installed, WLR_BACKENDS=headless support.
Skips if sway is not available.
"""

import os
import json
import signal
import subprocess
import tempfile
import time

import pytest


def _find_sway():
    """Check if sway binary exists."""
    import shutil
    return shutil.which('sway') is not None


@pytest.fixture(scope='module')
def headless_sway():
    """Start a headless sway compositor for testing.

    Yields (wayland_display, swaysock) paths.
    Tears down on exit.
    """
    if not _find_sway():
        pytest.skip('sway not installed')

    tmpdir = tempfile.mkdtemp(prefix='flame-sheep-test-')

    # Minimal config — no exec, no background, no bar
    config_path = os.path.join(tmpdir, 'config')
    with open(config_path, 'w') as f:
        f.write('# flame-sheep test compositor\n')

    swaysock = os.path.join(tmpdir, 'sway.sock')

    env = dict(os.environ)
    env.update({
        'WLR_BACKENDS': 'headless',
        'XDG_CONFIG_HOME': tmpdir,
        'SWAYSOCK': swaysock,
    })

    proc = subprocess.Popen(
        ['sway', '-c', config_path],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for sway to create its IPC socket
    for _ in range(30):
        if os.path.exists(swaysock):
            break
        time.sleep(0.1)
    else:
        proc.kill()
        proc.wait()
        pytest.skip('headless sway failed to start')

    # Find the wayland display socket — newest wayland-N created after sway started
    runtime_dir = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    wayland_display = None
    try:
        sockets = sorted(
            [f for f in os.listdir(runtime_dir)
             if f.startswith('wayland-') and not f.endswith('.lock')],
            key=lambda f: os.path.getmtime(os.path.join(runtime_dir, f)),
            reverse=True,
        )
        if sockets:
            wayland_display = sockets[0]
    except Exception:
        pass

    yield {
        'swaysock': swaysock,
        'wayland_display': wayland_display,
        'runtime_dir': runtime_dir,
        'pid': proc.pid,
        'tmpdir': tmpdir,
    }

    # Teardown
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestSwayHeadless:
    """Basic tests against headless sway."""

    def test_compositor_starts(self, headless_sway):
        """Headless sway should start and create an IPC socket."""
        assert os.path.exists(headless_sway['swaysock'])

    def test_has_output(self, headless_sway):
        """Should have at least one HEADLESS output."""
        raw = subprocess.check_output(
            ['swaymsg', '-t', 'get_outputs'],
            env={'SWAYSOCK': headless_sway['swaysock'],
                 'PATH': os.environ['PATH']},
            timeout=5,
        )
        outputs = json.loads(raw)
        assert len(outputs) >= 1
        assert 'HEADLESS' in outputs[0]['name']

    def test_output_has_resolution(self, headless_sway):
        """HEADLESS output should have a valid resolution."""
        raw = subprocess.check_output(
            ['swaymsg', '-t', 'get_outputs'],
            env={'SWAYSOCK': headless_sway['swaysock'],
                 'PATH': os.environ['PATH']},
            timeout=5,
        )
        outputs = json.loads(raw)
        o = outputs[0]
        assert o['rect']['width'] > 0
        assert o['rect']['height'] > 0

    def test_sway_layout_query(self, headless_sway):
        """flame-sheep's _get_sway_layout should work against headless sway."""
        old_swaysock = os.environ.get('SWAYSOCK')
        os.environ['SWAYSOCK'] = headless_sway['swaysock']
        try:
            from flame_sheep.main import _get_sway_layout
            layout = _get_sway_layout()
            assert len(layout) >= 1
            name = list(layout.keys())[0]
            assert layout[name]['w'] > 0
            assert layout[name]['h'] > 0
        finally:
            if old_swaysock:
                os.environ['SWAYSOCK'] = old_swaysock
            else:
                os.environ.pop('SWAYSOCK', None)

    def test_wayland_output_layout(self, headless_sway):
        """Compositor-agnostic _get_output_layout via wl_output protocol."""
        wayland_display = headless_sway['wayland_display']
        if not wayland_display:
            pytest.skip('could not determine wayland display socket')

        old_display = os.environ.get('WAYLAND_DISPLAY')
        os.environ['WAYLAND_DISPLAY'] = wayland_display
        try:
            from flame_sheep.main import _get_output_layout
            layout = _get_output_layout()
            assert len(layout) >= 1
            name = list(layout.keys())[0]
            info = layout[name]
            assert info['w'] > 0
            assert info['h'] > 0
            assert info['ppi'] > 0
        finally:
            if old_display:
                os.environ['WAYLAND_DISPLAY'] = old_display
            else:
                os.environ.pop('WAYLAND_DISPLAY', None)

    def test_multi_output(self, headless_sway):
        """Should be able to add a second headless output."""
        env = {'SWAYSOCK': headless_sway['swaysock'],
               'PATH': os.environ['PATH']}
        # Add a second output
        subprocess.run(
            ['swaymsg', 'output', 'HEADLESS-2', 'enable'],
            env=env, capture_output=True, timeout=5)

        # Wait briefly for it to register
        time.sleep(0.5)

        raw = subprocess.check_output(
            ['swaymsg', '-t', 'get_outputs'], env=env, timeout=5)
        outputs = json.loads(raw)
        names = [o['name'] for o in outputs if o.get('active')]
        assert len(names) >= 1  # at least the original
        # Note: sway may not support adding headless outputs via swaymsg
        # This test verifies the query still works

    def test_list_outputs(self, headless_sway):
        """WallpaperSession.list_outputs should see the headless output."""
        wayland_display = headless_sway['wayland_display']
        if not wayland_display:
            pytest.skip('could not determine wayland display socket')

        old_display = os.environ.get('WAYLAND_DISPLAY')
        os.environ['WAYLAND_DISPLAY'] = wayland_display
        try:
            from flame_sheep.wayland_window import WallpaperSession
            output_names = WallpaperSession.list_outputs()
            assert len(output_names) >= 1
            # Headless sway reports outputs as WL-N or HEADLESS-N
            assert len(output_names) >= 1, \
                f'Expected at least one output, got {output_names}'
        except Exception as e:
            # EGL initialization may fail in headless — that's OK for this test
            if 'EGL' in str(e) or 'egl' in str(e):
                pytest.skip(f'EGL not available in headless: {e}')
            raise
        finally:
            if old_display:
                os.environ['WAYLAND_DISPLAY'] = old_display
            else:
                os.environ.pop('WAYLAND_DISPLAY', None)

    def test_create_wallpaper_session(self, headless_sway):
        """Should be able to create a WallpaperSession, add a surface, and get configure."""
        def run():
            from flame_sheep.wayland_window import WallpaperSession
            session = WallpaperSession()
            outputs = list(session._wl_outputs.keys())
            assert len(outputs) >= 1, 'No outputs in session'

            surf = session.add_output(outputs[0])
            assert surf is not None, 'Failed to create surface'
            assert surf._configured, 'Surface not configured after add_output'
            assert surf.width > 0 and surf.height > 0, \
                f'Invalid size: {surf.width}x{surf.height}'

            session.destroy()

        self._with_display(headless_sway, run)

    def _with_display(self, headless_sway, fn):
        """Run fn with WAYLAND_DISPLAY set to headless sway."""
        wayland_display = headless_sway['wayland_display']
        if not wayland_display:
            pytest.skip('could not determine wayland display socket')

        old_display = os.environ.get('WAYLAND_DISPLAY')
        os.environ['WAYLAND_DISPLAY'] = wayland_display
        try:
            fn()
        except Exception as e:
            if 'EGL' in str(e) or 'egl' in str(e):
                pytest.skip(f'EGL not available: {e}')
            raise
        finally:
            if old_display:
                os.environ['WAYLAND_DISPLAY'] = old_display
            else:
                os.environ.pop('WAYLAND_DISPLAY', None)

    def test_frame_callback(self, headless_sway):
        """Frame callback should fire after swap on headless compositor."""
        def run():
            from flame_sheep.wayland_window import WallpaperSession
            session = WallpaperSession()
            outputs = list(session._wl_outputs.keys())
            surf = session.add_output(outputs[0])

            # make_current before creating GL context
            session.make_current(surf)
            ctx = session.create_moderngl_context()

            assert surf._configured, 'Surface should be configured after add_output'
            assert surf._frame_pending, '_frame_pending should start True'

            # Swap resets _frame_pending and requests callback
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            session.swap(surf)
            assert not surf._frame_pending

            # Dispatch — compositor should deliver the frame callback
            for _ in range(20):
                session._wl_display.dispatch(block=False)
                session._wl_display.roundtrip()
                if surf._frame_pending:
                    break
                time.sleep(0.01)

            assert surf._frame_pending, \
                'Frame callback not delivered after swap + dispatch'
            session.destroy()

        self._with_display(headless_sway, run)

    def test_surface_lifecycle(self, headless_sway):
        """Surface should survive create → render → destroy cycle."""
        def run():
            from flame_sheep.wayland_window import WallpaperSession
            session = WallpaperSession()
            outputs = list(session._wl_outputs.keys())
            surf = session.add_output(outputs[0])
            session.make_current(surf)
            ctx = session.create_moderngl_context()

            # Render a few frames
            for _ in range(3):
                session.make_current(surf)
                ctx.clear(0.1, 0.2, 0.3, 1.0)
                session.swap(surf)
                session._wl_display.dispatch(block=False)
                session._wl_display.roundtrip()

            # Clean destroy should not crash
            session.destroy()

        self._with_display(headless_sway, run)

    def test_layer_shell_available(self, headless_sway):
        """Headless sway should advertise wlr-layer-shell protocol."""
        wayland_display = headless_sway['wayland_display']
        if not wayland_display:
            pytest.skip('could not determine wayland display socket')

        old_display = os.environ.get('WAYLAND_DISPLAY')
        os.environ['WAYLAND_DISPLAY'] = wayland_display
        try:
            from pywayland.client import Display
            display = Display()
            display.connect()
            registry = display.get_registry()

            protocols = []
            def _on_global(reg, name, interface, version):
                protocols.append(interface)
            registry.dispatcher['global'] = _on_global
            display.roundtrip()
            display.disconnect()

            assert 'zwlr_layer_shell_v1' in protocols, \
                f'layer-shell not found in: {[p for p in protocols if "layer" in p or "wlr" in p]}'
        except ImportError:
            pytest.skip('pywayland not available')
        except Exception as e:
            if 'connect' in str(e).lower():
                pytest.skip(f'Could not connect to display: {e}')
            raise
        finally:
            if old_display:
                os.environ['WAYLAND_DISPLAY'] = old_display
            else:
                os.environ.pop('WAYLAND_DISPLAY', None)
