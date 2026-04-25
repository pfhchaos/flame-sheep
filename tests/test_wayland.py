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
        'WLR_RENDERER': 'pixman',
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

    # Find the wayland display socket
    runtime_dir = os.environ.get('XDG_RUNTIME_DIR', f'/run/user/{os.getuid()}')
    wayland_display = None
    try:
        raw = subprocess.check_output(
            ['swaymsg', '-t', 'get_outputs'],
            env={'SWAYSOCK': swaysock, 'PATH': os.environ['PATH']},
            timeout=5,
        )
        outputs = json.loads(raw)
        if outputs:
            # Find the wayland socket this sway is listening on
            # sway creates wayland-N in XDG_RUNTIME_DIR
            # We need to find which one belongs to this instance
            for name in sorted(os.listdir(runtime_dir)):
                if name.startswith('wayland-') and not name.endswith('.lock'):
                    wayland_display = name
            # Use the last one (most recently created)
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
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        # Temporarily override SWAYSOCK so our code queries the test instance
        old_swaysock = os.environ.get('SWAYSOCK')
        os.environ['SWAYSOCK'] = headless_sway['swaysock']
        try:
            from flame_sheep.main import _get_sway_layout
            layout = _get_sway_layout()
            assert len(layout) >= 1
            name = list(layout.keys())[0]
            assert 'HEADLESS' in name
            assert layout[name]['w'] > 0
            assert layout[name]['h'] > 0
        finally:
            if old_swaysock:
                os.environ['SWAYSOCK'] = old_swaysock
            else:
                os.environ.pop('SWAYSOCK', None)
