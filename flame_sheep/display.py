"""Wayland output discovery and wallpaper singleton management.

Functions used by `_run_wallpaper` at startup:
  - _monitor_cfg: read per-monitor config overrides
  - _get_output_layout: query Wayland for multi-monitor geometry
  - _swaymsg_fallback: sway-specific fallback when xdg-output positions fail
  - _ensure_singleton: pidfile-based mutual exclusion (one wallpaper at a time)
"""

from __future__ import annotations

import logging
import os
import time

from .config import cfg

log = logging.getLogger(__name__)


def _monitor_cfg(name: str, key: str, default=None):
    """Read per-monitor config: [monitors.NAME].key or [monitors].default_*."""
    mon = cfg.monitors if hasattr(cfg, 'monitors') else {}
    # Check per-monitor override first
    per = mon.get(name, {}) if isinstance(mon, dict) else getattr(mon, name, None)
    if per is not None:
        if isinstance(per, dict):
            val = per.get(key)
        else:
            val = getattr(per, key, None)
        if val is not None:
            return val
    # Fall back to default
    if isinstance(mon, dict):
        return mon.get(f'default_{key}', mon.get(key, default))
    return getattr(mon, f'default_{key}', getattr(mon, key, default))


def _get_output_layout() -> dict[str, dict]:
    """
    Query Wayland outputs for geometry via wl_output + xdg-output-manager.

    Uses xdg-output-manager-v1 for logical positions (compositor-agnostic),
    falling back to wl_output.geometry and then swaymsg.

    Returns {name: {x, y, w, h, ppi, phys_w_mm, phys_h_mm}}.
    """
    import math
    from pywayland.client import Display
    from pywayland.protocol.wayland import WlOutput

    display = Display()
    display.connect()
    registry = display.get_registry()
    outputs = []
    xdg_output_manager = None

    # Try to bind xdg-output-manager for logical positions
    try:
        from .protocol.xdg_output_unstable_v1.zxdg_output_manager_v1 import ZxdgOutputManagerV1
        _has_xdg_output = True
    except ImportError:
        _has_xdg_output = False

    def _on_global(reg, name, interface, version):
        nonlocal xdg_output_manager
        if interface == WlOutput.name:
            out = reg.bind(name, WlOutput, min(version, 4))
            outputs.append(out)
        elif _has_xdg_output and interface == ZxdgOutputManagerV1.name:
            xdg_output_manager = reg.bind(name, ZxdgOutputManagerV1, min(version, 3))

    registry.dispatcher['global'] = _on_global
    display.roundtrip()

    # Collect output info via events
    output_info = {}

    def _make_handlers(out):
        info = {'name': None, 'x': 0, 'y': 0, 'phys_w_mm': 0, 'phys_h_mm': 0,
                'mode_w': 0, 'mode_h': 0, 'xdg_x': None, 'xdg_y': None}
        output_info[id(out)] = info

        def _on_geometry(output, x, y, phys_w, phys_h, subpixel, make, model, transform):
            info['x'] = x
            info['y'] = y
            info['phys_w_mm'] = phys_w
            info['phys_h_mm'] = phys_h
            info['transform'] = transform

        def _on_mode(output, flags, width, height, refresh):
            if flags & 0x1:  # WL_OUTPUT_MODE_CURRENT
                info['mode_w'] = width
                info['mode_h'] = height

        def _on_name(output, name):
            info['name'] = name

        out.dispatcher['geometry'] = _on_geometry
        out.dispatcher['mode'] = _on_mode
        out.dispatcher['name'] = _on_name

        # If xdg-output-manager is available, get logical position
        if xdg_output_manager is not None:
            xdg_out = xdg_output_manager.get_xdg_output(out)

            def _on_logical_position(xdg_output, x, y):
                info['xdg_x'] = x
                info['xdg_y'] = y

            def _on_logical_size(xdg_output, w, h):
                pass  # we use mode size instead

            def _on_done(xdg_output):
                pass

            xdg_out.dispatcher['logical_position'] = _on_logical_position
            xdg_out.dispatcher['logical_size'] = _on_logical_size
            xdg_out.dispatcher['done'] = _on_done

    for out in outputs:
        _make_handlers(out)

    display.roundtrip()
    display.disconnect()

    result = {}
    for info in output_info.values():
        name = info['name']
        if not name:
            continue
        w = info['mode_w']
        h = info['mode_h']
        if w == 0 or h == 0:
            continue

        # Use xdg-output position if available, else wl_output geometry
        x = info['xdg_x'] if info['xdg_x'] is not None else info['x']
        y = info['xdg_y'] if info['xdg_y'] is not None else info['y']

        # Compute PPI from physical size or fallback
        if info['phys_w_mm'] > 0 and info['phys_h_mm'] > 0:
            phys_w_mm = info['phys_w_mm']
            phys_h_mm = info['phys_h_mm']
            diag_mm = math.sqrt(phys_w_mm**2 + phys_h_mm**2)
            diag_px = math.sqrt(w**2 + h**2)
            ppi = diag_px / (diag_mm / 25.4) if diag_mm > 0 else 96.0
        else:
            diag_px = math.sqrt(w**2 + h**2)
            diag_inches = _monitor_cfg(name, 'diagonal', 27.0)
            ppi = diag_px / diag_inches
            phys_w_mm = w / ppi * 25.4
            phys_h_mm = h / ppi * 25.4

        # Account for rotation: swap physical w/h if transform is 90 or 270
        transform = info.get('transform', 0)
        if transform in (1, 3, 6, 7):  # 90°, 270°, flipped variants
            phys_w_mm, phys_h_mm = phys_h_mm, phys_w_mm
            w, h = h, w  # swap pixel dimensions to match physical

        result[name] = {
            'x': x, 'y': y,
            'w': w, 'h': h,
            'ppi': ppi,
            'phys_w_mm': phys_w_mm,
            'phys_h_mm': phys_h_mm,
        }

    # If xdg-output didn't work and positions are all zero, try swaymsg
    if result and all(g['x'] == 0 and g['y'] == 0 for g in result.values()):
        sway_result = _swaymsg_fallback(result)
        if sway_result:
            return sway_result

    return result


def _swaymsg_fallback(wl_result: dict[str, dict]) -> dict[str, dict]:
    """swaymsg fallback for output positions — sway-specific."""
    import json, subprocess, math
    try:
        raw = subprocess.check_output(['swaymsg', '-t', 'get_outputs'], timeout=3)
        outputs = json.loads(raw)
        result = {}
        for o in outputs:
            if not o.get('active'):
                continue
            r = o['rect']
            name = o['name']
            if wl_result and name in wl_result:
                phys_w_mm = wl_result[name]['phys_w_mm']
                phys_h_mm = wl_result[name]['phys_h_mm']
                ppi = wl_result[name]['ppi']
            else:
                mode = o.get('current_mode', {})
                native_w = mode.get('width', r['width'])
                native_h = mode.get('height', r['height'])
                diag_px = math.sqrt(native_w**2 + native_h**2)
                diag_inches = _monitor_cfg(name, 'diagonal', 27.0)
                ppi = diag_px / diag_inches
                phys_w_mm = r['width'] / ppi * 25.4
                phys_h_mm = r['height'] / ppi * 25.4
            result[name] = {
                'x': r['x'], 'y': r['y'],
                'w': r['width'], 'h': r['height'],
                'ppi': ppi,
                'phys_w_mm': phys_w_mm,
                'phys_h_mm': phys_h_mm,
            }
        return result
    except Exception as e:
        log.error(f'swaymsg fallback failed: {e}')
        return {}


def _ensure_singleton() -> None:
    """Kill any existing flame-sheep wallpaper instance.

    Uses a pidfile at ~/.local/share/flame-sheep/pid. If a previous
    instance is still running, SIGTERM it and wait briefly for cleanup.
    """
    import signal
    pid_path = os.path.expanduser('~/.local/share/flame-sheep/pid')
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    # Check for existing instance
    try:
        with open(pid_path, 'r') as f:
            old_pid = int(f.read().strip())
        # Check if it's actually running
        os.kill(old_pid, 0)
        if old_pid == os.getpid():
            pass  # that's us (re-exec after VT switch)
        else:
            log.info(f'killing previous instance (pid {old_pid})')
            os.kill(old_pid, signal.SIGTERM)
        # Wait briefly for it to die
        for _ in range(20):
            time.sleep(0.1)
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                break
    except (FileNotFoundError, ValueError, ProcessLookupError):
        pass  # no previous instance

    # Write our PID
    with open(pid_path, 'w') as f:
        f.write(str(os.getpid()))
