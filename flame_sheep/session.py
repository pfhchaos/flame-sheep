"""Logind session monitoring for VT switch detection.

Polls the session Active property via logind D-Bus to detect VT switches.
The render loop checks is_active() before any GL calls.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


class SessionMonitor:
    """Monitor logind session active state for VT switch detection."""

    def __init__(self) -> None:
        self._props: Any = None
        self._last_check: float = 0.0
        self._cached_active: bool = True
        self._CHECK_INTERVAL: float = 0.0  # no caching — always poll fresh

        try:
            import dbus
            bus = dbus.SystemBus()
            login1 = bus.get_object('org.freedesktop.login1',
                                     '/org/freedesktop/login1')
            manager = dbus.Interface(login1, 'org.freedesktop.login1.Manager')
            session_path = manager.GetSessionByPID(0)
            session = bus.get_object('org.freedesktop.login1', session_path)
            self._props = dbus.Interface(session,
                                          'org.freedesktop.DBus.Properties')
            log.info(f'[session] monitoring {session_path}')
        except Exception as e:
            log.warning(f'[session] logind monitoring unavailable: {e}')

    def start(self) -> None:
        pass  # no thread needed — we poll

    def stop(self) -> None:
        pass

    @property
    def gpu_paused(self) -> bool:
        """True when the session is inactive (VT switched away)."""
        return not self.is_active()

    def is_active(self) -> bool:
        """Check if the session is active. Cached with 50ms TTL."""
        if self._props is None:
            return True  # can't check, assume active

        now = time.monotonic()
        if now - self._last_check < self._CHECK_INTERVAL:
            return self._cached_active

        self._last_check = now
        try:
            active = bool(self._props.Get(
                'org.freedesktop.login1.Session', 'Active'))
            if active != self._cached_active:
                if active:
                    log.info('[session] GPU resumed (VT switch back)')
                else:
                    log.info('[session] GPU paused (VT switch away)')
            self._cached_active = active
            return active
        except Exception:
            return self._cached_active
