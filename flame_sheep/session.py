"""Logind session monitoring for VT switch detection.

Listens to org.freedesktop.login1.Session on the system bus for
PauseDevice/ResumeDevice signals. When the GPU device is paused
(VT switch away), sets a flag the render loop can check.
"""

import logging
import threading

log = logging.getLogger(__name__)


class SessionMonitor:
    """Monitor logind session for GPU device pause/resume (VT switch)."""

    def __init__(self):
        self.gpu_paused = False
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name='session-monitor')
        self._thread.start()

    def stop(self):
        self._running = False

    def _listen(self):
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            from gi.repository import GLib

            DBusGMainLoop(set_as_default=True)
            bus = dbus.SystemBus()

            # Find our session path
            login1 = bus.get_object(
                'org.freedesktop.login1',
                '/org/freedesktop/login1')
            manager = dbus.Interface(login1, 'org.freedesktop.login1.Manager')

            # Get session for our PID
            session_path = manager.GetSessionByPID(0)  # 0 = current process
            session = bus.get_object('org.freedesktop.login1', session_path)

            def on_pause(major, minor, pause_type):
                # DRM devices are major 226
                if major == 226:
                    self.gpu_paused = True
                    log.info(f'[session] GPU paused (VT switch away)')

            def on_resume(major, minor, fd):
                if major == 226:
                    self.gpu_paused = False
                    log.info(f'[session] GPU resumed (VT switch back)')

            session.connect_to_signal('PauseDevice', on_pause,
                                       dbus_interface='org.freedesktop.login1.Session')
            session.connect_to_signal('ResumeDevice', on_resume,
                                       dbus_interface='org.freedesktop.login1.Session')

            log.info(f'[session] monitoring {session_path} for VT switches')

            loop = GLib.MainLoop()
            while self._running:
                ctx = loop.get_context()
                ctx.iteration(True)  # blocking, handles signals

        except Exception as e:
            log.warning(f'[session] logind monitoring unavailable: {e}')
