"""MPRIS listener — watches D-Bus for media player events.

Subscribes to PropertiesChanged and Seeked signals from any MPRIS-compatible
player on the session bus. Detects:
  - Track changes → 'song' command
  - Playback paused/stopped → 'pause' command
  - Playback resumed → 'resume' command
  - Seek → 'seek' command (resets tempo/drop state)

Runs a GLib main loop in a daemon thread.
"""

import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class MprisListener:
    """Listen for MPRIS events via D-Bus and write to the control pipe."""

    DEBOUNCE_SEC = 2.0

    def __init__(self, ctl_path: str | Path):
        self._ctl_path = Path(ctl_path)
        self._thread: threading.Thread | None = None
        self._loop = None  # GLib MainLoop

        # State
        self._last_track: str | None = None
        self._last_status: str | None = None
        self._last_track_time: float = 0.0
        self._first_track = True

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='mpris-listener')
        self._thread.start()

    def stop(self):
        if self._loop is not None:
            self._loop.quit()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _write_ctl(self, command: str):
        try:
            if self._ctl_path.exists():
                with open(self._ctl_path, 'w') as f:
                    f.write(command + '\n')
        except OSError as e:
            log.debug(f'MPRIS ctl write failed: {e}')

    def _run(self):
        """Set up D-Bus signal handlers and run the GLib main loop."""
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            from gi.repository import GLib
        except ImportError as e:
            log.warning(f'MPRIS listener disabled — missing dependency: {e}')
            return

        DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()

        # Subscribe to PropertiesChanged on all MPRIS players
        bus.add_signal_receiver(
            self._on_properties_changed,
            signal_name='PropertiesChanged',
            dbus_interface='org.freedesktop.DBus.Properties',
            path='/org/mpris/MediaPlayer2',
        )

        # Subscribe to Seeked signal
        bus.add_signal_receiver(
            self._on_seeked,
            signal_name='Seeked',
            dbus_interface='org.mpris.MediaPlayer2.Player',
            path='/org/mpris/MediaPlayer2',
        )

        log.info('MPRIS listener started (D-Bus)')
        self._loop = GLib.MainLoop()
        self._loop.run()
        log.info('MPRIS listener stopped')

    def _on_properties_changed(self, interface_name, changed, invalidated):
        """Handle PropertiesChanged signal from MPRIS players."""
        if interface_name != 'org.mpris.MediaPlayer2.Player':
            return

        # Playback status change
        if 'PlaybackStatus' in changed:
            status = str(changed['PlaybackStatus'])
            if status != self._last_status:
                prev = self._last_status
                self._last_status = status

                if status in ('Paused', 'Stopped'):
                    self._write_ctl('pause')
                    log.info(f'[MPRIS] {status.lower()}')
                elif status == 'Playing' and prev in ('Paused', 'Stopped', None):
                    self._write_ctl('resume')
                    log.info('[MPRIS] resumed')

        # Metadata change (track)
        if 'Metadata' in changed:
            metadata = changed['Metadata']
            artist_list = metadata.get('xesam:artist', [])
            artist = str(artist_list[0]) if artist_list else ''
            title = str(metadata.get('xesam:title', ''))

            if not artist or not title:
                return

            track_id = f'{artist} - {title}'
            now = time.monotonic()

            if track_id != self._last_track:
                self._last_track = track_id
                if self._first_track:
                    self._first_track = False
                    log.info(f'[MPRIS] initial track: {track_id}')
                elif now - self._last_track_time > self.DEBOUNCE_SEC:
                    self._last_track_time = now
                    self._write_ctl('song')
                    log.info(f'[MPRIS] track: {track_id}')

    def _on_seeked(self, position_us):
        """Handle Seeked signal — user scrubbed to a new position."""
        self._write_ctl('seek')
        log.info(f'[MPRIS] seeked to {int(position_us) / 1e6:.1f}s')
