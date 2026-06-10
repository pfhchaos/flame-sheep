"""Audio daemon client — reads shared memory + dbus for audio analysis.

The wallpaper's sole production audio path (as of 2026-06-09 — the
in-process AudioProcessor fallback was removed once auto-reconnect
landed).

Survives daemon restarts AND startup race conditions: subscribes to
`NameOwnerChanged` on the session bus and rebuilds its shm + signal
subscriptions whenever the daemon respawns under the well-known name.
During any gap (daemon not yet started, daemon being restarted),
`drain()` returns an empty snapshot so the render loop keeps ticking
at idle.

Initialization raises only on session-bus-unreachable — a hard failure
that nothing in the wallpaper can recover from anyway.
"""

from __future__ import annotations

import json
import logging
import mmap
import threading

import numpy as np

from flame_sheep_audio import AudioSnapshot
from flame_sheep_audio.shm_layout import (
    SHM_NAME, ShmLayout, ShmReader, compute_layout,
)
from flame_sheep_audio.dbus_service import BUS_NAME, OBJ_PATH, IFACE

log = logging.getLogger(__name__)


class SessionBusUnavailable(Exception):
    """Raised when the dbus session bus itself cannot be reached.

    This is a fundamental failure — the wallpaper has no fallback path
    for it (the in-process AudioProcessor fallback was removed when
    daemon auto-reconnect landed). Caller should treat this as fatal
    or surface a clear error to the user.
    """
    pass


class AudioDaemonClient:
    """Consumer client for the audio daemon.

    Connects to the daemon's shared memory and dbus service.
    Provides drain() returning AudioSnapshot, same as AudioProcessor.

    Automatically reconnects when the daemon respawns. State is
    protected by an internal lock so the dbus mainloop thread can
    swap the mmap while the render thread calls drain().
    """

    def __init__(self):
        # State protected by self._lock for cross-thread safety
        # (dbus mainloop swaps mmap; render thread reads).
        self._lock = threading.Lock()
        self._mmap: mmap.mmap | None = None
        self._reader: ShmReader | None = None
        self._layout: ShmLayout | None = None
        self._schema: dict | None = None
        self._n_bins: int = 108
        self._band_names: list[str] = []
        self._bin_freqs: np.ndarray = np.zeros(0, dtype=np.float32)
        # SongStart events delivered via dbus, drained on next read.
        self._pending_song_starts: list[tuple[str, str]] = []

        # Set up GLib main loop before SessionBus so MPRIS can reuse it
        # later with signal receivers attached.
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            DBusGMainLoop(set_as_default=True)
            self._bus = dbus.SessionBus()
        except Exception as e:
            raise SessionBusUnavailable(f'Cannot reach session bus: {e}')

        # Watch the daemon's well-known name. NameOwnerChanged fires
        # whenever the owner of BUS_NAME shifts — daemon dies, daemon
        # restarts, etc. We rebuild on each transition.
        try:
            self._bus.add_signal_receiver(
                self._on_name_owner_changed,
                signal_name='NameOwnerChanged',
                dbus_interface='org.freedesktop.DBus',
                arg0=BUS_NAME,
            )
        except Exception as e:
            # Without this signal, reconnect is impossible — keep going
            # so initial connect can still succeed, but log loudly.
            log.warning('NameOwnerChanged subscription failed (%s); '
                        'wallpaper will need a manual restart if the '
                        'audio daemon respawns', e)

        # Initial connection attempt. If the daemon isn't running yet,
        # don't raise — sit in disconnected state and wait for the
        # NameOwnerChanged callback to fire when it shows up. drain()
        # returns empty snapshots until then; render loop idles cleanly.
        if not self._connect():
            log.info('audio daemon not running yet; will connect when '
                     'it appears on dbus (%s)', BUS_NAME)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> bool:
        """Resolve daemon, mmap its shm, subscribe to SongStart. Returns
        True on success, False if the daemon isn't there. Idempotent —
        tears down any existing connection first."""
        self._disconnect()
        try:
            import dbus
            proxy = self._bus.get_object(BUS_NAME, OBJ_PATH)
            iface = dbus.Interface(proxy, IFACE)
            schema_json = str(iface.GetSchema())
            shm_name = str(iface.GetShmName())
        except Exception as e:
            log.debug('audio daemon not reachable via dbus: %s', e)
            return False

        schema = json.loads(schema_json)

        try:
            import os
            from flame_sheep_audio.shm_layout import SHM_SIZE
            shm_path = f'/dev/shm/{shm_name}'
            fd = os.open(shm_path, os.O_RDONLY)
            new_mmap = mmap.mmap(fd, SHM_SIZE, access=mmap.ACCESS_READ)
            os.close(fd)
        except (FileNotFoundError, OSError) as e:
            log.warning('daemon dbus reachable but shm %r missing: %s',
                        shm_name, e)
            return False

        n_bins = schema['n_bins']
        band_names = schema['band_names']
        layout = compute_layout(n_bins, band_names)
        reader = ShmReader(layout, new_mmap)

        # Derive bin frequencies from engine type + bin count.
        if n_bins == 108:  # CQT default
            try:
                from flame_sheep_audio import CqtEngine
                bin_freqs = CqtEngine().bin_centers
            except ImportError:
                bin_freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        else:
            from flame_sheep_audio import FREQS
            bin_freqs = (FREQS[:n_bins] if len(FREQS) >= n_bins
                         else np.linspace(20, 20000, n_bins).astype(np.float32))

        # Re-subscribe to SongStart — the previous subscription died
        # with the old daemon instance.
        try:
            self._bus.add_signal_receiver(
                self._on_song_start,
                signal_name='SongStart',
                dbus_interface=IFACE,
                bus_name=BUS_NAME,
            )
        except Exception:
            pass  # song-start events are optional backup

        with self._lock:
            self._schema = schema
            self._n_bins = n_bins
            self._band_names = band_names
            self._layout = layout
            self._mmap = new_mmap
            self._reader = reader
            self._bin_freqs = bin_freqs

        # Expose band config for orchestrator.band_config property.
        # Done outside the lock; band_config doesn't change across
        # daemon restarts (it's daemon-side config).
        from flame_sheep_audio import default_band_config
        self._band_config = default_band_config()

        log.info('Connected to audio daemon (shm=%s, %d bins, %d bands)',
                 shm_name, n_bins, len(band_names))
        return True

    def _disconnect(self) -> None:
        """Tear down the shm mapping. Safe to call when not connected."""
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.close()
                except Exception:
                    pass
            self._mmap = None
            self._reader = None
            self._layout = None

    def _on_name_owner_changed(self, name: str, old_owner: str,
                                new_owner: str) -> None:
        """Daemon's well-known name changed owners.

        - new_owner empty: daemon died. Tear down.
        - new_owner non-empty: daemon back. Reconnect.
        """
        if str(name) != BUS_NAME:
            return
        if not str(new_owner):
            log.info('Audio daemon disappeared from dbus (%s); '
                     'render will idle until reconnect', BUS_NAME)
            self._disconnect()
        else:
            log.info('Audio daemon back on dbus (%s, new owner=%s); '
                     'reconnecting', BUS_NAME, new_owner)
            if not self._connect():
                # Race: name was owned briefly but service not ready.
                # Will retry on the next NameOwnerChanged (or stay
                # disconnected if the daemon dies again).
                log.warning('reconnect raced; staying disconnected '
                             'until next NameOwnerChanged')

    # ------------------------------------------------------------------
    # Public API (same shape as AudioProcessor)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """No-op — daemon is already running (or will reconnect)."""
        pass

    def stop(self) -> None:
        """Disconnect from shared memory. Does NOT stop the daemon."""
        self._disconnect()

    def drain(self) -> AudioSnapshot:
        """Read current audio state from shared memory.

        Returns an empty AudioSnapshot if the daemon is currently gone
        — the render loop keeps ticking at idle until reconnect lands.
        """
        with self._lock:
            reader = self._reader
        if reader is None:
            return self._empty_snapshot()

        try:
            snap = reader.read_snapshot()
        except (BufferError, ValueError, OSError):
            # mmap got invalidated mid-read (rare; daemon respawned
            # between the lock acquire and the read). Return empty
            # this tick; the NameOwnerChanged callback will rebuild.
            return self._empty_snapshot()

        # Inject any song_start events from dbus
        from flame_sheep_audio import BeatEvent
        for _title, _artist in self._pending_song_starts:
            snap.events.append(BeatEvent(kind='song_start', energy=0.0))
        self._pending_song_starts.clear()
        return snap

    def _empty_snapshot(self) -> AudioSnapshot:
        """Idle-state AudioSnapshot used while disconnected."""
        return AudioSnapshot(
            spectrum=np.zeros(self._n_bins, dtype=np.float32),
            waveform=np.zeros(0, dtype=np.float32),
            stability=np.zeros(self._n_bins, dtype=np.float32),
            sustained=np.zeros(self._n_bins, dtype=np.float32))

    @property
    def bin_freqs(self) -> np.ndarray:
        return self._bin_freqs

    def song_started(self) -> None:
        """Signal new song to daemon. TODO: forward via dbus."""
        pass

    def reset_bands(self) -> None:
        """Reset adaptive bands. TODO: forward via dbus."""
        pass

    def hint_tempo(self, bpm: float) -> None:
        """Provide tempo hint. TODO: forward via dbus."""
        pass

    def reset_tempo(self) -> None:
        """Reset tempo state. TODO: forward via dbus."""
        pass

    def _on_song_start(self, title: str, artist: str) -> None:
        self._pending_song_starts.append((title, artist))
