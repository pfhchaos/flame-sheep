"""Audio daemon client — reads shared memory + dbus for audio analysis.

Drop-in replacement for AudioProcessor in the orchestrator. Presents
the same start()/stop()/drain() interface so the orchestrator doesn't
need to know which backend is active.

Usage:
    try:
        audio = AudioDaemonClient()
        audio.start()  # no-op, daemon is already running
    except AudioDaemonUnavailable:
        audio = AudioProcessor(...)  # fall back to in-process
"""

from __future__ import annotations

import json
import logging
import mmap

import numpy as np

from flame_sheep_audio._types import AudioSnapshot
from flame_sheep_audio.shm_layout import (
    SHM_NAME, ShmLayout, ShmReader, compute_layout,
)
from flame_sheep_audio.dbus_service import BUS_NAME, OBJ_PATH, IFACE

log = logging.getLogger(__name__)


class AudioDaemonUnavailable(Exception):
    """Raised when the audio daemon is not running."""
    pass


class AudioDaemonClient:
    """Consumer client for the audio daemon.

    Connects to the daemon's shared memory and dbus service.
    Provides drain() returning AudioSnapshot, same as AudioProcessor.
    """

    def __init__(self):
        # Connect to dbus and get schema.
        # Set up the GLib main loop BEFORE creating the SessionBus singleton
        # so MPRIS can reuse it later with signal receivers attached.
        try:
            import dbus
            from dbus.mainloop.glib import DBusGMainLoop
            DBusGMainLoop(set_as_default=True)
            bus = dbus.SessionBus()
            proxy = bus.get_object(BUS_NAME, OBJ_PATH)
            iface = dbus.Interface(proxy, IFACE)
            schema_json = str(iface.GetSchema())
            shm_name = str(iface.GetShmName())
        except Exception as e:
            raise AudioDaemonUnavailable(f'Cannot connect to audio daemon: {e}')

        schema = json.loads(schema_json)
        self._schema = schema

        # Connect to shared memory (raw /dev/shm to avoid resource tracker)
        try:
            import os
            from flame_sheep_audio.shm_layout import SHM_SIZE
            shm_path = f'/dev/shm/{shm_name}'
            fd = os.open(shm_path, os.O_RDONLY)
            self._mmap = mmap.mmap(fd, SHM_SIZE, access=mmap.ACCESS_READ)
            os.close(fd)
        except (FileNotFoundError, OSError):
            raise AudioDaemonUnavailable(f'Shared memory {shm_name!r} not found')

        # Build layout from schema
        n_bins = schema['n_bins']
        band_names = schema['band_names']
        self._layout = compute_layout(n_bins, band_names)
        self._reader = ShmReader(self._layout, self._mmap)

        # Subscribe to dbus signals for song_start events
        # (these come from MPRIS via the daemon, not from shmem)
        self._pending_song_starts: list[tuple[str, str]] = []
        try:
            bus.add_signal_receiver(
                self._on_song_start,
                signal_name='SongStart',
                dbus_interface=IFACE,
                bus_name=BUS_NAME,
            )
        except Exception:
            pass  # dbus signals are optional backup

        # Expose band config for orchestrator.band_config property
        from flame_sheep_audio._band_config import default_band_config
        self._band_config = default_band_config()

        # Derive bin frequencies from engine type + bin count
        # CQT: 9 octaves × 12 bins, starting at ~32Hz
        # FFT: linear spacing 0..24000Hz
        if n_bins == 108:  # CQT default
            try:
                from flame_sheep_audio._cqt_engine import CqtEngine
                engine = CqtEngine()
                self._bin_freqs = engine.bin_centers
            except ImportError:
                self._bin_freqs = np.linspace(20, 20000, n_bins).astype(np.float32)
        else:
            from flame_sheep_audio._constants import FREQS
            self._bin_freqs = FREQS[:n_bins] if len(FREQS) >= n_bins else np.linspace(20, 20000, n_bins).astype(np.float32)

        log.info('Connected to audio daemon (shm=%s, %d bins, %d bands)',
                 shm_name, n_bins, len(band_names))

    def start(self) -> None:
        """No-op — daemon is already running."""
        pass

    def stop(self) -> None:
        """Disconnect from shared memory. Does NOT stop the daemon."""
        try:
            self._mmap.close()
        except Exception:
            pass

    def drain(self) -> AudioSnapshot:
        """Read current audio state from shared memory.

        Returns AudioSnapshot with events from the ring buffer.
        """
        snap = self._reader.read_snapshot()

        # Inject any song_start events from dbus
        from flame_sheep_audio._types import BeatEvent
        for title, artist in self._pending_song_starts:
            snap.events.append(BeatEvent(kind='song_start', energy=0.0))
        self._pending_song_starts.clear()

        return snap

    @property
    def bin_freqs(self) -> np.ndarray:
        return self._bin_freqs

    def reset_tempo(self) -> None:
        """Request tempo reset via dbus (if supported)."""
        pass  # TODO: add ResetTempo dbus method to daemon

    def _on_song_start(self, title: str, artist: str) -> None:
        self._pending_song_starts.append((title, artist))
