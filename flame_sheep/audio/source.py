"""Signal sources — abstract audio input from capture method.

PipeWireSource: real audio from sounddevice/PipeWire.
FeedSource: manual PCM injection for testing.
"""

import threading
import numpy as np
from collections import deque
from typing import Protocol

import sounddevice as sd

from ._constants import SAMPLE_RATE, BLOCK_SIZE, FFT_SIZE


class SignalSource(Protocol):
    """Protocol for audio signal sources."""

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> np.ndarray | None:
        """Return FFT_SIZE samples if available, else None."""
        ...

    def feed(self, pcm: np.ndarray) -> None:
        """Push PCM samples (for test sources). May be a no-op."""
        ...


class PipeWireSource:
    """Real audio capture via sounddevice (PipeWire/PulseAudio/ALSA)."""

    def __init__(self, device: str | int | None = None):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=FFT_SIZE)
        self._new_audio = False

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time, status):
        with self._lock:
            self._buffer.extend(indata[:, 0])
            self._new_audio = True

    def start(self):
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()

    def read(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buffer) < FFT_SIZE or not self._new_audio:
                return None
            self._new_audio = False
            return np.array(self._buffer, dtype=np.float32)

    def feed(self, pcm: np.ndarray):
        """Push PCM samples directly (bypass sounddevice)."""
        with self._lock:
            self._buffer.extend(pcm)
            self._new_audio = True


class FeedSource:
    """Test signal source — accepts PCM via feed(), no audio hardware."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=FFT_SIZE)
        self._new_audio = False

    def start(self): pass
    def stop(self): pass

    def read(self) -> np.ndarray | None:
        with self._lock:
            if len(self._buffer) < FFT_SIZE or not self._new_audio:
                return None
            self._new_audio = False
            return np.array(self._buffer, dtype=np.float32)

    def feed(self, pcm: np.ndarray):
        with self._lock:
            self._buffer.extend(pcm)
            self._new_audio = True
