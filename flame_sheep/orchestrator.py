"""Orchestrator — central coordinator for flame-sheep.

Owns the audio engine, control pipe, MPRIS listener, and session monitor.
Distributes audio state and events to registered consumers (wallpaper,
debug overlay, logger, etc.) without any rendering or visualization logic.

Consumers register for per-consumer event queues and command callbacks.
The orchestrator's tick() updates shared audio state and dispatches
control pipe commands to all registered handlers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from flame_sheep_audio import (
    AudioProcessor, SyntheticAudioProcessor, AudioSnapshot, BeatEvent, DEFAULT_DEVICE,
    BandConfig,
)
from .control import ControlPipe, ControlEvent
from .mpris import MprisListener
from .session import SessionMonitor

log = logging.getLogger(__name__)


@dataclass
class TimestampedEvent:
    """Beat event with reception timestamp for staleness detection."""
    event: BeatEvent
    timestamp: float  # time.perf_counter() when received by orchestrator


class Orchestrator:
    """Central coordinator — owns audio + infrastructure, distributes state.

    Consumers call register() to get an event queue ID, then drain_events()
    each frame. Continuous features are on audio_state (overwritten each tick).
    Command handlers are registered via on_command().
    """

    def __init__(self, audio_device: str | int | None = DEFAULT_DEVICE,
                 test_audio: bool = False,
                 clock: Callable[[], float] | None = None,
                 spectrum_engine: str = 'cqt') -> None:
        self._clock: Callable[[], float] = clock or time.perf_counter

        # Audio engine — try daemon client first, fall back to in-process
        self.audio: AudioProcessor | SyntheticAudioProcessor
        self._using_daemon = False
        if test_audio:
            self.audio = SyntheticAudioProcessor(
                low_interval=0.5,
                mid_interval=1.0,
                high_interval=0.25,
                clock=clock,
            )
        else:
            try:
                from .audio_client import AudioDaemonClient
                self.audio = AudioDaemonClient()
                self._using_daemon = True
                log.info('using audio daemon')
            except Exception:
                engine = None
                if spectrum_engine == 'cqt':
                    try:
                        from flame_sheep_audio._cqt_engine import CqtEngine
                        engine = CqtEngine()
                        log.info('spectrum engine: CQT (rt-cqt SlidingCqt)')
                    except ImportError:
                        log.warning('CQT requested but prtcqt not available, falling back to octave bank')
                if spectrum_engine == 'octave_bank' or engine is None:
                    from flame_sheep_audio._octave_bank import OctaveBankEngine
                    engine = OctaveBankEngine()
                    log.info('spectrum engine: octave bank')
                self.audio = AudioProcessor(device=audio_device, spectrum_engine=engine)
                log.info(f'audio device: {audio_device!r}')

        # Shared audio state (read by consumers, overwritten each tick)
        self.audio_state = AudioSnapshot()

        # Per-consumer event queues
        self._consumers: dict[str, deque[TimestampedEvent]] = {}

        # Command callbacks
        self._command_handlers: dict[str, list[Callable[[ControlEvent], None]]] = {}

        # Infrastructure
        self.control = ControlPipe()
        self.mpris = MprisListener(ctl_path=self.control.pipe_path)
        self.session = SessionMonitor()

    @property
    def band_config(self) -> BandConfig:
        """Expose audio engine's band configuration to consumers."""
        return self.audio._band_config

    def register(self, consumer_id: str) -> str:
        """Register a consumer. Returns the consumer_id for drain_events()."""
        self._consumers[consumer_id] = deque(maxlen=1000)
        return consumer_id

    def on_command(self, command: str, handler: Callable[[ControlEvent], None]) -> None:
        """Register a callback for a control pipe command.

        Multiple handlers per command are allowed — all are called.
        """
        self._command_handlers.setdefault(command, []).append(handler)

    def drain_events(self, consumer_id: str) -> list[TimestampedEvent]:
        """Drain and return pending events for a consumer."""
        q = self._consumers[consumer_id]
        events = list(q)
        q.clear()
        return events

    def tick(self) -> None:
        """Update shared audio state and dispatch events/commands.

        Called by whoever drives the main loop (render loop, test harness).
        """
        # Drain audio engine
        snap = self.audio.drain()
        self.audio_state = snap

        # Distribute beat events to all consumer queues
        now = self._clock()
        for event in snap.events:
            te = TimestampedEvent(event=event, timestamp=now)
            for q in self._consumers.values():
                q.append(te)

        # Process control pipe commands
        for cmd in self.control.poll_all():
            log.info(f'[ctl] {cmd.command} {" ".join(cmd.args)}')
            for handler in self._command_handlers.get(cmd.command, []):
                handler(cmd)

    def start(self) -> None:
        """Start all owned subsystems."""
        self.audio.start()
        self.control.start()
        self.mpris.start()
        self.session.start()
        log.info(f'orchestrator started. Control pipe: {self.control.pipe_path}')

    def stop(self) -> None:
        """Stop all owned subsystems."""
        self.mpris.stop()
        self.session.stop()
        self.control.stop()
        self.audio.stop()
