"""flame-sheep visualization state machine.

FlameSheepCore owns the visual axes (genome, palette, zoom, brightness,
detail) and produces a FrameState per tick. It consumes audio via the
Orchestrator. It does NOT hold a renderer or GL context — that lives in
each window/wallpaper surface.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import cfg
from .genome import Genome
from flame_sheep_audio import BeatEvent, AudioState
from flame_sheep_audio.mode import Mode
from .orchestrator import Orchestrator
from .axes.zoom_axis import ZoomAxis
from .axes.brightness_axis import BrightnessAxis
from .axes.detail_axis import DetailAxis
from .axes.genome_axis import GenomeAxis
from .axes.palette_axis import PaletteAxis
from .role_mapper import RoleMapper

if TYPE_CHECKING:
    from .storage import Library

log = logging.getLogger(__name__)


class FlameSheepCore:
    """
    Visualization state — axes, drift mode, genome morphing.
    Consumes an Orchestrator for audio state and events.
    Does NOT hold a renderer or GL context — that lives in each window.

    Each frame, call:
      genome, spectrum = core.tick(frame_time)   # advance state, get current genome
    Then render with those values in your per-window renderer.

    Call .force_genome_swap() to kick a degenerate image.
    """

    def __init__(self, orchestrator: Orchestrator,
                 lib: Library | None = None,
                 clock: Callable[[], float] | None = None,
                 genome_factory: Callable[[], Genome] | None = None) -> None:
        self._orch = orchestrator
        self._consumer_id: str = orchestrator.register('wallpaper')
        self._clock: Callable[[], float] = clock or time.perf_counter
        self.rng: np.random.Generator = np.random.default_rng()
        self._lib = lib

        # --- Visual axes ---
        if genome_factory:
            factory = genome_factory
        elif lib is not None and lib.genome_count() > 0:
            # Pre-load top genomes — no DB access from prefetch thread
            _pool = [lib.load_genome(gid) for gid, _ in lib.top_genomes(n=50)]
            factory = lambda: _pool[int(self.rng.integers(0, len(_pool)))]
        else:
            factory = lambda: Genome.random(self.rng)
        # Role mapping (band names → visual roles)
        role_cfg = cfg.roles
        role_map = {k: getattr(role_cfg, k) for k in ('downbeat', 'backbeat', 'subdivision', 'energy')}
        self._role = RoleMapper(role_map)

        bin_freqs = orchestrator.audio.bin_freqs
        self._genome_axis = GenomeAxis(genome_factory=factory, role=self._role, lib=lib, rng=self.rng)
        self._palette_axis = PaletteAxis(
            initial_palette=self._genome_axis.current_genome.palette,
            role=self._role, lib=lib, rng=self.rng, freqs=bin_freqs)
        self._zoom_axis = ZoomAxis(role=self._role)
        self._brightness_axis = BrightnessAxis(role=self._role)
        self._detail_axis = DetailAxis(role=self._role)
        self._pending_song_start = False

    @dataclass
    class FrameState:
        """Per-frame output from tick() — three orthogonal axes + audio."""
        genome: 'Genome'          # low axis: transforms, zoom, rotation, center (includes zoom pulse)
        palette: np.ndarray       # mid axis: 256x3 float32 color palette
        spectrum: np.ndarray      # raw FFT spectrum for audio-reactive tonemap
        brightness: float         # RMS-driven brightness for tonemap
        iterations: int           # chaos game iterations — scales with bass energy

    def tick(self, frame_time: float) -> 'FlameSheepCore.FrameState':
        """Advance visualization one frame.
        Returns FrameState with three independent axes.
        """
        snap = self._orch.audio_state
        timestamped = self._orch.drain_events(self._consumer_id)
        now  = self._clock()

        # Collect beat events from timestamped wrappers
        events = [te.event for te in timestamped]
        if self._pending_song_start:
            events.insert(0, BeatEvent(kind='song_start', energy=0.0))
            self._pending_song_start = False

        # Build AudioState for axes
        audio = AudioState(
            events=events,
            spectrum=snap.spectrum,
            bands=snap.bands,
            centroid=snap.centroid,
            centroid_delta=snap.centroid_delta,
            centroid_rms=snap.centroid_rms,
            centroid_harmonic_rms=snap.centroid_harmonic_rms,
            slow_centroid_harmonic_rms=snap.slow_centroid_harmonic_rms,
            percussiveness=snap.percussiveness,
            spectral_novelty=snap.spectral_novelty,
            section_change=snap.section_change,
            bpm=snap.bpm,
            effective_bpm=snap.effective_bpm,
            tempo_confidence=snap.tempo_confidence,
            tempo_saturated=snap.tempo_saturated,
            break_intensity=snap.break_intensity,
            mode=snap.mode,
        )

        # Cross-band inhibition: when multiple bands fire simultaneously,
        # suppress weaker events that are likely bleed from the dominant hit.
        # Events within 60% of the strongest are kept (independent hits).
        if len(audio.events) > 1:
            best_energy = max(e.energy for e in audio.events)
            audio.events = [e for e in audio.events
                            if e.energy > best_energy * 0.6]

        current_mode = Mode(snap.mode)

        # Tick all axes — genome_axis handles mode internally
        self._brightness_axis.tick(audio, frame_time, now)
        self._detail_axis.tick(audio, frame_time, now)
        self._genome_axis.tick(audio, frame_time, now)

        if current_mode == Mode.BEAT:
            self._palette_axis.tick(audio, frame_time, now)
            self._zoom_axis.tick(audio, frame_time, now)

        # Assemble frame state
        frame = self.FrameState(
            genome=self.current_genome,  # overwritten by contribute
            palette=self._palette_axis.palette_current,  # overwritten by contribute
            spectrum=snap.spectrum,
            brightness=self._brightness_axis.brightness,
            iterations=self._detail_axis.iterations,
        )

        self._genome_axis.contribute(frame)
        self._palette_axis.contribute(frame)
        self._zoom_axis.contribute(frame)

        return frame

    # --- Backward-compat properties delegating to axes ---

    @property
    def palette_t(self):
        return self._palette_axis.palette_t

    @property
    def palette_target(self):
        return self._palette_axis.palette_target

    @property
    def zoom_boost(self):
        return self._zoom_axis.zoom_boost

    @zoom_boost.setter
    def zoom_boost(self, value):
        self._zoom_axis.zoom_boost = value

    @property
    def current_genome(self):
        return self._genome_axis.current_genome

    @current_genome.setter
    def current_genome(self, value):
        self._genome_axis.current_genome = value

    @property
    def target_genome(self):
        return self._genome_axis.target_genome

    @target_genome.setter
    def target_genome(self, value):
        self._genome_axis.target_genome = value

    @property
    def morph_t(self):
        return self._genome_axis.morph_t

    @morph_t.setter
    def morph_t(self, value):
        self._genome_axis.morph_t = value

    @property
    def morph_speed(self):
        return self._genome_axis.morph_speed

    @morph_speed.setter
    def morph_speed(self, value):
        self._genome_axis.morph_speed = value

    @property
    def needs_walker_reset(self):
        return self._genome_axis.needs_walker_reset

    @needs_walker_reset.setter
    def needs_walker_reset(self, value):
        self._genome_axis.needs_walker_reset = value

    @property
    def active_loop_id(self):
        return self._genome_axis.active_loop_id

    @property
    def active_genome_db_id(self) -> int | None:
        """DB ID of the genome currently dominant on screen."""
        ga = self._genome_axis
        if ga.morph_t < 0.5:
            g = ga.current_genome
        else:
            g = ga.target_genome
        return g.db_id if g is not None else None

    def force_genome_swap(self) -> None:
        """Immediately swap to a new genome — call when image looks degenerate."""
        self._genome_axis.force_swap()

    def load_loop(self, loop_id: int) -> None:
        self._genome_axis.load_loop(loop_id)

    def next_loop(self) -> None:
        self._genome_axis.next_loop()

    def user_next(self) -> None:
        self._genome_axis.user_next()

    def song_started(self) -> None:
        """Signal new song started — resets tempo, bands, drop detectors, mode.
        Injects a song_start event on the next tick via _pending_song_start.
        """
        self._orch.audio.song_started()
        self._orch.audio.reset_bands()
        self._pending_song_start = True
        log.info('[song] reset tempo, bands, drop detectors, mode')

    def hint_tempo(self, bpm: float) -> None:
        """Provide tempo hint from external source."""
        self._orch.audio.hint_tempo(bpm)
        log.info(f'[tempo] hint: {bpm:.1f} BPM')
