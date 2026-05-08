"""Genome axis — downbeat/drop events drive genome morphing and swapping."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from collections import deque
from typing import TYPE_CHECKING

from flame_sheep.config import cfg
from flame_sheep.role_mapper import RoleMapper, DOWNBEAT, BACKBEAT
from flame_sheep.loops import loop_sequence, cycle_length
import numpy as np

log = logging.getLogger(__name__)

from flame_sheep_audio import AudioState, BeatEvent
from flame_sheep_audio import HOP_SIZE, SAMPLE_RATE
from flame_sheep_audio.response import MelCentroid, Delta, AsymmetricEnvelope
from flame_sheep.genome import Genome
from flame_sheep.variations import Variation

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore
    from flame_sheep.storage import Library


class GenomeAxis:
    """Downbeat -> genome morph/swap. Manages genome lifecycle, loops, prefetch.

    Responds to event types:
      - downbeat: count toward swap, pulse morph speed
      - drop: force swap + freeze morph for N bars
      - song_start: reset drop/downbeat state

    Continuous features:
      - percussiveness: scales morph speed (low = slow drift, high = snappy)
      - centroid_delta: triggers swaps in low-percussiveness mode

    State:
        current_genome, target_genome: endpoints of the current morph
        morph_t: 0..1 progress through current morph
        morph_speed: how fast morph_t advances per frame
    """

    LOOP_HISTORY_SIZE = 8

    @property
    def DRIFT_MORPH_SPEED(self) -> float:
        return cfg.genome.drift_morph_speed

    @property
    def LOW_MORPH_PULSE(self) -> float:
        return cfg.genome.low_morph_pulse

    @property
    def BREAK_DECAY(self) -> float:
        return cfg.genome.break_decay

    @property
    def DENSITY_MORPH_SCALE(self) -> float:
        return cfg.genome.density_morph_scale

    @property
    def STRONG_BEAT_THRESHOLD(self) -> float:
        return cfg.genome.strong_beat_threshold

    @property
    def DENSITY_DAMPING(self) -> float:
        return cfg.genome.density_damping

    @property
    def CENTROID_SWAP_THRESHOLD(self) -> float:
        return cfg.genome.centroid_swap_threshold

    @property
    def ROTATION_SPEED(self) -> float:
        return cfg.genome.rotation_speed

    @property
    def ROTATION_BEAT_BOOST(self) -> float:
        return cfg.genome.rotation_beat_boost

    @property
    def MIN_GENOME_DISTANCE(self) -> float:
        return cfg.drift.min_genome_distance

    def __init__(self, genome_factory: Callable[[], Genome],
                 role: RoleMapper,
                 lib: Library | None = None,
                 rng: np.random.Generator | None = None,
                 freqs: np.ndarray | None = None) -> None:
        self.enabled = True
        self._genome_factory = genome_factory
        self._role = role
        self._lib = lib
        self.rng = rng or np.random.default_rng()

        # Morph state
        self.current_genome = self._genome_factory()
        self.target_genome = self._genome_factory()
        self.morph_t = 0.0
        self.morph_speed = self.DRIFT_MORPH_SPEED

        # Prefetch
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

        # Downbeat energy tracking (for strong beat detection)
        self._recent_downbeat_energy = 0.5

        # Section change: consumed on next beat to trigger loop swap
        self._section_change_pending = False
        self.SECTION_CHANGE_THRESHOLD = 0.6
        self._section_warmup = 0      # frames since last reset
        self._section_cooldown = 0    # frames since last section swap
        self.SECTION_COOLDOWN = 480   # ~8s at 60fps (~4 bars at 120 BPM)

        # Loop playback
        self._loop_genomes: list[Genome] = []
        self._loop_structure: str = 'cyclic'
        self._loop_sequence = loop_sequence([], 'cyclic')
        self._loop_step: int = 0  # steps since cycle start
        self._loop_cycle_len: int = 0
        self.active_loop_id: int | None = None
        self._loop_history: deque[int] = deque(maxlen=self.LOOP_HISTORY_SIZE)

        # Walker reset flag (consumed by renderer)
        self.needs_walker_reset = False

        # Mel-space centroid tracking (perceptually uniform, fixes HF bias)
        self._mel_centroid = MelCentroid(freqs) if freqs is not None else None
        self._mel_delta = Delta()

        # Affine rotation — continuous spin within each genome (à la Electric Sheep)
        self._rotation_phase = 0.0          # radians, wraps at 2π
        # Beat boost envelope: instant attack, decay over 1 beat
        self._rotation_boost = AsymmetricEnvelope(
            attack=0.001, release=1.0, hop_time=HOP_SIZE / SAMPLE_RATE, unit='beats')

        # Timing
        self._last_downbeat_time = 0.0
        self._break_damping = 1.0

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        self._section_warmup += 1
        self._section_cooldown += 1

        # Compute mel-space centroid delta (perceptually uniform)
        if len(audio.spectrum) > 0 and self._mel_centroid is not None:
            mel_centroid = self._mel_centroid.compute(audio.spectrum)
        else:
            mel_centroid = 0.0
        mel_delta = self._mel_delta.update(mel_centroid)

        # Detect section change — suppress during warmup and cooldown
        WARMUP_FRAMES = 1800  # ~30s at 60fps
        if (
            audio.section_change > self.SECTION_CHANGE_THRESHOLD
            and self._section_warmup > WARMUP_FRAMES
            and self._section_cooldown > self.SECTION_COOLDOWN
        ):
            if not self._section_change_pending:
                self._section_change_pending = True
                log.info(
                    f"[section] change detected ({audio.section_change:.3f}), "
                    f"will swap loop on next strong beat"
                )

        # Handle discrete events
        for event in audio.events:
            if event.kind == self._role.band_for_role(DOWNBEAT):
                self._handle_downbeat(event, clock, audio)
            elif event.kind == "song_start":
                self._handle_song_start()

        # Energy mode: centroid derivative drives morph speed continuously
        # (no discrete swaps — just smooth drift tracking tonal movement)
        if audio.mode == 'energy':
            # Gate centroid by energy — ignore delta when centroid RMS is low
            # (noise floor wandering, consonant bursts)
            if audio.centroid_harmonic_rms > 0.01:
                # mel delta ~0-200 for normal music (vs 0-5000 in Hz space)
                centroid_drive = min(1.0, mel_delta / 100.0)
                self.morph_speed = self.DRIFT_MORPH_SPEED + centroid_drive * 0.003
            else:
                self.morph_speed = self.DRIFT_MORPH_SPEED
        else:
            # Beat mode fallback: low-band density swap on large centroid shift
            low_density = self._role.band_state(audio, DOWNBEAT).onset_density
            # mel delta threshold: ~50 mel ≈ one octave shift at 200 Hz
            if (
                low_density < cfg.genome.centroid_swap_density_gate
                and mel_delta > self.CENTROID_SWAP_THRESHOLD
                and self.morph_t > 0.3
            ):
                self.current_genome = self.current_genome.lerp(
                    self.target_genome, self.morph_t
                )
                self._swap_next_genome()
                self.morph_t = 0.0
                self.morph_speed = 0.02
                log.debug(
                    f"[centroid swap] mel_delta={mel_delta:.1f} "
                    f"low_density={low_density:.2f}"
                )

        # Break damping: exponential slowdown during breaks, symmetric recovery
        if audio.break_intensity > 0:
            self._break_damping *= self.BREAK_DECAY
        else:
            self._break_damping = min(1.0, self._break_damping / self.BREAK_DECAY)

        # Advance morph — speed set by low-band events, damped by breaks
        self.morph_t = min(
            1.0, self.morph_t + self.morph_speed * self._break_damping
        )

        if self.morph_t >= 1.0:
            self.current_genome = self.target_genome
            self.morph_t = 0.0
            self._swap_next_genome()
            self.morph_speed = self.DRIFT_MORPH_SPEED

        # Ramp morph speed toward density-driven baseline
        # Downbeat + backbeat drive morph (rhythm section), subdivision drives zoom
        rhythm_density = (
            self._role.band_state(audio, DOWNBEAT).onset_density
            + self._role.band_state(audio, BACKBEAT).onset_density * 0.5
        )
        density_speed = (
            self.DRIFT_MORPH_SPEED + rhythm_density * self.DENSITY_MORPH_SCALE
        )
        # Blend toward baseline — ramps up after swap, decays down after pulse
        self.morph_speed = 0.95 * self.morph_speed + 0.05 * density_speed

        # Advance affine rotation phase — base speed + beat boost envelope
        boost = self._rotation_boost.update(0.0, bpm=max(audio.effective_bpm, 60.0))
        rotation_speed = self.ROTATION_SPEED + boost * self.ROTATION_BEAT_BOOST
        self._rotation_phase += rotation_speed * self._break_damping
        TWO_PI = 2.0 * np.pi
        if self._rotation_phase >= TWO_PI:
            self._rotation_phase -= TWO_PI

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        frame.genome = self.current_genome.lerp(self.target_genome, self.morph_t)
        # Apply affine rotation for organic movement
        if self._rotation_phase != 0.0:
            frame.genome = frame.genome.rotated(self._rotation_phase)
        # Slowly nudge center toward origin — keeps attractor on screen
        CENTER_PULL = 0.002  # ~1% per frame toward center
        frame.genome.center = frame.genome.center * (1.0 - CENTER_PULL)

    # --- Event handlers ---

    def _handle_downbeat(self, event: BeatEvent, clock: float, audio: AudioState | None = None) -> None:
        since = clock - self._last_downbeat_time
        self._last_downbeat_time = clock

        self._recent_downbeat_energy = self._recent_downbeat_energy * 0.8 + event.energy * 0.2

        downbeat_density = self._role.band_state(audio, DOWNBEAT).onset_density if audio else 0.0
        density_scale = 1.0 / (1.0 + downbeat_density * self.DENSITY_DAMPING)

        # Section change pending → consume on any low-band onset (don't wait for strong beat)
        if self._section_change_pending and self._lib is not None:
            self._section_change_pending = False
            self._section_cooldown = 0  # start cooldown
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t
            )
            self.next_loop()
            self.morph_t = 0.0
            log.info(f"[SWAP+LOOP]  section change consumed on beat")
            return

        # Strong beat → swap direction (harder to trigger at high density)
        swap_thresh = self.STRONG_BEAT_THRESHOLD * (1.0 + downbeat_density * 0.5)
        if event.energy > self._recent_downbeat_energy * swap_thresh:
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self.morph_t
            )
            self._swap_next_genome()
            self.morph_t = 0.0
            dist = self.current_genome.distance(self.target_genome)
            log.debug(
                f"[SWAP]  +{since:.3f}s  energy={event.energy:.2f}  "
                f"avg={self._recent_downbeat_energy:.2f}  dist={dist:.3f}"
            )
        else:
            # Normal downbeat — pulse morph speed and rotation (scaled by density)
            self.morph_speed = min(
                0.15,
                self.morph_speed + event.energy * self.LOW_MORPH_PULSE * density_scale,
            )
            self._rotation_boost.update(event.energy * density_scale,
                                       bpm=max(audio.effective_bpm, 60.0))
            log.debug(f"[downbeat]  +{since:.3f}s  energy={event.energy:.2f}")

    def _handle_song_start(self) -> None:
        """Reset state for new song — swap loop + reset energy tracking."""
        self._last_downbeat_time = 0.0
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        self._section_change_pending = False
        self._section_warmup = 0
        # New song, new loop
        if self._lib is not None and self._lib.loop_count() > 1:
            self.next_loop()
            log.info(f"[song_start] switched to loop #{self.active_loop_id}")

    def accept_handoff(self, genome: Genome, loop_id: int | None = None) -> None:
        """Receive genome from drift mode on transition back to active."""
        self.current_genome = genome
        self.morph_t = 0.0
        self.morph_speed = self.DRIFT_MORPH_SPEED
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        if loop_id is not None and loop_id != self.active_loop_id:
            self.load_loop(loop_id)
        # Prefetch a target so morphing can begin immediately
        # (don't set target = current, that freezes until first beat)
        self._swap_next_genome()

    def force_swap(self) -> None:
        """Immediately swap to a new genome."""
        self.current_genome = self.current_genome.lerp(self.target_genome, self.morph_t)
        self._swap_next_genome()
        self.morph_t = 0.0
        self.morph_speed = 0.15
        self.needs_walker_reset = True
        log.info("force swap")

    # --- Loop management ---

    def _load_top_loop(self) -> None:
        if self._lib is None or self._lib.loop_count() < 1:
            self._loop_genomes = []
            self.active_loop_id = None
            self._prefetch_genome()
            return
        self.next_loop()

    def load_loop(self, loop_id: int) -> None:
        self._start_loop(loop_id)

    def _start_loop(self, loop_id: int) -> None:
        items = self._lib.load_loop(loop_id)
        self._loop_genomes = [genome for _, genome, _ in items]
        self._loop_structure = self._lib.loop_type(loop_id)
        n = len(self._loop_genomes)

        # Create sequence generator and skip to random start
        self._loop_sequence = loop_sequence(self._loop_genomes, self._loop_structure)
        self._loop_cycle_len = cycle_length(n, self._loop_structure)
        start = int(self.rng.integers(0, n))
        # Advance sequence to start position
        for _ in range(start):
            next(self._loop_sequence)
        self._loop_step = start

        self.active_loop_id = loop_id
        self.current_genome = self._loop_genomes[start]
        self.target_genome = next(self._loop_sequence)
        self._loop_step += 1
        self.morph_t = 0.0
        self.morph_speed = 0.05
        self.needs_walker_reset = True
        log.info(f"[loop] loaded #{loop_id} ({self._loop_structure}, "
                 f"{n} genomes, start={start})")
        for i, g in enumerate(self._loop_genomes):
            marker = " <--" if i == start else ""
            log.debug(f"  [{i}] {_describe_genome(g)}{marker}")

    def next_loop(self) -> None:
        if self._lib is None or self._lib.loop_count() < 1:
            return
        top = self._lib.top_loops(n=20)
        candidates = [(lid, info) for lid, info in top if lid not in self._loop_history]
        if not candidates:
            candidates = [
                (lid, info) for lid, info in top if lid != self.active_loop_id
            ]
        if not candidates:
            candidates = top
        fitnesses = np.array([max(info["fitness"], 0.01) for _, info in candidates])
        weights = fitnesses / fitnesses.sum()
        idx = self.rng.choice(len(candidates), p=weights)
        lid, info = candidates[idx]
        self._load_and_track(lid, info["fitness"])

    def _load_and_track(self, loop_id: int, fitness: float | None = None) -> None:
        self._loop_history.append(loop_id)
        self.load_loop(loop_id)
        if fitness is not None:
            log.debug(f"[loop] fitness={fitness:.3f}")

    # --- Genome prefetch ---

    def _prefetch_genome(self) -> None:
        current_snapshot = self.current_genome
        factory = self._genome_factory

        def _gen():
            for _ in range(20):
                g = factory()
                if g.distance(current_snapshot) >= self.MIN_GENOME_DISTANCE:
                    break
            with self._genome_lock:
                self._next_genome = g

        threading.Thread(target=_gen, daemon=True).start()

    def _swap_next_genome(self) -> None:
        if self._loop_genomes:
            self.target_genome = next(self._loop_sequence)
            self._loop_step += 1
            if self._loop_step >= self._loop_cycle_len:
                self._loop_step = 0
                log.debug(f"[loop] cycle complete ({self._loop_structure}, "
                          f"{len(self._loop_genomes)} genomes)")
        else:
            with self._genome_lock:
                if self._next_genome is not None:
                    self.target_genome = self._next_genome
                    self._next_genome = None
                else:
                    self.target_genome = self._genome_factory()
            self._prefetch_genome()
            log.debug(f"[genome] {_describe_genome(self.target_genome)}")


# Reverse map: variation index -> name
_VAR_NAMES = {v: k.lower() for k, v in vars(Variation).items() if isinstance(v, int)}


def _describe_genome(g: Genome) -> str:
    """One-line summary: top variations per transform, sorted by weight."""
    parts = []
    for i, t in enumerate(sorted(g.transforms, key=lambda t: -t.weight)):
        if t.weight < 0.05:
            continue
        active = [
            (w, _VAR_NAMES.get(j, f"v{j}"))
            for j, w in enumerate(t.variations)
            if w > 0.01
        ]
        active.sort(key=lambda x: -x[0])
        names = "+".join(n for _, n in active[:3])
        parts.append(f"{names}({t.weight:.2f})")
    return "  ".join(parts) if parts else "(empty)"
