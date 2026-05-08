"""Genome axis — mode-aware genome morphing with loop sequencing.

Two-state morph cycle:
  MORPHING: morph_t advances at fixed speed (cfg.drift.morph_speed).
            On completion → swap genome → DWELL.
  DWELL:    waiting for a mode-specific trigger to start next morph:
            - Beat:   next beat event releases dwell
            - Energy: centroid shift releases dwell (initially same as idle)
            - Idle:   fixed duration timeout releases dwell

The mode (idle/energy/beat) is read from AudioState each frame. Mode
changes are invisible in the visual output — same genome, same morph
position, only what additional behaviors are active changes.

Beat mode adds: beat event processing, rotation boost, section change.
Energy mode adds: centroid shift response (future refinement target).
Idle mode adds: loop cycling via CYCLES_PER_LOOP counter.
"""

from __future__ import annotations

import enum
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
from flame_sheep_audio.mode import Mode
from flame_sheep_audio.response import MelCentroid, Delta, AsymmetricEnvelope
from flame_sheep.genome import Genome
from flame_sheep.variations import Variation

if TYPE_CHECKING:
    from flame_sheep.main import FlameSheepCore
    from flame_sheep.storage import Library


class _MorphState(enum.Enum):
    MORPHING = 'morphing'
    DWELL = 'dwell'


class GenomeAxis:
    """Genome axis — manages genome lifecycle across all three modes.

    Morph cycle: MORPHING → DWELL → MORPHING (repeat)
    Morph speed is fixed across all modes. The dwell trigger varies by mode.
    """

    LOOP_HISTORY_SIZE = 8

    @property
    def BREAK_DECAY(self) -> float:
        return cfg.genome.break_decay

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
    def MORPH_SPEED(self) -> float:
        return cfg.drift.morph_speed

    @property
    def MIN_GENOME_DISTANCE(self) -> float:
        return cfg.drift.min_genome_distance

    @property
    def CYCLES_PER_LOOP(self) -> int:
        return cfg.drift.cycles_per_loop

    @property
    def DWELL_BEATS(self) -> int:
        return cfg.genome.dwell_beats

    def __init__(self, genome_factory: Callable[[], Genome],
                 role: RoleMapper,
                 lib: Library | None = None,
                 rng: np.random.Generator | None = None) -> None:
        self.enabled = True
        self._genome_factory = genome_factory
        self._role = role
        self._lib = lib
        self.rng = rng or np.random.default_rng()

        # --- Mode ---
        self._mode = Mode.IDLE
        self._playback_paused = False

        # --- Genome state ---
        self.current_genome = self._genome_factory()
        self.target_genome = self._genome_factory()

        # --- Morph state (two-state: MORPHING / DWELL) ---
        self._morph_state = _MorphState.DWELL
        self._morph_t = 0.0
        self._dwell_start: float = 0.0

        # --- Prefetch ---
        self._next_genome: Genome | None = None
        self._genome_lock = threading.Lock()

        # --- Beat mode state ---
        self._recent_downbeat_energy = 0.5
        self._section_change_pending = False
        self.SECTION_CHANGE_THRESHOLD = 0.6
        self._section_warmup = 0
        self._section_cooldown = 0
        self.SECTION_COOLDOWN = 480
        self._break_damping = 1.0

        # --- Loop playback ---
        self._loop_genomes: list[Genome] = []
        self._loop_structure: str = 'cyclic'
        self._loop_sequence = loop_sequence([], 'cyclic')
        self._loop_step: int = 0
        self._loop_cycle_len: int = 0
        self.active_loop_id: int | None = None
        self._loop_history: deque[int] = deque(maxlen=self.LOOP_HISTORY_SIZE)

        # --- Idle loop cycling ---
        self._idle_loop_cycles = 0

        # Walker reset flag (consumed by renderer)
        self.needs_walker_reset = False

        # --- Mel-space centroid tracking (lazy init on first spectrum) ---
        self._mel_centroid: MelCentroid | None = None
        self._mel_delta = Delta()

        # --- Rotation ---
        _hop_time = HOP_SIZE / SAMPLE_RATE
        self._rotation_boost = AsymmetricEnvelope(
            attack=0.001, release=1.0, hop_time=_hop_time, unit='beats')
        self._rotation_phase = 0.0

        # --- Timing ---
        self._last_downbeat_time = 0.0

        if lib is not None and lib.loop_count() > 0:
            self._load_top_loop()
        else:
            self._prefetch_genome()

    # --- Properties ---

    @property
    def morph_t(self) -> float:
        return self._morph_t

    @morph_t.setter
    def morph_t(self, value: float) -> None:
        self._morph_t = value

    # --- Main tick ---

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # --- Mode tracking ---
        new_mode = Mode(audio.mode)
        if new_mode != self._mode:
            self._on_mode_change(self._mode, new_mode, clock)
        self._mode = new_mode

        self._section_warmup += 1
        self._section_cooldown += 1

        # --- Mel-space centroid delta (lazy init on first valid spectrum) ---
        if len(audio.spectrum) > 0 and self._mel_centroid is None:
            from flame_sheep_audio.response import MelCentroid as _MC
            # Derive freqs from spectrum length — matches whatever engine is active
            n_bins = len(audio.spectrum)
            freqs = np.linspace(0, SAMPLE_RATE / 2, n_bins)
            self._mel_centroid = _MC(freqs)
        if len(audio.spectrum) > 0 and self._mel_centroid is not None:
            mel_centroid = self._mel_centroid.compute(audio.spectrum)
        else:
            mel_centroid = 0.0
        mel_delta = self._mel_delta.update(mel_centroid)

        # --- Section change detection (beat mode only) ---
        if self._mode == Mode.BEAT:
            WARMUP_FRAMES = 1800
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

        # --- Handle discrete events ---
        for event in audio.events:
            if self._mode == Mode.BEAT:
                if event.kind == self._role.band_for_role(DOWNBEAT):
                    self._handle_downbeat(event, clock, audio)
                elif event.kind == "song_start":
                    self._handle_song_start(clock)
            elif event.kind == "song_start":
                self._handle_song_start(clock)

        # --- Centroid swap (beat + energy modes) ---
        if self._mode in (Mode.BEAT, Mode.ENERGY):
            low_density = self._role.band_state(audio, DOWNBEAT).onset_density
            if (
                low_density < cfg.genome.centroid_swap_density_gate
                and mel_delta > self.CENTROID_SWAP_THRESHOLD
                and self._morph_t > 0.3
            ):
                self.current_genome = self.current_genome.lerp(
                    self.target_genome, self._morph_t
                )
                self._swap_next_genome()
                self._morph_state = _MorphState.DWELL
                self._dwell_start = clock
                self._morph_t = 0.0
                log.debug(
                    f"[centroid swap] mel_delta={mel_delta:.1f} "
                    f"low_density={low_density:.2f}"
                )

        # --- Break damping (beat mode only) ---
        if self._mode == Mode.BEAT:
            if audio.break_intensity > 0:
                self._break_damping *= self.BREAK_DECAY
            else:
                self._break_damping = min(1.0, self._break_damping / self.BREAK_DECAY)
        else:
            self._break_damping = 1.0

        bpm = max(audio.effective_bpm, 60.0)

        # --- Morph advancement (same speed, all modes) ---
        if self._morph_state == _MorphState.MORPHING:
            self._morph_t = min(1.0, self._morph_t + self.MORPH_SPEED)
            if self._morph_t >= 1.0:
                # Morph complete → swap genome → enter dwell
                self.current_genome = self.target_genome
                self._swap_next_genome()
                self._morph_t = 0.0
                self._morph_state = _MorphState.DWELL
                self._dwell_start = clock

                # Idle loop cycling
                if self._mode == Mode.IDLE and self._loop_genomes and self._lib is not None:
                    if self._loop_step == 0:
                        self._idle_loop_cycles += 1
                        if self._idle_loop_cycles >= self.CYCLES_PER_LOOP:
                            self._idle_loop_cycles = 0
                            self._next_idle_loop()

        elif self._morph_state == _MorphState.DWELL:
            # Mode-specific dwell release
            if self._mode == Mode.BEAT:
                # Beat releases dwell (handled in _handle_downbeat)
                pass
            elif self._mode == Mode.ENERGY:
                # Centroid shift releases dwell (handled in centroid swap above)
                # Fallback: same timeout as idle
                dwell_duration = self.DWELL_BEATS * 60.0 / bpm
                if clock - self._dwell_start >= dwell_duration:
                    self._morph_state = _MorphState.MORPHING
                    log.debug("[dwell] energy timeout → MORPHING")
            elif self._mode == Mode.IDLE:
                # Fixed duration timeout
                dwell_duration = self.DWELL_BEATS * 60.0 / bpm
                if clock - self._dwell_start >= dwell_duration:
                    self._morph_state = _MorphState.MORPHING
                    log.debug("[dwell] idle timeout → MORPHING")

        # --- Rotation (all modes) ---
        boost = self._rotation_boost.update(0.0, bpm=bpm)
        rotation_speed = self.ROTATION_SPEED + boost * self.ROTATION_BEAT_BOOST
        self._rotation_phase += rotation_speed * self._break_damping
        TWO_PI = 2.0 * np.pi
        if self._rotation_phase >= TWO_PI:
            self._rotation_phase -= TWO_PI

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        """Write interpolated genome to frame. Same for all modes."""
        frame.genome = self.current_genome.lerp(self.target_genome, self._morph_t)
        # Pull center toward origin — don't rotate it, because the
        # static centroid doesn't represent the rotational center of mass.
        # Once we have rotation-accumulated scoring this can be smarter.
        CENTER_PULL = 0.002
        frame.genome.center = frame.genome.center * (1.0 - CENTER_PULL)
        if self._rotation_phase != 0.0:
            frame.genome = frame.genome.rotated(self._rotation_phase)

    # --- Mode transitions ---

    def _on_mode_change(self, old: Mode, new: Mode, clock: float) -> None:
        """Handle mode transition. Visual output is continuous — no genome
        reset, no handoff, just change what behaviors are active."""
        if old == Mode.BEAT and new != Mode.BEAT:
            # Leaving beat mode: if dwell was waiting for a beat, release it
            if self._morph_state == _MorphState.DWELL:
                self._morph_state = _MorphState.MORPHING
                log.debug("[mode] beat→other: released dwell")

        if new == Mode.IDLE:
            self._idle_loop_cycles = 0

    # --- Playback hints ---

    def on_playback_paused(self) -> None:
        self._playback_paused = True

    def on_playback_resumed(self) -> None:
        self._playback_paused = False

    # --- Event handlers ---

    def _handle_downbeat(self, event: BeatEvent, clock: float, audio: AudioState | None = None) -> None:
        since = clock - self._last_downbeat_time
        self._last_downbeat_time = clock

        self._recent_downbeat_energy = self._recent_downbeat_energy * 0.8 + event.energy * 0.2

        downbeat_density = self._role.band_state(audio, DOWNBEAT).onset_density if audio else 0.0
        density_scale = 1.0 / (1.0 + downbeat_density * self.DENSITY_DAMPING)

        # Section change pending → consume on any low-band onset
        if self._section_change_pending and self._lib is not None:
            self._section_change_pending = False
            self._section_cooldown = 0
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self._morph_t
            )
            self.next_loop()
            self._morph_state = _MorphState.DWELL
            self._dwell_start = clock
            self._morph_t = 0.0
            log.info(f"[SWAP+LOOP]  section change consumed on beat")
            return

        bpm = max(audio.effective_bpm, 60.0) if audio else 120.0
        kick = event.energy * density_scale

        # Beat releases dwell → start morphing
        if self._morph_state == _MorphState.DWELL:
            self._morph_state = _MorphState.MORPHING
            log.debug(f"[downbeat] dwell released by beat, energy={event.energy:.2f}")
        else:
            # Normal beat during morph — boost rotation
            self._rotation_boost.update(kick, bpm=bpm)
            log.debug(f"[downbeat]  +{since:.3f}s  energy={event.energy:.2f}")

    def _handle_song_start(self, clock: float, mode_hint: str | None = None) -> None:
        """Reset state for new song."""
        if mode_hint is not None:
            self._mode = Mode(mode_hint)

        self._last_downbeat_time = clock
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        self._section_change_pending = False
        self._section_warmup = 0
        self._morph_state = _MorphState.DWELL
        self._dwell_start = clock
        self._morph_t = 0.0
        # New song, new loop
        if self._lib is not None and self._lib.loop_count() > 1:
            self.next_loop()
            log.info(f"[song_start] switched to loop #{self.active_loop_id}")

    def accept_handoff(self, genome: Genome, loop_id: int | None = None) -> None:
        """Receive genome from external source (e.g., test injection)."""
        self.current_genome = genome
        self._morph_state = _MorphState.DWELL
        self._morph_t = 0.0
        self._recent_downbeat_energy = 0.5
        self._break_damping = 1.0
        if loop_id is not None and loop_id != self.active_loop_id:
            self.load_loop(loop_id)
        self._swap_next_genome()

    def force_swap(self) -> None:
        """Immediately swap to a new genome."""
        self.current_genome = self.current_genome.lerp(
            self.target_genome, self._morph_t)
        self._swap_next_genome()
        self._morph_state = _MorphState.DWELL
        self._morph_t = 0.0
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

        self._loop_sequence = loop_sequence(self._loop_genomes, self._loop_structure)
        self._loop_cycle_len = cycle_length(n, self._loop_structure)
        start = int(self.rng.integers(0, n))
        for _ in range(start):
            next(self._loop_sequence)
        self._loop_step = start

        self.active_loop_id = loop_id
        self.current_genome = self._loop_genomes[start]
        self.target_genome = next(self._loop_sequence)
        self._loop_step += 1
        self._morph_state = _MorphState.DWELL
        self._morph_t = 0.0
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

    def _next_idle_loop(self) -> None:
        """Switch to next loop during idle mode."""
        if self._lib is None or self._lib.loop_count() < 1:
            return
        top = self._lib.top_loops(n=20)
        candidates = [(lid, info) for lid, info in top
                      if lid not in self._loop_history]
        if not candidates:
            candidates = [(lid, info) for lid, info in top
                          if lid != self.active_loop_id]
        if not candidates:
            candidates = top
        fitnesses = np.array([max(info['fitness'], 0.01)
                              for _, info in candidates])
        weights = fitnesses / fitnesses.sum()
        idx = self.rng.choice(len(candidates), p=weights)
        lid, info = candidates[idx]
        self._loop_history.append(lid)

        items = self._lib.load_loop(lid)
        self._loop_genomes = [genome for _, genome, _ in items]
        self._loop_structure = self._lib.loop_type(lid)
        n = len(self._loop_genomes)
        self._loop_cycle_len = cycle_length(n, self._loop_structure)
        self._loop_sequence = loop_sequence(self._loop_genomes, self._loop_structure)
        start = int(self.rng.integers(0, n))
        for _ in range(start):
            next(self._loop_sequence)
        self._loop_step = start
        self.active_loop_id = lid
        self.current_genome = self._loop_genomes[start]
        self.target_genome = next(self._loop_sequence)
        self._loop_step += 1
        self._morph_t = 0.0
        log.info(f'[idle] switched to loop #{lid} '
                 f'({self._loop_structure}, fitness={info["fitness"]:.3f}, {n} genomes)')

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
