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

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from flame_sheep.config import cfg
from flame_sheep.runtime.role_mapper import RoleMapper, DOWNBEAT, BACKBEAT
from flame_sheep.axes._morph_cycle import MorphCycle
from flame_sheep.axes._beat_responder import BeatResponder, BeatAction
from flame_sheep.axes._rotation_driver import RotationDriver
from flame_sheep.axes._centroid_swap import CentroidSwap
import numpy as np

log = logging.getLogger(__name__)

from flame_sheep_audio import AudioState
from flame_sheep_audio.mode import Mode
from flame_sheep.genome import Genome
from flame_sheep.variations import Variation

if TYPE_CHECKING:
    from flame_sheep.runtime import FlameSheepCore
    from flame_sheep.storage import Library


class GenomeAxis:
    """Genome axis — manages genome lifecycle across all three modes.

    Morph cycle: MORPHING → DWELL → MORPHING (repeat)
    Morph speed is fixed across all modes. The dwell trigger varies by mode.
    """

    LOOP_HISTORY_SIZE = 8

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
                 rng: np.random.Generator | None = None,
                 is_pipeline_ready: Callable[[Genome, Genome], bool] | None = None,
                 on_pipeline_request: Callable[[Genome, Genome], None] | None = None,
                 ) -> None:
        """is_pipeline_ready: optional predicate the axis consults
        before committing to a new target_genome. If supplied and the
        candidate's pipeline isn't ready, the axis defers the swap
        (keeps current target / no morph start). The wallpaper passes
        this so cold genomes never get selected — render loop only
        ever sees genomes whose shaders are compiled.

        on_pipeline_request: optional side-effect called whenever a
        candidate is rejected for not being ready. Caller wires this
        to enqueue a precompile request, so the not-ready set shrinks
        over time. Without it, deferred swaps never get unstuck.

        Both default to None for backward compat (tests, batch tools
        that don't need shader gating); when both are absent the axis
        behaves exactly as before."""
        self.enabled = True
        self._genome_factory = genome_factory
        self._role = role
        self._lib = lib
        self.rng = rng or np.random.default_rng()
        self._is_pipeline_ready = is_pipeline_ready
        self._on_pipeline_request = on_pipeline_request
        # Step-3 shader-warming A/B hook: called at each loop-mode arrival, just
        # before the render-side sequence advances, so the wallpaper can fire a
        # PLAN request to the worker and later compare the worker's targets
        # against its own. Default None → no-op (behavior unchanged).
        self._on_arrival: Callable[[Genome, int | None, int, Genome | None], None] | None = None
        # Step-4 S3: the worker's pre-decided-target provider. get_target(
        # arrival_id, branch) -> Genome | None (None = not answered yet -> defer).
        # When None (precompile unavailable), FOLLOW falls back to the render-side
        # loop pick so the wallpaper still runs. Wired by the wallpaper.
        self._plan_provider: Callable[[int | None, str], Genome | None] | None = None
        self._follow_deferring = False

        # --- Mode ---
        self._mode = Mode.IDLE
        self._playback_paused = False

        # --- Genome state ---
        self.current_genome = self._genome_factory()
        self.target_genome = self._genome_factory()

        # --- Morph state ---
        self._morph = MorphCycle()
        self._next_loop_pending = False

        # --- Beat responder ---
        self._beat = BeatResponder(role)

        # --- Loop player ---
        from flame_sheep.axes._loop_player import LoopPlayer
        self._loop = LoopPlayer(lib, genome_factory, self.current_genome, self.rng)

        # Walker reset flag (consumed by renderer)
        self.needs_walker_reset = False

        # --- Centroid swap ---
        self._centroid = CentroidSwap()

        # --- External display pin ---
        # When set, contribute() bypasses morph/loop selection and uses
        # this genome as both current and target. Rotation phase from
        # _rotation.tick() still applies (audio-reactive rotation
        # continues), and downstream axes (ZoomAxis, PaletteAxis) still
        # contribute their effects normally — only the genome IDENTITY
        # is locked. Cleared by FlameSheepCore.clear_pin().
        self._pinned_genome: Genome | None = None

        # --- Rotation ---
        self._rotation = RotationDriver()

        # Boot: load loop or start prefetch
        if lib is not None and lib.loop_count() > 0:
            target = self._loop.next_loop(self.current_genome)
            if target is not None:
                self.target_genome = target
                self._morph.start_morph()
        else:
            self._loop._prefetch_genome(self.current_genome)

    # --- Properties ---

    @property
    def morph_t(self) -> float:
        return self._morph.t

    @morph_t.setter
    def morph_t(self, value: float) -> None:
        self._morph.t = value

    @property
    def active_loop_id(self) -> int | None:
        return self._loop.active_loop_id

    @active_loop_id.setter
    def active_loop_id(self, value: int | None) -> None:
        self._loop.active_loop_id = value

    # --- Main tick ---

    def tick(self, audio: AudioState, dt: float, clock: float) -> None:
        # --- Mode tracking ---
        new_mode = Mode(audio.mode)
        if new_mode != self._mode:
            self._on_mode_change(self._mode, new_mode, clock)
        self._mode = new_mode

        # --- Beat processing ---
        if self._mode == Mode.BEAT:
            self._beat.tick(audio)
            self._process_beat_events(audio, clock)
        else:
            self._beat.break_damping = 1.0
            # Song start events still processed outside beat mode
            for event in audio.events:
                if event.kind == "song_start":
                    self._handle_song_start(clock)

        # --- Centroid swap (ENERGY mode only) ---
        # A centroid/timbral shift is ENERGY's reactivity: WALK toward a spectral
        # neighbor. BEAT does NOT swap on centroid — its loop advances on the beat
        # (kicks), not on timbre. (Firing in beat too was drift; it both broke the
        # rhythm-only loop and walked onto un-warmed genomes = cold compiles.)
        if self._mode == Mode.ENERGY:
            low_density = self._role.band_state(audio, DOWNBEAT).onset_density
            if self._centroid.tick(audio.spectrum, low_density, self._morph.t):
                self.current_genome = self.current_genome.lerp(
                    self.target_genome, self._morph.t
                )
                self._graph_walk_next()
                self._morph.commit_swap(clock)
                log.debug("[centroid swap] triggered")

        bpm = max(audio.effective_bpm, 60.0)

        # --- Morph advancement (same speed, all modes) ---
        if self._morph.morphing:
            self._morph.advance()
            if self._morph.is_complete():
                if self._mode == Mode.BEAT:
                    self._morph.enter_swap_ready()
                else:
                    # Idle/energy ARRIVE: fire the PLAN for the just-arrived
                    # genome and enter DWELL. These modes WALK (loops are a
                    # beat-mode construct — they encode musical cadence, which
                    # idle/energy don't have), so there is no loop-change here; the
                    # steady pick at dwell-release is a one-step walk. Walking also
                    # surfaces NON-loop genomes for the user to vote on (GA
                    # coverage) — see DECISIONS 2026-10-12.
                    self.current_genome = self.target_genome
                    self._emit_arrival()
                    self._morph.commit_swap(clock)

        elif self._morph.in_dwell:
            # Mode-specific dwell release
            if self._mode == Mode.BEAT:
                # Beat releases dwell (handled in event loop above)
                pass
            elif self._mode in (Mode.ENERGY, Mode.IDLE):
                # Timeout → DEPART: pick the steady FOLLOW/WALK target now, then
                # morph. The pick moved here from morph-complete so the dwell is
                # the worker's think-window (S3). Behaviorally identical while the
                # source is still render-side: same deterministic next genome.
                dwell_duration = self.DWELL_BEATS * 60.0 / bpm
                if clock - self._morph.dwell_start >= dwell_duration:
                    if self._pick_next_target():
                        self._morph.release_dwell()
                        log.debug(f"[dwell] {self._mode.value} timeout → MORPHING")
                    # else: DEFER — stay in DWELL (affines rotate), retry next tick

        # --- Rotation (all modes) ---
        self._rotation.tick(bpm, self._beat.break_damping)

    def contribute(self, frame: FlameSheepCore.FrameState) -> None:
        """Write interpolated genome to frame. Same for all modes.

        When pinned (via FlameSheepCore.pin_genome), uses the pinned
        genome instead of the morph-interpolated current/target. Rotation
        phase still applies so the wallpaper keeps spinning, and
        downstream axes (ZoomAxis pulse, etc.) still get to mutate the
        pinned genome's per-frame state.
        """
        if self._pinned_genome is not None:
            frame.genome = self._pinned_genome
        else:
            frame.genome = self.current_genome.lerp(self.target_genome, self._morph.t)
        if self._rotation.phase != 0.0:
            frame.genome = frame.genome.rotated(self._rotation.phase)

    # --- Mode transitions ---

    def _on_mode_change(self, old: Mode, new: Mode, clock: float) -> None:
        """Handle mode transition. Visual output is continuous — no genome
        reset, no handoff, just change what behaviors are active."""
        if old == Mode.BEAT and new != Mode.BEAT:
            # Leaving beat mode: if dwell was waiting for a beat, release it
            if self._morph.in_dwell:
                self._morph.release_dwell()
                log.debug("[mode] beat→other: released dwell")

        if new == Mode.BEAT and not self._loop.has_loop:
            # Entering beat mode without a loop (e.g., from graph walk).
            # Find the nearest loop via the transition graph.
            self.next_loop()
            log.info("[mode] →beat: loaded loop via graph transition")


    # --- Playback hints ---

    def on_playback_paused(self) -> None:
        self._playback_paused = True

    def on_playback_resumed(self) -> None:
        self._playback_paused = False

    # --- Event handlers ---

    def _process_beat_events(self, audio: AudioState, clock: float) -> None:
        """Process beat events in beat mode via BeatResponder."""
        kick_band = self._role.band_for_role(DOWNBEAT)
        for event in audio.events:
            if event.kind == kick_band:
                action = self._beat.process_kick(
                    event, self._morph.in_swap_ready, self._morph.in_dwell)
                if action == BeatAction.COMMIT_SWAP:
                    # ARRIVE: fire the PLAN request for the just-arrived genome.
                    # Loop CHANGE morphs immediately (BREAK); otherwise enter DWELL
                    # and defer the FOLLOW pick to the dwell-release kick.
                    self.current_genome = self.target_genome
                    self._emit_arrival()
                    if self._next_loop_pending:
                        self._next_loop_pending = False
                        self.next_loop()
                        self._morph.start_morph()
                        log.debug(f'[swap on beat] next loop pending, kick energy={event.energy:.2f}')
                    else:
                        self._morph.commit_swap(clock)
                        log.debug(f'[arrive on beat] kick energy={event.energy:.2f}')
                elif action == BeatAction.RELEASE_DWELL:
                    # DEPART on the kick: pick the steady FOLLOW/WALK target, morph.
                    # If the worker hasn't answered, DEFER — stay dwelling, retry
                    # on the next kick.
                    if self._pick_next_target():
                        self._morph.release_dwell()
            elif event.kind == self._role.band_for_role(BACKBEAT):
                action = self._beat.process_downbeat(
                    event, clock, audio, self._lib is not None)
                if action == BeatAction.SECTION_CHANGE:
                    self.current_genome = self.current_genome.lerp(
                        self.target_genome, self._morph.t)
                    self.next_loop()
                    self._morph.commit_swap(clock)
                    log.info("[SWAP+LOOP] section change consumed on beat")
                else:
                    # Apply morph nudge + rotation boost
                    self._morph.nudge(self._beat.last_morph_nudge)
                    bpm = max(audio.effective_bpm, 60.0)
                    self._rotation.boost(
                        self._beat.last_morph_nudge / 0.05, bpm=bpm)
            elif event.kind == "song_start":
                self._handle_song_start(clock)

    def _handle_song_start(self, clock: float, mode_hint: str | None = None) -> None:
        """Reset state for new song."""
        if mode_hint is not None:
            self._mode = Mode(mode_hint)

        self._beat.reset_for_song(clock)
        self._morph.commit_swap(clock)
        # New song, new loop
        if self._lib is not None and self._lib.loop_count() > 1:
            self.next_loop()
            log.info(f"[song_start] switched to loop #{self.active_loop_id}")

    def accept_handoff(self, genome: Genome, loop_id: int | None = None) -> None:
        """Receive genome from external source (e.g., test injection)."""
        self.current_genome = genome
        self._morph.commit_swap(0.0)
        self._beat.recent_downbeat_energy = 0.5
        self._beat.break_damping = 1.0
        if loop_id is not None and loop_id != self.active_loop_id:
            self.load_loop(loop_id)
        self._swap_next_genome()

    def force_swap(self) -> None:
        """Immediately swap to a new genome."""
        self.current_genome = self.current_genome.lerp(
            self.target_genome, self._morph.t)
        self._swap_next_genome()
        self._morph.commit_swap(0.0)
        self.needs_walker_reset = True
        log.info("force swap")

    def user_next(self) -> None:
        """User-triggered 'next'. Mode-keyed: BEAT jumps to the next loop;
        IDLE/ENERGY walk on now (loops are beat-only). The not-beat branch forces
        an immediate walk rather than setting the loop-pending flag (which only
        BEAT consumes — leaving it set in idle/energy would linger).

        TODO (DECISIONS 2026-10-12): "next" probably wants a RANDOM JUMP to
        something DIFFERENT, not a similar graph-neighbor — revisit once directed
        walk / random-jump exists."""
        if self._mode != Mode.BEAT:
            # not-beat: walk on immediately
            self.current_genome = self.current_genome.lerp(
                self.target_genome, self._morph.t)
            self._graph_walk_next()
            self._morph.start_morph()
            return
        # beat: switch to the next loop
        if self._morph.in_dwell or self._morph.in_swap_ready:
            self.current_genome = self.target_genome or self.current_genome
            self.next_loop()
            self._morph.start_morph()
        else:
            # Mid-morph — flag it to switch when the morph completes.
            self._next_loop_pending = True

    # --- Loop management ---

    def load_loop(self, loop_id: int) -> None:
        target = self._loop.load_loop(loop_id, self.current_genome)
        self.target_genome = target
        self._morph.start_morph()

    def set_pipeline_readiness(
            self,
            *,
            is_ready: Callable[[Genome, Genome], bool],
            on_request: Callable[[Genome, Genome], None] | None = None) -> None:
        """Late-bind the pipeline-readiness predicate + (optional)
        not-ready callback. Used by wallpaper_vk after FlameSheepCore
        construction (the renderer's pipeline cache + precompile
        driver aren't available at axis-init time)."""
        self._is_pipeline_ready = is_ready
        self._on_pipeline_request = on_request

    def _accept_target(self, candidate: Genome) -> bool:
        """Returns True if `candidate` may be committed as target_genome.
        If a readiness predicate was supplied and the candidate isn't
        ready (shader uncompiled), request the compile and return False
        so the caller can skip the swap. Caller is then expected to
        leave target_genome unchanged — render continues on the current
        genome until precompile catches up.

        The predicate is called with BOTH current and candidate. This
        lets the wallpaper check not just target's own shader but also
        the union(current, candidate) shader needed during the morph
        (lerp intermediates have keep_vars = union of both endpoints).
        Skipping union check leaves a 500-800ms inline compile stall
        on the render thread at morph start.

        No fallback to "compile inline anyway" — if the system is
        compile-starved, staying on the current genome forever is the
        correct behavior (user-confirmed). Avoids the silent wrong-
        shader pattern where the render thread gets a target it can't
        actually render properly."""
        if self._is_pipeline_ready is None:
            return True
        if candidate is None:
            return True
        if self._is_pipeline_ready(candidate, self.current_genome):
            return True
        if self._on_pipeline_request is not None:
            try:
                self._on_pipeline_request(candidate, self.current_genome)
            except Exception:
                log.exception('[genome-axis] pipeline request failed')
        log.debug(f'[genome-axis] deferring swap; candidate '
                  f'#{getattr(candidate, "db_id", "?")} not yet compiled')
        return False

    def next_loop(self) -> None:
        ref = self.current_genome
        if ref.db_id is None and self.target_genome is not None:
            ref = self.target_genome
        target = self._loop.next_loop(ref)
        if target is not None and self._accept_target(target):
            self.target_genome = target
            self._morph.start_morph()

    def set_plan_hook(
            self,
            on_arrival: Callable[[Genome, int | None, int, Genome | None], None]
            ) -> None:
        """Late-bind the step-3 PLAN/A-B arrival hook (wired by the wallpaper
        after the precompile driver exists). ``on_arrival(arrival_genome,
        loop_id, loop_step, expected_follow)`` fires at each loop-mode arrival.
        Purely observational — it must not mutate axis state (step 3 keeps target
        authority render-side)."""
        self._on_arrival = on_arrival

    def set_plan_provider(
            self,
            get_target: Callable[[int | None, str], Genome | None]) -> None:
        """Late-bind the worker's pre-decided-target provider (step 4 S3). Wired
        by the wallpaper to the PlanCoordinator's cache. ``get_target(arrival_id,
        branch)`` returns the worker's FOLLOW/BREAK/WALK genome or None (not
        answered yet → the move path defers)."""
        self._plan_provider = get_target

    def _emit_arrival(self) -> None:
        """Fire the step-3 arrival hook, if wired, at a loop-mode arrival. Called
        BEFORE the sequence advances so ``loop_step`` is the position just
        arrived at and ``peek_next_in_sequence`` is the upcoming FOLLOW. Guarded
        to loop mode + a real db_id (the planner keys off the arrival id); lerp
        intermediates (centroid/section swaps, db_id None) are skipped."""
        if self._on_arrival is None or not self._loop.has_loop:
            return
        g = self.current_genome
        if g is None or getattr(g, 'db_id', None) is None:
            return
        try:
            expected_follow = self._loop.peek_next_in_sequence()
            self._on_arrival(g, self._loop.active_loop_id,
                             self._loop.loop_step, expected_follow)
        except Exception:
            log.exception('[genome-axis] arrival hook failed')

    def _swap_next_genome(self) -> None:
        candidate = self._loop.swap_next(self.current_genome)
        if self._accept_target(candidate):
            self.target_genome = candidate

    def _pick_next_target(self) -> bool:
        """Pick the next steady-cycle target at DEPARTURE (dwell-release / the
        release-kick). Movement is MODE-keyed: BEAT follows a loop (loops encode
        rhythmic cadence); IDLE/ENERGY one-step WALK (no cadence to follow, and
        walking surfaces non-loop genomes for voting — DECISIONS 2026-10-12).
        Returns False iff the pick DEFERRED (worker not answered / shader not
        warm) — caller stays in DWELL and retries, affines still rotating.

        In BEAT, the loop may also BREAK: should_exit_loop() picks a uniform exit
        position per loop entry (after MIN_EXIT_CYCLES), and on that step we leave
        the loop for a new one instead of following. Loop CHANGES stay beat-only —
        idle/energy never reach this branch, they just WALK."""
        if self._mode == Mode.BEAT and self._loop.has_loop:
            if self._loop.should_exit_loop():
                return self._break_pick()    # leave the loop (BREAK)
            return self._follow_pick()   # beat: follow the loop (S3 worker FOLLOW)
        self._graph_walk_next()          # idle/energy or no-loop: one-step walk
        return True

    def _break_pick(self) -> bool:
        """Leave the current loop (BREAK). Render-side next_loop for now (the
        worker-ships-the-loop flip is a follow-up). Returns True if a new loop
        target was committed, False to defer."""
        ref = self.current_genome
        if ref.db_id is None and self.target_genome is not None:
            ref = self.target_genome
        target = self._loop.next_loop(ref)
        if target is not None and self._accept_target(target):
            self.target_genome = target
            return True
        return False

    def _follow_pick(self) -> bool:
        """FOLLOW using the worker's pre-decided target (step 4 S3). The render
        thread does NO loop computation for the pick — it consumes the worker's
        serialized genome and only keeps the loop_step bookkeeping. DEFERS (False)
        when the worker hasn't answered for this arrival or the shader isn't warm.
        With no provider wired (precompile unavailable) it falls back to the
        render-side loop pick so the wallpaper still runs."""
        if self._plan_provider is None:
            self._swap_next_genome()               # render-side FOLLOW (no worker)
            return True
        arrival_id = getattr(self.current_genome, 'db_id', None)
        target = (self._plan_provider(arrival_id, 'follow')
                  if arrival_id is not None else None)
        if target is None:
            if not self._follow_deferring:
                self._follow_deferring = True
                log.warning('[warming] FOLLOW not ready from worker; deferring '
                            '(arrival #%s) — dwelling until it lands', arrival_id)
            return False
        self._follow_deferring = False
        if not self._accept_target(target):
            return False                            # shader cold: gate defers
        self.target_genome = target
        self._loop.advance_step()                  # render keeps loop_step
        return True

    def _graph_walk_next(self) -> None:
        candidate = self._loop.graph_walk(self.current_genome)
        if self._accept_target(candidate):
            self.target_genome = candidate


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
