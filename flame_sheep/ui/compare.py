"""Comparison mode for pairwise genome rating.

CompareMode manages pair selection (active learning, multi-strategy) and
vote handling.

Pair selection uses scores from one or more models stored per genome
(legacy `cnn_score` column + named entries in `cnn_scores_detail` JSON).
With N models available, strategies rotate through:
  - `uncertainty:<model>` for each model (N strategies)
  - `disagreement:<m1>:<m2>` for each pair of models (C(N,2) strategies)
  - `random` baseline

Disagreement is confidence-weighted: `score = -d_m1 * d_m2` where
`d_m = z_m(a) - z_m(b)`. Positive when models disagree on direction,
magnitude scales with each model's preference strength — so two confident
opposite picks score MUCH higher than two uncertain picks that happen to
differ.

All strategies apply a within-pair diversity bonus using
transitions.signature_distance (cheap bag comparison): pairs whose two
genomes are graph-neighbors (low signature distance) carry less label
information per vote — the user can't easily tell two near-clones apart,
so we re-rank toward pairs whose members are structurally distinct from
each other.

CompareRenderer manages the visual split — lazy renderer creation at half
resolution, offset-based dual chaos game dispatch, and split-screen
tonemap of the largest output surface.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..genome import Genome
from ..rendering import FlameRenderer, GpuContext, Viewport
from ..transitions import variation_signature, signature_distance

if TYPE_CHECKING:
    from ..storage import Library

log = logging.getLogger(__name__)

# Legacy key name used for the deployed model's score (from `cnn_score`
# column) when no aux models are backfilled to cnn_scores_detail. Lets
# the strategy framework operate uniformly even in single-model mode.
LEGACY_MODEL_NAME = 'deployed_legacy'


@dataclass
class PairState:
    """Current comparison pair."""
    left: Genome | None = None
    right: Genome | None = None
    left_id: int | None = None
    right_id: int | None = None


class CompareMode:
    """Active-learning pairwise comparison manager.

    Generates pair-selection strategies from the set of available models
    (uncertainty per model + confidence-weighted disagreement per model
    pair + random baseline). Rotates strategies every `votes_per_strategy`
    votes. Applies DPP-style diversity penalty against recently-shown
    genomes via variation-signature distance.

    Tracks compared pairs across sessions (seeded from pairwise_ratings).
    """

    def __init__(self, lib: Library, score_fn=None,
                 votes_per_strategy: int = 5,
                 skip_progress_weight: float = 0.5,
                 diversity_weight: float = 0.3,
                 candidate_pool: int = 40):
        """
        Args:
            votes_per_strategy: rotation budget per strategy, measured
                in "votes-equivalent." A real vote contributes 1.0; a
                skip contributes `skip_progress_weight`. Strategy
                rotates when accumulated progress reaches this value.
            skip_progress_weight: how much a skip contributes to
                strategy rotation, in vote-equivalents. Default 0.5 →
                10 skips OR 5 votes triggers rotation. Rationale: a
                skip is itself signal ("these aren't worth
                distinguishing"); if a strategy keeps surfacing
                skip-worthy pairs, rotate AWAY from it.
            diversity_weight: λ multiplier on within-pair signature
                distance bonus. Larger = stronger preference for pairs
                whose two genomes are structurally distinct from each
                other (more label info per vote)
            candidate_pool: how many top candidates per strategy to
                re-rank by within-pair diversity. Larger = more
                diversity choices, higher per-pick cost
        """
        self.lib = lib
        self._score_fn = score_fn  # legacy callable: Genome -> float
        self.pair = PairState()

        # Compared pairs persist across sessions (seeded from DB) so
        # restarting compare-mode doesn't replay the same pair every time
        self._compared: set[tuple[int, int]] = self._load_compared_pairs()
        log.info(f'[compare] loaded {len(self._compared)} prior pairs')

        self.rng = np.random.default_rng()

        # Multi-model state
        self._scores_by_gid: dict[int, dict[str, float]] = {}
        self._models: list[str] = []
        # Pre-computed z-scored arrays (gid_list, z_score_array) per model
        self._zscores: dict[str, tuple[list[int], np.ndarray]] = {}
        # Strategy rotation
        self._strategies: list[tuple[str, Callable]] = []
        self._strategy_idx = 0
        self._strategy_progress = 0.0  # accumulates 1.0 per vote, skip_progress_weight per skip
        self._votes_per_strategy = votes_per_strategy
        self._skip_progress_weight = skip_progress_weight
        # Diversity re-rank parameters
        self._diversity_weight = diversity_weight
        self._candidate_pool = candidate_pool
        # Signature cache — avoid re-computing per pick_pair call
        self._signature_cache: dict[int, str] = {}

    def _load_compared_pairs(self) -> set[tuple[int, int]]:
        """Pull every (winner, loser) pair from pairwise_ratings as a
        canonical (min, max) tuple set. Persisted across sessions so
        re-running compare doesn't replay pairs the user already judged."""
        rows = self.lib.conn.execute(
            'SELECT winner_id, loser_id FROM pairwise_ratings'
        ).fetchall()
        return {(min(w, l), max(w, l)) for w, l in rows}

    def set_score_fn(self, fn) -> None:
        """Set the scoring function (genome_id -> float)."""
        self._score_fn = fn

    # --- Score loading + strategy setup ---

    def _load_model_scores(self) -> None:
        """Read per-genome multi-model scores from DB.

        Combines the legacy `cnn_score` column (single deployed model) and
        the `cnn_scores_detail` JSON dict (multiple named models). The
        legacy column's score is stored under LEGACY_MODEL_NAME so the
        strategy framework has at least one model to work with even when
        no aux models are backfilled.

        `_all_active_gids` always reflects every active rendered genome,
        even unscored ones — used by the random strategy as a fallback
        when no model has scored anything yet (fresh library).
        """
        rows = self.lib.conn.execute(
            '''SELECT g.id, g.cnn_score, g.cnn_scores_detail FROM genomes g
                JOIN genome_blobs b ON b.genome_id = g.id
               WHERE b.render_static IS NOT NULL
                 AND COALESCE(g.archived, 0) = 0'''
        ).fetchall()

        self._scores_by_gid = {}
        self._all_active_gids: list[int] = []
        for gid, cnn_score, detail_json in rows:
            self._all_active_gids.append(gid)
            scores: dict[str, float] = {}
            if detail_json:
                try:
                    d = json.loads(detail_json)
                    for k, v in d.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            scores[k] = float(v)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Legacy cnn_score column is included ONLY if no named-model
            # entries exist for this genome. Otherwise it's redundant with
            # whichever named entry corresponds to the deployed scorer,
            # and including both inflates the strategy set with a trivial
            # disagreement:deployed:deployed_legacy slot.
            if not scores and cnn_score is not None:
                scores[LEGACY_MODEL_NAME] = float(cnn_score)
            if scores:
                self._scores_by_gid[gid] = scores

        # Discover model names — only include ones with reasonable coverage
        # (≥ 50% of scored genomes). Models with sparse coverage produce
        # unstable z-score distributions.
        n_total = len(self._scores_by_gid)
        model_coverage: dict[str, int] = {}
        for scores in self._scores_by_gid.values():
            for k in scores:
                model_coverage[k] = model_coverage.get(k, 0) + 1
        self._models = sorted(
            m for m, c in model_coverage.items() if c >= n_total * 0.5
        )
        log.info(f'[compare] loaded scores for {n_total} genomes, '
                 f'models: {self._models} '
                 f'(coverage filtered from {sorted(model_coverage)})')

        # Pre-compute z-scored arrays per model for fast disagreement math
        for m in self._models:
            gids_with_m = [gid for gid, s in self._scores_by_gid.items()
                           if m in s]
            scores_arr = np.array(
                [self._scores_by_gid[gid][m] for gid in gids_with_m],
                dtype=np.float64,
            )
            std = scores_arr.std()
            if std < 1e-9:
                z = scores_arr - scores_arr.mean()
            else:
                z = (scores_arr - scores_arr.mean()) / std
            self._zscores[m] = (gids_with_m, z)

    def _build_strategies(self) -> None:
        """Generate selection strategies from the model set:
          N uncertainty (one per model)
          + C(N,2) disagreement (one per ordered pair)
          + 1 random baseline
        """
        strategies: list[tuple[str, Callable]] = []
        for m in self._models:
            strategies.append((f'uncertainty:{m}',
                               lambda m=m: self._pick_uncertainty(m)))
        for i in range(len(self._models)):
            for j in range(i + 1, len(self._models)):
                m1, m2 = self._models[i], self._models[j]
                strategies.append((f'disagreement:{m1}:{m2}',
                                   lambda m1=m1, m2=m2: self._pick_disagreement(m1, m2)))
        strategies.append(('random', self._pick_random))
        self._strategies = strategies
        log.info(f'[compare] {len(strategies)} strategies: '
                 f'{[s[0] for s in strategies]}')

    # --- Diversity / signature helpers ---

    def _signature(self, gid: int) -> str:
        sig = self._signature_cache.get(gid)
        if sig is None:
            try:
                g = self.lib.load_genome(gid)
                sig = variation_signature(g)
            except Exception:
                sig = ''
            self._signature_cache[gid] = sig
        return sig

    def _pair_diversity(self, gid_a: int, gid_b: int) -> float:
        """Within-pair structural distance — bonus for pairs whose
        two genomes use different variations (high signal per vote).

        Pairs that are graph-neighbors (low signature distance) are
        near-clones the user can barely tell apart; preferring distant
        pairs surfaces comparisons where the user has a clear opinion.
        """
        sig_a = self._signature(gid_a)
        sig_b = self._signature(gid_b)
        return float(signature_distance(sig_a, sig_b))

    def _set_pair(self, gid_a: int, gid_b: int) -> None:
        # Randomize left/right so display position doesn't bias the vote
        if self.rng.random() > 0.5:
            gid_a, gid_b = gid_b, gid_a
        self.pair.left = self.lib.load_genome(gid_a)
        self.pair.right = self.lib.load_genome(gid_b)
        self.pair.left_id = gid_a
        self.pair.right_id = gid_b

    def _is_fresh(self, gid_a: int, gid_b: int) -> bool:
        key = (min(gid_a, gid_b), max(gid_a, gid_b))
        return key not in self._compared

    # --- Strategy implementations ---

    def _pick_uncertainty(self, model: str) -> tuple[int, int] | None:
        """Smallest score-gap adjacent pair in `model`'s ranking,
        with diversity re-rank over the top `candidate_pool` candidates.
        """
        gids, _z = self._zscores.get(model, ([], np.array([])))
        if len(gids) < 2:
            return None
        # Sort gids by raw score (z-scores have same order)
        sorted_gids = sorted(gids,
                             key=lambda g: self._scores_by_gid[g][model])
        # Walk adjacent pairs, collect uncompared ones with their gaps
        candidates: list[tuple[float, int, int]] = []
        for i in range(len(sorted_gids) - 1):
            ga, gb = sorted_gids[i], sorted_gids[i + 1]
            if not self._is_fresh(ga, gb):
                continue
            gap = abs(self._scores_by_gid[ga][model]
                      - self._scores_by_gid[gb][model])
            candidates.append((gap, ga, gb))
            if len(candidates) >= self._candidate_pool:
                break
        if not candidates:
            return None
        # Normalize both terms to [0, 1] so diversity_weight is interpretable
        # as "fraction of the final score the diversity term contributes."
        max_gap = max(c[0] for c in candidates) or 1.0
        pair_divs = [self._pair_diversity(c[1], c[2]) for c in candidates]
        max_div = max(pair_divs) or 1.0
        scored = []
        for (gap, ga, gb), div in zip(candidates, pair_divs):
            base = 1.0 - (gap / max_gap)  # in [0, 1] — small gap = high base
            div_norm = div / max_div       # in [0, 1]
            combined = base + self._diversity_weight * div_norm
            scored.append((combined, ga, gb, gap, div))
        scored.sort(key=lambda x: -x[0])
        _, ga, gb, gap, div = scored[0]
        log.info(f'[compare] uncertainty:{model} pair #{ga} vs #{gb} '
                 f'gap={gap:.3f} pair_diversity={div:.2f}')
        return ga, gb

    def _pick_disagreement(self, m1: str, m2: str) -> tuple[int, int] | None:
        """Pair (a, b) maximizing -d_m1 * d_m2 where d_m = z_m(a) - z_m(b).

        Confident-disagreement: positive only when models pick opposite
        sides; magnitude scales with each model's preference strength.
        Two confident-but-opposite models score MUCH higher than two
        uncertain-and-noisy models that happen to differ.

        O(N) implementation: per-genome 'split' value = z_m1(g) - z_m2(g).
        Sort by split. Top genomes are "m1 prefers, m2 doesn't"; bottom
        are the opposite. Best disagreement pair = top × bottom.
        """
        gids_m1, z_m1 = self._zscores.get(m1, ([], np.array([])))
        gids_m2, z_m2 = self._zscores.get(m2, ([], np.array([])))
        if len(gids_m1) < 2 or len(gids_m2) < 2:
            return None
        # Intersect on gids that have both models' scores
        common = set(gids_m1) & set(gids_m2)
        if len(common) < 2:
            return None
        z1_map = dict(zip(gids_m1, z_m1))
        z2_map = dict(zip(gids_m2, z_m2))
        splits = sorted(
            ((z1_map[g] - z2_map[g], g) for g in common),
            key=lambda x: x[0],
        )
        # Pair top-of-split (m1 prefers strongly relative to m2) with
        # bottom-of-split (m2 prefers strongly relative to m1)
        # Take a pool of top-K and bottom-K, try cross-products in
        # decreasing disagreement magnitude with diversity re-rank.
        K = min(self._candidate_pool, len(common) // 2)
        top = splits[-K:]      # high split — m1 likes
        bot = splits[:K]        # low split — m2 likes

        candidates: list[tuple[float, int, int]] = []
        for s_t, g_t in top[::-1]:    # walk top from most-extreme
            for s_b, g_b in bot:      # walk bot from most-extreme
                if not self._is_fresh(g_t, g_b):
                    continue
                # Confidence-weighted disagreement: -d_m1 * d_m2
                d_m1 = z1_map[g_t] - z1_map[g_b]
                d_m2 = z2_map[g_t] - z2_map[g_b]
                score = -d_m1 * d_m2  # positive when signs disagree
                if score <= 0:
                    continue  # not actually disagreement
                candidates.append((score, g_t, g_b))
                if len(candidates) >= self._candidate_pool:
                    break
            if len(candidates) >= self._candidate_pool:
                break
        if not candidates:
            return None
        max_score = max(c[0] for c in candidates) or 1.0
        pair_divs = [self._pair_diversity(c[1], c[2]) for c in candidates]
        max_div = max(pair_divs) or 1.0
        scored = []
        for (s, ga, gb), div in zip(candidates, pair_divs):
            base = s / max_score          # in (0, 1]
            div_norm = div / max_div       # in [0, 1]
            combined = base + self._diversity_weight * div_norm
            scored.append((combined, ga, gb, s, div))
        scored.sort(key=lambda x: -x[0])
        _, ga, gb, s, div = scored[0]
        log.info(f'[compare] disagreement:{m1}:{m2} pair #{ga} vs #{gb} '
                 f'score={s:.3f} pair_diversity={div:.2f}')
        return ga, gb

    def _pick_random(self) -> tuple[int, int] | None:
        # Fall back to the full active set if no genomes have scores yet
        # (fresh library, before score_worker has filled cnn_score)
        gids = list(self._scores_by_gid.keys()) or self._all_active_gids
        if len(gids) < 2:
            return None
        rng = self.rng
        for _ in range(50):  # bounded attempts to find an uncompared pair
            ga, gb = rng.choice(gids, 2, replace=False)
            if self._is_fresh(int(ga), int(gb)):
                log.info(f'[compare] random pair #{ga} vs #{gb}')
                return int(ga), int(gb)
        return None

    # --- Main entry ---

    def pick_pair(self) -> PairState:
        """Select next pair via the current strategy.

        First call lazily loads scores + builds strategies. Each call
        increments the in-strategy vote counter; when it reaches
        votes_per_strategy, advance to the next strategy in rotation.
        """
        if not self._scores_by_gid:
            self._load_model_scores()
            self._build_strategies()

        if not self._strategies:
            log.warning('[compare] no strategies available — empty model set')
            return self.pair

        # Rotate strategy if we've exhausted the current one's vote budget
        if self._strategy_progress >= self._votes_per_strategy:
            self._strategy_idx = (self._strategy_idx + 1) % len(self._strategies)
            self._strategy_progress = 0.0

        # Try current strategy, fall back through subsequent ones if it
        # can't produce a fresh pair (e.g., all candidates compared)
        for attempt in range(len(self._strategies)):
            idx = (self._strategy_idx + attempt) % len(self._strategies)
            name, fn = self._strategies[idx]
            try:
                pair = fn()
            except Exception as e:
                log.warning(f'[compare] strategy {name} crashed: {e}')
                pair = None
            if pair is not None:
                if attempt > 0:
                    log.info(f'[compare] strategy {self._strategies[self._strategy_idx][0]} '
                             f'exhausted, fell back to {name}')
                self._set_pair(*pair)
                return self.pair

        log.warning('[compare] all strategies exhausted — no fresh pairs')
        return self.pair

    def on_left_wins(self) -> None:
        """Record left genome as winner, advance to next pair."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.left_id, self.pair.right_id)
        log.info(f'[compare] left #{self.pair.left_id} wins over '
                 f'#{self.pair.right_id}')
        self._strategy_progress += 1.0
        self.pick_pair()

    def on_right_wins(self) -> None:
        """Record right genome as winner, advance to next pair."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.right_id, self.pair.left_id)
        log.info(f'[compare] right #{self.pair.right_id} wins over '
                 f'#{self.pair.left_id}')
        self._strategy_progress += 1.0
        self.pick_pair()

    def on_skip(self) -> None:
        """Skip this pair — mark seen for the session (not persisted),
        advance to next pair. Skips contribute fractional progress
        (`skip_progress_weight`, default 0.5) toward the strategy
        rotation budget — "these aren't worth distinguishing" is itself
        signal that the current strategy may be surfacing low-value
        pairs and we should rotate away from it."""
        if self.pair.left_id is not None and self.pair.right_id is not None:
            pair_key = (min(self.pair.left_id, self.pair.right_id),
                        max(self.pair.left_id, self.pair.right_id))
            self._compared.add(pair_key)
        self._strategy_progress += self._skip_progress_weight
        self.pick_pair()

    def _record(self, winner_id: int, loser_id: int) -> None:
        """Store pairwise comparison result."""
        pair_key = (min(winner_id, loser_id), max(winner_id, loser_id))
        self._compared.add(pair_key)

        # Stamp the current generation. Pairwise comparisons are self-
        # contained and don't NEED the tag for training (they're durable
        # across generations), but the generation is useful for analysis
        # and weighting decisions later. See
        # docs/generational_architecture.md.
        gen = self.lib.get_current_generation()
        self.lib.conn.execute(
            '''INSERT INTO pairwise_ratings (winner_id, loser_id, source, generation)
               VALUES (?, ?, 'compare', ?)''',
            (winner_id, loser_id, gen),
        )
        self.lib.conn.commit()



class CompareRenderer:
    """Half-resolution renderer for the side-by-side compare view.

    Lazily created on first compare-mode entry. Uses an offset-based dual
    histogram (one buffer, two halves) so a single render program can
    dispatch and tonemap both genomes without rebinding SSBOs — works
    around Mesa's per-program SSBO binding cache on Arc.
    """

    CMP_SCALE = 2  # half-resolution renderer
    PERF_LOG_INTERVAL = 60  # frames between per-stage timing summaries

    def __init__(self, ctx, viewports: dict, surfaces: dict, first_surf,
                 canvas_ppmm: float):
        self.ctx = ctx
        self.viewports = viewports
        self.surfaces = surfaces
        self.first_surf = first_surf
        self.canvas_ppmm = canvas_ppmm
        self.renderer: FlameRenderer | None = None
        self.surf_name: str | None = None  # output we render compare on
        self.needs_reset = False

        # Per-stage frame timing — accumulated then logged + reset every
        # PERF_LOG_INTERVAL frames. Investigating why compare-mode
        # framerate is significantly worse than main mode despite the
        # quarter-resolution renderer.
        self._perf_frames = 0
        self._perf_accum: dict[str, float] = {}

    def _accum(self, stage: str, dt: float) -> None:
        self._perf_accum[stage] = self._perf_accum.get(stage, 0.0) + dt

    def _maybe_log_perf(self) -> None:
        self._perf_frames += 1
        if self._perf_frames < self.PERF_LOG_INTERVAL:
            return
        n = self._perf_frames
        # Headline metrics: frame budget, GPU-sync wait, measured CPU.
        # frame_total + swap are populated by the render-loop closure;
        # everything else is per-stage CPU work measured here.
        frame_total = self._perf_accum.pop('frame_total', 0.0) / n * 1000
        swap = self._perf_accum.pop('swap', 0.0) / n * 1000
        cpu_stages = sum(self._perf_accum.values()) / n * 1000
        other = max(0, frame_total - swap - cpu_stages)
        parts = sorted(self._perf_accum.items(),
                       key=lambda kv: -kv[1])
        s = '  '.join(f'{k}={v / n * 1000:.2f}ms' for k, v in parts)
        log.info(
            f'[compare perf {n} frames]  '
            f'frame_total={frame_total:.2f}ms  '
            f'swap={swap:.2f}ms  '
            f'cpu={cpu_stages:.2f}ms  '
            f'other={other:.2f}ms  ({s})'
        )
        self._perf_frames = 0
        self._perf_accum.clear()

    def ensure_renderer(self, main_renderer: FlameRenderer) -> None:
        """Create the compare renderer the first time it's needed."""
        if self.renderer is not None:
            return
        center_name = max(self.viewports, key=lambda n: self.viewports[n].w)
        center_surf = self.surfaces.get(center_name, self.first_surf)
        cmp_w = center_surf.width // 2 // self.CMP_SCALE
        cmp_h = center_surf.height // self.CMP_SCALE
        self.renderer = FlameRenderer(GpuContext(self.ctx, cmp_w, cmp_h,
                                                  ppmm=self.canvas_ppmm / self.CMP_SCALE))
        self.renderer.blur_radius = 0.0
        # Restore main renderer's bindings after our pipeline creation
        main_renderer.bind_buffers()
        log.info(f'[compare] created renderer at {cmp_w}x{cmp_h}')

    def reclaim_bindings(self) -> None:
        """Re-bind compare renderer's SSBOs after main renderer's bindings
        clobbered ours (Mesa per-program binding cache workaround)."""
        if self.renderer is None:
            return
        self.renderer.bind_buffers()
        # CPU-side zero to ensure clean state regardless of binding cache
        n_px = self.renderer.canvas_w * self.renderer.canvas_h
        self.renderer.histogram_buf.write(
            np.zeros(n_px * 2, dtype=np.uint32).tobytes())

    def dispatch(self, pair: PairState, frame, rotation_phase: float) -> None:
        """Run chaos game for left + right genomes into offset halves of
        the dual histogram."""
        if self.renderer is None or pair.left is None or pair.right is None:
            return
        import time
        cr = self.renderer
        rot = rotation_phase

        t0 = time.perf_counter()
        left_g = pair.left.rotated(rot) if rot != 0.0 else pair.left
        right_g = pair.right.rotated(rot) if rot != 0.0 else pair.right
        n_px = cr.canvas_w * cr.canvas_h
        self._accum('cpu_rotate', time.perf_counter() - t0)

        # Cache which output surface gets the compare view (largest viewport)
        if self.surf_name is None:
            self.surf_name = max(self.viewports, key=lambda n: self.viewports[n].w)

        if self.needs_reset:
            cr.ensure_double_histogram()
            cr.histogram_buf.write(
                np.zeros(n_px * 4, dtype=np.uint32).tobytes())
            cr.reset_walkers()
            self.needs_reset = False

        # Left genome: offset=0
        t = time.perf_counter()
        cr.set_histogram_offset(0)
        self._accum('set_offset', time.perf_counter() - t)
        t = time.perf_counter()
        cr.upload_audio(frame.spectrum)
        self._accum('upload_audio', time.perf_counter() - t)
        t = time.perf_counter()
        cr.upload_genome(left_g)
        self._accum('upload_genome', time.perf_counter() - t)
        t = time.perf_counter()
        cr.upload_palette(frame.palette)
        self._accum('upload_palette', time.perf_counter() - t)
        t = time.perf_counter()
        cr.clear_histogram(decay=0.3)
        self._accum('clear_hist', time.perf_counter() - t)
        t = time.perf_counter()
        cr.dispatch_chaos_game(iterations=frame.iterations)
        self._accum('chaos_game', time.perf_counter() - t)
        t = time.perf_counter()
        self.ctx.memory_barrier()
        self._accum('barrier', time.perf_counter() - t)

        # Right genome: offset=n_pixels
        t = time.perf_counter()
        cr.set_histogram_offset(n_px)
        self._accum('set_offset', time.perf_counter() - t)
        t = time.perf_counter()
        cr.upload_genome(right_g)
        self._accum('upload_genome', time.perf_counter() - t)
        t = time.perf_counter()
        cr.upload_palette(frame.palette)
        self._accum('upload_palette', time.perf_counter() - t)
        t = time.perf_counter()
        cr.clear_histogram(decay=0.3)
        self._accum('clear_hist', time.perf_counter() - t)
        t = time.perf_counter()
        cr.dispatch_chaos_game(iterations=frame.iterations)
        self._accum('chaos_game', time.perf_counter() - t)
        t = time.perf_counter()
        self.ctx.memory_barrier()
        self._accum('barrier', time.perf_counter() - t)

    def tonemap_surface(self, surf, frame) -> None:
        """Tonemap the dual histogram into a split-screen view on `surf`."""
        if self.renderer is None:
            return
        import time
        cr = self.renderer
        half_w = surf.width // 2
        n_px = cr.canvas_w * cr.canvas_h
        cr_vp = Viewport(0, 0, cr.canvas_w, cr.canvas_h)

        # Left half: offset=0
        t = time.perf_counter()
        cr.set_histogram_offset(0)
        self._accum('set_offset', time.perf_counter() - t)
        t = time.perf_counter()
        cr.reduce_histogram_max()
        self._accum('reduce_max', time.perf_counter() - t)
        t = time.perf_counter()
        cr.render_tonemap(cr_vp, surf.width, surf.height,
                         brightness=frame.brightness,
                         screen_rect=(0, 0, half_w, surf.height))
        self._accum('tonemap', time.perf_counter() - t)

        # Right half: offset=n_pixels
        t = time.perf_counter()
        cr.set_histogram_offset(n_px)
        self._accum('set_offset', time.perf_counter() - t)
        t = time.perf_counter()
        cr.reduce_histogram_max()
        self._accum('reduce_max', time.perf_counter() - t)
        t = time.perf_counter()
        cr.render_tonemap(cr_vp, surf.width, surf.height,
                         brightness=frame.brightness,
                         screen_rect=(half_w, 0, surf.width - half_w, surf.height))
        self._accum('tonemap', time.perf_counter() - t)

        # Log + reset every PERF_LOG_INTERVAL frames
        self._maybe_log_perf()
