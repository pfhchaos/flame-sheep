"""Comparison mode for pairwise genome rating.

CompareMode manages pair selection and active learning (which pair to
present next, vote handling).

CompareRenderer manages the visual split — lazy renderer creation at half
resolution, offset-based dual chaos game dispatch, and split-screen
tonemap of the largest output surface.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .genome import Genome
from .renderer import FlameRenderer, Viewport

if TYPE_CHECKING:
    from .storage import Library

log = logging.getLogger(__name__)


@dataclass
class PairState:
    """Current comparison pair."""
    left: Genome | None = None
    right: Genome | None = None
    left_id: int | None = None
    right_id: int | None = None


class CompareMode:
    """Active-learning pairwise comparison manager.

    Picks pairs where the CNN scorer is most uncertain (smallest score gap).
    Tracks which genomes have been compared to avoid repeats.
    """

    def __init__(self, lib: Library, score_fn=None):
        self.lib = lib
        self._score_fn = score_fn  # callable: Genome -> float (CNN score)
        self.pair = PairState()
        self._compared: set[tuple[int, int]] = set()
        self._score_cache: dict[int, float] = {}
        self._candidates: list[tuple[int, float]] = []  # (genome_id, score)
        self._candidate_idx = 0
        self.rng = np.random.default_rng()

    def set_score_fn(self, fn) -> None:
        """Set the scoring function (genome_id -> float)."""
        self._score_fn = fn

    def _build_candidates(self) -> None:
        """Score all genomes and sort by score for pair selection."""
        # Get all genomes with renders. The render presence check goes to
        # genome_blobs (where the column is dense) — the main genomes table
        # only has cheap scalars after blob separation.
        rows = self.lib.conn.execute(
            '''SELECT g.id, g.cnn_score FROM genomes g
                JOIN genome_blobs b ON b.genome_id = g.id
               WHERE b.render_static IS NOT NULL
                 AND COALESCE(g.archived, 0) = 0
               ORDER BY g.id'''
        ).fetchall()

        scored = []
        n_scored = 0
        for gid, cnn_score in rows:
            if cnn_score is not None:
                scored.append((gid, float(cnn_score)))
                n_scored += 1
            elif self._score_fn is not None:
                try:
                    g = self.lib.load_genome(gid)
                    s = self._score_fn(g)
                    scored.append((gid, s))
                    n_scored += 1
                except Exception:
                    continue
            else:
                # No score — assign random value for diversity
                scored.append((gid, self.rng.random()))

        if n_scored < len(scored) * 0.5:
            # Most genomes unscored — shuffle for diversity instead of
            # clustering at score=0 which causes repeated pairs
            self.rng.shuffle(scored)
            log.info(f'[compare] {n_scored}/{len(scored)} scored, using random order')
        else:
            scored.sort(key=lambda x: x[1])
        self._candidates = scored
        self._candidate_idx = 0
        log.info(f'[compare] built candidate pool: {len(scored)} genomes')

    def pick_pair(self) -> PairState:
        """Select a pair where the scorer is most uncertain.

        Finds adjacent genomes in the score ranking (smallest gap)
        that haven't been compared yet.
        """
        if not self._candidates:
            self._build_candidates()

        if len(self._candidates) < 2:
            log.warning('[compare] not enough candidates for comparison')
            return self.pair

        # Find the pair with smallest score gap that hasn't been compared
        best_pair = None
        best_gap = float('inf')

        # Search from where we left off to avoid always showing the same pair
        n = len(self._candidates)
        for offset in range(n - 1):
            i = (self._candidate_idx + offset) % (n - 1)
            gid_a, score_a = self._candidates[i]
            gid_b, score_b = self._candidates[i + 1]

            pair_key = (min(gid_a, gid_b), max(gid_a, gid_b))
            if pair_key in self._compared:
                continue

            gap = abs(score_a - score_b)
            if gap < best_gap:
                best_gap = gap
                best_pair = (gid_a, gid_b)
                self._candidate_idx = i + 1
                break  # take the first uncompared adjacent pair

        if best_pair is None:
            # All adjacent pairs compared — pick random
            gid_a, gid_b = self.rng.choice(n, 2, replace=False)
            best_pair = (self._candidates[gid_a][0], self._candidates[gid_b][0])
            log.debug('[compare] all adjacent pairs seen, using random')

        # Load the genomes
        gid_a, gid_b = best_pair
        # Randomize left/right so position doesn't bias
        if self.rng.random() > 0.5:
            gid_a, gid_b = gid_b, gid_a

        self.pair.left = self.lib.load_genome(gid_a)
        self.pair.right = self.lib.load_genome(gid_b)
        self.pair.left_id = gid_a
        self.pair.right_id = gid_b

        log.info(f'[compare] pair: #{gid_a} vs #{gid_b} (gap={best_gap:.3f})')
        return self.pair

    def on_left_wins(self) -> None:
        """Record left genome as winner."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.left_id, self.pair.right_id)
        # Keep winner, replace loser
        old_right = self.pair.right_id
        self._pick_opponent('right')
        log.info(f'[compare] left #{self.pair.left_id} wins over #{old_right}')

    def on_right_wins(self) -> None:
        """Record right genome as winner."""
        if self.pair.left_id is None or self.pair.right_id is None:
            return
        self._record(self.pair.right_id, self.pair.left_id)
        # Keep winner, replace loser
        old_left = self.pair.left_id
        self._pick_opponent('left')
        log.info(f'[compare] right #{self.pair.right_id} wins over #{old_left}')

    def on_skip(self) -> None:
        """Skip this pair — replace both."""
        self.pick_pair()

    def _record(self, winner_id: int, loser_id: int) -> None:
        """Store pairwise comparison result."""
        pair_key = (min(winner_id, loser_id), max(winner_id, loser_id))
        self._compared.add(pair_key)

        # Store in pairwise_ratings table
        self.lib.conn.execute(
            '''INSERT INTO pairwise_ratings (winner_id, loser_id, source)
               VALUES (?, ?, 'compare')''',
            (winner_id, loser_id),
        )
        self.lib.conn.commit()

    def _pick_opponent(self, replace_side: str) -> None:
        """Replace the losing side with a new genome near the winner in score."""
        winner_id = self.pair.left_id if replace_side == 'right' else self.pair.right_id

        # Find the winner's position in the candidate list
        winner_score = None
        for i, (gid, score) in enumerate(self._candidates):
            if gid == winner_id:
                winner_score = score
                break

        if winner_score is None:
            # Winner not in candidate list — pick random
            self.pick_pair()
            return

        # Find nearest uncompared genome to the winner
        best = None
        best_gap = float('inf')
        for gid, score in self._candidates:
            if gid == winner_id or gid == self.pair.left_id or gid == self.pair.right_id:
                continue
            pair_key = (min(gid, winner_id), max(gid, winner_id))
            if pair_key in self._compared:
                continue
            gap = abs(score - winner_score)
            if gap < best_gap:
                best_gap = gap
                best = gid

        if best is None:
            self.pick_pair()
            return

        genome = self.lib.load_genome(best)
        if replace_side == 'left':
            self.pair.left = genome
            self.pair.left_id = best
        else:
            self.pair.right = genome
            self.pair.right_id = best

        log.debug(f'[compare] new opponent #{best} (gap={best_gap:.3f})')


class CompareRenderer:
    """Half-resolution renderer for the side-by-side compare view.

    Lazily created on first compare-mode entry. Uses an offset-based dual
    histogram (one buffer, two halves) so a single render program can
    dispatch and tonemap both genomes without rebinding SSBOs — works
    around Mesa's per-program SSBO binding cache on Arc.
    """

    CMP_SCALE = 2  # half-resolution renderer

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

    def ensure_renderer(self, main_renderer: FlameRenderer) -> None:
        """Create the compare renderer the first time it's needed."""
        if self.renderer is not None:
            return
        center_name = max(self.viewports, key=lambda n: self.viewports[n].w)
        center_surf = self.surfaces.get(center_name, self.first_surf)
        cmp_w = center_surf.width // 2 // self.CMP_SCALE
        cmp_h = center_surf.height // self.CMP_SCALE
        self.renderer = FlameRenderer(self.ctx, cmp_w, cmp_h)
        self.renderer.blur_radius = 0.0
        self.renderer.set_ppmm(self.canvas_ppmm / self.CMP_SCALE)
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
        cr = self.renderer
        rot = rotation_phase
        left_g = pair.left.rotated(rot) if rot != 0.0 else pair.left
        right_g = pair.right.rotated(rot) if rot != 0.0 else pair.right
        n_px = cr.canvas_w * cr.canvas_h

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
        cr.set_histogram_offset(0)
        cr.upload_audio(frame.spectrum)
        cr.upload_genome(left_g)
        cr.upload_palette(frame.palette)
        cr.clear_histogram(decay=0.3)
        cr.dispatch_chaos_game(iterations=frame.iterations)
        self.ctx.memory_barrier()

        # Right genome: offset=n_pixels
        cr.set_histogram_offset(n_px)
        cr.upload_genome(right_g)
        cr.upload_palette(frame.palette)
        cr.clear_histogram(decay=0.3)
        cr.dispatch_chaos_game(iterations=frame.iterations)
        self.ctx.memory_barrier()

    def tonemap_surface(self, surf, frame) -> None:
        """Tonemap the dual histogram into a split-screen view on `surf`."""
        if self.renderer is None:
            return
        cr = self.renderer
        half_w = surf.width // 2
        n_px = cr.canvas_w * cr.canvas_h
        cr_vp = Viewport(0, 0, cr.canvas_w, cr.canvas_h)

        # Left half: offset=0
        cr.set_histogram_offset(0)
        cr.reduce_histogram_max()
        cr.render_tonemap(cr_vp, surf.width, surf.height,
                         brightness=frame.brightness,
                         screen_rect=(0, 0, half_w, surf.height))

        # Right half: offset=n_pixels
        cr.set_histogram_offset(n_px)
        cr.reduce_histogram_max()
        cr.render_tonemap(cr_vp, surf.width, surf.height,
                         brightness=frame.brightness,
                         screen_rect=(half_w, 0, surf.width - half_w, surf.height))
