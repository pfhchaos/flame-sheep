"""Wallpaper command handlers — replaces 18 inline closures in _run_wallpaper.

Closure state becomes explicit instance attributes. The render loop reads
from `WallpaperCommands` properties (quit_requested, comparing, etc.) and
delegates handler dispatch to register_all().
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

import numpy as np

from .renderer import FlameRenderer

if TYPE_CHECKING:
    from .core import FlameSheepCore
    from .storage import Library
    from .orchestrator import Orchestrator
    from .compare import CompareMode

log = logging.getLogger(__name__)


class WallpaperCommands:
    """Owns command handler state and dispatches commands from the orchestrator.

    State that used to live in `_run_wallpaper` closures is now explicit on
    this class. The render loop reads from `quit_requested`, `comparing`,
    `compare_needs_reset`, `compare_renderer`, `compare_mode` directly.
    """

    CMP_SCALE = 2

    def __init__(self,
                 core: FlameSheepCore,
                 lib: Library | None,
                 orch: Orchestrator,
                 renderer: FlameRenderer,
                 ctx,
                 viewports: dict,
                 surfaces: dict,
                 first_surf,
                 canvas_ppmm: float):
        self.core = core
        self.lib = lib
        self.orch = orch
        self.renderer = renderer
        self.ctx = ctx
        self.viewports = viewports
        self.surfaces = surfaces
        self.first_surf = first_surf
        self.canvas_ppmm = canvas_ppmm

        # Quit signal — render loop reads this each iteration
        self.quit_requested = False

        # Evolution state
        self._vote_count = 0
        self._votes_per_evolve = 5
        self._evolve_count = 0
        self._evolving = False

        # Compare mode state — render loop reads these
        self.compare_mode: CompareMode | None = None
        self.compare_renderer: FlameRenderer | None = None
        self.comparing = False
        self.compare_needs_reset = False

    def register_all(self) -> None:
        """Wire all handlers to the orchestrator."""
        self.orch.on_command('quit', self._handle_quit)
        self.orch.on_command('swap', self._handle_swap)
        self.orch.on_command('like', self._handle_like)
        self.orch.on_command('dislike', self._handle_dislike)
        self.orch.on_command('next', self._handle_next)
        self.orch.on_command('song', self._handle_song)
        self.orch.on_command('tempo', self._handle_tempo)
        self.orch.on_command('pause', self._handle_pause)
        self.orch.on_command('resume', self._handle_resume)
        self.orch.on_command('seek', self._handle_seek)
        self.orch.on_command('config', self._handle_config)
        self.orch.on_command('evolve', self._handle_evolve)
        self.orch.on_command('compare', self._handle_compare)
        self.orch.on_command('left', self._handle_left)
        self.orch.on_command('right', self._handle_right)
        self.orch.on_command('skip', self._handle_skip)
        self.orch.on_command('wallpaper', self._handle_wallpaper)

    # --- Evolution ---

    def _maybe_evolve(self) -> None:
        """Trigger evolution if enough votes have accumulated."""
        self._vote_count += 1
        if self._evolving:
            log.debug(f'[evolve] already running, vote queued ({self._vote_count})')
            return
        if self._vote_count < self._votes_per_evolve:
            log.debug(f'[evolve] {self._vote_count}/{self._votes_per_evolve} votes until next evolution')
            return
        if self.lib is None or self.lib.loop_count() < 2:
            log.warning('not enough loops to evolve')
            return
        self._vote_count = 0
        self._evolve_count += 1
        if self._evolve_count >= 3:
            self._votes_per_evolve = 3
        self._evolving = True
        log.info(f'[evolve] starting evolution cycle {self._evolve_count} in subprocess...')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'flame_sheep', '--evolve', '--loop-length', str(6)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

        def _wait_evolve():
            out, _ = proc.communicate()
            if out:
                for line in out.decode().strip().split('\n'):
                    log.info(f'[evolve] {line}')
            self._evolving = False

        threading.Thread(target=_wait_evolve, daemon=True).start()

    # --- Core commands ---

    def _handle_quit(self, event) -> None:
        self.quit_requested = True

    def _handle_swap(self, event) -> None:
        self.core.force_genome_swap()

    def _handle_like(self, event) -> None:
        gid = self.core.active_genome_db_id
        if gid is not None and self.lib is not None:
            self.lib.rate('genome', gid, +1)
            log.debug(f'[ctl] liked genome #{gid}')
            self._maybe_evolve()
        else:
            log.warning('no active genome to rate')

    def _handle_dislike(self, event) -> None:
        gid = self.core.active_genome_db_id
        if gid is not None and self.lib is not None:
            self.lib.rate('genome', gid, -1)
            log.debug(f'[ctl] disliked genome #{gid}')
            self._maybe_evolve()
        self.core.user_next()

    def _handle_next(self, event) -> None:
        self.core.user_next()
        log.debug(f'[ctl] next loop #{self.core.active_loop_id}')

    def _handle_song(self, event) -> None:
        self.core.song_started()
        self.core.force_genome_swap()

    def _handle_tempo(self, event) -> None:
        if event.args:
            try:
                bpm = float(event.args[0])
                self.core.hint_tempo(bpm)
            except ValueError:
                log.info(f'[ctl] invalid tempo: {event.args[0]}')

    def _handle_pause(self, event) -> None:
        self.core._genome_axis.on_playback_paused()

    def _handle_resume(self, event) -> None:
        self.core._genome_axis.on_playback_resumed()

    def _handle_seek(self, event) -> None:
        self.orch.audio.reset_tempo()
        log.debug('[ctl] seek — reset tempo + drop state')

    def _handle_config(self, event) -> None:
        if event.args and event.args[0] == 'reload':
            from .config import cfg
            from flame_sheep_audio.config import cfg as audio_cfg
            cfg.reload()
            audio_cfg.reload()
            log.debug('[ctl] config reloaded (viz + audio)')

    def _handle_evolve(self, event) -> None:
        """Force an evolution cycle regardless of vote count."""
        self._vote_count = self._votes_per_evolve
        self._maybe_evolve()

    # --- Compare mode ---

    def _ensure_compare_renderer(self) -> None:
        if self.compare_renderer is not None:
            return
        center_name = max(self.viewports, key=lambda n: self.viewports[n].w)
        center_surf = self.surfaces.get(center_name, self.first_surf)
        cmp_w = center_surf.width // 2 // self.CMP_SCALE
        cmp_h = center_surf.height // self.CMP_SCALE
        self.compare_renderer = FlameRenderer(self.ctx, cmp_w, cmp_h)
        self.compare_renderer.blur_radius = 0.0
        self.compare_renderer.set_ppmm(self.canvas_ppmm / self.CMP_SCALE)
        # Restore main renderer's bindings
        self.renderer.bind_buffers()
        log.info(f'[compare] created renderer at {cmp_w}x{cmp_h}')

    def _handle_compare(self, event) -> None:
        from .compare import CompareMode
        if self.comparing:
            return
        import time
        _t0 = time.perf_counter()
        self._ensure_compare_renderer()
        _t1 = time.perf_counter()
        log.info(f'[compare] renderer: {(_t1-_t0)*1000:.0f}ms')
        self.compare_mode = CompareMode(self.lib)
        _t2 = time.perf_counter()
        log.info(f'[compare] CompareMode init: {(_t2-_t1)*1000:.0f}ms')
        self.compare_mode.pick_pair()
        _t3 = time.perf_counter()
        log.info(f'[compare] pick_pair: {(_t3-_t2)*1000:.0f}ms')
        self.comparing = True
        self.compare_needs_reset = True
        # Reclaim compare renderer's SSBO bindings after main renderer's exit rebind
        if self.compare_renderer is not None:
            self.compare_renderer.bind_buffers()
            # CPU-side zero to ensure clean state regardless of binding cache
            n_px = self.compare_renderer.canvas_w * self.compare_renderer.canvas_h
            self.compare_renderer.histogram_buf.write(
                np.zeros(n_px * 2, dtype=np.uint32).tobytes())
        log.info('[ctl] entered compare mode')

    def _handle_left(self, event) -> None:
        if self.comparing and self.compare_mode:
            self.compare_mode.on_left_wins()
            self.compare_needs_reset = True

    def _handle_right(self, event) -> None:
        if self.comparing and self.compare_mode:
            self.compare_mode.on_right_wins()
            self.compare_needs_reset = True

    def _handle_skip(self, event) -> None:
        if self.comparing and self.compare_mode:
            self.compare_mode.on_skip()
            self.compare_needs_reset = True

    def _handle_wallpaper(self, event) -> None:
        if self.comparing:
            self.comparing = False
            # Restore main renderer to normal mode
            self.renderer.set_histogram_offset(0)
            self.renderer.bind_buffers()
            self.renderer.reset_walkers()
            self.core.needs_walker_reset = True
            log.info('[ctl] exited compare mode')
