"""Supervisor for the background GPU genome-render worker.

Mirror of PrecompileDriver's role but for the GL/EGL render worker
(flame_sheep.genome.render_worker.BackgroundGpuRenderer):

- Owns a PauseFlag + Policy (LOW priority — more aggressive throttling
  than precompile, since user notices new shader compiles more than
  they notice the catalog rendering itself)
- Runs a policy-tick thread that updates the flag from the shared
  WallpaperSignalReader
- Spawns the render worker with the flag path

The render worker process itself is still multiprocessing.Process —
last time we tried subprocess.Popen with EGL/GL it crashed sway via
the Mesa-Xe + fork issue. Worth re-checking if/when the renderer
ports to Vulkan, but until then multiprocessing is the working path.
"""
from __future__ import annotations

import logging
import threading
import time

from .pause_flag import PauseFlag
from .policy import Policy
from .wallpaper_signal import WallpaperSignalReader

log = logging.getLogger(__name__)


class RenderWorkerSupervisor:
    """Wraps BackgroundGpuRenderer + throttle plumbing."""

    def __init__(self, db_path: str,
                  signal_reader: WallpaperSignalReader,
                  *,
                  slow_threshold: float = 0.45,
                  pause_threshold: float = 0.70,
                  sample_interval_s: float = 0.25,
                  pause_hold_s: float = 3.0):
        """slow_threshold / pause_threshold: lower than precompile's
        defaults (0.55 / 0.80) because render_worker is LOW priority —
        user doesn't notice the score DB filling, but does notice
        wallpaper jitter, so we'd rather throttle the renderer earlier.

        pause_hold_s 3s (vs precompile's 2s) — render worker's per-unit
        cost is larger (full genome render + tonemap + PNG encode), so
        an in-flight render that slipped through is more disruptive.
        Hold pause longer to give the wallpaper recovery room."""
        self._db_path = db_path
        self._flag = PauseFlag('render_worker')
        self._policy = Policy(signal_reader,
                                sample_interval_s=sample_interval_s,
                                slow_threshold=slow_threshold,
                                pause_threshold=pause_threshold,
                                pause_hold_s=pause_hold_s)
        self._stop = threading.Event()
        self._policy_t: threading.Thread | None = None
        self._renderer = None

    def start(self) -> None:
        # Import here so we don't drag in moderngl/EGL at module import
        from ..genome.render_worker import BackgroundGpuRenderer
        self._renderer = BackgroundGpuRenderer(
            self._db_path,
            pause_flag_path=str(self._flag.path),
        )
        self._renderer.start()
        self._policy_t = threading.Thread(
            target=self._policy_loop,
            name='render-policy',
            daemon=True)
        self._policy_t.start()
        log.info('[render-supervisor] started; pause flag = %s',
                 self._flag.path)

    def _policy_loop(self):
        interval = self._policy.sample_interval_s
        while not self._stop.is_set():
            state = self._policy.tick()
            self._flag.set(state)
            time.sleep(interval)

    def drain_state_distribution(self) -> dict:
        """Pass-through to the policy — see Policy.drain_state_distribution."""
        return self._policy.drain_state_distribution()

    def stop(self) -> None:
        self._stop.set()
        if self._renderer is not None:
            try:
                self._renderer.stop()
            except Exception:
                log.exception('[render-supervisor] renderer stop failed')
        self._flag.close()
        log.info('[render-supervisor] stopped')
