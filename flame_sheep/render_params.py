"""Default render parameters for the chaos game.

These were historically defined alongside the GL FlameRenderer. They
moved here so they can outlive the GL stack — surviving consumers
(genome scoring, catalog generation, render worker, axes/detail_axis)
all import from here.
"""

# Default walker count — matches viz_authoring.vk.ChaosGame's default
# and the GL renderer's prior value. Picked for parallelism on modern
# GPUs (~65k walkers gives enough concurrency to saturate Arc / 4090).
N_WALKERS = 1024 * 64

# Default iterations per walker per frame for offline / fixed-cadence
# rendering. Live wallpaper varies between LIVE_ITER_MIN and
# LIVE_ITER_MAX based on audio energy (see axes/detail_axis.py).
N_ITERS = 150

# Live render iteration range. DetailAxis maps audio energy into this
# range — quiet audio → MIN (sparse / wispy), loud audio → MAX
# (dense / saturated).
LIVE_ITER_MIN = 100
LIVE_ITER_MAX = 500
