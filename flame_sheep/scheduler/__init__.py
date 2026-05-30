"""flame-sheep scheduler — internal-only first cut.

Goal: arbitrate GPU access between the wallpaper render path (must
keep up with vsync) and best-effort work (shader pre-compile,
background catalog rendering, offline training when we wire it in).

This package starts with the primitive everything else builds on:
reading GPU utilization from the kernel without shelling out. See
gpu_load.py.
"""
