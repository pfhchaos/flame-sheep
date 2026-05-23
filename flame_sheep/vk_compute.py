"""Shim re-exporting wallpaper_ml.vk_compute.

vk_compute moved to the wallpaper_ml package in the Stage 1 reorg.
This shim keeps `from flame_sheep.vk_compute import ...` working for
any consumer that wasn't updated in the same pass.
"""
from wallpaper_ml.vk_compute import *  # noqa: F401,F403
from wallpaper_ml.vk_compute import VkCompute, VkBuffer  # noqa: F401
