"""Shim re-exporting the wallpaper_ml package.

The Vulkan ML framework moved to the standalone wallpaper_ml package
in the Stage 1 reorg. This shim keeps `from flame_sheep.wallpaper_ml
import ...` working for any consumer that wasn't updated in the same
pass.
"""
from wallpaper_ml import (  # noqa: F401
    VkBuffer,
    VkCompute,
    VkConv2d,
    VkGAPLinear,
    VkGAPMLP,
    VkGRU,
    VkLayer,
    VkLinear,
    VkModel,
    bce_loss_dispatch,
    build_beat_crnn,
    build_cnn_scorer,
    cross_entropy_dispatch,
)
from wallpaper_ml.layers import SHADER_DIR, RNN_SHADER_DIR  # noqa: F401
