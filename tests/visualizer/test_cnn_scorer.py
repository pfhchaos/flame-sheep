"""
Tests for cnn_scorer: model architecture, weight loading, input preparation.

Requires PyTorch for model tests. Input prep tests use numpy only.
"""

import io
import struct
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.cnn_scorer import (
    rgb_to_hsl,
    _build_4ch_hsl,
    _default_weights_path,
)

# Guard torch-dependent tests
torch = pytest.importorskip('torch')
from flame_sheep.cnn_scorer import (
    AestheticNetVk,
    load_vk_weights,
    _prepare_input,
    _prepare_input_domain,
    load_model,
)


# ----------------------------------------------------------------
# rgb_to_hsl (pure numpy)
# ----------------------------------------------------------------

class TestRgbToHsl:
    def test_output_shape(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        hsl = rgb_to_hsl(rgb)
        assert hsl.shape == (32, 32, 3)
        assert hsl.dtype == np.float32

    def test_white(self):
        white = np.ones((1, 1, 3), dtype=np.float32)
        hsl = rgb_to_hsl(white)
        assert hsl[0, 0, 2] == pytest.approx(1.0)  # L = 1

    def test_black(self):
        black = np.zeros((1, 1, 3), dtype=np.float32)
        hsl = rgb_to_hsl(black)
        assert hsl[0, 0, 2] == pytest.approx(0.0)  # L = 0

    def test_red(self):
        red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        hsl = rgb_to_hsl(red)
        assert hsl[0, 0, 0] == pytest.approx(0.0, abs=0.01)  # H = 0
        assert hsl[0, 0, 1] == pytest.approx(1.0, abs=0.01)  # S = 1
        assert hsl[0, 0, 2] == pytest.approx(0.5, abs=0.01)  # L = 0.5

    def test_bounded(self):
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        hsl = rgb_to_hsl(rgb)
        assert hsl.min() >= 0.0
        assert hsl.max() <= 1.0


# ----------------------------------------------------------------
# Model architecture
# ----------------------------------------------------------------

class TestAestheticNetVk:
    def test_default_is_25k(self):
        model = AestheticNetVk()
        total = sum(p.numel() for p in model.parameters())
        assert total == 24665

    def test_25k_forward_shape(self):
        model = AestheticNetVk()
        x = torch.randn(2, 4, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2,)

    def test_55k_layers(self):
        layers = AestheticNetVk.MODEL_CONFIGS[55141]
        model = AestheticNetVk(layers=layers)
        total = sum(p.numel() for p in model.parameters())
        assert total == 55141

    def test_100k_layers(self):
        layers = AestheticNetVk.MODEL_CONFIGS[97713]
        model = AestheticNetVk(layers=layers)
        total = sum(p.numel() for p in model.parameters())
        assert total == 97713

    def test_all_configs_have_correct_param_count(self):
        for expected_params, layers in AestheticNetVk.MODEL_CONFIGS.items():
            model = AestheticNetVk(layers=layers)
            actual = sum(p.numel() for p in model.parameters())
            assert actual == expected_params, f'Config {expected_params}: got {actual}'


# ----------------------------------------------------------------
# Weight loading
# ----------------------------------------------------------------

class TestLoadVkWeights:
    def _make_weights(self, n_params):
        """Generate random weight file with correct param count."""
        rng = np.random.default_rng(42)
        weights = rng.standard_normal(n_params).astype(np.float32)
        path = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
        np.save(path.name, weights)
        return Path(path.name)

    def test_load_25k(self):
        path = self._make_weights(24665)
        model = AestheticNetVk()
        load_vk_weights(model, path)
        total = sum(p.numel() for p in model.parameters())
        assert total == 24665

    def test_load_55k_auto_reconfigures(self):
        path = self._make_weights(55141)
        model = AestheticNetVk()  # starts as 25K
        load_vk_weights(model, path)  # should reconfigure to 55K
        total = sum(p.numel() for p in model.parameters())
        assert total == 55141

    def test_load_100k_auto_reconfigures(self):
        path = self._make_weights(97713)
        model = AestheticNetVk()
        load_vk_weights(model, path)
        total = sum(p.numel() for p in model.parameters())
        assert total == 97713

    def test_unknown_size_raises(self):
        path = self._make_weights(12345)
        model = AestheticNetVk()
        with pytest.raises(ValueError, match='Unknown model size'):
            load_vk_weights(model, path)

    def test_loaded_weights_differ_from_init(self):
        path = self._make_weights(24665)
        model = AestheticNetVk()
        init_w = next(model.parameters()).detach().clone()
        load_vk_weights(model, path)
        loaded_w = next(model.parameters()).detach()
        assert not torch.equal(init_w, loaded_w)


# ----------------------------------------------------------------
# Input preparation
# ----------------------------------------------------------------

class TestPrepareInput:
    def _make_png(self, w=32, h=32, mode='RGB'):
        """Generate a minimal PNG blob."""
        from PIL import Image
        img = Image.new(mode, (w, h), color=(128,) * len(mode))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    def test_rgb_output_shape(self):
        static = self._make_png(mode='RGB')
        swept = self._make_png(mode='L')
        tensor = _prepare_input(static, swept)
        assert tensor.shape == (1, 4, 256, 256)

    def test_rgb_output_range(self):
        static = self._make_png(mode='RGB')
        swept = self._make_png(mode='L')
        tensor = _prepare_input(static, swept)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_rgb_channels_are_rgb_plus_swept(self):
        """First 3 channels should be RGB, 4th should be swept grayscale."""
        static = self._make_png(mode='RGB')
        swept = self._make_png(mode='L')
        tensor = _prepare_input(static, swept)
        # All channels should have data (not zeros)
        for c in range(4):
            assert tensor[0, c].sum() > 0


class TestPrepareInputDomain:
    def _make_hist_static_blob(self, h=64, w=64):
        hits = np.random.randint(0, 1000, (h, w), dtype=np.uint32)
        colors = np.random.randint(0, 1_000_000, (h, w), dtype=np.uint32)
        header = struct.pack('II', h, w)
        return zlib.compress(header + hits.tobytes() + colors.tobytes())

    def _make_hist_blob(self, h=64, w=64, dtype=np.uint32):
        arr = np.random.randint(0, 1000, (h, w)).astype(dtype)
        header = struct.pack('II', h, w)
        return zlib.compress(header + arr.tobytes())

    def test_output_shape(self):
        static = self._make_hist_static_blob()
        swept = self._make_hist_blob()
        tensor = _prepare_input_domain(static, swept)
        assert tensor.shape == (1, 4, 256, 256)

    def test_output_range(self):
        static = self._make_hist_static_blob()
        swept = self._make_hist_blob()
        tensor = _prepare_input_domain(static, swept)
        assert tensor.min() >= -1e-6
        assert tensor.max() <= 1.0 + 1e-6

    def test_with_first_hit(self):
        static = self._make_hist_static_blob()
        swept = self._make_hist_blob()
        first_hit = self._make_hist_blob(dtype=np.uint8)
        tensor = _prepare_input_domain(static, swept, hist_first_hit=first_hit)
        assert tensor.shape == (1, 4, 256, 256)
        # A channel (index 3) should be nonzero
        assert tensor[0, 3].sum() > 0

    def test_without_first_hit_alpha_zero(self):
        static = self._make_hist_static_blob()
        swept = self._make_hist_blob()
        tensor = _prepare_input_domain(static, swept, hist_first_hit=None)
        # A channel should be zeros
        assert tensor[0, 3].sum() == 0


# ----------------------------------------------------------------
# Default weights path
# ----------------------------------------------------------------

class TestDefaultWeightsPath:
    def test_returns_path(self):
        path = _default_weights_path()
        assert isinstance(path, Path)

    def test_path_has_npy_extension(self):
        path = _default_weights_path()
        assert path.suffix == '.npy'
