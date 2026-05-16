"""CNN aesthetic scorer for flame fractal genomes.

Siamese pairwise ranking model trained on Electric Sheep crowd ratings.
Two inputs per genome: static render (HSL) + swept render (grayscale),
concatenated as 4-channel input: [H, S_swept, L, reserved].

HSL encoding: luminance carries structure, hue encodes palette position
(invariant to hue rotation), swept render packed into saturation channel
(original S is always 1.0 from rainbow palette → zero information).

Architecture: ~24K params (Vulkan) or ~105K (PyTorch w/ batchnorm).
Designed for CPU inference (<1ms per genome).

Training uses Bradley-Terry pairwise ranking loss within same generation
(voter populations vary across generations).
"""
from __future__ import annotations

import csv
import importlib.resources
import io
from pathlib import Path

import numpy as np


def rgb_to_hsl(rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float32 RGB [0,1] to (H, W, 3) HSL [0,1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    L = (max_c + min_c) / 2.0
    delta = max_c - min_c

    S = np.where(delta < 1e-8, 0.0,
                 np.where(L <= 0.5,
                          delta / (max_c + min_c + 1e-8),
                          delta / (2.0 - max_c - min_c + 1e-8)))

    H = np.zeros_like(L)
    mask = delta > 1e-8
    rm = mask & (max_c == r)
    H[rm] = ((g[rm] - b[rm]) / delta[rm]) % 6.0
    gm = mask & (max_c == g)
    H[gm] = (b[gm] - r[gm]) / delta[gm] + 2.0
    bm = mask & (max_c == b)
    H[bm] = (r[bm] - g[bm]) / delta[bm] + 4.0
    H = H / 6.0
    H[H < 0] += 1.0

    return np.stack([H, S, L], axis=-1).astype(np.float32)


def _build_4ch_hsl(rgb: np.ndarray, swept: np.ndarray) -> np.ndarray:
    """Build 4-channel HSL input from RGB static + grayscale swept.

    Channels: [H, S_swept, L, 0]
    H = hue from static render (palette position)
    S_swept = swept rotation render (replaces uninformative saturation)
    L = luminance from static render (structure)
    """
    hsl = rgb_to_hsl(rgb)
    out = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    out[:, :, 0] = hsl[:, :, 0]  # H
    out[:, :, 1] = swept          # S_swept
    out[:, :, 2] = hsl[:, :, 2]  # L
    return out


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # Stubs so module can be imported without torch (for type checking, etc.)
    class _Stub:
        Module = object
        Linear = BatchNorm2d = Conv2d = ReLU = Sequential = None
    nn = _Stub()
    Dataset = object


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AestheticNet(nn.Module):
    """Lightweight CNN that maps a 256×256×4 image to a scalar aesthetic score.

    4 channels = RGB from static render + grayscale from swept render.
    ~105K parameters.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=2, padding=1),    # → H/2
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, 3, stride=2, padding=1),   # → H/4
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → H/8
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → H/16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 4, 256, 256) → (B,) scores."""
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pool → (B, 64)
        return self.head(x).squeeze(-1)


class AestheticNetVk(nn.Module):
    """No-batchnorm variant matching the Vulkan compute trainer.

    Architecture auto-detected from weight file size. Supports:
      25K: 4→8→16→32→64, linear 64→1
      55K: 4→12→24→48→96, linear 96→1
     100K: 4→16→32→64→128, linear 128→1
    """

    MODEL_CONFIGS = {
        24665: [( 4,  8, 3, 2, 1), ( 8, 16, 3, 2, 1), (16, 32, 3, 2, 1), (32,  64, 3, 2, 1)],
        55141: [( 4, 12, 3, 2, 1), (12, 24, 3, 2, 1), (24, 48, 3, 2, 1), (48,  96, 3, 2, 1)],
        97713: [( 4, 16, 3, 2, 1), (16, 32, 3, 2, 1), (32, 64, 3, 2, 1), (64, 128, 3, 2, 1)],
    }

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = self.MODEL_CONFIGS[24665]  # default 25K
        self.LAYERS = layers
        convs = []
        for c_in, c_out, k, s, p in layers:
            convs.append(nn.Conv2d(c_in, c_out, k, stride=s, padding=p))
            convs.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*convs)
        self.head = nn.Linear(layers[-1][1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 4, 256, 256) → (B,) scores."""
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pool
        return self.head(x).squeeze(-1)


def load_vk_weights(model: AestheticNetVk, npy_path: str | Path) -> None:
    """Load flat .npy weights from Vulkan trainer into AestheticNetVk.

    Auto-detects model size from weight count and reconfigures the model
    if needed. Weight layout: [conv0_w, conv0_b, conv1_w, conv1_b, ..., linear_w, linear_b]
    """
    flat = np.load(npy_path).astype(np.float32)
    n_params = len(flat)

    # Auto-detect and reconfigure model if size doesn't match
    if n_params not in AestheticNetVk.MODEL_CONFIGS:
        raise ValueError(f'Unknown model size: {n_params} params. '
                         f'Expected one of {list(AestheticNetVk.MODEL_CONFIGS.keys())}')
    layers = AestheticNetVk.MODEL_CONFIGS[n_params]
    expected = sum(co*ci*k*k + co for ci,co,k,_,_ in layers) + layers[-1][1] + 1
    assert n_params == expected, f'Param count mismatch: {n_params} vs {expected}'

    # Rebuild model if layer config changed
    if model.LAYERS != layers:
        model.__init__(layers=layers)

    state = {}
    offset = 0

    # Conv layers: features.0, features.2, features.4, features.6 (skip ReLU indices)
    for i, (c_in, c_out, k, _, _) in enumerate(model.LAYERS):
        key_prefix = f'features.{i * 2}'
        n_w = c_out * c_in * k * k
        state[f'{key_prefix}.weight'] = torch.from_numpy(
            flat[offset:offset + n_w].reshape(c_out, c_in, k, k).copy())
        offset += n_w
        state[f'{key_prefix}.bias'] = torch.from_numpy(
            flat[offset:offset + c_out].copy())
        offset += c_out

    # Linear head
    n_linear = model.LAYERS[-1][1]
    state['head.weight'] = torch.from_numpy(
        flat[offset:offset + n_linear].reshape(1, n_linear).copy())
    offset += n_linear
    state['head.bias'] = torch.from_numpy(flat[offset:offset + 1].copy())
    offset += 1

    assert offset == n_params, f'Weight offset mismatch: {offset} vs {n_params}'
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SheepDataset(Dataset):
    """Load pre-rendered Electric Sheep images with ratings.

    Each item is (image_4ch, rating, generation).
    image_4ch: (4, 256, 256) float32 tensor — RGB static + grayscale swept.
    """

    def __init__(self, manifest_path: str | Path, image_dir: str | Path | None = None,
                 augment: bool = False, preload: bool = False,
                 image_size: int = 256):
        self.image_dir = Path(image_dir) if image_dir else Path(manifest_path).parent
        self.augment = augment
        self.image_size = image_size

        self.entries = []  # (static_path, swept_path, rating, generation)
        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((
                    row['static_path'],
                    row['swept_path'],
                    int(row['rating']),
                    int(row['generation']),
                ))

        # Optional: preload all images into RAM (~4.7GB for 18K genomes)
        self._cache: list[torch.Tensor] | None = None
        if preload:
            self.preload()

    def preload(self):
        """Load all images into RAM. ~256KB per genome × 18K = ~4.7GB."""
        import sys
        self._cache = []
        for i, (static_name, swept_name, _, _) in enumerate(self.entries):
            self._cache.append(self._load_4ch(static_name, swept_name))
            if (i + 1) % 1000 == 0:
                print(f'  preloading: {i + 1}/{len(self.entries)}', file=sys.stderr)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self._cache is not None:
            img = self._cache[idx].clone() if self.augment else self._cache[idx]
        else:
            static_name, swept_name, _, _ = self.entries[idx]
            img = self._load_4ch(static_name, swept_name)
        if self.augment and torch.rand(1).item() > 0.5:
            img = img.flip(-1)  # horizontal flip
        _, rating, gen = self.entries[idx][1], self.entries[idx][2], self.entries[idx][3]
        return img, rating, gen

    def _load_4ch(self, static_name: str, swept_name: str) -> torch.Tensor:
        """Load static RGB + swept grayscale → (4, size, size) HSL input."""
        from PIL import Image

        sz = self.image_size
        static_img = Image.open(self.image_dir / static_name).convert('RGB').resize((sz, sz), Image.LANCZOS)
        swept_img = Image.open(self.image_dir / swept_name).convert('L').resize((sz, sz), Image.LANCZOS)

        rgb = np.array(static_img, dtype=np.float32) / 255.0
        swept_arr = np.array(swept_img, dtype=np.float32) / 255.0

        combined = _build_4ch_hsl(rgb, swept_arr)
        return torch.from_numpy(combined.transpose(2, 0, 1))

    def by_generation(self) -> dict[int, list[int]]:
        """Return {generation: [indices]} for within-generation pair sampling."""
        gen_map: dict[int, list[int]] = {}
        for i, (_, _, _, gen) in enumerate(self.entries):
            gen_map.setdefault(gen, []).append(i)
        return gen_map


# ---------------------------------------------------------------------------
# Pair sampler
# ---------------------------------------------------------------------------

class PairSampler:
    """Generate within-generation pairs for pairwise ranking training.

    Only pairs where rating_A != rating_B are useful.
    Samples uniformly across generations to avoid large-generation bias.
    """

    def __init__(self, dataset: SheepDataset, pairs_per_epoch: int = 50000,
                 min_rating_gap: int = 0):
        self.dataset = dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.min_rating_gap = min_rating_gap
        self.gen_map = dataset.by_generation()
        self.rng = np.random.default_rng()

        # Pre-compute ratings per generation for fast pair sampling
        self.gen_ratings: dict[int, list[tuple[int, int]]] = {}
        for gen, indices in self.gen_map.items():
            rated = [(i, dataset.entries[i][2]) for i in indices]
            # Sort by rating for efficient pair sampling
            rated.sort(key=lambda x: x[1])
            if len(rated) >= 2:
                self.gen_ratings[gen] = rated

        self.generations = list(self.gen_ratings.keys())

    def sample_pairs(self) -> list[tuple[int, int]]:
        """Return list of (winner_idx, loser_idx) pairs."""
        pairs = []
        per_gen = max(1, self.pairs_per_epoch // len(self.generations))

        for gen in self.generations:
            rated = self.gen_ratings[gen]
            n = len(rated)
            for _ in range(per_gen):
                # Pick two distinct indices
                a, b = self.rng.choice(n, size=2, replace=False)
                idx_a, rating_a = rated[a]
                idx_b, rating_b = rated[b]
                if abs(rating_a - rating_b) <= self.min_rating_gap:
                    continue
                if rating_a > rating_b:
                    pairs.append((idx_a, idx_b))
                else:
                    pairs.append((idx_b, idx_a))

        self.rng.shuffle(pairs)
        return pairs


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

def _default_weights_path() -> Path:
    """Resolve bundled weights via importlib.resources.

    Prefers personal fine-tuned weights if available, falls back to base.
    """
    personal = importlib.resources.files('flame_sheep.data').joinpath('cnn_scorer_personal_vk.npy')
    if Path(str(personal)).exists():
        return Path(str(personal))
    ref = importlib.resources.files('flame_sheep.data').joinpath('cnn_scorer_vk.npy')
    return Path(str(ref))


def _prepare_input(static_png: bytes, swept_png: bytes) -> torch.Tensor:
    """Convert rendered PNGs to model input tensor (1, 4, 256, 256) in HSL."""
    from PIL import Image

    static_img = Image.open(io.BytesIO(static_png)).convert('RGB').resize((256, 256), Image.LANCZOS)
    swept_img = Image.open(io.BytesIO(swept_png)).convert('L').resize((256, 256), Image.LANCZOS)

    rgb = np.array(static_img, dtype=np.float32) / 255.0
    swept_arr = np.array(swept_img, dtype=np.float32) / 255.0

    combined = _build_4ch_hsl(rgb, swept_arr)
    return torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0)


def _prepare_input_domain(hist_static: bytes, hist_swept: bytes,
                          hist_first_hit: bytes | None = None) -> torch.Tensor:
    """Convert histogram DB blobs to model input tensor (1, 4, 256, 256).

    Uses scoring_channels.normalize_channels() for domain-native representation:
    H=palette index, S=swept density, L=hit count structure, A=first-hit emergence.
    """
    from .scoring_channels import (
        unpack_static_histogram, unpack_histogram, normalize_channels,
    )

    hits, colors = unpack_static_histogram(hist_static)
    swept = unpack_histogram(hist_swept)
    first_hit = unpack_histogram(hist_first_hit, dtype=np.uint8) if hist_first_hit else None

    channels = normalize_channels(hits, colors, swept, first_hit, output_size=256)
    return torch.from_numpy(channels).unsqueeze(0)  # (1, 4, 256, 256)


def score_genome(static_png: bytes, swept_png: bytes,
                 model_path: str | Path | None = None) -> float:
    """Score a single genome from its rendered PNGs.

    Args:
        static_png: PNG bytes of static color render
        swept_png: PNG bytes of swept grayscale render
        model_path: path to trained weights (.pt or .npy). Defaults to
            ~/.local/share/flame-sheep/cnn_scorer_vk.npy

    Returns:
        Scalar aesthetic score (higher = more aesthetic).
    """
    if not _HAS_TORCH:
        raise ImportError('PyTorch required for CNN scoring')

    model = load_model(model_path)
    tensor = _prepare_input(static_png, swept_png)

    with torch.no_grad():
        return model(tensor).item()


def load_model(model_path: str | Path | None = None) -> AestheticNetVk | AestheticNet:
    """Load a trained model for batch inference.

    Auto-detects format: .npy loads AestheticNetVk (Vulkan weights),
    .pt loads AestheticNet (PyTorch weights).
    """
    if not _HAS_TORCH:
        raise ImportError('PyTorch required for CNN scoring')

    path = Path(model_path) if model_path else _default_weights_path()

    if path.suffix == '.npy':
        model = AestheticNetVk()
        load_vk_weights(model, path)
    else:
        model = AestheticNet()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    model.eval()
    return model
