"""CNN aesthetic scorer for flame fractal genomes.

Siamese pairwise ranking model trained on Electric Sheep crowd ratings.
Two inputs per genome: static render (RGB) + swept render (grayscale),
concatenated as 4-channel input.

Architecture: ~105K params, designed for CPU inference (<1ms per genome).

Training uses Bradley-Terry pairwise ranking loss within same generation
(voter populations vary across generations).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

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
            nn.Conv2d(4, 16, 3, stride=2, padding=1),   # → 128×128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # → 64×64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # → 32×32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1), # → 16×16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, 4, 256, 256) → (B,) scores."""
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pool → (B, 128)
        return self.head(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SheepDataset(Dataset):
    """Load pre-rendered Electric Sheep images with ratings.

    Each item is (image_4ch, rating, generation).
    image_4ch: (4, 256, 256) float32 tensor — RGB static + grayscale swept.
    """

    def __init__(self, manifest_path: str | Path, image_dir: str | Path | None = None,
                 augment: bool = False, preload: bool = False):
        self.image_dir = Path(image_dir) if image_dir else Path(manifest_path).parent
        self.augment = augment

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
        """Load static RGB + swept grayscale → (4, 256, 256) float32."""
        from PIL import Image

        static_img = Image.open(self.image_dir / static_name).convert('RGB')
        swept_img = Image.open(self.image_dir / swept_name).convert('L')

        static_arr = np.array(static_img, dtype=np.float32) / 255.0  # (256, 256, 3)
        swept_arr = np.array(swept_img, dtype=np.float32) / 255.0    # (256, 256)

        # Stack: (256, 256, 3) + (256, 256, 1) → (256, 256, 4) → (4, 256, 256)
        combined = np.concatenate([static_arr, swept_arr[:, :, None]], axis=2)
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

    def __init__(self, dataset: SheepDataset, pairs_per_epoch: int = 50000):
        self.dataset = dataset
        self.pairs_per_epoch = pairs_per_epoch
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
                if rating_a == rating_b:
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

def score_genome(static_png: bytes, swept_png: bytes,
                 model_path: str | Path) -> float:
    """Score a single genome from its rendered PNGs.

    Args:
        static_png: PNG bytes of static color render
        swept_png: PNG bytes of swept grayscale render
        model_path: path to trained weights (.pt file)

    Returns:
        Scalar aesthetic score (higher = more aesthetic).
    """
    if not _HAS_TORCH:
        raise ImportError('PyTorch required for CNN scoring')

    from PIL import Image

    static_img = Image.open(io.BytesIO(static_png)).convert('RGB')
    swept_img = Image.open(io.BytesIO(swept_png)).convert('L')

    static_arr = np.array(static_img, dtype=np.float32) / 255.0
    swept_arr = np.array(swept_img, dtype=np.float32) / 255.0

    combined = np.concatenate([static_arr, swept_arr[:, :, None]], axis=2)
    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0)  # (1, 4, 256, 256)

    model = AestheticNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()

    with torch.no_grad():
        score = model(tensor).item()

    return score


def load_model(model_path: str | Path) -> AestheticNet:
    """Load a trained model for batch inference."""
    if not _HAS_TORCH:
        raise ImportError('PyTorch required for CNN scoring')

    model = AestheticNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model
