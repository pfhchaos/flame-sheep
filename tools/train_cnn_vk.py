#!/usr/bin/env python3
"""Train CNN aesthetic scorer on GPU via Vulkan compute shaders.

No PyTorch dependency at training time — forward and backward passes
run as Vulkan compute shader dispatches on the Arc A770.

Usage:
    python tools/train_cnn_vk.py --data ~/datasets/esheep-cnn/
    python tools/train_cnn_vk.py --data ~/datasets/esheep-cnn/ --epochs 30 --lr 0.001
"""
from __future__ import annotations

import argparse
import csv
import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wallpaper_ml.vk_compute import VkCompute

log = logging.getLogger(__name__)

# Layer definitions: (in_channels, out_channels, kernel, stride, padding)
MODEL_CONFIGS = {
    '25k': [
        (4,  8,  3, 2, 1),
        (8,  16, 3, 2, 1),
        (16, 32, 3, 2, 1),
        (32, 64, 3, 2, 1),
    ],
    '55k': [
        (4,  12, 3, 2, 1),
        (12, 24, 3, 2, 1),
        (24, 48, 3, 2, 1),
        (48, 96, 3, 2, 1),
    ],
    '100k': [
        (4,  16, 3, 2, 1),
        (16, 32, 3, 2, 1),
        (32, 64, 3, 2, 1),
        (64, 128, 3, 2, 1),
    ],
}
LAYERS = MODEL_CONFIGS['25k']  # default, overridden by --model-size
MLP_HEAD = False  # overridden by --mlp-head
MLP_HIDDEN = 16


class ImageStore:
    """Lazy-loading image store. Loads from disk on demand with LRU cache.

    At 256×256×4×4 = 1MB per image, a 4GB cache holds ~4000 images.
    For 18K images, most batches hit cache after the first epoch.
    """

    def __init__(self, manifest_path: Path, image_dir: Path,
                 image_size: int, cache_gb: float = 4.0,
                 channels: str = 'rgb',
                 n_channels: int = 4,
                 normalization: tuple | None = None,
                 entries_filter: set[tuple[int, int]] | None = None,
                 input_channel_mask: np.ndarray | None = None):
        """
        Args:
            entries_filter: if provided, only manifest rows whose
                (generation, sheep_id) tuple is in this set are kept. Used
                for staged-training corpus subsetting (e.g., smooth-Q1 ES).
            input_channel_mask: per-channel multiplier (shape =
                (n_channels,)) applied to each batch before return. Used
                for staged-training input masking (e.g., [0, 1, 1, 0] for
                L+S-only stage).
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self._channels = channels
        self._n_channels = n_channels
        self._normalization = normalization
        self._input_channel_mask: np.ndarray | None = None
        if input_channel_mask is not None:
            arr = np.asarray(input_channel_mask, dtype=np.float32)
            if arr.shape != (n_channels,):
                raise ValueError(
                    f'input_channel_mask shape {arr.shape} != '
                    f'({n_channels},)')
            if not np.allclose(arr, 1.0):
                self._input_channel_mask = arr.reshape(1, n_channels, 1, 1)

        self.entries = []
        n_filtered = 0
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                gen = int(row['generation'])
                if entries_filter is not None:
                    sid = int(row['sheep_id'])
                    if (gen, sid) not in entries_filter:
                        n_filtered += 1
                        continue
                self.entries.append((
                    row['static_path'], row['swept_path'],
                    int(row['rating']), gen,
                ))
        if entries_filter is not None:
            print(f'  ImageStore: {len(self.entries)} kept, '
                  f'{n_filtered} filtered out')

        bytes_per_image = 4 * image_size * image_size * 4
        self.max_cache = int(cache_gb * 1e9 / bytes_per_image)
        self._cache: dict[int, np.ndarray] = {}
        self._access_order: list[int] = []

    def __len__(self):
        return len(self.entries)

    def _load_one(self, idx: int) -> np.ndarray:
        static, swept, _, _ = self.entries[idx]
        sz = self.image_size

        if self._channels == 'domain':
            from flame_sheep.genome.scoring.scoring_channels import (
                build_cnn_input, load_raw_histograms,
            )
            hist_name = static.replace('_static.png', '_hist.npz')
            hist_path = self.image_dir / hist_name
            if hist_path.exists():
                raw = load_raw_histograms(str(hist_path))
                return build_cnn_input(
                    raw['static_hits'], raw['static_colors'],
                    raw['swept_hits'], raw.get('first_hit'),
                    normalization=self._normalization,
                    output_size=sz,
                    n_channels=self._n_channels,
                )
            # Fall through to RGB if no .npz

        s_img = Image.open(self.image_dir / static).convert('RGB').resize(
            (sz, sz), Image.LANCZOS)
        w_img = Image.open(self.image_dir / swept).convert('L').resize(
            (sz, sz), Image.LANCZOS)
        img = np.zeros((4, sz, sz), dtype=np.float32)
        img[:3] = np.array(s_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img[3] = np.array(w_img, dtype=np.float32) / 255.0
        if self._normalization is not None:
            from flame_sheep.genome.scoring.scoring_channels import standardize_channels
            img = standardize_channels(img, *self._normalization)
        return img

    def get_batch(self, indices: list[int] | np.ndarray) -> np.ndarray:
        """Load a batch of images by index. Uses LRU cache.

        If input_channel_mask was set at construction time, it's applied
        to the returned batch (broadcast over batch and spatial dims).
        Cache stores unmasked images so mask changes don't require cache
        invalidation (though we don't currently support mask changes
        mid-training).
        """
        batch = np.zeros((len(indices), self._n_channels,
                          self.image_size, self.image_size),
                         dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx not in self._cache:
                # Evict oldest if cache full
                while len(self._cache) >= self.max_cache:
                    old = self._access_order.pop(0)
                    self._cache.pop(old, None)
                self._cache[idx] = self._load_one(idx)
            else:
                # Move to end of access order
                try:
                    self._access_order.remove(idx)
                except ValueError:
                    pass
            self._access_order.append(idx)
            batch[i] = self._cache[idx]
        if self._input_channel_mask is not None:
            batch = batch * self._input_channel_mask
        return batch


def load_images(manifest_path: Path, image_dir: Path,
                image_size: int) -> tuple[np.ndarray, list]:
    """Load all images as (N, 4, H, W) float32 + metadata.

    For backward compatibility. Use ImageStore for streaming.
    """
    store = ImageStore(manifest_path, image_dir, image_size, cache_gb=100)
    entries = store.entries

    images = np.zeros((len(entries), 4, image_size, image_size), dtype=np.float32)
    for i in range(len(entries)):
        images[i] = store._load_one(i)
        if (i + 1) % 2000 == 0:
            log.info('  loaded %d/%d images', i + 1, len(entries))

    return images, entries


def sample_pairs(entries: list, n_pairs: int, min_gap: float = 0.5,
                 exclude_gen: int | None = None) -> list[tuple[int, int]]:
    """Sample pairwise comparisons using composite scoring.

    Composite score = generation_rank + normalized_within_gen_rating.
    Later generations are globally better (that's what selection pressure means).
    Cross-generation pairs give strong signal; within-generation pairs add local detail.
    """
    rng = np.random.default_rng()

    # Build generation rank (sorted by generation number)
    gen_map: dict[int, list[int]] = {}
    for i, (_, _, _, gen) in enumerate(entries):
        if gen != exclude_gen:
            gen_map.setdefault(gen, []).append(i)

    sorted_gens = sorted(gen_map.keys())
    gen_rank = {g: rank for rank, g in enumerate(sorted_gens)}

    # Compute composite score for each entry:
    #   composite = generation_rank + normalized_within_gen_rating
    # Within a generation, ratings are normalized to [0, 1]
    composites = np.zeros(len(entries), dtype=np.float64)
    for gen, indices in gen_map.items():
        ratings = np.array([entries[i][2] for i in indices], dtype=np.float64)
        rmin, rmax = ratings.min(), ratings.max()
        if rmax > rmin:
            norm = (ratings - rmin) / (rmax - rmin)
        else:
            norm = np.full(len(ratings), 0.5)
        rank = gen_rank[gen]
        for j, idx in enumerate(indices):
            composites[idx] = rank + norm[j]

    # Sample pairs: winner has higher composite score
    all_indices = []
    for indices in gen_map.values():
        all_indices.extend(indices)
    all_indices = np.array(all_indices)
    n_total = len(all_indices)

    pairs = []
    for _ in range(n_pairs * 2):  # oversample, filter by gap
        a, b = rng.choice(n_total, 2, replace=False)
        ia, ib = all_indices[a], all_indices[b]
        ca, cb = composites[ia], composites[ib]
        gap = abs(ca - cb)
        if gap < min_gap:
            continue
        if ca > cb:
            pairs.append((ia, ib))
        else:
            pairs.append((ib, ia))
        if len(pairs) >= n_pairs:
            break

    rng.shuffle(pairs)
    return pairs



def _apply_curriculum_defaults(args) -> None:
    """Set sensible defaults for staged-training flags based on --curriculum-stage.

    Stages 1 and 2 both train on ES (manifest-based). Stage 3 (personal data,
    DB-based) is handled by finetune_cnn_vk.py.

    Channel order is [H, S, L, A] (matches the deployed model's
    normalization v3).
    """
    if args.curriculum_stage == 1:
        # L+S only on full ES. H, A input-masked; their conv1 weights stay
        # at random init since input × any-weight = zero gradient.
        if args.input_channel_mask is None:
            args.input_channel_mask = '0,1,1,0'
        if args.channels_count is None or args.channels_count == 4:
            args.channels_count = 4  # keep architecture at 4 channels
        # No per-channel LR scaling needed in stage 1 (input mask suffices).
    elif args.curriculum_stage == 2:
        # +H on smooth-Q1 ES. A still input-masked. H gets warmup LR boost
        # so its random-init weights find features before the loss landscape
        # locks them out.
        if args.input_channel_mask is None:
            args.input_channel_mask = '1,1,1,0'
        if args.per_channel_lr_mult_warmup is None:
            args.per_channel_lr_mult_warmup = '3,1,1,0'
        if args.per_channel_lr_mult_settle is None:
            args.per_channel_lr_mult_settle = '1,1,1,0'
        if args.warmup_epochs == 0:
            args.warmup_epochs = 5
        if args.es_where is None:
            args.es_where = 'palette_mean_step <= 0.049'


def _parse_csv_floats(s: str | None, expected_len: int,
                      arg_name: str) -> np.ndarray | None:
    """Parse a comma-separated float list. Returns None if input is None."""
    if s is None:
        return None
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != expected_len:
        raise SystemExit(
            f'{arg_name}: expected {expected_len} values, got {len(parts)} '
            f'({s!r})')
    return np.array([float(p) for p in parts], dtype=np.float32)


def _load_es_filter(db_path: Path, where_clause: str) -> set[tuple[int, int]]:
    """Run SELECT generation, sheep_id WHERE <clause> against esheep.db."""
    import sqlite3
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            f"SELECT generation, sheep_id FROM sheep WHERE {where_clause}"
        ).fetchall()
    finally:
        con.close()
    return {(g, s) for g, s in rows}


def main():
    parser = argparse.ArgumentParser(description='Train CNN scorer on GPU (Vulkan)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--pairs-per-epoch', type=int, default=5000)
    parser.add_argument('--val-gen', type=int, default=244)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--cache-gb', type=float, default=4.0,
                        help='Image cache size in GB (default: 4.0)')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model-size', type=str, default='25k',
                        choices=['25k', '55k', '100k'],
                        help='Model size: 25k / 55k / 100k')
    parser.add_argument('--mlp-head', action='store_true',
                        help='Use MLP head (Linear→ReLU→Linear) instead of single Linear')
    parser.add_argument('--channels', type=str, default='domain',
                        choices=['rgb', 'domain'],
                        help='Input channels: domain (histogram-based H/S/L[/A], default) '
                             'or rgb (deprecated, palette-index artifact in green channel)')
    parser.add_argument('--channels-count', type=int, default=4, choices=[3, 4],
                        help='Channel count for domain mode: 4 includes A '
                             '(first-hit emergence, sentinel on ES), 3 omits A '
                             'entirely. Use 3 for ES pretraining; 4 for personal.')
    parser.add_argument('--init-weights', type=str, default=None,
                        help='Load initial weights from .npy file (for resuming or fine-tuning)')

    # --- Staged training (channel curriculum) flags ---
    parser.add_argument('--curriculum-stage', type=int, default=None,
                        choices=[1, 2],
                        help='Convenience preset for staged training. Sets '
                             'sensible defaults for input-channel-mask, '
                             'per-channel LR multipliers, ES filter, and '
                             'warmup-epochs. Individual flags below override. '
                             'Stage 1: L+S only on full ES. Stage 2: +H on '
                             'smooth-Q1 ES. Stage 3 (personal data) is '
                             'handled by finetune_cnn_vk.py.')
    parser.add_argument('--input-channel-mask', type=str, default=None,
                        help='Comma-separated per-channel input multipliers, '
                             'e.g. "0,1,1,0" zeros out H and A channels. '
                             'Length must match --channels-count.')
    parser.add_argument('--per-channel-lr-mult-warmup', type=str, default=None,
                        help='Comma-separated per-input-channel LR multipliers '
                             'for conv1 during the first --warmup-epochs '
                             'epochs, e.g. "3,1,1,0".')
    parser.add_argument('--per-channel-lr-mult-settle', type=str, default=None,
                        help='Comma-separated per-input-channel LR multipliers '
                             'for conv1 after warmup elapses, e.g. "1,1,1,0".')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of epochs to use warmup LR multipliers '
                             'before switching to settle.')
    parser.add_argument('--es-where', type=str, default=None,
                        help='SQL WHERE predicate against the esheep.db '
                             'sheep table — only matching (gen, sheep_id) '
                             'rows are kept in the manifest. E.g. '
                             '"palette_mean_step <= 0.049" for smooth-Q1.')
    parser.add_argument('--es-db', type=str,
                        default=str(Path.home() / '.local/share/flame-sheep/esheep.db'),
                        help='Path to the esheep.db for --es-where lookups.')

    args = parser.parse_args()

    # Apply --curriculum-stage defaults BEFORE reading other flags so user-
    # passed values still take precedence.
    if args.curriculum_stage is not None:
        _apply_curriculum_defaults(args)

    global LAYERS, MLP_HEAD
    # rgb mode always uses 4 channels (R, G, B, swept). channels_count
    # only applies to domain mode. Normalize the arg so downstream code
    # (buffer alloc, forward shape) can read a single source of truth.
    if args.channels == 'rgb':
        args.channels_count = 4

    # Copy MODEL_CONFIGS entry so we don't mutate the module-level constant
    # when overriding the first layer's in_channels for 3-channel mode.
    LAYERS = list(MODEL_CONFIGS[args.model_size])
    if args.channels_count != LAYERS[0][0]:
        # Override first layer's in_channels to match the requested count.
        first = LAYERS[0]
        LAYERS[0] = (args.channels_count, first[1], first[2], first[3], first[4])
    MLP_HEAD = args.mlp_head

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S')

    data_dir = Path(args.data)
    output_path = Path(args.output) if args.output else data_dir / 'cnn_scorer_vk.npy'

    # Load normalization stats (zero mean / unit variance per channel) so
    # Kaiming weight init's distributional assumptions hold. Without this,
    # the sentinel-heavy domain channels (mean H ≈ 0.76) blow up the first
    # conv layer into a dead-ReLU collapse.
    #
    # Per-dataset: an ES manifest has different render parameters than the
    # user's library (different walker density / iteration count → different
    # channel distribution). Each dataset standardizes against its own stats
    # so the model always sees zero-mean / unit-variance regardless of
    # source. Features then transfer cleanly between pretrain and fine-tune.
    from flame_sheep.storage import NORMALIZATION_VERSION
    from flame_sheep.genome.scoring.scoring_channels import load_normalization_sidecar
    normalization = load_normalization_sidecar(data_dir, NORMALIZATION_VERSION)
    norm_source = None
    if normalization is not None:
        norm_source = f'sidecar {data_dir}/normalization_{NORMALIZATION_VERSION}.json'
    else:
        from flame_sheep.storage import Library as _Lib
        _lib_norm = _Lib()
        normalization = _lib_norm.get_normalization(NORMALIZATION_VERSION)
        _lib_norm.close()
        if normalization is not None:
            norm_source = f'Library metadata (no sidecar in {data_dir})'
    if normalization is None:
        log.warning('No normalization stats found — run '
                    'tools/compute_normalization_from_manifest.py on this '
                    'dataset, or tools/compute_normalization.py for the '
                    'Library. Training will proceed with identity '
                    'normalization (legacy mode).')
    else:
        log.info('Normalization %s from %s: mean=%s std=%s',
                 NORMALIZATION_VERSION, norm_source,
                 [f'{x:.4f}' for x in normalization[0]],
                 [f'{x:.4f}' for x in normalization[1]])

    # If 3-channel domain, trim normalization stats to first 3 entries.
    # Per-channel stats are independent so H/S/L slices are valid as-is —
    # no recomputation needed.
    if args.channels == 'domain' and args.channels_count == 3 and normalization is not None:
        normalization = (normalization[0][:3], normalization[1][:3])
        log.info('Trimmed normalization to 3 channels (H, S, L): '
                 'mean=%s std=%s',
                 [f'{x:.4f}' for x in normalization[0]],
                 [f'{x:.4f}' for x in normalization[1]])

    # --- Staged-training parsing (input mask + per-channel LR mults + ES filter) ---
    input_mask = _parse_csv_floats(args.input_channel_mask,
                                   args.channels_count,
                                   '--input-channel-mask')
    lr_mult_warmup = _parse_csv_floats(args.per_channel_lr_mult_warmup,
                                       args.channels_count,
                                       '--per-channel-lr-mult-warmup')
    lr_mult_settle = _parse_csv_floats(args.per_channel_lr_mult_settle,
                                       args.channels_count,
                                       '--per-channel-lr-mult-settle')
    es_filter: set[tuple[int, int]] | None = None
    if args.es_where:
        db_path = Path(args.es_db)
        if not db_path.exists():
            raise SystemExit(f'--es-db does not exist: {db_path}')
        es_filter = _load_es_filter(db_path, args.es_where)
        log.info('ES filter "%s": %d (gen, sheep_id) tuples',
                 args.es_where, len(es_filter))
    if input_mask is not None:
        log.info('Input channel mask: %s', input_mask.tolist())
    if lr_mult_warmup is not None:
        log.info('Per-channel LR mult (warmup %d ep): %s',
                 args.warmup_epochs, lr_mult_warmup.tolist())
    if lr_mult_settle is not None:
        log.info('Per-channel LR mult (settle): %s', lr_mult_settle.tolist())

    # Streaming image store (loads from disk on demand)
    store = ImageStore(data_dir / 'manifest.csv', data_dir, args.image_size,
                       cache_gb=args.cache_gb, channels=args.channels,
                       n_channels=args.channels_count,
                       normalization=normalization,
                       entries_filter=es_filter,
                       input_channel_mask=input_mask)
    entries = store.entries
    log.info('Dataset: %d genomes, image cache: %.1f GB (%d images)',
             len(entries), args.cache_gb, store.max_cache)

    # Init Vulkan
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)

    from wallpaper_ml import build_cnn_scorer
    model = build_cnn_scorer(gpu, LAYERS, batch_size=args.batch_size,
                              image_size=args.image_size, mlp_head=MLP_HEAD)

    if args.init_weights:
        from flame_sheep.genome.scoring.cnn_scorer import load_cnn_weights_file
        init_w, init_norm_version = load_cnn_weights_file(args.init_weights)
        if init_norm_version is not None and init_norm_version != NORMALIZATION_VERSION:
            raise RuntimeError(
                f'init_weights normalization mismatch: file is '
                f'{init_norm_version}, codebase is {NORMALIZATION_VERSION}. '
                f'Refusing to load — scores would be silently wrong.')
        model.load_weights(init_w)
        log.info('Loaded initial weights: %d params (norm_version=%s)',
                 len(init_w), init_norm_version)
    else:
        model.init_weights()

    log.info('Model: %d params (%s head)', model.param_count(),
             'MLP' if MLP_HEAD else 'linear')

    # Input buffer for batch uploads
    input_buf = gpu.create_buffer(
        args.batch_size * args.channels_count * args.image_size * args.image_size * 4)
    d_scores_buf = gpu.create_buffer(args.batch_size * 4)

    # Validation pairs — cross-gen composite, same method as training.
    # Hold out val_gen from training; val uses cross-gen pairs involving val_gen.
    val_pairs = sample_pairs(entries, 2000, min_gap=0.5, exclude_gen=None)
    # Keep only pairs where at least one side is from val_gen
    val_pairs = [(w, l) for w, l in val_pairs
                 if entries[w][3] == args.val_gen or entries[l][3] == args.val_gen]
    log.info('Validation pairs: %d (gen %d)', len(val_pairs), args.val_gen)

    best_val_acc = 0.0
    patience = 0

    # Stage-training per-channel LR multipliers on conv1 (model.layers[0]).
    # Switches at epoch == args.warmup_epochs + 1 from warmup → settle. Both
    # may be None — then this is a no-op.
    conv1 = model.layers[0]
    current_lr_mult: str = 'none'

    def _set_lr_mult(mults: np.ndarray | None, phase: str) -> None:
        nonlocal current_lr_mult
        if mults is None or current_lr_mult == phase:
            return
        conv1.set_per_input_channel_lr_mult(mults)
        log.info('Conv1 per-channel LR mult → %s: %s', phase, mults.tolist())
        current_lr_mult = phase

    # Apply initial (warmup if configured, else settle, else none)
    if args.warmup_epochs > 0 and lr_mult_warmup is not None:
        _set_lr_mult(lr_mult_warmup, 'warmup')
    elif lr_mult_settle is not None:
        _set_lr_mult(lr_mult_settle, 'settle')

    for epoch in range(1, args.epochs + 1):
        # Switch warmup → settle at the boundary
        if (args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1
                and lr_mult_settle is not None):
            _set_lr_mult(lr_mult_settle, 'settle')

        t0 = time.time()

        # Sample training pairs from ALL generations except val
        train_pairs = sample_pairs(entries, args.pairs_per_epoch, min_gap=0,
                                   exclude_gen=args.val_gen)

        # Train
        epoch_loss = 0.0
        n_batches = 0
        rng = np.random.default_rng()
        pair_arr = np.array(train_pairs)
        rng.shuffle(pair_arr)

        for i in range(0, len(pair_arr), args.batch_size):
            batch = pair_arr[i:i + args.batch_size]
            if len(batch) < 2:
                continue
            B = len(batch)
            w_imgs = store.get_batch(batch[:, 0])
            l_imgs = store.get_batch(batch[:, 1])

            # Pad to batch_size if needed
            if B < args.batch_size:
                pad_shape = (args.batch_size - B, *w_imgs.shape[1:])
                w_imgs = np.concatenate([w_imgs, np.zeros(pad_shape, dtype=np.float32)])
                l_imgs = np.concatenate([l_imgs, np.zeros(pad_shape, dtype=np.float32)])

            # Forward winner → backward winner → forward loser → backward loser
            # This avoids saving/restoring activations between passes
            model.zero_grad()

            gpu.upload(input_buf, w_imgs)
            w_out = model.forward(input_buf, args.batch_size, (args.channels_count, args.image_size, args.image_size))
            w_scores = gpu.download(w_out, np.float32, args.batch_size)[:B]

            gpu.upload(input_buf, l_imgs)
            l_out = model.forward(input_buf, args.batch_size, (args.channels_count, args.image_size, args.image_size))
            l_scores = gpu.download(l_out, np.float32, args.batch_size)[:B]

            # Margin ranking loss
            margin = 1.0
            diff = w_scores - l_scores
            losses = np.maximum(0, margin - diff)
            loss = losses.mean()
            if np.isnan(loss):
                log.warning('NaN loss at batch %d, skipping', i)
                continue

            active = (losses > 0).astype(np.float32)
            d_l_scores = np.zeros(args.batch_size, dtype=np.float32)
            d_l_scores[:B] = active / B

            # Backward loser (state is from loser forward, which just ran)
            gpu.upload(d_scores_buf, d_l_scores)
            model.backward(d_scores_buf, args.batch_size)

            # Re-forward winner to restore layer state, then backward
            gpu.upload(input_buf, w_imgs)
            model.forward(input_buf, args.batch_size, (args.channels_count, args.image_size, args.image_size))
            d_w_scores = np.zeros(args.batch_size, dtype=np.float32)
            d_w_scores[:B] = -active / B
            gpu.upload(d_scores_buf, d_w_scores)
            model.backward(d_scores_buf, args.batch_size)

            model.sgd_step(args.lr)

            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validate
        val_indices = sorted(set(w for w, _ in val_pairs) | set(l for _, l in val_pairs))
        val_scores = {}
        for i in range(0, len(val_indices), args.batch_size):
            batch_idx = val_indices[i:i + args.batch_size]
            batch_imgs = store.get_batch(batch_idx)
            if len(batch_imgs) < args.batch_size:
                pad = np.zeros((args.batch_size - len(batch_imgs), *batch_imgs.shape[1:]),
                               dtype=np.float32)
                batch_imgs = np.concatenate([batch_imgs, pad])
            gpu.upload(input_buf, batch_imgs)
            out = model.forward(input_buf, args.batch_size,
                                (args.channels_count, args.image_size, args.image_size))
            scores = gpu.download(out, np.float32, args.batch_size)
            for j, idx in enumerate(batch_idx):
                val_scores[idx] = scores[j]

        correct = sum(1 for w, l in val_pairs if val_scores.get(w, 0) > val_scores.get(l, 0))
        val_acc = correct / max(len(val_pairs), 1)

        elapsed = time.time() - t0
        log.info('Epoch %2d/%d  loss=%.4f  val_acc=%.3f  cache=%d  (%.0fs)',
                 epoch, args.epochs, avg_loss, val_acc, len(store._cache), elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            weights = model.save_weights()
            from flame_sheep.genome.scoring.cnn_scorer import save_cnn_weights_file
            save_cnn_weights_file(str(output_path), weights, NORMALIZATION_VERSION)
            log.info('  -> saved (best val_acc=%.3f)', best_val_acc)
        else:
            patience += 1
            if patience >= 15:
                log.info('Early stopping (15 epochs without improvement)')
                break

    log.info('Training complete. Best val accuracy: %.3f', best_val_acc)
    gpu.destroy()


if __name__ == '__main__':
    main()
