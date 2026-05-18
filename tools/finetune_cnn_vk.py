#!/usr/bin/env python3
"""Fine-tune CNN aesthetic scorer on personal preferences from DB.

Reads pairwise_ratings (A/B comparisons) and genome ratings (thumbs up/down)
directly from the library DB. No pre-exported dataset needed.

Usage:
    python tools/finetune_cnn_vk.py
    python tools/finetune_cnn_vk.py --epochs 30 --lr 0.0003
    python tools/finetune_cnn_vk.py --model-size 55k --base-weights path/to/55k.npy
"""
from __future__ import annotations

import argparse
import io
import logging
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.vk_compute import VkCompute

# Reuse training infrastructure from train_cnn_vk
from train_cnn_vk import MODEL_CONFIGS, MLP_HIDDEN

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB data loading
# ---------------------------------------------------------------------------

def _load_image_rgb(static_png: bytes, swept_png: bytes | None,
                    image_size: int) -> np.ndarray:
    """Convert DB render blobs to (4, H, W) float32 RGB+swept input."""
    sz = image_size
    s_img = Image.open(io.BytesIO(static_png)).convert('RGB').resize(
        (sz, sz), Image.LANCZOS)
    img = np.zeros((4, sz, sz), dtype=np.float32)
    img[:3] = np.array(s_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if swept_png is not None:
        w_img = Image.open(io.BytesIO(swept_png)).convert('L').resize(
            (sz, sz), Image.LANCZOS)
        img[3] = np.array(w_img, dtype=np.float32) / 255.0
    return img


def _load_image_domain(hist_static: bytes, hist_swept: bytes,
                       hist_first_hit: bytes | None,
                       image_size: int) -> np.ndarray:
    """Convert histogram blobs to (4, H, W) float32 domain-native input."""
    from flame_sheep.scoring_channels import (
        unpack_static_histogram, unpack_histogram, normalize_channels,
    )
    hits, colors = unpack_static_histogram(hist_static)
    swept = unpack_histogram(hist_swept)
    first_hit = unpack_histogram(hist_first_hit, dtype=np.uint8) if hist_first_hit else None
    return normalize_channels(hits, colors, swept, first_hit, output_size=image_size)


class DbImageStore:
    """Lazy-loading image store backed by the library DB."""

    def __init__(self, db_path: str, image_size: int, channels: str = 'rgb',
                 cache_gb: float = 2.0):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.image_size = image_size
        self._channels = channels

        bytes_per_image = 4 * image_size * image_size * 4
        self.max_cache = int(cache_gb * 1e9 / bytes_per_image)
        self._cache: dict[int, np.ndarray] = {}
        self._access_order: list[int] = []

    def get(self, genome_id: int) -> np.ndarray | None:
        """Load a genome's image by ID. Returns (4, H, W) float32 or None."""
        if genome_id in self._cache:
            return self._cache[genome_id]

        if self._channels == 'domain':
            row = self.conn.execute(
                'SELECT hist_static, hist_swept, hist_first_hit FROM genome_blobs WHERE genome_id=?',
                (genome_id,)).fetchone()
            if row is None or row['hist_static'] is None or row['hist_swept'] is None:
                return None
            img = _load_image_domain(
                row['hist_static'], row['hist_swept'], row['hist_first_hit'],
                self.image_size)
        else:
            row = self.conn.execute(
                'SELECT render_static, render_swept FROM genome_blobs WHERE genome_id=?',
                (genome_id,)).fetchone()
            if row is None or row['render_static'] is None:
                return None
            img = _load_image_rgb(
                row['render_static'], row['render_swept'], self.image_size)

        # LRU cache
        while len(self._cache) >= self.max_cache:
            old = self._access_order.pop(0)
            self._cache.pop(old, None)
        self._cache[genome_id] = img
        self._access_order.append(genome_id)
        return img

    def get_batch(self, ids: list[int]) -> np.ndarray:
        """Load a batch of images. Returns (B, 4, H, W) float32."""
        batch = np.zeros((len(ids), 4, self.image_size, self.image_size),
                         dtype=np.float32)
        for i, gid in enumerate(ids):
            img = self.get(gid)
            if img is not None:
                batch[i] = img
        return batch

    def close(self):
        self.conn.close()


def load_training_pairs(db_path: str) -> tuple[list[tuple[int, int]], dict]:
    """Load all training pairs from DB.

    Returns:
        pairs: list of (winner_id, loser_id)
        stats: dict with counts
    """
    import sqlite3
    conn = sqlite3.connect(db_path)

    # 1. Direct pairwise comparisons
    pairwise = conn.execute(
        'SELECT winner_id, loser_id FROM pairwise_ratings'
    ).fetchall()
    pairwise_pairs = [(r[0], r[1]) for r in pairwise]

    # 2. Thumbs up/down → synthetic pairs
    ratings = conn.execute('''
        SELECT target_id, SUM(rating) as net
        FROM ratings WHERE target_type='genome'
        GROUP BY target_id
    ''').fetchall()

    liked_ids = [r[0] for r in ratings if r[1] > 0]
    disliked_ids = [r[0] for r in ratings if r[1] < 0]

    # Filter to genomes that have renders
    rendered = set(r[0] for r in conn.execute(
        'SELECT genome_id FROM genome_blobs WHERE render_static IS NOT NULL'
    ).fetchall())
    liked_ids = [gid for gid in liked_ids if gid in rendered]
    disliked_ids = [gid for gid in disliked_ids if gid in rendered]

    # Sample liked-vs-disliked pairs (cap to avoid overwhelming pairwise data)
    rng = np.random.default_rng()
    thumbs_pairs = []
    if liked_ids and disliked_ids:
        # Generate up to 2x the pairwise count from thumbs
        max_thumbs = len(pairwise_pairs) * 2
        for _ in range(max_thumbs):
            w = rng.choice(liked_ids)
            l = rng.choice(disliked_ids)
            thumbs_pairs.append((w, l))

    all_pairs = pairwise_pairs + thumbs_pairs
    rng.shuffle(all_pairs)

    conn.close()

    stats = {
        'pairwise': len(pairwise_pairs),
        'thumbs_liked': len(liked_ids),
        'thumbs_disliked': len(disliked_ids),
        'thumbs_pairs': len(thumbs_pairs),
        'total': len(all_pairs),
    }
    return all_pairs, stats


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-5s %(message)s',
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description='Fine-tune CNN scorer on personal preferences from DB')
    parser.add_argument('--base-weights', type=str,
                        default=str(Path(__file__).resolve().parent.parent /
                                    'flame_sheep/data/cnn_scorer_vk.npy'),
                        help='Base weights to fine-tune from')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Fraction of pairs held out for validation')
    parser.add_argument('--model-size', type=str, default=None,
                        choices=['25k', '55k', '100k'],
                        help='Model size (auto-detected from base weights if omitted)')
    parser.add_argument('--mlp-head', action='store_true',
                        help='Use MLP head (auto-detected from base weights if omitted)')
    parser.add_argument('--channels', type=str, default='rgb',
                        choices=['rgb', 'domain'],
                        help='Input channels: rgb (RGB+swept) or domain (histogram H/S/L/A)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output weights path (default: cnn_scorer_personal_vk.npy)')
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / 'flame_sheep/data/cnn_scorer_personal_vk.npy')

    # Detect model size from base weights
    base_weights = np.load(args.base_weights).astype(np.float32)
    n_params = len(base_weights)
    log.info('Base weights: %d params from %s', n_params, args.base_weights)

    import train_cnn_vk

    if args.model_size:
        train_cnn_vk.LAYERS = MODEL_CONFIGS[args.model_size]
        train_cnn_vk.MLP_HEAD = args.mlp_head
    else:
        # Auto-detect model size and head type from param count
        matched = False
        for size_name, config in MODEL_CONFIGS.items():
            C = config[-1][1]
            conv_params = sum(co*ci*k*k + co for ci,co,k,_,_ in config)
            # Check linear head
            if conv_params + C + 1 == n_params:
                train_cnn_vk.LAYERS = config
                train_cnn_vk.MLP_HEAD = False
                log.info('Auto-detected: %s linear (%d params)', size_name, n_params)
                matched = True
                break
            # Check MLP head
            mlp_params = C * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN + 1
            if conv_params + mlp_params == n_params:
                train_cnn_vk.LAYERS = config
                train_cnn_vk.MLP_HEAD = True
                log.info('Auto-detected: %s MLP (%d params)', size_name, n_params)
                matched = True
                break
        if not matched:
            log.error('Cannot auto-detect model size for %d params. '
                      'Known linear: %s, MLP: %s', n_params,
                      [sum(co*ci*k*k+co for ci,co,k,_,_ in c)+c[-1][1]+1 for c in MODEL_CONFIGS.values()],
                      [sum(co*ci*k*k+co for ci,co,k,_,_ in c)+c[-1][1]*MLP_HIDDEN+MLP_HIDDEN+MLP_HIDDEN+1
                       for c in MODEL_CONFIGS.values()])
            sys.exit(1)

    # Load training data from DB
    from flame_sheep.storage import _db_path
    db = str(_db_path())
    all_pairs, stats = load_training_pairs(db)
    log.info('Training data: %d pairwise + %d thumbs-derived = %d total pairs',
             stats['pairwise'], stats['thumbs_pairs'], stats['total'])
    log.info('Thumbs: %d liked, %d disliked genomes',
             stats['thumbs_liked'], stats['thumbs_disliked'])

    if len(all_pairs) < 10:
        log.error('Not enough training pairs')
        sys.exit(1)

    # Train/val split (by pair, not by genome — some genomes appear in both)
    rng = np.random.default_rng(42)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - args.val_fraction))
    train_pairs = [all_pairs[i] for i in indices[:split]]
    val_pairs = [all_pairs[i] for i in indices[split:]]
    log.info('Split: %d train, %d val', len(train_pairs), len(val_pairs))

    # Image store
    store = DbImageStore(db, args.image_size, channels=args.channels)

    # Init GPU model
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)

    import train_cnn_vk
    from flame_sheep.wallpaper_ml import build_cnn_scorer
    model = build_cnn_scorer(gpu, train_cnn_vk.LAYERS,
                              batch_size=args.batch_size,
                              image_size=args.image_size,
                              mlp_head=train_cnn_vk.MLP_HEAD)
    model.load_weights(base_weights)
    log.info('Loaded base weights: %d params', model.param_count())

    # Input/gradient buffers
    input_buf = gpu.create_buffer(args.batch_size * 4 * args.image_size * args.image_size * 4)
    d_scores_buf = gpu.create_buffer(args.batch_size * 4)

    best_acc = 0.0
    best_weights = None
    patience = 0
    max_patience = 8

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Shuffle training pairs each epoch
        rng.shuffle(train_pairs)

        # Train
        epoch_loss = 0.0
        n_batches = 0
        BS = args.batch_size
        IMG_SZ = args.image_size
        for i in range(0, len(train_pairs), BS):
            batch = train_pairs[i:i + BS]
            if len(batch) < 2:
                continue

            B = len(batch)
            winner_ids = [w for w, _ in batch]
            loser_ids = [l for _, l in batch]
            w_imgs = store.get_batch(winner_ids)
            l_imgs = store.get_batch(loser_ids)

            # Pad to batch_size if needed
            if B < BS:
                pad_shape = (BS - B, *w_imgs.shape[1:])
                w_imgs = np.concatenate([w_imgs, np.zeros(pad_shape, dtype=np.float32)])
                l_imgs = np.concatenate([l_imgs, np.zeros(pad_shape, dtype=np.float32)])

            model.zero_grad()

            # Forward winner
            gpu.upload(input_buf, w_imgs)
            w_out = model.forward(input_buf, BS, (4, IMG_SZ, IMG_SZ))
            w_scores = gpu.download(w_out, np.float32, BS)[:B]

            # Forward loser
            gpu.upload(input_buf, l_imgs)
            l_out = model.forward(input_buf, BS, (4, IMG_SZ, IMG_SZ))
            l_scores = gpu.download(l_out, np.float32, BS)[:B]

            # Margin ranking loss
            margin = 1.0
            diff = w_scores - l_scores
            losses = np.maximum(0, margin - diff)
            loss = losses.mean()
            if np.isnan(loss):
                continue
            active = (losses > 0).astype(np.float32)

            # Backward loser (state from loser forward)
            d_l = np.zeros(BS, dtype=np.float32)
            d_l[:B] = active / B
            gpu.upload(d_scores_buf, d_l)
            model.backward(d_scores_buf, BS)

            # Re-forward winner, then backward
            gpu.upload(input_buf, w_imgs)
            model.forward(input_buf, BS, (4, IMG_SZ, IMG_SZ))
            d_w = np.zeros(BS, dtype=np.float32)
            d_w[:B] = -active / B
            gpu.upload(d_scores_buf, d_w)
            model.backward(d_scores_buf, BS)

            model.sgd_step(args.lr)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation: pairwise accuracy (batch single images, padded to BS)
        correct = 0
        n_val = 0
        val_cache = {}
        val_indices = sorted(set(w for w, _ in val_pairs) | set(l for _, l in val_pairs))
        for vi in range(0, len(val_indices), BS):
            batch_idx = val_indices[vi:vi + BS]
            batch_imgs = store.get_batch(batch_idx)
            if len(batch_imgs) < BS:
                pad = np.zeros((BS - len(batch_imgs), *batch_imgs.shape[1:]),
                               dtype=np.float32)
                batch_imgs = np.concatenate([batch_imgs, pad])
            gpu.upload(input_buf, batch_imgs)
            out = model.forward(input_buf, BS, (4, IMG_SZ, IMG_SZ))
            scores = gpu.download(out, np.float32, BS)
            for j, idx in enumerate(batch_idx):
                val_cache[idx] = scores[j]
        for w_id, l_id in val_pairs:
            if w_id in val_cache and l_id in val_cache:
                if val_cache[w_id] > val_cache[l_id]:
                    correct += 1
                n_val += 1
        val_acc = correct / max(n_val, 1)

        elapsed = time.time() - t0
        saved = ''
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.save_weights()
            np.save(output, best_weights)
            saved = f'  -> saved (best={best_acc:.3f})'
            patience = 0
        else:
            patience += 1

        log.info('Epoch %2d/%d  loss=%.4f  val_acc=%.3f  (%d pairs, %.0fs)%s',
                 epoch, args.epochs, avg_loss, val_acc, n_val, elapsed, saved)

        if patience >= max_patience:
            log.info('Early stopping after %d epochs without improvement', max_patience)
            break

    log.info('Best val accuracy: %.3f', best_acc)
    log.info('Weights saved to: %s', output)

    store.close()
    gpu.destroy()


if __name__ == '__main__':
    main()
