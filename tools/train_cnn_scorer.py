#!/usr/bin/env python3
"""Train the CNN aesthetic scorer on Electric Sheep ratings.

Pairwise ranking: for each pair (A, B) where A is rated higher than B,
train the model so score(A) > score(B). Uses Bradley-Terry loss
(BCE on sigmoid of score difference).

Pairs are sampled within same generation only — voter populations
vary across generations.

Usage:
    python tools/train_cnn_scorer.py --data ~/datasets/esheep-cnn/
    python tools/train_cnn_scorer.py --data ~/datasets/esheep-cnn/ --epochs 30 --val-gen 244
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flame_sheep.cnn_scorer import AestheticNet, SheepDataset, PairSampler

log = logging.getLogger(__name__)


def build_image_tensor(dataset: SheepDataset) -> torch.Tensor:
    """Stack all dataset images into a single (N, 4, 256, 256) float16 tensor.

    Stored as float16 to halve memory (~11GB vs ~22GB for 18K images).
    Cast to float32 per-batch during training.
    """
    if dataset._cache is not None:
        return torch.stack(dataset._cache).half()
    images = []
    for i in range(len(dataset)):
        img, _, _ = dataset[i]
        images.append(img)
    return torch.stack(images).half()


def pairwise_accuracy(model: AestheticNet, images: torch.Tensor,
                      pairs: list[tuple[int, int]], device: str = 'cpu',
                      batch_size: int = 512) -> float:
    """Compute fraction of pairs where model ranks the winner higher."""
    model.eval()

    # Score all images in batches
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].float().to(device)
            all_scores.append(model(batch).cpu())
    scores = torch.cat(all_scores)

    correct = sum(1 for w, l in pairs if scores[w] > scores[l])
    return correct / max(len(pairs), 1)


def train_epoch(model: AestheticNet, images: torch.Tensor,
                pairs: list[tuple[int, int]], optimizer: torch.optim.Optimizer,
                device: str = 'cpu', batch_size: int = 64,
                augment: bool = True) -> float:
    """Train one epoch of pairwise ranking. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Shuffle pairs
    rng = np.random.default_rng()
    pair_arr = np.array(pairs)
    rng.shuffle(pair_arr)

    for i in range(0, len(pair_arr), batch_size):
        batch_pairs = pair_arr[i:i + batch_size]
        w_idx = batch_pairs[:, 0]
        l_idx = batch_pairs[:, 1]

        w_batch = images[w_idx].float().to(device)
        l_batch = images[l_idx].float().to(device)

        # Random horizontal flip for augmentation
        if augment:
            flip_mask = torch.rand(len(w_batch)) > 0.5
            w_batch[flip_mask] = w_batch[flip_mask].flip(-1)
            flip_mask = torch.rand(len(l_batch)) > 0.5
            l_batch[flip_mask] = l_batch[flip_mask].flip(-1)

        w_scores = model(w_batch)
        l_scores = model(l_batch)

        # Bradley-Terry loss: -log(σ(score_winner - score_loser))
        loss = F.binary_cross_entropy_with_logits(
            w_scores - l_scores,
            torch.ones_like(w_scores),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='Train CNN aesthetic scorer')
    parser.add_argument('--data', type=str, required=True,
                        help='Directory with manifest.csv and rendered images')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for trained weights (default: data/cnn_scorer.pt)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--pairs-per-epoch', type=int, default=50000)
    parser.add_argument('--val-gen', type=int, default=None,
                        help='Hold out this generation for validation (default: largest)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )

    data_dir = Path(args.data)
    manifest = data_dir / 'manifest.csv'
    output_path = Path(args.output) if args.output else data_dir / 'cnn_scorer.pt'

    # Load dataset (preload all images into RAM for fast training)
    log.info('Loading dataset from %s...', manifest)
    dataset = SheepDataset(manifest, augment=True, preload=True)
    log.info('Loaded %d genomes (preloaded into RAM)', len(dataset))

    # Split by generation
    gen_map = dataset.by_generation()
    gen_sizes = {g: len(indices) for g, indices in gen_map.items()}
    log.info('Generations: %s', {g: len(v) for g, v in gen_map.items()})

    # Pick validation generation
    if args.val_gen:
        val_gen = args.val_gen
    else:
        # Use largest generation as validation
        val_gen = max(gen_sizes, key=gen_sizes.get)
    log.info('Validation generation: %d (%d genomes)', val_gen, gen_sizes.get(val_gen, 0))

    # Create train/val datasets by masking
    train_indices = []
    val_indices = []
    for gen, indices in gen_map.items():
        if gen == val_gen:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    # Build sub-datasets that share the preloaded cache
    train_entries = [dataset.entries[i] for i in train_indices]
    val_entries = [dataset.entries[i] for i in val_indices]

    train_cache = [dataset._cache[i] for i in train_indices] if dataset._cache else None
    val_cache = [dataset._cache[i] for i in val_indices] if dataset._cache else None

    train_dataset = SheepDataset.__new__(SheepDataset)
    train_dataset.image_dir = dataset.image_dir
    train_dataset.augment = True
    train_dataset.entries = train_entries
    train_dataset._cache = train_cache

    val_dataset = SheepDataset.__new__(SheepDataset)
    val_dataset.image_dir = dataset.image_dir
    val_dataset.augment = False
    val_dataset.entries = val_entries
    val_dataset._cache = val_cache

    train_sampler = PairSampler(train_dataset, pairs_per_epoch=args.pairs_per_epoch)
    val_sampler = PairSampler(val_dataset, pairs_per_epoch=min(5000, len(val_indices) * 5))

    val_pairs = val_sampler.sample_pairs()
    log.info('Train: %d genomes, Val: %d genomes (%d pairs)',
             len(train_entries), len(val_entries), len(val_pairs))

    # Pre-stack images into tensors for fast training
    log.info('Building image tensors...')
    train_images = build_image_tensor(train_dataset)
    val_images = build_image_tensor(val_dataset)
    log.info('Train tensor: %s, Val tensor: %s', train_images.shape, val_images.shape)

    # Model
    model = AestheticNet().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info('Model: %d parameters', n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Sample new training pairs each epoch
        train_pairs = train_sampler.sample_pairs()
        loss = train_epoch(model, train_images, train_pairs, optimizer,
                          device=args.device, batch_size=args.batch_size)
        scheduler.step()

        # Validate
        val_acc = pairwise_accuracy(model, val_images, val_pairs, device=args.device)
        elapsed = time.time() - t0

        log.info('Epoch %2d/%d  loss=%.4f  val_acc=%.3f  lr=%.1e  (%.1fs)',
                 epoch, args.epochs, loss, val_acc,
                 optimizer.param_groups[0]['lr'], elapsed)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            log.info('  → saved (best val_acc=%.3f)', best_val_acc)

    log.info('Training complete. Best val accuracy: %.3f', best_val_acc)
    log.info('Weights saved to: %s', output_path)

    # Final sanity check: score some extremes
    log.info('--- Sanity check: top vs bottom rated ---')
    model.load_state_dict(torch.load(output_path, map_location=args.device, weights_only=True))
    model.eval()

    # Find highest and lowest rated in validation set
    val_by_rating = sorted(enumerate(val_entries), key=lambda x: x[1][2], reverse=True)
    top_5 = val_by_rating[:5]
    bottom_5 = val_by_rating[-5:]

    with torch.no_grad():
        for label, group in [('TOP', top_5), ('BOTTOM', bottom_5)]:
            for local_idx, (_, _, rating, gen) in group:
                score = model(val_images[local_idx].unsqueeze(0).to(args.device)).item()
                log.info('  %s  rating=%3d  score=%.3f', label, rating, score)


if __name__ == '__main__':
    main()
