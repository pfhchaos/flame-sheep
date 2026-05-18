#!/usr/bin/env python3
"""Benchmark a CNN scorer model: inference cost, accuracy, score distribution.

Usage:
    python tools/benchmark_model.py --weights flame_sheep/data/cnn_scorer_vk.npy
    python tools/benchmark_model.py --weights ~/datasets/esheep-cnn/cnn_scorer_domain_25k.npy --channels domain
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import resource
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def measure_inference(weights_path: str, n_runs: int = 1000) -> dict:
    """Measure CPU time and RSS for inference."""
    import torch
    from flame_sheep.cnn_scorer import AestheticNetVk, load_vk_weights

    model = AestheticNetVk()
    load_vk_weights(model, weights_path)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(1, 4, 256, 256)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    # Measure RSS before
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

    # Timed run — measure wall time and CPU time
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    cpu_after = time.process_time()
    wall_after = time.perf_counter()

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

    return {
        'params': n_params,
        'layers': str(model.LAYERS),
        'n_runs': n_runs,
        'cpu_time_ms': (cpu_after - cpu_before) / n_runs * 1000,
        'wall_time_ms': (wall_after - wall_before) / n_runs * 1000,
        'peak_rss_mb': rss_after / 1024,
        'throughput_per_sec': n_runs / (wall_after - wall_before),
    }


def measure_es_accuracy(weights_path: str, data_dir: str,
                        channels: str = 'rgb') -> dict:
    """Measure cross-gen validation accuracy on Electric Sheep data."""
    import torch
    from flame_sheep.cnn_scorer import AestheticNetVk, load_vk_weights
    from PIL import Image

    model = AestheticNetVk()
    load_vk_weights(model, weights_path)
    model.eval()

    # Load manifest
    manifest = Path(data_dir) / 'manifest.csv'
    entries = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            entries.append((row['static_path'], row['swept_path'],
                            int(row['rating']), int(row['generation'])))

    # Build composite scores
    gen_map = {}
    for i, (_, _, _, gen) in enumerate(entries):
        gen_map.setdefault(gen, []).append(i)
    sorted_gens = sorted(gen_map.keys())
    gen_rank = {g: rank for rank, g in enumerate(sorted_gens)}

    composites = np.zeros(len(entries))
    for gen, indices in gen_map.items():
        ratings = np.array([entries[i][2] for i in indices], dtype=np.float64)
        rmin, rmax = ratings.min(), ratings.max()
        norm = (ratings - rmin) / (rmax - rmin) if rmax > rmin else np.full(len(ratings), 0.5)
        for j, idx in enumerate(indices):
            composites[idx] = gen_rank[gen] + norm[j]

    # Sample unrestricted cross-gen pairs
    rng = np.random.default_rng(42)
    all_idx = list(range(len(entries)))
    pairs = []
    for _ in range(20000):
        a, b = rng.choice(len(all_idx), 2, replace=False)
        if abs(composites[a] - composites[b]) < 0.5:
            continue
        if composites[a] > composites[b]:
            pairs.append((a, b))
        else:
            pairs.append((b, a))
        if len(pairs) >= 2000:
            break

    # Score unique genomes
    unique_idx = sorted(set(w for w, _ in pairs) | set(l for _, l in pairs))

    def load_image(idx):
        static, swept, _, _ = entries[idx]
        img_dir = Path(data_dir)

        if channels == 'domain':
            from flame_sheep.scoring_channels import normalize_channels, load_raw_histograms
            hist_name = static.replace('_static.png', '_hist.npz')
            hist_path = img_dir / hist_name
            if hist_path.exists():
                raw = load_raw_histograms(str(hist_path))
                arr = normalize_channels(
                    raw['static_hits'], raw['static_colors'],
                    raw['swept_hits'], raw.get('first_hit'),
                    output_size=256)
                return torch.from_numpy(arr).unsqueeze(0)

        s = Image.open(img_dir / static).convert('RGB').resize((256, 256), Image.LANCZOS)
        w = Image.open(img_dir / swept).convert('L').resize((256, 256), Image.LANCZOS)
        arr = np.zeros((4, 256, 256), dtype=np.float32)
        arr[:3] = np.array(s, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr[3] = np.array(w, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    scores = {}
    for i, idx in enumerate(unique_idx):
        inp = load_image(idx)
        with torch.no_grad():
            scores[idx] = model(inp).item()
        if (i + 1) % 500 == 0:
            log.info('  scored %d/%d', i + 1, len(unique_idx))

    correct = sum(1 for w, l in pairs if scores[w] > scores[l])
    acc = correct / len(pairs)

    # Score distribution
    all_scores = list(scores.values())

    return {
        'es_val_pairs': len(pairs),
        'es_val_accuracy': acc,
        'score_mean': float(np.mean(all_scores)),
        'score_std': float(np.std(all_scores)),
        'score_min': float(np.min(all_scores)),
        'score_max': float(np.max(all_scores)),
        'score_range': float(np.max(all_scores) - np.min(all_scores)),
    }


def measure_personal_accuracy(weights_path: str,
                               channels: str = 'rgb') -> dict:
    """Measure pairwise accuracy against personal votes."""
    import torch
    import sqlite3
    import io
    from flame_sheep.cnn_scorer import AestheticNetVk, load_vk_weights
    from PIL import Image

    model = AestheticNetVk()
    load_vk_weights(model, weights_path)
    model.eval()

    db = sqlite3.connect(str(Path.home() / '.local/share/flame-sheep/library.db'))

    # Get pairwise ratings
    pairs = db.execute(
        'SELECT winner_id, loser_id FROM pairwise_ratings'
    ).fetchall()

    if not pairs:
        # Fall back to genome ratings
        rows = db.execute(
            '''SELECT target_id, SUM(rating) as net
               FROM ratings WHERE target_type = 'genome'
               GROUP BY target_id HAVING SUM(rating) != 0'''
        ).fetchall()
        liked = [gid for gid, net in rows if net > 0]
        disliked = [gid for gid, net in rows if net < 0]
        pairs = []
        rng = np.random.default_rng(42)
        for _ in range(min(2000, len(liked) * len(disliked))):
            w = rng.choice(liked)
            l = rng.choice(disliked)
            pairs.append((w, l))

    if not pairs:
        return {'personal_pairs': 0, 'personal_accuracy': None}

    # Score unique genomes
    unique_ids = sorted(set(w for w, _ in pairs) | set(l for _, l in pairs))

    def load_genome_image(gid):
        if channels == 'domain':
            row = db.execute(
                'SELECT hist_static, hist_swept, hist_first_hit FROM genome_blobs WHERE genome_id = ?',
                (gid,)
            ).fetchone()
            if row is None or row[0] is None or row[1] is None:
                return None
            from flame_sheep.cnn_scorer import _prepare_input_domain
            return _prepare_input_domain(row[0], row[1], row[2])

        row = db.execute(
            'SELECT render_static, render_swept FROM genome_blobs WHERE genome_id = ?',
            (gid,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        s = Image.open(io.BytesIO(row[0])).convert('RGB').resize((256, 256), Image.LANCZOS)
        w = Image.open(io.BytesIO(row[1])).convert('L').resize((256, 256), Image.LANCZOS)
        arr = np.zeros((4, 256, 256), dtype=np.float32)
        arr[:3] = np.array(s, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr[3] = np.array(w, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    scores = {}
    for gid in unique_ids:
        inp = load_genome_image(gid)
        if inp is not None:
            with torch.no_grad():
                scores[gid] = model(inp).item()

    valid_pairs = [(w, l) for w, l in pairs if w in scores and l in scores]
    correct = sum(1 for w, l in valid_pairs if scores[w] > scores[l])
    acc = correct / max(len(valid_pairs), 1)

    db.close()

    return {
        'personal_pairs': len(valid_pairs),
        'personal_accuracy': acc,
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description='Benchmark CNN scorer model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to .npy weights file')
    parser.add_argument('--data', type=str, default=str(Path.home() / 'datasets/esheep-cnn'),
                        help='Path to ES training data')
    parser.add_argument('--channels', type=str, default='rgb',
                        choices=['rgb', 'domain'],
                        help='Input channel format')
    parser.add_argument('--n-infer', type=int, default=1000,
                        help='Number of inference runs for timing')
    parser.add_argument('--skip-es', action='store_true',
                        help='Skip ES validation (slow)')
    parser.add_argument('--skip-personal', action='store_true',
                        help='Skip personal accuracy test')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results as JSON')
    args = parser.parse_args()

    results = {'weights': args.weights, 'channels': args.channels}

    # Inference benchmark
    log.info('=== Inference benchmark ===')
    infer = measure_inference(args.weights, n_runs=args.n_infer)
    results.update(infer)
    log.info(f'  Params: {infer["params"]:,}')
    log.info(f'  CPU time: {infer["cpu_time_ms"]:.2f}ms/genome')
    log.info(f'  Wall time: {infer["wall_time_ms"]:.2f}ms/genome')
    log.info(f'  Peak RSS: {infer["peak_rss_mb"]:.0f}MB')
    log.info(f'  Throughput: {infer["throughput_per_sec"]:.0f}/sec')

    # ES accuracy
    if not args.skip_es:
        log.info('=== ES validation accuracy ===')
        es = measure_es_accuracy(args.weights, args.data, args.channels)
        results.update(es)
        log.info(f'  Accuracy: {es["es_val_accuracy"]:.3f} ({es["es_val_pairs"]} pairs)')
        log.info(f'  Score distribution: {es["score_mean"]:.3f} ± {es["score_std"]:.3f} '
                 f'[{es["score_min"]:.3f}, {es["score_max"]:.3f}]')

    # Personal accuracy
    if not args.skip_personal:
        log.info('=== Personal accuracy ===')
        personal = measure_personal_accuracy(args.weights, args.channels)
        results.update(personal)
        if personal['personal_accuracy'] is not None:
            log.info(f'  Accuracy: {personal["personal_accuracy"]:.3f} '
                     f'({personal["personal_pairs"]} pairs)')
        else:
            log.info('  No personal votes available')

    # Summary
    log.info('=== Summary ===')
    log.info(f'  Model: {results["params"]:,} params')
    log.info(f'  Cost: {results["cpu_time_ms"]:.1f}ms CPU, {results["peak_rss_mb"]:.0f}MB RSS')
    if 'es_val_accuracy' in results:
        log.info(f'  ES accuracy: {results["es_val_accuracy"]:.1%}')
    if results.get('personal_accuracy') is not None:
        log.info(f'  Personal accuracy: {results["personal_accuracy"]:.1%}')
    if 'score_std' in results:
        log.info(f'  Score spread: {results["score_std"]:.3f} std, {results["score_range"]:.3f} range')

    # Save JSON
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f'Results saved to {args.output}')
    else:
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
