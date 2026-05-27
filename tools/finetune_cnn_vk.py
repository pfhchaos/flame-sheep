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
from wallpaper_ml.vk_compute import VkCompute

# Reuse training infrastructure from train_cnn_vk
from train_cnn_vk import MODEL_CONFIGS, MLP_HIDDEN, _parse_csv_floats

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
                       image_size: int,
                       normalization: tuple | None = None,
                       n_channels: int = 4) -> np.ndarray:
    """Convert histogram blobs to (n_channels, H, W) float32 domain input.

    n_channels=4 (default): H, S, L, A. n_channels=3: drops A entirely.

    With normalization=(mean, std), applies sentinel-aware standardization
    (hit pixels z-scored, sentinels snapped to SENTINEL_STANDARDIZED).
    Without normalization, returns the raw channels at native sentinel
    encoding (1.0 = never hit) for inspection/visualization.
    """
    from flame_sheep.genome.scoring.scoring_channels import (
        unpack_static_histogram, unpack_histogram, build_cnn_input,
    )
    hits, colors = unpack_static_histogram(hist_static)
    swept = unpack_histogram(hist_swept)
    first_hit = unpack_histogram(hist_first_hit, dtype=np.uint8) if hist_first_hit else None
    return build_cnn_input(hits, colors, swept, first_hit,
                           normalization=normalization, output_size=image_size,
                           n_channels=n_channels)


class DbImageStore:
    """Lazy-loading image store backed by the library DB.

    normalization: optional (mean, std) tuple of (4,) arrays. If provided,
    every loaded image is standardized so model inputs match what Kaiming
    weight init expects (zero mean / unit variance per channel). Cached
    after standardization, so changing this in the middle of a run won't
    affect already-loaded images.
    """

    def __init__(self, db_path: str, image_size: int, channels: str = 'rgb',
                 cache_gb: float = 2.0,
                 normalization: tuple | None = None,
                 n_channels: int = 4):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.image_size = image_size
        self._channels = channels
        self._n_channels = n_channels
        self._normalization = normalization

        bytes_per_image = n_channels * image_size * image_size * 4
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
            # Domain path standardizes inside _load_image_domain (sentinel-aware).
            img = _load_image_domain(
                row['hist_static'], row['hist_swept'], row['hist_first_hit'],
                self.image_size, normalization=self._normalization,
                n_channels=self._n_channels)
        else:
            row = self.conn.execute(
                'SELECT render_static, render_swept FROM genome_blobs WHERE genome_id=?',
                (genome_id,)).fetchone()
            if row is None or row['render_static'] is None:
                return None
            img = _load_image_rgb(
                row['render_static'], row['render_swept'], self.image_size)
            # RGB path: no sentinel concept, use legacy whole-image standardize.
            if self._normalization is not None:
                from flame_sheep.genome.scoring.scoring_channels import standardize_channels
                img = standardize_channels(img, *self._normalization)

        # LRU cache
        while len(self._cache) >= self.max_cache:
            old = self._access_order.pop(0)
            self._cache.pop(old, None)
        self._cache[genome_id] = img
        self._access_order.append(genome_id)
        return img

    def get_batch(self, ids: list[int]) -> np.ndarray:
        """Load a batch of images. Returns (B, n_channels, H, W) float32."""
        batch = np.zeros((len(ids), self._n_channels,
                          self.image_size, self.image_size),
                         dtype=np.float32)
        for i, gid in enumerate(ids):
            img = self.get(gid)
            if img is not None:
                batch[i] = img
        return batch

    def close(self):
        self.conn.close()


def load_training_pairs(db_path: str, mode: str = 'mixed') -> tuple[list[tuple[int, int]], dict]:
    """Load training pairs from DB.

    Args:
        db_path: Path to library DB
        mode: 'mixed' (pairwise + thumbs), 'thumbs' (thumbs only), 'pairwise' (pairwise only)

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
    #
    # A thumb is a corpus-relative judgment: "this is liked compared to
    # whatever else was in front of me when I rated it." Two rules govern
    # what's valid training signal:
    #
    # (a) Pair only within a generation. A gen-1 liked vs a gen-1 disliked
    #     were contemporaries — valid pair. A gen-1 liked vs a gen-2
    #     disliked never co-existed in the same corpus — invalid pair
    #     (training poison). Past-generation thumbs stay valid forever as
    #     long as we don't cross-pair them.
    # (b) Gen 0 is excluded entirely. Bootstrap/seed corpus that was so
    #     mixed it actively poisoned earlier training runs.
    #
    # Archived status is NOT a filter: downvoted genomes get archived by
    # the pruner, but their renders survive and their thumbs are exactly
    # the negative signal we need. The renders-exist filter below is the
    # only existence check that matters.
    #
    # Pairwise pairs (above) accumulate across generations — they're
    # self-contained 2-genome comparisons whose meaning doesn't depend
    # on the surrounding population.
    from collections import defaultdict

    ratings_by_gen = conn.execute('''
        SELECT target_id, generation, SUM(rating) as net
        FROM ratings
        WHERE target_type='genome' AND generation > 0
        GROUP BY target_id, generation
    ''').fetchall()

    liked_by_gen: dict[int, list[int]] = defaultdict(list)
    disliked_by_gen: dict[int, list[int]] = defaultdict(list)
    for target_id, gen, net in ratings_by_gen:
        if net > 0:
            liked_by_gen[gen].append(target_id)
        elif net < 0:
            disliked_by_gen[gen].append(target_id)

    # Filter each gen to genomes that have renders
    rendered = set(r[0] for r in conn.execute(
        'SELECT genome_id FROM genome_blobs WHERE render_static IS NOT NULL'
    ).fetchall())
    all_gens = set(liked_by_gen) | set(disliked_by_gen)
    for gen in all_gens:
        liked_by_gen[gen] = [gid for gid in liked_by_gen[gen] if gid in rendered]
        disliked_by_gen[gen] = [gid for gid in disliked_by_gen[gen] if gid in rendered]

    # Usable gens have both sides — needed to form intra-gen pairs.
    usable_gens = sorted(g for g in all_gens
                         if liked_by_gen[g] and disliked_by_gen[g])
    total_K = sum(len(liked_by_gen[g]) for g in usable_gens)
    total_M = sum(len(disliked_by_gen[g]) for g in usable_gens)
    total_KM = sum(len(liked_by_gen[g]) * len(disliked_by_gen[g])
                   for g in usable_gens)

    if usable_gens:
        per_gen_summary = ', '.join(
            f'gen{g}: {len(liked_by_gen[g])}L×{len(disliked_by_gen[g])}D'
            for g in usable_gens)
        log.info('Thumbs by generation: %s', per_gen_summary)

    # Per-pair weights — info-content based, with a global scale factor
    # that preserves backward-compatible loss magnitude (so existing LR
    # tuning still applies).
    #
    # Information content:
    #   - A real pairwise compare = 1 independent judgment.
    #   - A synth pair from gen g comes from a pool of K_g+M_g judgments
    #     producing K_g*M_g pair constraints — each pair is worth
    #     (K_g+M_g)/(K_g*M_g) effective samples.
    #
    # Naive convention "synth=1.0 baseline, real=K*M/(K+M)" was exact for
    # a single generation but over-weights low-density gens when pooled
    # (gen with small K_g*M_g/(K_g+M_g) ratio gets same weight per pair as
    # high-density gen, contradicting the info content).
    #
    # Per-gen convention: synth_g = (K_g+M_g)/(K_g*M_g), real = 1.0. This
    # is info-exact but shrinks aggregate loss scale by total_KM/total_K+M,
    # which would require LR retuning.
    #
    # Compromise: scale every weight by total_KM/(total_K+total_M). For a
    # single gen this gives synth=1.0, real=K*M/(K+M) — identical to the
    # original convention. For multi-gen it redistributes weight within
    # the synth pool according to per-gen info density, while keeping the
    # synth-vs-real aggregate balance constant. LR-stable.
    if total_K > 0 and total_M > 0:
        scale = total_KM / (total_K + total_M)
    else:
        scale = 1.0
    real_weight = scale

    # Synthesize every unique intra-gen pair exactly once. No oversampling
    # — duplicating a pair would just inflate its gradient weight without
    # adding information (the K_g+M_g underlying judgments are fixed).
    thumbs_triples = []
    if usable_gens:
        for gen in usable_gens:
            K_g = len(liked_by_gen[gen])
            M_g = len(disliked_by_gen[gen])
            w_g = ((K_g + M_g) / (K_g * M_g)) * scale
            for w in liked_by_gen[gen]:
                for l in disliked_by_gen[gen]:
                    thumbs_triples.append((w, l, w_g))

    rng = np.random.default_rng()

    pairwise_triples = [(w, l, real_weight) for (w, l) in pairwise_pairs]

    if mode == 'thumbs':
        all_pairs = thumbs_triples
    elif mode == 'pairwise':
        all_pairs = pairwise_triples
    else:  # mixed
        all_pairs = pairwise_triples + thumbs_triples
    rng.shuffle(all_pairs)

    conn.close()

    stats = {
        'pairwise': len(pairwise_pairs),
        'thumbs_liked': total_K,
        'thumbs_disliked': total_M,
        'thumbs_pairs': len(thumbs_triples),
        'thumbs_usable_gens': usable_gens,
        'total': len(all_pairs),
        'mode': mode,
        'real_pair_weight': real_weight,
    }
    return all_pairs, stats


def _backup_existing_weights(output: Path) -> None:
    """If a weights file already exists at the output path, copy it
    aside with a timestamp suffix. Cheap insurance — a bad fine-tune
    is then one `cp` away from being reverted. Always run, not opt-in.
    """
    if not output.exists():
        log.info('No existing weights at %s — fresh output, no backup needed',
                 output)
        return
    import shutil
    import time
    stamp = time.strftime('%Y%m%d_%H%M%S')
    backup = output.with_suffix(output.suffix + f'.backup_{stamp}')
    shutil.copy2(output, backup)
    log.info('Backed up existing weights: %s → %s', output, backup.name)


def _snapshot_voted_genome_scores(db_path: str) -> dict:
    """Snapshot the current cnn_score for every genome the user has
    directly rated (thumbs up or down). After training, comparing old
    vs new scores on the same genomes is the single best aggregate
    signal that the new model actually learned what was taught — more
    informative than val_acc alone, which can move on noise.
    """
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.execute('''
        SELECT g.id, g.cnn_score, r.rating
          FROM genomes g
          JOIN ratings r ON r.target_id = g.id
         WHERE r.target_type = 'genome'
           AND r.source = 'direct'
           AND COALESCE(g.archived, 0) = 0
           AND g.cnn_score IS NOT NULL
         ORDER BY r.rating DESC, g.id
    ''').fetchall()
    conn.close()
    snapshot = {}
    for gid, score, rating in rows:
        snapshot[gid] = {'before_score': float(score), 'rating': int(rating)}
    log.info('Snapshot: %d directly-rated genomes (%d up, %d down)',
             len(snapshot),
             sum(1 for v in snapshot.values() if v['rating'] > 0),
             sum(1 for v in snapshot.values() if v['rating'] < 0))
    return snapshot


def _report_score_changes(snapshot: dict, gpu, model, store,
                           input_buf, batch_size: int,
                           image_size: int, n_channels: int = 4) -> None:
    """After training, forward each snapshot genome through the model
    (now holding the best-val-acc weights) and print before/after
    scores grouped by rating. Tells you whether the new model is
    actually doing what the votes asked for — more informative than
    aggregate val_acc, which can drift on noise.
    """
    if not snapshot:
        return
    import numpy as np
    gids = list(snapshot.keys())
    for i in range(0, len(gids), batch_size):
        batch_ids = gids[i:i + batch_size]
        batch_imgs = store.get_batch(batch_ids)
        # Mark genomes the store couldn't load so we don't count them.
        loaded_count = len(batch_imgs)
        for j, gid in enumerate(batch_ids):
            if j >= loaded_count:
                snapshot[gid]['after_score'] = None
        if loaded_count == 0:
            continue
        if loaded_count < batch_size:
            pad = np.zeros((batch_size - loaded_count, *batch_imgs.shape[1:]),
                            dtype=np.float32)
            batch_imgs = np.concatenate([batch_imgs, pad])
        gpu.upload(input_buf, batch_imgs)
        out = model.forward(input_buf, batch_size, (n_channels, image_size, image_size))
        scores = gpu.download(out, np.float32, batch_size)
        for j in range(loaded_count):
            snapshot[batch_ids[j]]['after_score'] = float(scores[j])

    valid = [v for v in snapshot.values() if v.get('after_score') is not None]
    n_up = sum(1 for v in valid if v['rating'] > 0)
    n_dn = sum(1 for v in valid if v['rating'] < 0)
    n_up_moved_up = sum(1 for v in valid
                        if v['rating'] > 0 and v['after_score'] > v['before_score'])
    n_dn_moved_dn = sum(1 for v in valid
                        if v['rating'] < 0 and v['after_score'] < v['before_score'])
    mean_up_delta = (sum(v['after_score'] - v['before_score']
                          for v in valid if v['rating'] > 0)
                     / max(n_up, 1))
    mean_dn_delta = (sum(v['after_score'] - v['before_score']
                          for v in valid if v['rating'] < 0)
                     / max(n_dn, 1))
    log.info('=== Voted-genome score changes (new model vs old) ===')
    log.info('  upvoted   moved up:   %d/%d (%.0f%%)  mean Δ = %+.3f',
             n_up_moved_up, n_up,
             100 * n_up_moved_up / max(n_up, 1), mean_up_delta)
    log.info('  downvoted moved down: %d/%d (%.0f%%)  mean Δ = %+.3f',
             n_dn_moved_dn, n_dn,
             100 * n_dn_moved_dn / max(n_dn, 1), mean_dn_delta)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-5s %(message)s',
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description='Fine-tune CNN scorer on personal preferences from DB')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Skip --base-weights, init the model from Kaiming. '
                             'Use this when starting fresh against the current '
                             'normalization version. Loading legacy .npy as base '
                             'weights with new standardization gives the model '
                             'two contradictory things to learn at once and '
                             'training stalls.')
    parser.add_argument('--base-weights', type=str,
                        default=str(Path(__file__).resolve().parent.parent /
                                    'flame_sheep/data/cnn_scorer_vk.npy'),
                        help='Base weights to fine-tune from')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max-patience', type=int, default=8,
                        help='Early-stop after this many epochs without '
                             'val_acc improvement (default 8). At low LR '
                             '(1e-4 or 1e-5) per-epoch progress is small '
                             'enough that noise can break short streaks — '
                             'bump to 15-20 for proper fine-tuning LRs so '
                             'slow real learning has time to show through.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--val-fraction', type=float, default=0.15,
                        help='Fraction of pairs held out for validation')
    parser.add_argument('--val-seed', type=int, default=42,
                        help='Seed for the train/val split RNG (default 42). '
                             'Independent of --seed which controls model init.')
    parser.add_argument('--model-size', type=str, default=None,
                        choices=['25k', '55k', '100k'],
                        help='Model size (auto-detected from base weights if omitted)')
    parser.add_argument('--mlp-head', action='store_true',
                        help='Use MLP head (auto-detected from base weights if omitted)')
    parser.add_argument('--channels', type=str, default='domain',
                        choices=['rgb', 'domain'],
                        help='Input channels: domain (histogram H/S/L[/A], default) or rgb (deprecated, palette-index artifact in green channel)')
    parser.add_argument('--channels-count', type=int, default=4, choices=[3, 4],
                        help='Channel count for the target model. Default 4 '
                             '(H/S/L/A) — what personal-data renders provide. '
                             'Use 3 only if you specifically want to drop A '
                             'from the target architecture. If base weights '
                             'have fewer channels than the target, conv1 is '
                             'expanded automatically (see --new-channel-init).')
    parser.add_argument('--new-channel-init', type=str, default='zero',
                        choices=['zero', 'matched', 'kaiming'],
                        help='How to initialize the added conv1 channel planes '
                             'when expanding base weights. zero (default): '
                             'planes start at 0, model learns from gradient '
                             'flow only (slow under low LR). matched: sample '
                             'from N(0, std) matched to existing channels — '
                             'gives the new column parity-scale contribution '
                             'from epoch 1. kaiming: fresh fan_in-scaled '
                             'init. See cnn_scorer.expand_conv1_channels.')
    parser.add_argument('--data-mode', type=str, default='mixed',
                        choices=['mixed', 'thumbs', 'pairwise'],
                        help='Training data: mixed (default), thumbs only (curriculum stage 1), pairwise only (stage 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for weight init (only used with --from-scratch)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output weights path (default: cnn_scorer_personal_vk.npy)')

    # --- Staged training (channel curriculum) — per-channel LR mult for conv1 ---
    parser.add_argument('--per-channel-lr-mult-warmup', type=str, default=None,
                        help='Comma-separated per-input-channel LR multipliers '
                             'for conv1 during the first --warmup-epochs '
                             'epochs, e.g. "1,1,1,3" boosts the A channel.')
    parser.add_argument('--per-channel-lr-mult-settle', type=str, default=None,
                        help='Comma-separated per-input-channel LR multipliers '
                             'for conv1 after warmup elapses, e.g. "1,1,1,1".')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of epochs to use warmup LR multipliers '
                             'before switching to settle.')

    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        Path(__file__).resolve().parent.parent / 'flame_sheep/data/cnn_scorer_personal_vk.npy')

    # Pre-flight safety: back up the existing weights so a bad fine-tune
    # is one `cp` away from reverting. Always-on, not opt-in.
    _backup_existing_weights(output)

    # Load base weights + normalization version stamp (None for legacy .npy).
    # --from-scratch skips this and Kaiming-inits the model after we know
    # the layer config (decided below from --model-size).
    from flame_sheep.genome.scoring.cnn_scorer import (
        load_cnn_weights_file, save_cnn_weights_file,
    )
    base_weights = None
    base_norm_version = None
    if not args.from_scratch:
        base_weights, base_norm_version = load_cnn_weights_file(args.base_weights)
        n_params = len(base_weights)
        log.info('Base weights: %d params from %s (norm_version=%s)',
                 n_params, args.base_weights, base_norm_version)
    else:
        log.info('--from-scratch: skipping --base-weights, will Kaiming-init')

    # rgb mode is always 4-channel; reject mismatched flag
    if args.channels == 'rgb' and args.channels_count != 4:
        log.error('--channels rgb is always 4-channel (R, G, B, swept); '
                  'cannot use --channels-count %d. Drop the flag or use '
                  '--channels domain.', args.channels_count)
        sys.exit(1)

    import train_cnn_vk

    base_in_channels = None
    if args.model_size:
        # Manual override: trust user. Channel count comes from the config.
        chosen = list(MODEL_CONFIGS[args.model_size])
        first = chosen[0]
        chosen[0] = (args.channels_count, first[1], first[2], first[3], first[4])
        train_cnn_vk.LAYERS = chosen
        train_cnn_vk.MLP_HEAD = args.mlp_head
        if base_weights is not None:
            # Best-effort detect base channel count from param count vs known shapes.
            for ch in (3, 4):
                conv_params = sum(co*ci*k*k + co
                                  for (ci, co, k, _, _) in
                                  [(ch, first[1], first[2], first[3], first[4])] + chosen[1:])
                C = chosen[-1][1]
                if (conv_params + C + 1 == n_params or
                    conv_params + C * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN + 1 == n_params):
                    base_in_channels = ch
                    break
            if base_in_channels is None:
                log.warning('Could not infer base channel count from %d params; '
                            'assuming target (%d). If wrong, expansion will fail.',
                            n_params, args.channels_count)
                base_in_channels = args.channels_count
    elif base_weights is None:
        log.error('--from-scratch requires --model-size to be specified '
                  '(no base weights to auto-detect from).')
        sys.exit(1)
    else:
        # Auto-detect model size, head type, AND base channel count from param
        # count. Iterate over plausible base-channel values (3, 4) since the
        # ES base is 3-channel and the personal model is 4-channel.
        matched = False
        for ch in (args.channels_count, 3, 4):
            for size_name, config in MODEL_CONFIGS.items():
                # Override first layer's in_channels for the trial
                first = config[0]
                trial_config = [(ch, first[1], first[2], first[3], first[4])] + list(config[1:])
                C = trial_config[-1][1]
                conv_params = sum(co*ci*k*k + co for ci,co,k,_,_ in trial_config)
                if conv_params + C + 1 == n_params:
                    target_first = (args.channels_count, first[1], first[2], first[3], first[4])
                    train_cnn_vk.LAYERS = [target_first] + list(config[1:])
                    train_cnn_vk.MLP_HEAD = False
                    base_in_channels = ch
                    log.info('Auto-detected: %s linear, base channels=%d → target %d (%d params)',
                             size_name, ch, args.channels_count, n_params)
                    matched = True
                    break
                mlp_params = C * MLP_HIDDEN + MLP_HIDDEN + MLP_HIDDEN + 1
                if conv_params + mlp_params == n_params:
                    target_first = (args.channels_count, first[1], first[2], first[3], first[4])
                    train_cnn_vk.LAYERS = [target_first] + list(config[1:])
                    train_cnn_vk.MLP_HEAD = True
                    base_in_channels = ch
                    log.info('Auto-detected: %s MLP, base channels=%d → target %d (%d params)',
                             size_name, ch, args.channels_count, n_params)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            log.error('Cannot auto-detect model size + channels for %d params. '
                      'Tried base channels (3, 4) × configs.', n_params)
            sys.exit(1)

    # If base channels < target, expand conv1 with new input planes per
    # the chosen init strategy.
    if base_weights is not None and base_in_channels < args.channels_count:
        from flame_sheep.genome.scoring.cnn_scorer import expand_conv1_channels
        log.info('Expanding conv1 channels: %d → %d (init=%s)',
                 base_in_channels, args.channels_count, args.new_channel_init)
        base_weights = expand_conv1_channels(
            base_weights, base_in_channels, args.channels_count,
            train_cnn_vk.LAYERS,
            new_channel_init=args.new_channel_init,
            seed=args.seed)
        n_params = len(base_weights)
        log.info('Expanded base weights: %d params', n_params)

    # Load training data from DB
    from flame_sheep.storage import _db_path
    db = str(_db_path())

    # Snapshot pre-training scores for every directly-rated genome so
    # we can report before/after changes after training completes. The
    # gold-standard signal that the new model actually learned what
    # was taught.
    voted_snapshot = _snapshot_voted_genome_scores(db)

    all_pairs, stats = load_training_pairs(db, mode=args.data_mode)
    log.info('Training data: %d pairwise + %d thumbs-derived = %d total pairs',
             stats['pairwise'], stats['thumbs_pairs'], stats['total'])
    log.info('Thumbs: %d liked, %d disliked genomes',
             stats['thumbs_liked'], stats['thumbs_disliked'])

    if len(all_pairs) < 10:
        log.error('Not enough training pairs')
        sys.exit(1)

    # Train/val split (by pair, not by genome — some genomes appear in both)
    rng = np.random.default_rng(args.val_seed)
    indices = np.arange(len(all_pairs))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - args.val_fraction))
    train_pairs = [all_pairs[i] for i in indices[:split]]
    val_pairs = [all_pairs[i] for i in indices[split:]]
    log.info('Split: %d train, %d val', len(train_pairs), len(val_pairs))

    # Load normalization stats. If present, all model inputs get
    # standardized (zero mean / unit variance per channel) so Kaiming
    # init's assumptions hold. Trained weights are stamped with the
    # version so they can never be silently loaded with mismatched stats.
    from flame_sheep.storage import Library as _Lib, NORMALIZATION_VERSION
    _lib_norm = _Lib()
    normalization = _lib_norm.get_normalization(NORMALIZATION_VERSION)
    _lib_norm.close()
    if normalization is None:
        log.warning('No normalization stats — run tools/compute_normalization.py first. '
                    'Training will proceed with identity normalization (legacy mode).')
    else:
        log.info('Normalization %s: mean=%s std=%s',
                 NORMALIZATION_VERSION,
                 [f'{x:.4f}' for x in normalization[0]],
                 [f'{x:.4f}' for x in normalization[1]])

    if base_norm_version is not None and base_norm_version != NORMALIZATION_VERSION:
        raise RuntimeError(
            f'Base weights normalization version mismatch: weights are '
            f'{base_norm_version}, codebase expects {NORMALIZATION_VERSION}. '
            f'Refusing to load — output scores would be silently wrong. '
            f'Either bump NORMALIZATION_VERSION and rerun '
            f'tools/compute_normalization.py, or load weights that match.')

    # Trim normalization to target channel count (per-channel stats are
    # independent, so slicing the first N entries is exact).
    if (args.channels == 'domain' and args.channels_count == 3
            and normalization is not None):
        normalization = (normalization[0][:3], normalization[1][:3])
        log.info('Trimmed normalization to 3 channels (H, S, L)')

    # Image store
    store = DbImageStore(db, args.image_size, channels=args.channels,
                         n_channels=args.channels_count,
                         normalization=normalization)

    # Init GPU model
    gpu = VkCompute()
    log.info('GPU: %s', gpu.device_name)

    import train_cnn_vk
    from wallpaper_ml import build_cnn_scorer
    model = build_cnn_scorer(gpu, train_cnn_vk.LAYERS,
                              batch_size=args.batch_size,
                              image_size=args.image_size,
                              mlp_head=train_cnn_vk.MLP_HEAD)
    if base_weights is not None:
        model.load_weights(base_weights)
        log.info('Loaded base weights: %d params', model.param_count())
    else:
        model.init_weights(seed=args.seed)
        log.info('Initialized fresh weights: %d params (seed=%d)', model.param_count(), args.seed)

    # Input/gradient buffers
    input_buf = gpu.create_buffer(
        args.batch_size * args.channels_count * args.image_size * args.image_size * 4)
    d_scores_buf = gpu.create_buffer(args.batch_size * 4)

    best_acc = 0.0
    best_weights = None
    patience = 0
    max_patience = args.max_patience

    # Stage-3 (and general) per-channel LR mult support on conv1. When
    # expanding 3→4 channels (A added), A's weights start at random init —
    # give them a higher LR for the first `warmup_epochs` so they find
    # features before the loss landscape locks them out.
    conv1 = model.layers[0]
    lr_mult_warmup = _parse_csv_floats(args.per_channel_lr_mult_warmup,
                                       args.channels_count,
                                       '--per-channel-lr-mult-warmup')
    lr_mult_settle = _parse_csv_floats(args.per_channel_lr_mult_settle,
                                       args.channels_count,
                                       '--per-channel-lr-mult-settle')
    current_lr_mult: list[str] = ['none']

    def _set_lr_mult(mults: np.ndarray | None, phase: str) -> None:
        if mults is None or current_lr_mult[0] == phase:
            return
        conv1.set_per_input_channel_lr_mult(mults)
        log.info('Conv1 per-channel LR mult → %s: %s', phase, mults.tolist())
        current_lr_mult[0] = phase

    if args.warmup_epochs > 0 and lr_mult_warmup is not None:
        _set_lr_mult(lr_mult_warmup, 'warmup')
    elif lr_mult_settle is not None:
        _set_lr_mult(lr_mult_settle, 'settle')

    for epoch in range(1, args.epochs + 1):
        if (args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1
                and lr_mult_settle is not None):
            _set_lr_mult(lr_mult_settle, 'settle')

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
            # Pairs are now (winner_id, loser_id, weight) triples — weight
            # is 1.0 for synthesized thumbs pairs, K*M/(K+M) for real
            # pairwise compares. Without it, synthesized pairs dominate
            # the loss by row count despite carrying less independent
            # information.
            winner_ids = [t[0] for t in batch]
            loser_ids = [t[1] for t in batch]
            pair_weights = np.array([t[2] for t in batch], dtype=np.float32)
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
            w_out = model.forward(input_buf, BS, (args.channels_count, IMG_SZ, IMG_SZ))
            w_scores = gpu.download(w_out, np.float32, BS)[:B]

            # Forward loser
            gpu.upload(input_buf, l_imgs)
            l_out = model.forward(input_buf, BS, (args.channels_count, IMG_SZ, IMG_SZ))
            l_scores = gpu.download(l_out, np.float32, BS)[:B]

            # Weighted margin ranking loss. Each pair contributes its loss
            # scaled by its weight, and the gradient is similarly scaled.
            # Loss is reported as the weighted mean (sum of weighted losses
            # divided by sum of weights) so it stays scale-comparable
            # across batches with different pair-type mixes.
            margin = 1.0
            diff = w_scores - l_scores
            raw_losses = np.maximum(0, margin - diff)
            losses = raw_losses * pair_weights
            weight_sum = float(pair_weights.sum())
            loss = float(losses.sum() / max(weight_sum, 1e-9))
            if np.isnan(loss):
                continue
            # active * weight: pairs with no loss contribute zero gradient;
            # pairs with loss contribute their weighted gradient. Divide
            # by weight_sum (not B) so the gradient magnitude is the
            # weighted-mean gradient.
            active_w = ((raw_losses > 0).astype(np.float32) * pair_weights
                        / max(weight_sum, 1e-9))

            # Backward loser (state from loser forward)
            d_l = np.zeros(BS, dtype=np.float32)
            d_l[:B] = active_w
            gpu.upload(d_scores_buf, d_l)
            model.backward(d_scores_buf, BS)

            # Re-forward winner, then backward
            gpu.upload(input_buf, w_imgs)
            model.forward(input_buf, BS, (args.channels_count, IMG_SZ, IMG_SZ))
            d_w = np.zeros(BS, dtype=np.float32)
            d_w[:B] = -active_w
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
        # val_pairs are (winner, loser, weight) triples — unpack the
        # first two for accuracy. Weights aren't used in val_acc
        # reporting; we treat each pair equally there since the question
        # is "fraction of pairs ranked correctly" not "weighted loss."
        val_indices = sorted(set(t[0] for t in val_pairs)
                             | set(t[1] for t in val_pairs))
        for vi in range(0, len(val_indices), BS):
            batch_idx = val_indices[vi:vi + BS]
            batch_imgs = store.get_batch(batch_idx)
            if len(batch_imgs) < BS:
                pad = np.zeros((BS - len(batch_imgs), *batch_imgs.shape[1:]),
                               dtype=np.float32)
                batch_imgs = np.concatenate([batch_imgs, pad])
            gpu.upload(input_buf, batch_imgs)
            out = model.forward(input_buf, BS, (args.channels_count, IMG_SZ, IMG_SZ))
            scores = gpu.download(out, np.float32, BS)
            for j, idx in enumerate(batch_idx):
                val_cache[idx] = scores[j]
        for w_id, l_id, _ in val_pairs:
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
            save_cnn_weights_file(output, best_weights, NORMALIZATION_VERSION)
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

    # Restore the best weights into the in-process model and report
    # how every directly-rated genome moved under the new model.
    if best_weights is not None:
        model.load_weights(best_weights)
    _report_score_changes(voted_snapshot, gpu, model, store,
                          input_buf, BS, IMG_SZ,
                          n_channels=args.channels_count)

    # Trigger the generation-advance pipeline — ONLY if this run wrote to
    # the deployed weights path. Experiment runs (output to a separate
    # path, e.g. ~/datasets/esheep-cnn/ for fine-tune A/B testing) must
    # NOT hijack production: the score worker hot-reloads on the deployed
    # file's mtime, so latching for an experiment that produced unrelated
    # weights would kick off rescoring with whatever was on disk at the
    # deployed path (which doesn't reflect this experiment at all).
    #
    # Match on resolved canonical path of the deployed location (handles
    # .npy vs .npz, relative paths, symlinks).
    deployed = (Path(__file__).resolve().parent.parent /
                'flame_sheep/data/cnn_scorer_personal_vk').resolve()
    output_resolved = output.resolve()
    is_deployment = (output_resolved.with_suffix('') == deployed)

    if best_weights is not None and is_deployment:
        try:
            from flame_sheep.storage import Library
            _lib = Library()
            target = _lib.get_current_generation() + 1
            won = _lib.cas_gen_advance_state('idle', 'awaiting_rescore',
                                              target=target)
            if won:
                log.info('Latched gen advance: idle → awaiting_rescore '
                         '(target gen %d). Score + pruner workers will '
                         'pick it up.', target)
            else:
                existing = _lib.get_gen_advance_state()
                log.warning('Could not latch gen advance: existing state '
                            'is %r (not idle). Inspect with '
                            'tools/gen_state.py', existing)
            _lib.close()
        except Exception:
            log.exception('Failed to latch gen advance (training succeeded)')
    elif best_weights is not None:
        log.info('Experiment run (output != deployed weights path); '
                 'NOT latching gen advance. Output: %s', output_resolved)

    store.close()
    gpu.destroy()


if __name__ == '__main__':
    main()
