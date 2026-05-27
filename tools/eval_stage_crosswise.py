"""Disambiguate stage 1 vs stage 2 by evaluating both on both val sets.

Stage 1's training-time val (0.586) used full ES gen244 cross-gen pairs.
Stage 2's training-time val (0.613) used smooth-Q1 gen244 cross-gen pairs
— a different population (~80% of gen244 is NOT smooth-Q1).

This script runs a 2×2 grid (stage × val_set) so we can read:
  - Stage 2 on full ES val: apples-to-apples with stage 1's reported 0.586
  - Stage 1 on smooth-Q1 val: apples-to-apples with stage 2's reported 0.613

Each stage uses its training-time input mask (stage 1: [0,1,1,0], stage 2:
[1,1,1,0]) — feeding mismatched inputs would corrupt the result (e.g.,
stage 1's H weights are at random init; feeding real H into them adds noise).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_cnn_vk import (
    ImageStore, MODEL_CONFIGS, sample_pairs, _load_es_filter,
)

DATA_DIR = Path.home() / 'datasets/esheep-cnn'
STAGE1_W = DATA_DIR / 'cnn_stage1_LS_full_es.npz'
STAGE2_W = DATA_DIR / 'cnn_stage2_LSH_smooth.npz'
ES_DB = Path.home() / '.local/share/flame-sheep/esheep.db'
SMOOTH_FILTER = 'palette_mean_step <= 0.049'

VAL_GEN = 244
IMAGE_SIZE = 256
BATCH_SIZE = 8
N_VAL_PAIRS = 2000

# Stage configurations
STAGES = {
    'stage1': {
        'weights': STAGE1_W,
        'input_mask': np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    },
    'stage2': {
        'weights': STAGE2_W,
        'input_mask': np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    },
}

VAL_SETS = {
    'full_es': None,  # no filter
    'smooth_q1': SMOOTH_FILTER,
}


def evaluate(stage_name: str, val_set_name: str, gpu, normalization) -> float:
    """Returns pairwise val_acc."""
    stage_cfg = STAGES[stage_name]
    es_filter = (_load_es_filter(ES_DB, VAL_SETS[val_set_name])
                 if VAL_SETS[val_set_name] else None)

    store = ImageStore(
        DATA_DIR / 'manifest.csv', DATA_DIR, IMAGE_SIZE,
        cache_gb=4.0, channels='domain', n_channels=4,
        normalization=normalization,
        entries_filter=es_filter,
        input_channel_mask=stage_cfg['input_mask'],
    )
    entries = store.entries

    # Build val pairs: same approach as trainer (cross-gen, at least one side
    # from VAL_GEN, min_gap=0.5 composite score)
    rng = np.random.default_rng(42)  # deterministic across stages
    all_pairs = sample_pairs(entries, N_VAL_PAIRS, min_gap=0.5,
                             exclude_gen=None)
    val_pairs = [(w, l) for w, l in all_pairs
                 if entries[w][3] == VAL_GEN or entries[l][3] == VAL_GEN]

    # Build model + load weights
    from wallpaper_ml import build_cnn_scorer
    from flame_sheep.genome.scoring.cnn_scorer import load_cnn_weights_file

    LAYERS = list(MODEL_CONFIGS['25k'])
    model = build_cnn_scorer(gpu, LAYERS, batch_size=BATCH_SIZE,
                             image_size=IMAGE_SIZE, mlp_head=True)
    weights, _ = load_cnn_weights_file(str(stage_cfg['weights']))
    model.load_weights(weights)

    input_buf = gpu.create_buffer(BATCH_SIZE * 4 * IMAGE_SIZE * IMAGE_SIZE * 4)

    # Forward pass over val pairs
    correct = 0
    total = 0
    for i in range(0, len(val_pairs), BATCH_SIZE):
        batch = val_pairs[i:i + BATCH_SIZE]
        if len(batch) < 2:
            continue
        B = len(batch)
        batch_arr = np.array(batch)
        w_imgs = store.get_batch(batch_arr[:, 0])
        l_imgs = store.get_batch(batch_arr[:, 1])

        if B < BATCH_SIZE:
            pad_shape = (BATCH_SIZE - B, *w_imgs.shape[1:])
            w_imgs = np.concatenate(
                [w_imgs, np.zeros(pad_shape, dtype=np.float32)])
            l_imgs = np.concatenate(
                [l_imgs, np.zeros(pad_shape, dtype=np.float32)])

        gpu.upload(input_buf, w_imgs)
        w_out = model.forward(input_buf, BATCH_SIZE,
                              (4, IMAGE_SIZE, IMAGE_SIZE))
        w_scores = gpu.download(w_out, np.float32, BATCH_SIZE)[:B]

        gpu.upload(input_buf, l_imgs)
        l_out = model.forward(input_buf, BATCH_SIZE,
                              (4, IMAGE_SIZE, IMAGE_SIZE))
        l_scores = gpu.download(l_out, np.float32, BATCH_SIZE)[:B]

        correct += int(np.sum(w_scores > l_scores))
        total += B

    return correct / total if total > 0 else 0.0, len(val_pairs)


def main():
    from wallpaper_ml.vk_compute import VkCompute
    from flame_sheep.storage import NORMALIZATION_VERSION
    from flame_sheep.genome.scoring.scoring_channels import load_normalization_sidecar

    normalization = load_normalization_sidecar(DATA_DIR, NORMALIZATION_VERSION)
    print(f'Normalization {NORMALIZATION_VERSION}: '
          f'mean={[f"{x:.4f}" for x in normalization[0]]} '
          f'std={[f"{x:.4f}" for x in normalization[1]]}\n')

    gpu = VkCompute()
    print(f'GPU: {gpu.device_name}\n')

    print('=== Stage × val-set pairwise accuracy ===\n')
    print(f'{"":>10} | {"full_es":>14} | {"smooth_q1":>14}')
    print(f'{"-" * 10}-+-{"-" * 14}-+-{"-" * 14}')
    results = {}
    for stage_name in STAGES:
        row = [stage_name]
        for val_set_name in VAL_SETS:
            acc, n_pairs = evaluate(stage_name, val_set_name, gpu,
                                    normalization)
            results[(stage_name, val_set_name)] = (acc, n_pairs)
            row.append(f'{acc:.4f} (n={n_pairs})')
        print(f'{row[0]:>10} | {row[1]:>14} | {row[2]:>14}')

    # Read the answer
    s1_full = results[('stage1', 'full_es')][0]
    s1_smooth = results[('stage1', 'smooth_q1')][0]
    s2_full = results[('stage2', 'full_es')][0]
    s2_smooth = results[('stage2', 'smooth_q1')][0]

    print('\n=== Interpretation ===')
    print(f'Stage 1 → Stage 2 on full_es:    {s1_full:.4f} → {s2_full:.4f}  '
          f'(Δ {s2_full - s1_full:+.4f})  ← H benefit, apples-to-apples')
    print(f'Stage 1 → Stage 2 on smooth_q1:  {s1_smooth:.4f} → {s2_smooth:.4f}  '
          f'(Δ {s2_smooth - s1_smooth:+.4f})  ← H benefit on smooth subset')
    print(f'full_es → smooth_q1 (stage 1):   {s1_full:.4f} → {s1_smooth:.4f}  '
          f'(Δ {s1_smooth - s1_full:+.4f})  ← val-set shift only')
    print(f'full_es → smooth_q1 (stage 2):   {s2_full:.4f} → {s2_smooth:.4f}  '
          f'(Δ {s2_smooth - s2_full:+.4f})  ← val-set shift only')


if __name__ == '__main__':
    main()
