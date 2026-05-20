#!/usr/bin/env python3
"""CPU eval of the beat RNN — read a saved checkpoint, run a forward
pass over a validation subset on CPU (numpy), report per-class
precision/recall/F1 + confusion matrix.

CPU so it doesn't contend with the GPU when training is running.
Slower than GPU eval but the model is tiny and the val sample is
bounded — typically a minute or two of wall time for a sample of
200 batches.

Usage:
    python tools/eval_beat_rnn_cpu.py \\
        --weights ~/datasets/beat-labels/beat_rnn_v1.npz \\
        --data ~/datasets/beat-labels/ \\
        --max-val-batches 200 \\
        --chunk-len 256 --batch-size 64 --hidden 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the loader so we don't duplicate the train/val split logic.
from train_beat_rnn import (
    load_dataset, make_chunks, materialize_batch, LazyFileCache,
    format_val_metrics,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def linear_forward(x, W, b, relu=False):
    """x: (B, in) → y: (B, out). W: (in, out)."""
    y = x @ W + b
    if relu:
        y = np.maximum(y, 0.0)
    return y


def gru_forward_step(x, h_prev, W, U, bias, hidden_size, input_size):
    """One GRU step. PyTorch convention with split input/hidden biases.

    x: (B, input_size), h_prev: (B, hidden_size)
    W: (3, input_size, hidden_size) — gates z, r, h
    U: (3, hidden_size, hidden_size) — gates z, r, h
    bias: (6, hidden_size) — [bW_z, bW_r, bW_h, bU_z, bU_r, bU_h]
    """
    wx_z = x @ W[0]
    wx_r = x @ W[1]
    wx_h = x @ W[2]
    uh_z = h_prev @ U[0]
    uh_r = h_prev @ U[1]
    uh_n = h_prev @ U[2]

    z = sigmoid(wx_z + uh_z + bias[0] + bias[3])
    r = sigmoid(wx_r + uh_r + bias[1] + bias[4])
    # PyTorch GRU candidate: tanh(W_n@x + b_Wn + r * (U_n@h + b_Un))
    h_hat = np.tanh(wx_h + bias[2] + r * (uh_n + bias[5]))
    h_new = (1.0 - z) * h_hat + z * h_prev
    return h_new


def unpack_weights(flat: np.ndarray, input_size: int, proj_size: int,
                   hidden_size: int, n_classes: int):
    """Unpack the flat (18899,) weight array into named numpy arrays.

    Layout (must match the order layers were created in build_beat_crnn):
      1. VkLinear(input_size → proj_size): W (input_size, proj_size), b (proj_size)
      2. VkGRU(proj_size → hidden_size): W (3, proj_size, hidden_size),
                                          U (3, hidden_size, hidden_size),
                                          bias (6, hidden_size)
      3. VkLinear(hidden_size → n_classes): W (hidden_size, n_classes), b (n_classes)
    """
    offset = 0

    # linear_in
    n_w = input_size * proj_size
    n_b = proj_size
    W_in = flat[offset:offset + n_w].reshape(input_size, proj_size); offset += n_w
    b_in = flat[offset:offset + n_b]; offset += n_b

    # GRU
    n_W = 3 * proj_size * hidden_size
    n_U = 3 * hidden_size * hidden_size
    n_bias = 6 * hidden_size
    W_gru = flat[offset:offset + n_W].reshape(3, proj_size, hidden_size); offset += n_W
    U_gru = flat[offset:offset + n_U].reshape(3, hidden_size, hidden_size); offset += n_U
    bias_gru = flat[offset:offset + n_bias].reshape(6, hidden_size); offset += n_bias

    # linear_out
    n_w = hidden_size * n_classes
    n_b = n_classes
    W_out = flat[offset:offset + n_w].reshape(hidden_size, n_classes); offset += n_w
    b_out = flat[offset:offset + n_b]; offset += n_b

    if offset != len(flat):
        raise ValueError(f'Unexpected weight tail: {offset} parsed, {len(flat)} total')
    return W_in, b_in, W_gru, U_gru, bias_gru, W_out, b_out


def main():
    parser = argparse.ArgumentParser(
        description='CPU eval of a saved beat-RNN checkpoint.')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True,
                        help='Directory with .npz label files (same as training)')
    parser.add_argument('--max-val-batches', type=int, default=200,
                        help='Random sample of val batches to score (default 200)')
    parser.add_argument('--chunk-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--proj-size', type=int, default=32)
    parser.add_argument('--input-size', type=int, default=216)
    parser.add_argument('--n-classes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42,
                        help='Train/val split RNG seed; matches finetune default')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    # Load weights
    ckpt = np.load(args.weights)
    if 'weights' in ckpt.files:
        flat = ckpt['weights'].astype(np.float32)
        meta_epoch = int(ckpt['next_epoch']) if 'next_epoch' in ckpt.files else None
        meta_loss = float(ckpt['best_val_loss']) if 'best_val_loss' in ckpt.files else None
        print(f'Loaded {len(flat)} params from {args.weights} '
              f'(next_epoch={meta_epoch}, best_val_loss={meta_loss:.4f}'
              if meta_loss is not None else
              f'Loaded {len(flat)} params from {args.weights} (next_epoch={meta_epoch})',
              end='\n')
    else:
        # Legacy .npy: just raw weights
        flat = ckpt.astype(np.float32) if hasattr(ckpt, 'astype') else np.load(args.weights).astype(np.float32)
        print(f'Loaded {len(flat)} legacy weights from {args.weights}')

    W_in, b_in, W_gru, U_gru, bias_gru, W_out, b_out = unpack_weights(
        flat, args.input_size, args.proj_size, args.hidden, args.n_classes)

    # Reproduce the train/val split that training used.
    dataset = load_dataset(args.data, args.max_files)
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(len(dataset) * args.val_split))
    indices = rng.permutation(len(dataset))
    val_idx = indices[:n_val]
    val_data = [dataset[i] for i in val_idx]
    print(f'Val files: {len(val_data)} / {len(dataset)} total')

    # Build batch specs and random-sample for speed.
    val_batches = make_chunks(val_data, args.chunk_len, args.batch_size, rng)
    print(f'Val batches available: {len(val_batches)}; sampling {args.max_val_batches}')
    import random
    random.seed(args.seed)
    if args.max_val_batches < len(val_batches):
        val_batches = random.sample(val_batches, args.max_val_batches)

    cache = LazyFileCache(max_files=64)

    # 3x3 confusion matrix
    confusion = np.zeros((3, 3), dtype=np.int64)
    total_loss = 0.0
    n_frames = 0

    print(f'Running CPU forward over {len(val_batches)} batches...')
    for bi, batch_spec in enumerate(val_batches):
        inputs, labels = materialize_batch(batch_spec, val_data,
                                            args.chunk_len, cache)
        # inputs: (B, T, 216), labels: (B, T, 3)
        B = inputs.shape[0]
        T = inputs.shape[1]

        # Run the sequence through the model. Hidden starts at zero
        # for each chunk — same as training's `reset_hidden()`.
        h = np.zeros((B, args.hidden), dtype=np.float32)
        for t in range(T):
            x_t = inputs[:, t, :]                    # (B, 216)
            proj = linear_forward(x_t, W_in, b_in, relu=True)  # (B, 32)
            h = gru_forward_step(proj, h, W_gru, U_gru, bias_gru,
                                  args.hidden, args.proj_size)  # (B, 48)
            logits = linear_forward(h, W_out, b_out, relu=False)  # (B, 3)

            # Cross-entropy + confusion
            shifted = logits - logits.max(axis=1, keepdims=True)
            ex = np.exp(shifted)
            probs = ex / ex.sum(axis=1, keepdims=True)
            t_lbl = labels[:, t, :]                  # (B, 3) soft probs
            loss = -np.sum(t_lbl * np.log(probs + 1e-7)) / B
            total_loss += float(loss)
            n_frames += B

            pred = probs.argmax(axis=1)
            true = t_lbl.argmax(axis=1)
            for tc, pc in zip(true, pred):
                confusion[tc, pc] += 1

        if (bi + 1) % 20 == 0:
            print(f'  {bi + 1}/{len(val_batches)} batches')

    avg_loss = total_loss / max(len(val_batches) * T, 1)
    accuracy = float(np.trace(confusion) / max(confusion.sum(), 1))

    class_names = ('non-beat', 'beat', 'downbeat')
    per_class = {}
    for c, name in enumerate(class_names):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {'precision': precision, 'recall': recall,
                            'f1': f1, 'support': support,
                            'tp': tp, 'fp': fp, 'fn': fn}
    class_supports = [int(confusion[c, :].sum()) for c in range(3)]
    majority_class = int(np.argmax(class_supports))
    majority_baseline = class_supports[majority_class] / max(int(confusion.sum()), 1)
    metrics = {
        'confusion': confusion.tolist(),
        'class_names': class_names,
        'per_class': per_class,
        'majority_baseline': majority_baseline,
        'majority_class': class_names[majority_class],
    }

    print()
    print(f'val_loss={avg_loss:.4f}  val_acc={accuracy:.4f}  '
          f'n_frames={n_frames}')
    print(format_val_metrics(metrics))


if __name__ == '__main__':
    main()
