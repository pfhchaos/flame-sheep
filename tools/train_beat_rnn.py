#!/usr/bin/env python3
"""Train the beat detection CRNN via BPTT on BeatNet-distilled labels.

Uses the wallpaper-ml Vulkan compute framework for GPU training.
Input: .npz files from generate_beat_labels.py (spectrum + diff + soft labels).
Output: trained weight file for the BeatCRNN model.

Usage:
    python tools/train_beat_rnn.py ~/datasets/beat-labels/ -o flame_sheep/data/beat_rnn.npy
    python tools/train_beat_rnn.py ~/datasets/beat-labels/ --hidden 48 --epochs 50 --lr 0.001
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from flame_sheep.vk_compute import VkCompute
from flame_sheep.wallpaper_ml import build_beat_crnn, VkGRU


def load_dataset(data_dir: Path, max_files: int | None = None):
    """Load all .npz label files. Returns list of (spectrum, diff, labels) arrays."""
    files = sorted(data_dir.glob('*.npz'))
    if max_files:
        files = files[:max_files]

    dataset = []
    total_frames = 0
    for f in files:
        d = np.load(f)
        spec = d['spectrum']   # [T, 108]
        diff = d['diff']       # [T, 108]
        labels = d['labels']   # [T, 3]
        dataset.append((spec, diff, labels))
        total_frames += len(spec)

    print(f"Loaded {len(dataset)} files, {total_frames} total frames "
          f"({total_frames / 93.75 / 60:.1f} minutes)")
    return dataset


def make_chunks(dataset, chunk_len: int, batch_size: int, rng):
    """Extract random chunks from dataset for one epoch.

    Returns list of (input_batch, label_batch) where:
      input_batch: [B, T, 216] float32 (spectrum + diff concatenated)
      label_batch: [B, T, 3] float32 (soft probabilities)
    """
    # Collect all valid chunk start positions
    chunks = []
    for file_idx, (spec, diff, labels) in enumerate(dataset):
        n_frames = len(spec)
        n_chunks = n_frames // chunk_len
        for c in range(n_chunks):
            start = c * chunk_len
            chunks.append((file_idx, start))

    rng.shuffle(chunks)

    # Form batches
    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch_inputs = []
        batch_labels = []
        for j in range(batch_size):
            file_idx, start = chunks[i + j]
            spec, diff, labels = dataset[file_idx]
            end = start + chunk_len
            # Concatenate spectrum + diff → 216 features
            inp = np.concatenate([spec[start:end], diff[start:end]], axis=1)
            batch_inputs.append(inp)
            batch_labels.append(labels[start:end])

        batches.append((
            np.array(batch_inputs, dtype=np.float32),  # [B, T, 216]
            np.array(batch_labels, dtype=np.float32),  # [B, T, 3]
        ))

    return batches


def soft_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple:
    """Compute soft cross-entropy loss and gradient.

    Args:
        logits: [B, 3] raw model output (pre-softmax)
        targets: [B, 3] soft probability targets from BeatNet

    Returns:
        (loss_scalar, grad_logits [B, 3])
    """
    # Numerically stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / exp_x.sum(axis=1, keepdims=True)

    # KL divergence: sum(target * log(target / pred))
    # Gradient of CE w.r.t. logits: pred - target
    # Loss: -sum(target * log(pred))
    eps = 1e-7
    loss = -np.sum(targets * np.log(probs + eps)) / len(logits)
    grad = (probs - targets) / len(logits)

    return loss, grad.astype(np.float32)


def train_epoch(model, gpu, batches, lr: float, chunk_len: int):
    """Train one epoch. Returns average loss."""
    losses = []
    B = batches[0][0].shape[0] if batches else 1

    for batch_idx, (inputs, labels) in enumerate(batches):
        # inputs: [B, T, 216], labels: [B, T, 3]
        B_actual = inputs.shape[0]
        T = inputs.shape[1]

        model.zero_grad()

        # Reset GRU hidden state for each chunk
        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        # Forward through time: feed one frame at a time
        chunk_loss = 0.0
        all_grads = []  # collect per-frame output gradients

        # Forward pass: accumulate outputs
        frame_outputs = []
        for t in range(T):
            frame = inputs[:, t, :]  # [B, 216]
            in_buf = gpu.create_buffer(B_actual * 216 * 4)
            gpu.upload(in_buf, frame.ravel())

            out_buf = model.forward(in_buf, B_actual, (216,))
            logits = gpu.download(out_buf, np.float32, B_actual * 3).reshape(B_actual, 3)
            frame_outputs.append(logits)

            # Compute loss for this frame
            target = labels[:, t, :]  # [B, 3]
            loss, grad = soft_cross_entropy(logits, target)
            chunk_loss += loss
            all_grads.append(grad)

        chunk_loss /= T
        losses.append(chunk_loss)

        # Backward pass: BPTT
        # We need to backprop through the full sequence.
        # The model's backward() handles BPTT for VkGRU internally.
        # But we need to feed gradients from each timestep.
        #
        # For the output linear layer, we backprop the last frame's gradient,
        # then the GRU backward unrolls all cached timesteps.
        # Actually for proper BPTT with per-frame loss, we need to sum gradients.
        #
        # Approach: backprop each frame's output gradient through the output linear,
        # accumulating into the GRU's gradient. The GRU's backward handles temporal deps.
        #
        # Simplified approach for now: only backprop from the LAST frame.
        # TODO: proper per-frame gradient accumulation requires extending VkModel.

        # Use mean gradient across all frames as approximation
        mean_grad = np.mean(all_grads, axis=0).astype(np.float32)  # [B, 3]
        grad_buf = gpu.create_buffer(B_actual * 3 * 4)
        gpu.upload(grad_buf, mean_grad.ravel())

        model.backward(grad_buf, B_actual)
        model.sgd_step(lr)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{len(batches)} loss={chunk_loss:.4f}")

    return np.mean(losses) if losses else 0.0


def validate(model, gpu, batches, chunk_len: int):
    """Run validation, return average loss and beat detection accuracy."""
    losses = []
    correct_beats = 0
    total_beats = 0

    for inputs, labels in batches:
        B = inputs.shape[0]
        T = inputs.shape[1]

        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        for t in range(T):
            frame = inputs[:, t, :]
            in_buf = gpu.create_buffer(B * 216 * 4)
            gpu.upload(in_buf, frame.ravel())

            out_buf = model.forward(in_buf, B, (216,))
            logits = gpu.download(out_buf, np.float32, B * 3).reshape(B, 3)

            target = labels[:, t, :]
            loss, _ = soft_cross_entropy(logits, target)
            losses.append(loss)

            # Accuracy: compare argmax
            pred_class = logits.argmax(axis=1)
            true_class = target.argmax(axis=1)
            correct_beats += (pred_class == true_class).sum()
            total_beats += B

    avg_loss = np.mean(losses) if losses else 0.0
    accuracy = correct_beats / max(total_beats, 1)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train beat detection RNN')
    parser.add_argument('data_dir', type=Path, help='Directory with .npz label files')
    parser.add_argument('-o', '--output', type=Path,
                        default=Path('flame_sheep/data/beat_rnn.npy'),
                        help='Output weights file')
    parser.add_argument('--hidden', type=int, default=48,
                        help='GRU hidden size (default: 48)')
    parser.add_argument('--chunk-len', type=int, default=128,
                        help='BPTT chunk length in frames (default: 128 = ~1.4s)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load data
    dataset = load_dataset(args.data_dir, args.max_files)
    if not dataset:
        print("No data files found!", file=sys.stderr)
        sys.exit(1)

    # Train/val split
    n_val = max(1, int(len(dataset) * args.val_split))
    indices = rng.permutation(len(dataset))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    print(f"Train: {len(train_data)} files, Val: {len(val_data)} files")

    # Build model
    gpu = VkCompute()
    model = build_beat_crnn(gpu, input_size=216, hidden_size=args.hidden,
                            n_classes=3, batch_size=args.batch_size,
                            max_seq_len=args.chunk_len)
    model.init_weights(seed=args.seed)
    print(f"Model: {model.param_count()} parameters (hidden={args.hidden})")
    print(f"Chunk length: {args.chunk_len} frames ({args.chunk_len / 93.75:.2f}s)")
    print()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_batches = make_chunks(train_data, args.chunk_len, args.batch_size, rng)
        val_batches = make_chunks(val_data, args.chunk_len, args.batch_size, rng)

        train_loss = train_epoch(model, gpu, train_batches, args.lr, args.chunk_len)
        val_loss, val_acc = validate(model, gpu, val_batches, args.chunk_len)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            weights = model.save_weights()
            np.save(args.output, weights)
            print(f"  -> saved best weights ({len(weights)} params)")

        print()

    print(f"Training complete. Best val_loss={best_val_loss:.4f}")
    print(f"Weights saved to: {args.output}")


if __name__ == '__main__':
    main()
