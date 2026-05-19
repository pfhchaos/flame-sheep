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
    """Scan .npz label files and return list of (path, n_frames) tuples.

    Lazy: array data is NOT loaded here. The 7.5K-file beat corpus expands
    to ~170 GB if all arrays are materialized — would OOM any reasonable
    machine. Instead we record file paths + frame counts (cheap, ~one
    np.load per file just to read array shapes), and load actual arrays
    on demand during batching via LazyFileCache.
    """
    files = sorted(data_dir.glob('*.npz'))
    if max_files:
        files = files[:max_files]

    n_total = len(files)
    print(f"Scanning {n_total} .npz files from {data_dir}...")
    report_every = max(1, n_total // 20)

    dataset = []
    total_frames = 0
    for i, f in enumerate(files):
        with np.load(f) as d:
            n_frames = len(d['spectrum'])
        dataset.append((f, n_frames))
        total_frames += n_frames
        if (i + 1) % report_every == 0 or i + 1 == n_total:
            print(f"  scanned {i + 1}/{n_total} ({total_frames} frames so far)")

    print(f"Scanned {len(dataset)} files, {total_frames} total frames "
          f"({total_frames / 93.75 / 60:.1f} minutes)")
    return dataset


class LazyFileCache:
    """LRU cache of decompressed .npz arrays, keyed by file path.

    .npz is compressed zip, so np.load(mmap_mode='r') silently no-ops —
    the data has to be decompressed into a real ndarray on access. This
    cache holds the most-recently-used N files' (spec, diff, labels)
    tuples, evicting oldest entries when the cache fills. At ~10 MB per
    file × 64 cached, that's a ~640 MB working set instead of all-files-
    in-RAM (~170 GB for the full corpus).

    Cache shared across train and validate so files don't repeatedly
    reload between phases.
    """

    def __init__(self, max_files: int = 64):
        self.max_files = max_files
        self._cache: dict = {}
        self._order: list = []

    def get(self, path):
        key = str(path)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        d = np.load(path)
        # Copy to plain ndarrays so the underlying npz file handle can close.
        arrays = (np.array(d['spectrum']), np.array(d['diff']), np.array(d['labels']))
        self._cache[key] = arrays
        self._order.append(key)
        while len(self._cache) > self.max_files:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        return arrays


def make_chunks(dataset, chunk_len: int, batch_size: int, rng):
    """Build chunk specs (NOT materialized arrays) for one epoch.

    Returns list of batch specs, each spec a list of (file_idx, start)
    tuples. Materialize per-batch via materialize_batch() at iteration
    time — keeps memory footprint bounded by LazyFileCache rather than
    growing with epoch size.
    """
    chunks = []
    for file_idx, (_, n_frames) in enumerate(dataset):
        n_chunks = n_frames // chunk_len
        for c in range(n_chunks):
            chunks.append((file_idx, c * chunk_len))

    rng.shuffle(chunks)

    batches = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batches.append(chunks[i:i + batch_size])
    return batches


def materialize_batch(batch_spec, dataset, chunk_len: int, cache: LazyFileCache):
    """Load files for a single batch via cache, return (inputs, labels) arrays.

    inputs: [B, T, 216] float32 (spectrum + diff concatenated)
    labels: [B, T, 3]   float32 (soft probabilities)
    """
    batch_inputs = []
    batch_labels = []
    for file_idx, start in batch_spec:
        path, _ = dataset[file_idx]
        spec, diff, labels = cache.get(path)
        end = start + chunk_len
        inp = np.concatenate([spec[start:end], diff[start:end]], axis=1)
        batch_inputs.append(inp)
        batch_labels.append(labels[start:end])
    return (
        np.array(batch_inputs, dtype=np.float32),
        np.array(batch_labels, dtype=np.float32),
    )


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


def save_checkpoint(path: Path, weights: np.ndarray, next_epoch: int,
                    best_val_loss: float) -> None:
    """Save weights + training metadata in .npz format.

    Layout: {weights, next_epoch, best_val_loss}.

    `next_epoch` semantically means "first epoch to do on resume" — so an
    end-of-epoch save (just finished epoch N) stores next_epoch=N+1, but
    a mid-epoch save (still doing epoch N) stores next_epoch=N. Resume
    sets start_epoch = next_epoch directly. This way mid-epoch
    checkpoints correctly cause the resume to re-do that epoch's
    remaining data with the saved (mid-epoch-improved) weights, rather
    than skipping ahead.

    Atomic write via tmp + rename so partial writes don't corrupt a
    good checkpoint.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    np.savez(tmp, weights=weights, next_epoch=next_epoch,
             best_val_loss=best_val_loss)
    # numpy adds .npz if path didn't have one — pick whichever landed.
    actual_tmp = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + '.npz')
    actual_dst = path if path.suffix == '.npz' else path.with_suffix('.npz')
    actual_tmp.replace(actual_dst)


def load_checkpoint(path: Path):
    """Load checkpoint. Returns (weights, next_epoch, best_val_loss) or None.

    `next_epoch` is the index of the first epoch to (re)do on resume.
    Accepts both new .npz format (with metadata) and legacy formats
    (raw .npy weights, or .npz with the older 'epoch' field meaning
    "last completed epoch"). Legacy 'epoch' is converted: next_epoch =
    epoch + 1.
    Returns None if path doesn't exist or load fails.
    """
    # Allow either explicit .npz or fall back to whatever np.save wrote.
    candidates = [path]
    if path.suffix != '.npz':
        candidates.append(path.with_suffix('.npz'))
    if path.suffix != '.npy':
        candidates.append(path.with_suffix('.npy'))
    for p in candidates:
        if p.exists():
            path = p
            break
    else:
        return None
    try:
        obj = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"  WARN: couldn't load checkpoint at {path}: {e}", file=sys.stderr)
        return None
    if isinstance(obj, np.lib.npyio.NpzFile):
        weights = obj['weights'].astype(np.float32)
        if 'next_epoch' in obj.files:
            next_epoch = int(obj['next_epoch'])
        elif 'epoch' in obj.files:
            # Older field: 'epoch' meant "last completed epoch."
            next_epoch = int(obj['epoch']) + 1
        else:
            next_epoch = 0
        best_val_loss = (float(obj['best_val_loss'])
                         if 'best_val_loss' in obj.files else float('inf'))
        obj.close()
        return weights, next_epoch, best_val_loss
    # Raw .npy — weights only.
    return obj.astype(np.float32), 0, float('inf')


def train_epoch(model, gpu, batches, lr: float, chunk_len: int,
                in_buf, grad_buf, dataset, cache: LazyFileCache,
                val_batches=None, val_data=None, val_interval: float = 0.0,
                val_sample_size: int = 200,
                best_val_state=None, save_path: Path | None = None,
                epoch_num: int = 0):
    """Train one epoch. Reuses pre-allocated GPU buffers.

    `batches` is a list of batch specs (file_idx, start tuples).
    Arrays are materialized per-batch via the shared cache.

    Optional mid-epoch validation by wall time: if val_interval > 0,
    runs validate() whenever val_interval seconds have elapsed since the
    last validation (or since the epoch start), and saves the checkpoint
    via save_path if val_loss improves on best_val_state[0]. Wall time
    is more useful than batch count when batch processing time varies
    (different cache hit rates, GPU contention, etc.) — guarantees you
    get a checkpoint within a known time budget regardless of throughput.
    """
    import time
    losses = []
    mid_epoch_val_active = (val_batches and val_data is not None
                            and val_interval > 0 and best_val_state is not None
                            and save_path is not None)
    last_val_time = time.monotonic()

    for batch_idx, batch_spec in enumerate(batches):
        inputs, labels = materialize_batch(batch_spec, dataset, chunk_len, cache)
        # inputs: [B, T, 216], labels: [B, T, 3]
        B_actual = inputs.shape[0]
        T = inputs.shape[1]

        model.zero_grad()

        # Reset GRU hidden state for each chunk
        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        chunk_loss = 0.0
        all_grads = []

        for t in range(T):
            frame = inputs[:, t, :]  # [B, 216]
            gpu.upload(in_buf, frame.ravel())

            out_buf = model.forward(in_buf, B_actual, (216,))
            logits = gpu.download(out_buf, np.float32, B_actual * 3).reshape(B_actual, 3)

            target = labels[:, t, :]
            loss, grad = soft_cross_entropy(logits, target)
            chunk_loss += loss
            all_grads.append(grad)

        chunk_loss /= T
        losses.append(chunk_loss)

        # Use mean gradient across all frames as approximation for BPTT.
        # TODO: proper per-frame gradient accumulation through output linear.
        mean_grad = np.mean(all_grads, axis=0).astype(np.float32)
        gpu.upload(grad_buf, mean_grad.ravel())

        model.backward(grad_buf, B_actual)
        model.sgd_step(lr)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{len(batches)} loss={chunk_loss:.4f}")

        # Mid-epoch validation: runs whenever val_interval seconds have
        # elapsed since the last validation. Wall-time-based instead of
        # batch-count-based so the user gets a checkpoint within a known
        # time budget regardless of throughput swings.
        if mid_epoch_val_active and (time.monotonic() - last_val_time) >= val_interval:
            elapsed_min = (time.monotonic() - last_val_time) / 60.0
            # Sample a small subset of val for speed; end-of-epoch will run
            # the full val set for a high-confidence accuracy reading.
            val_loss, val_acc = validate(model, gpu, val_batches,
                                         chunk_len, in_buf, val_data, cache,
                                         max_batches=val_sample_size)
            print(f"    [mid-epoch +{elapsed_min:.1f}min] batch {batch_idx+1}/{len(batches)}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"(sampled {val_sample_size}/{len(val_batches)} val batches)")
            if val_loss < best_val_state[0]:
                best_val_state[0] = val_loss
                # next_epoch=epoch_num: still mid-epoch_num, retry it on resume.
                save_checkpoint(save_path, model.save_weights(),
                                epoch_num, val_loss)
                print(f"    [mid-epoch] -> saved best (val_loss={val_loss:.4f})")
            last_val_time = time.monotonic()

    return np.mean(losses) if losses else 0.0


def validate(model, gpu, batches, chunk_len: int, in_buf,
             dataset, cache: LazyFileCache,
             max_batches: int | None = None):
    """Run validation. Reuses pre-allocated input buffer.

    If max_batches is set and smaller than len(batches), randomly samples
    that many batches and runs validation only on the sample. Used for
    mid-epoch validation where the full val set would take hours (~15h on
    the full 751-file beat val set) and the user just wants a periodic
    val accuracy reading, not a high-confidence number. End-of-epoch
    validation passes max_batches=None to run the full set for an
    honest checkpoint metric.
    """
    if max_batches is not None and max_batches < len(batches):
        import random
        batches = random.sample(batches, max_batches)
    losses = []
    correct_beats = 0
    total_beats = 0

    for batch_spec in batches:
        inputs, labels = materialize_batch(batch_spec, dataset, chunk_len, cache)
        B = inputs.shape[0]
        T = inputs.shape[1]

        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        for t in range(T):
            frame = inputs[:, t, :]
            gpu.upload(in_buf, frame.ravel())

            out_buf = model.forward(in_buf, B, (216,))
            logits = gpu.download(out_buf, np.float32, B * 3).reshape(B, 3)

            target = labels[:, t, :]
            loss, _ = soft_cross_entropy(logits, target)
            losses.append(loss)

            pred_class = logits.argmax(axis=1)
            true_class = target.argmax(axis=1)
            correct_beats += (pred_class == true_class).sum()
            total_beats += B

    avg_loss = np.mean(losses) if losses else 0.0
    accuracy = correct_beats / max(total_beats, 1)
    return avg_loss, accuracy


def main():
    # Line-buffer stdout so progress prints flush on each newline.
    # Without this, output sits in a 4KB buffer and a GPU lockup leaves
    # no breadcrumb in the log file.
    sys.stdout.reconfigure(line_buffering=True)
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
    parser.add_argument('--val-interval', type=float, default=7200.0,
                        help='Mid-epoch validation cadence in seconds of '
                             'wall-clock training time. 0 disables. '
                             'Default: 7200 (2 hours)')
    parser.add_argument('--val-sample-size', type=int, default=200,
                        help='Number of val batches sampled per mid-epoch '
                             'validation pass. The full val set is ~21K '
                             'batches and takes hours to run; sampling 200 '
                             'gives a noisy but timely val_acc reading. '
                             'End-of-epoch validation always uses the full '
                             'val set. Default: 200')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore an existing checkpoint at --output and '
                             'start training from scratch (Kaiming init)')
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

    # Resume from checkpoint if it exists (unless --no-resume).
    start_epoch = 0
    best_val_loss = float('inf')
    if not args.no_resume:
        ckpt = load_checkpoint(args.output)
        if ckpt is not None:
            weights, next_epoch, best_val_loss = ckpt
            if len(weights) != model.param_count():
                print(f"WARN: checkpoint has {len(weights)} params but model "
                      f"expects {model.param_count()} — init from scratch instead",
                      file=sys.stderr)
                model.init_weights(seed=args.seed)
            else:
                model.load_weights(weights)
                start_epoch = next_epoch
                print(f"Resumed from checkpoint: next_epoch={next_epoch+1}, "
                      f"best_val_loss={best_val_loss:.4f}")
        else:
            model.init_weights(seed=args.seed)
            print(f"Fresh init (seed={args.seed})")
    else:
        model.init_weights(seed=args.seed)
        print(f"--no-resume: starting fresh (seed={args.seed})")

    print(f"Model: {model.param_count()} parameters (hidden={args.hidden})")
    print(f"Chunk length: {args.chunk_len} frames ({args.chunk_len / 93.75:.2f}s)")
    if args.val_interval > 0:
        print(f"Mid-epoch validation: every {args.val_interval / 60:.0f} minutes")
    print()

    # Pre-allocate GPU buffers once (reused across all batches/epochs)
    in_buf = gpu.create_buffer(args.batch_size * 216 * 4)
    grad_buf = gpu.create_buffer(args.batch_size * 3 * 4)

    # Lazy file cache shared across train + validate. Holds the N most
    # recently used files; cap at ~64 files (~640 MB) which is enough
    # working set for typical chunk shuffles without thrashing.
    file_cache = LazyFileCache(max_files=getattr(args, 'file_cache_size', 64))

    # Training loop. best_val_state is a one-element list so train_epoch
    # can mutate it across mid-epoch checkpoint saves and the main loop
    # sees the running best.
    best_val_state = [best_val_loss]
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_batches = make_chunks(train_data, args.chunk_len, args.batch_size, rng)
        val_batches = make_chunks(val_data, args.chunk_len, args.batch_size, rng)

        train_loss = train_epoch(model, gpu, train_batches, args.lr,
                                 args.chunk_len, in_buf, grad_buf,
                                 train_data, file_cache,
                                 val_batches=val_batches, val_data=val_data,
                                 val_interval=args.val_interval,
                                 val_sample_size=args.val_sample_size,
                                 best_val_state=best_val_state,
                                 save_path=args.output, epoch_num=epoch)
        val_loss, val_acc = validate(model, gpu, val_batches,
                                     args.chunk_len, in_buf,
                                     val_data, file_cache)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")

        # End-of-epoch save: store next_epoch=epoch+1 since this one is done.
        if val_loss < best_val_state[0]:
            best_val_state[0] = val_loss
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, val_loss)
            print(f"  -> saved best weights (val_loss={val_loss:.4f})")
        else:
            # Always save end-of-epoch progress so resume picks up at the
            # next epoch, even if val didn't improve this one.
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, best_val_state[0])

        print()

    print(f"Training complete. Best val_loss={best_val_state[0]:.4f}")
    print(f"Weights saved to: {args.output}")


if __name__ == '__main__':
    main()
