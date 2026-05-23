#!/usr/bin/env python3
"""Continuous-activation beat-RNN training.

Single output channel: beat probability per frame, regressed against
BeatNet's continuous beat-or-downbeat probability (1 - non_beat_prob).
Sigmoid + BCE loss. Sidesteps the 3-class imbalance that broke softmax
training even with class weights + focal loss.

Architecture mirrors the 3-class model except the final linear is
hidden→1 instead of hidden→3, and the loss head is bce_loss.comp
instead of cross_entropy.comp. The forward + backward seq-mode shaders
and BPTT pipeline are unchanged.

Validation metric is F-measure with 70ms tolerance per the standard
MIR convention (mir_eval.beat.f_measure): peak-pick the predicted
continuous score, match each predicted beat to the nearest unused
true beat, count as TP if within ±70ms.

Usage:
    python tools/train_beat_rnn_continuous.py ~/datasets/beat-labels/ \\
        -o ~/datasets/beat-labels/beat_rnn_continuous.npz \\
        --epochs 30 --batch-size 64 --chunk-len 256 --hidden 48 --lr 0.003
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wallpaper_ml.vk_compute import VkCompute
from wallpaper_ml import (
    build_beat_crnn, VkGRU, bce_loss_dispatch,
)
# Reuse infrastructure that doesn't change between 3-class and continuous modes.
from train_beat_rnn import (
    load_dataset, make_chunks, LazyFileCache, BatchPrepPipeline,
    save_checkpoint, load_checkpoint,
)


def materialize_batch_continuous(batch_spec, dataset, chunk_len: int,
                                  cache: LazyFileCache,
                                  target_col: int = 216):
    """Build a continuous-target batch.

    inputs: (B, T, 216) — spectrum + diff
    targets: (B, T) — single-channel target sliced from `target_col`

    Packed format conventions (see tools/pack_beat_labels.py):
    - 217-col packed: col 216 = beat_score (legacy v1 single-channel)
    - 218-col packed: col 216 = downbeat, col 217 = non-downbeat beat
      (current; both target columns coexist so two models can train
       from the same packed corpus, one per column).

    `target_col` selects which column the BCE loss targets. Default 216
    works for both v1 (beat_score) and v2 downbeat training.
    """
    batch_inputs = []
    batch_targets = []
    for file_idx, start in batch_spec:
        path, _ = dataset[file_idx]
        entry = cache.get(path)
        end = start + chunk_len
        if isinstance(entry, np.memmap) or (
                isinstance(entry, np.ndarray) and entry.ndim == 2
                and entry.shape[1] in (217, 218)):
            # Packed mmap format. View slicing is free.
            if target_col >= entry.shape[1]:
                raise ValueError(
                    f'{path.name}: target_col={target_col} out of range '
                    f'for packed shape {entry.shape}')
            batch_inputs.append(entry[start:end, :216])
            batch_targets.append(entry[start:end, target_col])
        else:
            # Legacy .npz format — only supports beat_score (col 216 equivalent).
            if target_col != 216:
                raise RuntimeError(
                    f'{path.name}: legacy .npz format only supports '
                    f'target_col=216; got {target_col}. Repack to .npy.')
            if 'beat_score' not in entry:
                raise RuntimeError(
                    f'{path.name} has no beat_score key — run '
                    f'tools/add_beat_score_to_labels.py first.')
            spec, diff = entry['spectrum'], entry['diff']
            beat_score = entry['beat_score']
            inp = np.concatenate([spec[start:end], diff[start:end]], axis=1)
            batch_inputs.append(inp)
            batch_targets.append(beat_score[start:end])
    return (
        np.array(batch_inputs, dtype=np.float32),
        np.array(batch_targets, dtype=np.float32),
    )


class ContinuousPrepPipeline:
    """Background-thread batch prep for continuous-mode training.

    Mirrors BatchPrepPipeline but uses materialize_batch_continuous and
    transposes targets to (T, B) instead of (T, B, 3).
    """

    def __init__(self, batches, dataset, chunk_len: int,
                 cache: LazyFileCache, max_queue: int = 2,
                 n_workers: int = 8, target_col: int = 216):
        import queue, threading
        from concurrent.futures import ThreadPoolExecutor
        self.batches = batches
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.cache = cache
        self.target_col = target_col
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        # n_workers threads decompress .npz files in parallel during
        # prefetch. zipfile's zlib calls release the GIL, so this scales
        # nearly linearly until I/O saturation. Cache size must be at
        # least batch_size to avoid intra-batch eviction.
        self._executor = ThreadPoolExecutor(max_workers=n_workers)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for batch_spec in self.batches:
            if self._stop.is_set():
                break
            # Prefetch unique files in parallel before slicing.
            # cache.get() is thread-safe; concurrent calls to the same
            # path coalesce (second waiter reuses the first's result).
            unique_paths = list({self.dataset[file_idx][0]
                                 for file_idx, _ in batch_spec})
            list(self._executor.map(self.cache.get, unique_paths))
            inputs, targets = materialize_batch_continuous(
                batch_spec, self.dataset, self.chunk_len, self.cache,
                target_col=self.target_col)
            inputs_TBF = np.ascontiguousarray(inputs.transpose(1, 0, 2))
            targets_TB = np.ascontiguousarray(targets.transpose(1, 0))
            self.queue.put((inputs_TBF, targets_TB))
        self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item

    def stop(self):
        self._stop.set()
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass
        self._executor.shutdown(wait=False)


def peak_pick(scores: np.ndarray, threshold: float = 0.3,
              min_distance: int = 5) -> np.ndarray:
    """Find local maxima above threshold in a 1D score array.

    A frame t is a peak if scores[t] >= scores[t-1] and >= scores[t+1]
    and scores[t] >= threshold. min_distance enforces a refractory
    period — within min_distance frames of a previous peak, additional
    peaks are suppressed. At our 93.75fps frame rate, min_distance=5
    is ~53ms — tighter than the 70ms F-measure tolerance, so no
    accidental TP doubling.

    Returns: indices of peak frames (1D int array).
    """
    if len(scores) < 3:
        return np.array([], dtype=np.int64)
    # Vectorized local-maxima check (interior frames only)
    interior = (scores[1:-1] >= scores[:-2]) & (scores[1:-1] >= scores[2:])
    above_thresh = scores[1:-1] >= threshold
    candidates = np.where(interior & above_thresh)[0] + 1
    # Refractory period: drop peaks within min_distance of a kept one
    if min_distance > 0 and len(candidates) > 0:
        kept = [candidates[0]]
        for c in candidates[1:]:
            if c - kept[-1] >= min_distance:
                kept.append(c)
        candidates = np.array(kept, dtype=np.int64)
    return candidates


def f_measure(pred_times: np.ndarray, true_times: np.ndarray,
              tolerance: float = 0.07) -> dict:
    """F-measure with ±tolerance seconds matching (mir_eval convention).

    Greedy nearest-unused-true match. Each predicted beat matches to
    the closest still-unmatched true beat if within tolerance; else
    counts as a false positive. Unmatched true beats are false
    negatives.
    """
    pred_times = np.sort(pred_times)
    true_times = np.sort(true_times)
    used_true = np.zeros(len(true_times), dtype=bool)
    tp = 0
    for p in pred_times:
        if len(true_times) == 0:
            break
        # Nearest true beat by absolute time
        diffs = np.abs(true_times - p)
        # Mask out already-used true beats
        diffs[used_true] = np.inf
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tolerance:
            used_true[idx] = True
            tp += 1
    fp = len(pred_times) - tp
    fn = len(true_times) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'n_pred': len(pred_times), 'n_true': len(true_times),
    }


OUR_FPS = 93.75  # matches generate_beat_labels.py


def validate_continuous(model, gpu, batches, chunk_len: int, in_buf,
                        dataset, cache: LazyFileCache,
                        max_batches: int | None = None,
                        peak_threshold: float = 0.3,
                        target_col: int = 216):
    """Run validation: peak-pick predicted scores, compute F1@70ms.

    Returns (avg_loss, f1, metrics_dict). metrics_dict has precision,
    recall, f1, plus tp/fp/fn aggregated across the val sample.
    """
    if max_batches is not None and max_batches < len(batches):
        import random
        batches = random.sample(batches, max_batches)
    losses = []
    total_tp = total_fp = total_fn = 0
    total_pred = total_true = 0

    for batch_spec in batches:
        inputs, targets = materialize_batch_continuous(
            batch_spec, dataset, chunk_len, cache, target_col=target_col)
        B = inputs.shape[0]
        T = inputs.shape[1]

        # Per-sequence collect logits across timesteps for peak-picking
        # afterwards. Shape will be (B, T) float.
        seq_logits = np.zeros((B, T), dtype=np.float32)

        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        for t in range(T):
            frame = inputs[:, t, :]
            gpu.upload(in_buf, frame.ravel())
            out_buf = model.forward(in_buf, B, (216,))
            logits = gpu.download(out_buf, np.float32, B).reshape(B)
            seq_logits[:, t] = logits

            # BCE loss for this timestep (CPU side, for reporting only)
            tgt = targets[:, t]
            z = logits
            abs_z = np.abs(z)
            max_z = np.maximum(z, 0.0)
            loss = max_z - z * tgt + np.log1p(np.exp(-abs_z))
            losses.append(float(loss.mean()))

        # Convert logits to probabilities, peak-pick per sequence
        probs = 1.0 / (1.0 + np.exp(-seq_logits))
        for b in range(B):
            pred_frames = peak_pick(probs[b], threshold=peak_threshold)
            # True beats: peak-pick the soft target the same way (target
            # was BeatNet's continuous beat-or-downbeat prob, which IS
            # peak-shaped at beat positions).
            true_frames = peak_pick(targets[b], threshold=0.5)
            pred_times = pred_frames / OUR_FPS
            true_times = true_frames / OUR_FPS
            m = f_measure(pred_times, true_times)
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']
            total_pred += m['n_pred']
            total_true += m['n_true']

    avg_loss = float(np.mean(losses)) if losses else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics = {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'n_pred': total_pred, 'n_true': total_true,
    }
    return avg_loss, f1, metrics


def format_continuous_metrics(m: dict) -> str:
    return (f'  F1={m["f1"]:.3f}  precision={m["precision"]:.3f}  '
            f'recall={m["recall"]:.3f}  '
            f'tp={m["tp"]} fp={m["fp"]} fn={m["fn"]}  '
            f'(pred={m["n_pred"]} true={m["n_true"]})')


def train_epoch_continuous(model, gpu, batches, lr: float, chunk_len: int,
                           in_buf, grad_buf, dataset, cache,
                           target_seq_buf, loss_acc_buf, grad_acc_buf,
                           input_seq_buf, gru_seq_out_buf,
                           val_batches=None, val_data=None,
                           val_interval: float = 0.0,
                           val_sample_size: int = 50,
                           best_val_state=None,
                           save_path: Path | None = None,
                           epoch_num: int = 0,
                           prep_workers: int = 8,
                           target_col: int = 216):
    """Train one epoch in continuous mode (single output channel, BCE)."""
    losses = []
    mid_epoch_val_active = (val_batches and val_data is not None
                            and val_interval > 0 and best_val_state is not None
                            and save_path is not None)
    last_val_time = time.monotonic()

    linear_in, gru, linear_out = model.layers

    prep = ContinuousPrepPipeline(batches, dataset, chunk_len, cache,
                                   max_queue=2, n_workers=prep_workers,
                                   target_col=target_col)
    n_batches = len(batches)

    prep_iter = iter(prep)
    for batch_idx in range(n_batches):
        try:
            inputs_TBF, targets_TB = next(prep_iter)
        except StopIteration:
            break

        B_actual = inputs_TBF.shape[1]
        T = inputs_TBF.shape[0]
        BT = B_actual * T

        model.zero_grad()
        gru.reset_hidden()
        gpu.zero_buffer(loss_acc_buf)
        gpu.zero_buffer(grad_acc_buf)

        gpu.upload(input_seq_buf, inputs_TBF.ravel())
        gpu.upload(target_seq_buf, targets_TB.ravel())

        # Forward — same seq-mode dispatches as 3-class. Final linear
        # produces (B*T, 1) logits since the model was built with
        # n_classes=1.
        linear_in.forward(input_seq_buf, BT, (linear_in.in_features,))
        gru.forward_sequence(linear_in.output_buf, gru_seq_out_buf,
                              B_actual, T)
        linear_out.forward(gru_seq_out_buf, BT, (gru.hidden_size,))

        # BCE loss + grad in one dispatch (single channel)
        bce_loss_dispatch(
            gpu, linear_out.output_buf, target_seq_buf,
            grad_acc_buf, loss_acc_buf,
            batch_size=BT, target_offset_floats=0, t_inv=1.0,
        )

        loss_per = gpu.download(loss_acc_buf, np.float32, BT)
        chunk_loss = float(loss_per.mean())
        losses.append(chunk_loss)

        # Backward — same shape as 3-class because grad_acc_buf is
        # sized for (B*T, n_classes) where n_classes=1 here so it's
        # just (B*T,) flat.
        linear_out.backward(grad_acc_buf, BT)
        gru.backward_sequence(linear_out._grad_input_buf, B_actual, T)
        linear_in.backward(gru._grad_input_buf, BT)
        model.sgd_step(lr)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{n_batches} loss={chunk_loss:.4f}")

        if mid_epoch_val_active and (time.monotonic() - last_val_time) >= val_interval:
            elapsed_min = (time.monotonic() - last_val_time) / 60.0
            val_loss, val_f1, val_metrics = validate_continuous(
                model, gpu, val_batches, chunk_len, in_buf,
                val_data, cache, max_batches=val_sample_size,
                target_col=target_col)
            print(f"    [mid-epoch +{elapsed_min:.1f}min] batch {batch_idx+1}/{n_batches}  "
                  f"val_loss={val_loss:.4f}  F1={val_f1:.3f}  "
                  f"(sampled {val_sample_size}/{len(val_batches)} val batches)")
            print(format_continuous_metrics(val_metrics))
            if val_loss < best_val_state[0]:
                best_val_state[0] = val_loss
                save_checkpoint(save_path, model.save_weights(),
                                epoch_num, val_loss)
                print(f"    [mid-epoch] -> saved best (val_loss={val_loss:.4f})")
            last_val_time = time.monotonic()

    return np.mean(losses) if losses else 0.0


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(
        description='Train continuous-activation beat RNN (single channel, BCE)')
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('-o', '--output', type=Path,
                        default=Path('flame_sheep/data/beat_rnn_continuous.npz'))
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--chunk-len', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-files', type=int, default=None)
    parser.add_argument('--val-interval', type=float, default=7200.0)
    parser.add_argument('--val-sample-size', type=int, default=50)
    parser.add_argument('--peak-threshold', type=float, default=0.3,
                        help='Threshold for peak-picking the predicted '
                             'continuous score during validation (default 0.3)')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--file-cache-size', type=int, default=512,
                        help='Max .npz files held decompressed in RAM. '
                             'Each file is ~10 MB. Must be >= batch_size to '
                             'avoid intra-batch eviction (default 512).')
    parser.add_argument('--target-col', type=int, default=216,
                        help='Column of the packed .npy to use as the BCE '
                             'target. 216 = downbeat (or v1 beat_score); '
                             '217 = non-downbeat beat (default 216 — train '
                             'two separate models, one per target column).')
    parser.add_argument('--prep-workers', type=int, default=8,
                        help='Threads decompressing .npz files in parallel '
                             'inside the prep pipeline (default 8).')
    args = parser.parse_args()
    if args.file_cache_size < args.batch_size:
        print(f'WARN: --file-cache-size ({args.file_cache_size}) < '
              f'--batch-size ({args.batch_size}); bumping cache to batch size.',
              file=sys.stderr)
        args.file_cache_size = args.batch_size

    rng = np.random.default_rng(args.seed)
    dataset = load_dataset(args.data_dir, args.max_files)
    if not dataset:
        print('No data files found.', file=sys.stderr)
        sys.exit(1)

    # Same train/val split as 3-class training (seed=42 permutation).
    n_val = max(1, int(len(dataset) * args.val_split))
    indices = rng.permutation(len(dataset))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    print(f'Train: {len(train_data)} files, Val: {len(val_data)} files')

    gpu = VkCompute()
    # n_classes=1: continuous single-channel output. seq_mode=True so
    # linear layers' output buffers are sized for B*T flat batches.
    model = build_beat_crnn(gpu, input_size=216, hidden_size=args.hidden,
                             n_classes=1, batch_size=args.batch_size,
                             max_seq_len=args.chunk_len, seq_mode=True)

    start_epoch = 0
    best_val_loss = float('inf')
    if not args.no_resume:
        ckpt = load_checkpoint(args.output)
        if ckpt is not None:
            weights, next_epoch_, best_val_loss = ckpt
            if len(weights) == model.param_count():
                model.load_weights(weights)
                start_epoch = next_epoch_
                print(f'Resumed from checkpoint: next_epoch={next_epoch_+1}, '
                      f'best_val_loss={best_val_loss:.4f}')
            else:
                print(f'WARN: checkpoint param mismatch — fresh init')
                model.init_weights(seed=args.seed)
        else:
            model.init_weights(seed=args.seed)
            print(f'Fresh init (seed={args.seed})')
    else:
        model.init_weights(seed=args.seed)
        print(f'--no-resume: fresh init (seed={args.seed})')

    print(f'Model: {model.param_count()} parameters (hidden={args.hidden}, '
          f'output=1)')
    print(f'Chunk length: {args.chunk_len} frames ({args.chunk_len / OUR_FPS:.2f}s)')
    if args.val_interval > 0:
        print(f'Mid-epoch validation: every {args.val_interval / 60:.0f} minutes')
    print()

    BT = args.batch_size * args.chunk_len
    in_buf = gpu.create_buffer(args.batch_size * 216 * 4)
    grad_buf = gpu.create_buffer(args.batch_size * 4)  # not used in seq path
    input_seq_buf = gpu.create_buffer(BT * 216 * 4)
    target_seq_buf = gpu.create_buffer(BT * 4)         # (T, B) single channel
    gru_seq_out_buf = gpu.create_buffer(BT * args.hidden * 4)
    loss_acc_buf = gpu.create_buffer(BT * 4)
    grad_acc_buf = gpu.create_buffer(BT * 4)           # (T, B) grad on logit

    file_cache = LazyFileCache(max_files=args.file_cache_size)

    best_val_state = [best_val_loss]
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        train_batches = make_chunks(train_data, args.chunk_len,
                                     args.batch_size, rng)
        val_batches = make_chunks(val_data, args.chunk_len,
                                   args.batch_size, rng)

        train_loss = train_epoch_continuous(
            model, gpu, train_batches, args.lr, args.chunk_len,
            in_buf, grad_buf, train_data, file_cache,
            target_seq_buf, loss_acc_buf, grad_acc_buf,
            input_seq_buf, gru_seq_out_buf,
            val_batches=val_batches, val_data=val_data,
            val_interval=args.val_interval,
            val_sample_size=args.val_sample_size,
            best_val_state=best_val_state,
            save_path=args.output, epoch_num=epoch,
            prep_workers=args.prep_workers,
            target_col=args.target_col,
        )
        val_loss, val_f1, val_metrics = validate_continuous(
            model, gpu, val_batches, args.chunk_len, in_buf,
            val_data, file_cache, peak_threshold=args.peak_threshold,
            target_col=args.target_col,
        )
        print(f'  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'F1={val_f1:.3f}')
        print(format_continuous_metrics(val_metrics))

        if val_loss < best_val_state[0]:
            best_val_state[0] = val_loss
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, val_loss)
            print(f'  -> saved best weights (val_loss={val_loss:.4f})')
        else:
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, best_val_state[0])

        print()

    print(f'Training complete. Best val_loss={best_val_state[0]:.4f}')
    print(f'Weights saved to: {args.output}')


if __name__ == '__main__':
    main()
