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
    build_beat_crnn, build_beat_crnn_multidepth, MultiDepthBeatRNN,
    VkGRU, bce_loss_dispatch, bce_loss_multich_dispatch,
    ChannelSpec, CorpusSchema, load_schema, SCHEMA_FILENAME,
)
# Reuse infrastructure that doesn't change between 3-class and continuous modes.
from train_beat_rnn import (
    load_dataset, make_chunks, LazyFileCache, BatchPrepPipeline,
    save_checkpoint, load_checkpoint,
)


# Schema for the hierarchical 3-head retrain. Used by both this trainer
# (when --n-classes 3 and the data dir is packed format) AND by
# tools/pack_corpus.py to know what columns to lay out at pack time.
# Keep this constant in sync with the trainer's expectations.
BEAT_RNN_3HEAD_SCHEMA = CorpusSchema(channels=[
    ChannelSpec('spectrum',    'spectrum',    108),
    ChannelSpec('diff',        'diff',        108),
    ChannelSpec('labels_hier', 'labels_hier', 3),
])


def materialize_batch_continuous(batch_spec, dataset, chunk_len: int,
                                  cache: LazyFileCache,
                                  target_col: int = 216,
                                  n_classes: int = 1,
                                  label_key: str = 'beat_score',
                                  schema: CorpusSchema | None = None):
    """Build a continuous-target batch.

    inputs: (B, T, 216) — spectrum + diff
    targets: (B, T) for n_classes=1, or (B, T, n_classes) for n_classes>1

    Single-channel (n_classes=1):
        `target_col` selects which column the BCE loss targets. Default 216
        works for both v1 (beat_score) and v2 downbeat training.

    Multi-channel (n_classes=3, label_key='labels_hier'):
        If the corpus is packed with a schema (schema != None), reads the
        named `label_key` channel directly from the packed columns —
        fast (memmap view, no decompression). Otherwise reads the
        `labels_hier` field from the source .npz — slower fallback that
        works against unpacked corpora.
    """
    batch_inputs = []
    batch_targets = []
    for file_idx, start in batch_spec:
        path, _ = dataset[file_idx]
        entry = cache.get(path)
        end = start + chunk_len

        packed = isinstance(entry, np.memmap) or (
            isinstance(entry, np.ndarray) and entry.ndim == 2)

        if n_classes > 1:
            if packed and schema is not None:
                # Schema-aware packed path — view slicing, no copies.
                chunk = entry[start:end]
                spec_slice = chunk[:, schema.column_range('spectrum')]
                diff_slice = chunk[:, schema.column_range('diff')]
                target_slice = chunk[:, schema.column_range(label_key)]
                if target_slice.shape[1] != n_classes:
                    raise RuntimeError(
                        f'{path.name}: schema channel {label_key!r} has '
                        f'{target_slice.shape[1]} cols, n_classes={n_classes}')
                inp = np.concatenate([spec_slice, diff_slice], axis=1)
                batch_inputs.append(inp)
                batch_targets.append(target_slice)
            elif packed:
                raise RuntimeError(
                    f'{path.name}: n_classes>1 against packed format '
                    f'requires schema; either repack with tools/'
                    f'pack_corpus.py --schema beat_rnn_3head or use '
                    f'the unpacked .npz corpus.')
            else:
                # Unpacked .npz fallback.
                if label_key not in entry:
                    raise RuntimeError(
                        f'{path.name} has no {label_key!r} key — '
                        f'regenerate labels with '
                        f'tools/generate_beat_labels.py to add '
                        f'hierarchical 3-channel targets.')
                spec, diff = entry['spectrum'], entry['diff']
                labels_hier = entry[label_key]
                if labels_hier.shape[1] != n_classes:
                    raise RuntimeError(
                        f'{path.name}: {label_key} has '
                        f'{labels_hier.shape[1]} channels, '
                        f'n_classes={n_classes}')
                inp = np.concatenate(
                    [spec[start:end], diff[start:end]], axis=1)
                batch_inputs.append(inp)
                batch_targets.append(labels_hier[start:end])
        elif packed and entry.shape[1] >= 217:
            # Packed mmap format. View slicing is free. Includes both
            # legacy formats (217/218 cols) and the new schema-driven
            # packs (≥219). target_col-based slicing still works for
            # single-channel validation against multi-head packs since
            # col 216 = downbeat in both the legacy 218 layout and the
            # current 3-head schema (where labels_hier starts at 216).
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
                 n_workers: int = 8, target_col: int = 216,
                 n_classes: int = 1, label_key: str = 'beat_score',
                 schema: CorpusSchema | None = None):
        import queue, threading
        from concurrent.futures import ThreadPoolExecutor
        self.batches = batches
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.cache = cache
        self.target_col = target_col
        self.n_classes = n_classes
        self.label_key = label_key
        self.schema = schema
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
                target_col=self.target_col,
                n_classes=self.n_classes, label_key=self.label_key,
                schema=self.schema)
            inputs_TBF = np.ascontiguousarray(inputs.transpose(1, 0, 2))
            if self.n_classes > 1:
                # (B, T, C) → (T, B, C). Per-timestep slice in train loop
                # is targets_TBC[t] which is (B, C) flat after ravel.
                targets_TB = np.ascontiguousarray(targets.transpose(1, 0, 2))
            else:
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


# Per-head peak-pick thresholds for the 3-head model. The downbeat head's
# sigmoid output never gets near 0.3 because BeatNet's softmax assigns at
# most ~0.37 probability to downbeats (the three classes compete). Using
# threshold 0.3 against this head guarantees an uncrossable bar — the
# original mistake the 2026-05-29 hierarchical retrain investigation
# surfaced (post-hoc per-head F1: downbeat 0.04 → 0.41 just by dropping
# the threshold to 0.15). Defaults below are calibrated against the
# actual label distribution per head; modify in source if your label
# generator settings produce different ranges.
DEFAULT_HEAD_THRESHOLDS_3HEAD = {
    0: 0.15,  # downbeat — label max ~0.37, need much lower threshold
    1: 0.30,  # any-beat — label max ~0.93, v1-comparable
    2: 0.30,  # any-onset — same shape as any-beat
}

# Channel names for log output (only used in n_classes=3 mode).
HEAD_NAMES_3HEAD = {0: 'downbeat', 1: 'any-beat', 2: 'any-onset'}


# Multi-depth: head index (= GRU layer depth, 0=shallow → 2=deep) →
# labels_hier column. Match each head to the task whose temporal
# context need fits its depth:
#   head 0 (shallow GRU) ← col 2 = any-onset  (local, fast cue)
#   head 1 (medium GRU)  ← col 1 = any-beat   (periodicity)
#   head 2 (deep GRU)    ← col 0 = downbeat   (longer-memory meter cue)
MULTIDEPTH_HEAD_TO_LABEL_COL = {0: 2, 1: 1, 2: 0}
MULTIDEPTH_HEAD_NAMES = {0: 'onset', 1: 'beat', 2: 'downbeat'}
# Per-head thresholds for the multi-depth model. Inherits the same
# rationale as the 3-head DEFAULT_HEAD_THRESHOLDS_3HEAD: downbeat
# probabilities top out lower than the others so they need a smaller
# threshold. Map back by head index (depth), not label index.
MULTIDEPTH_HEAD_THRESHOLDS = {0: 0.30, 1: 0.30, 2: 0.15}


def _f1_for_head(probs: np.ndarray, targets: np.ndarray,
                 pred_threshold: float, label_threshold: float) -> dict:
    """Peak-pick predicted probs + labels for one head, compute F1@70ms.

    probs / targets shape: (B, T) float per-sequence. Aggregates tp/fp/fn
    across all sequences in the batch. Returns the standard metrics dict.
    """
    total_tp = total_fp = total_fn = total_pred = total_true = 0
    B = probs.shape[0]
    for b in range(B):
        pred_frames = peak_pick(probs[b], threshold=pred_threshold)
        true_frames = peak_pick(targets[b], threshold=label_threshold)
        pred_times = pred_frames / OUR_FPS
        true_times = true_frames / OUR_FPS
        m = f_measure(pred_times, true_times)
        total_tp += m['tp']
        total_fp += m['fp']
        total_fn += m['fn']
        total_pred += m['n_pred']
        total_true += m['n_true']
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'n_pred': total_pred, 'n_true': total_true,
    }


def validate_continuous(model, gpu, batches, chunk_len: int, in_buf,
                        dataset, cache: LazyFileCache,
                        max_batches: int | None = None,
                        peak_threshold: float = 0.3,
                        target_col: int = 216,
                        n_classes: int = 1,
                        label_key: str = 'beat_score',
                        schema: 'CorpusSchema | None' = None,
                        head_thresholds: dict[int, float] | None = None):
    """Run validation: peak-pick predicted scores, compute F1@70ms.

    Single-channel mode (n_classes=1, default): single-head F1 against
    `target_col` with `peak_threshold` for both predictions and labels.
    Returns (avg_loss, f1, metrics_dict) — backward-compatible shape.

    Multi-head mode (n_classes>1): per-head F1 with per-head thresholds
    from `head_thresholds` (defaults to DEFAULT_HEAD_THRESHOLDS_3HEAD).
    Returns (avg_loss, headline_f1, metrics_dict) where:
      - headline_f1 is the col-1 (any-beat) F1 — most v1-comparable
      - metrics_dict has aggregate {tp,fp,fn,precision,recall,f1,n_pred,
        n_true} computed by summing across all heads (back-compat with
        format_continuous_metrics) PLUS a 'per_head' sub-dict keyed by
        head index with that head's individual metrics.

    The per-head distinction matters because head outputs cover different
    dynamic ranges (downbeat softmax peaks at ~0.37; any-beat at ~0.93).
    A single global threshold makes the rare-positive head look broken
    even when the model has learned it correctly.
    """
    if max_batches is not None and max_batches < len(batches):
        import random
        batches = random.sample(batches, max_batches)

    if n_classes > 1 and head_thresholds is None:
        head_thresholds = DEFAULT_HEAD_THRESHOLDS_3HEAD

    losses = []
    # Aggregate across heads for the back-compat top-level metrics dict.
    # Per-head metrics tracked separately when n_classes>1.
    total_tp = total_fp = total_fn = total_pred = total_true = 0
    per_head_tp = [0] * n_classes
    per_head_fp = [0] * n_classes
    per_head_fn = [0] * n_classes
    per_head_pred = [0] * n_classes
    per_head_true = [0] * n_classes

    for batch_spec in batches:
        if n_classes == 1:
            inputs, targets = materialize_batch_continuous(
                batch_spec, dataset, chunk_len, cache, target_col=target_col)
        else:
            inputs, targets = materialize_batch_continuous(
                batch_spec, dataset, chunk_len, cache,
                n_classes=n_classes, label_key=label_key, schema=schema)
        B = inputs.shape[0]
        T = inputs.shape[1]

        # Per-sequence collect logits across timesteps for peak-picking
        # afterwards. (B, T) for single, (B, T, C) for multi.
        if n_classes == 1:
            seq_logits = np.zeros((B, T), dtype=np.float32)
        else:
            seq_logits = np.zeros((B, T, n_classes), dtype=np.float32)

        for layer in model.layers:
            if isinstance(layer, VkGRU):
                layer.reset_hidden()

        for t in range(T):
            frame = inputs[:, t, :]
            gpu.upload(in_buf, frame.ravel())
            out_buf = model.forward(in_buf, B, (216,))
            logits = gpu.download(out_buf, np.float32, B * n_classes)
            if n_classes == 1:
                logits = logits.reshape(B)
                seq_logits[:, t] = logits
            else:
                logits = logits.reshape(B, n_classes)
                seq_logits[:, t, :] = logits

            # BCE loss for this timestep (CPU side, for reporting only)
            if n_classes == 1:
                tgt = targets[:, t]
                z = logits
            else:
                tgt = targets[:, t, :]
                z = logits
            abs_z = np.abs(z)
            max_z = np.maximum(z, 0.0)
            loss = max_z - z * tgt + np.log1p(np.exp(-abs_z))
            losses.append(float(loss.mean()))

        # Convert logits to probabilities, peak-pick per sequence
        probs = 1.0 / (1.0 + np.exp(-seq_logits))

        if n_classes == 1:
            # Single-channel — existing behavior unchanged.
            m = _f1_for_head(probs, targets,
                              pred_threshold=peak_threshold,
                              label_threshold=0.5)
            total_tp += m['tp']
            total_fp += m['fp']
            total_fn += m['fn']
            total_pred += m['n_pred']
            total_true += m['n_true']
        else:
            # Multi-head — per-head F1 with per-head thresholds.
            # Aggregate-across-heads totals get computed once at the
            # end from per_head_*; no need to update them per batch.
            for c in range(n_classes):
                thr = head_thresholds.get(c, 0.3)
                m = _f1_for_head(probs[:, :, c], targets[:, :, c],
                                  pred_threshold=thr,
                                  label_threshold=thr)
                per_head_tp[c] += m['tp']
                per_head_fp[c] += m['fp']
                per_head_fn[c] += m['fn']
                per_head_pred[c] += m['n_pred']
                per_head_true[c] += m['n_true']

    avg_loss = float(np.mean(losses)) if losses else 0.0

    if n_classes == 1:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics = {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'n_pred': total_pred, 'n_true': total_true,
        }
        return avg_loss, f1, metrics

    # Multi-head: build per-head metrics + aggregate-across-heads totals.
    per_head = {}
    headline_f1 = 0.0
    for c in range(n_classes):
        tp, fp, fn = per_head_tp[c], per_head_fp[c], per_head_fn[c]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_head[c] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'n_pred': per_head_pred[c], 'n_true': per_head_true[c],
        }
        # Headline = any-beat (col 1) — most directly comparable to v1's
        # F1 since v1's beat_score equaled the col-0 + col-1 sum (i.e.
        # 1 - non_beat_prob, which is exactly what col 1 measures).
        if c == 1:
            headline_f1 = f1

    agg_tp = sum(per_head_tp)
    agg_fp = sum(per_head_fp)
    agg_fn = sum(per_head_fn)
    agg_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    agg_rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) else 0.0
    metrics = {
        'tp': agg_tp, 'fp': agg_fp, 'fn': agg_fn,
        'precision': agg_prec, 'recall': agg_rec, 'f1': agg_f1,
        'n_pred': sum(per_head_pred), 'n_true': sum(per_head_true),
        'per_head': per_head,
    }
    return avg_loss, headline_f1, metrics


def format_continuous_metrics(m: dict,
                              head_names: dict[int, str] | None = None) -> str:
    """Pretty-print validation metrics. For multi-head (n_classes>1) the
    per-head breakdown shows where each head actually lands so you can
    spot patterns like "downbeat F1 collapses because the threshold is
    wrong for that head's output range".

    head_names overrides the head-index→label mapping for the per-head
    rows. Default is HEAD_NAMES_3HEAD (single-arch labels_hier columns:
    downbeat/any-beat/any-onset). Multi-depth callers pass
    MULTIDEPTH_HEAD_NAMES (depth-ordered: onset/beat/downbeat).
    """
    if head_names is None:
        head_names = HEAD_NAMES_3HEAD
    lines = [
        f'  F1={m["f1"]:.3f}  precision={m["precision"]:.3f}  '
        f'recall={m["recall"]:.3f}  '
        f'tp={m["tp"]} fp={m["fp"]} fn={m["fn"]}  '
        f'(pred={m["n_pred"]} true={m["n_true"]})'
    ]
    if 'per_head' in m:
        for c, h in m['per_head'].items():
            name = head_names.get(c, f'head{c}')
            lines.append(
                f'    [{name:<10}] F1={h["f1"]:.3f}  '
                f'P={h["precision"]:.3f}  R={h["recall"]:.3f}  '
                f'tp={h["tp"]} fp={h["fp"]} fn={h["fn"]}  '
                f'(pred={h["n_pred"]} true={h["n_true"]})'
            )
    return '\n'.join(lines)


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
                           target_col: int = 216,
                           n_classes: int = 1,
                           label_key: str = 'beat_score',
                           schema: CorpusSchema | None = None):
    """Train one epoch in continuous mode (single output channel, BCE)."""
    losses = []
    mid_epoch_val_active = (val_batches and val_data is not None
                            and val_interval > 0 and best_val_state is not None
                            and save_path is not None)
    last_val_time = time.monotonic()

    linear_in, gru, linear_out = model.layers

    prep = ContinuousPrepPipeline(batches, dataset, chunk_len, cache,
                                   max_queue=2, n_workers=prep_workers,
                                   target_col=target_col,
                                   n_classes=n_classes, label_key=label_key,
                                   schema=schema)
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

        # Forward — same seq-mode dispatches regardless of n_classes.
        # Final linear produces (B*T, n_classes) logits.
        linear_in.forward(input_seq_buf, BT, (linear_in.in_features,))
        gru.forward_sequence(linear_in.output_buf, gru_seq_out_buf,
                              B_actual, T)
        linear_out.forward(gru_seq_out_buf, BT, (gru.hidden_size,))

        # BCE loss + grad. Single-channel path uses the original shader;
        # multi-channel routes through bce_loss_multich_dispatch which
        # treats each channel as an independent sigmoid + BCE (no softmax
        # constraint between them).
        if n_classes == 1:
            bce_loss_dispatch(
                gpu, linear_out.output_buf, target_seq_buf,
                grad_acc_buf, loss_acc_buf,
                batch_size=BT, target_offset_floats=0, t_inv=1.0,
            )
        else:
            bce_loss_multich_dispatch(
                gpu, linear_out.output_buf, target_seq_buf,
                grad_acc_buf, loss_acc_buf,
                batch_size=BT, n_classes=n_classes,
                target_offset_floats=0, t_inv=1.0,
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
                target_col=target_col,
                n_classes=n_classes, label_key=label_key, schema=schema)
            f1_label = ('F1[any-beat]' if n_classes > 1 else 'F1')
            print(f"    [mid-epoch +{elapsed_min:.1f}min] batch {batch_idx+1}/{n_batches}  "
                  f"val_loss={val_loss:.4f}  {f1_label}={val_f1:.3f}  "
                  f"(sampled {val_sample_size}/{len(val_batches)} val batches)")
            print(format_continuous_metrics(val_metrics))
            if val_loss < best_val_state[0]:
                best_val_state[0] = val_loss
                save_checkpoint(save_path, model.save_weights(),
                                epoch_num, val_loss,
                                input_size=linear_in.in_features,
                                proj_size=linear_in.out_features,
                                hidden_size=gru.hidden_size,
                                n_classes=linear_out.out_features)
                print(f"    [mid-epoch] -> saved best (val_loss={val_loss:.4f})")
            last_val_time = time.monotonic()

    return np.mean(losses) if losses else 0.0


def validate_multidepth(model: MultiDepthBeatRNN, gpu, batches,
                        chunk_len: int, input_seq_buf,
                        per_head_logits_host,
                        dataset, cache: LazyFileCache,
                        max_batches: int | None = None,
                        schema: 'CorpusSchema | None' = None,
                        head_thresholds: dict[int, float] | None = None
                        ) -> tuple[float, float, dict]:
    """Validate the multi-depth model — per-head F1 + summed BCE loss.

    Multi-depth always operates on labels_hier (3 channels). One forward
    pass per chunk via forward_multidepth (seq mode); per-head BCE on
    the CPU side mirroring validate_continuous's reporting math.

    Returns (avg_loss, headline_f1, metrics_dict) with the same shape as
    validate_continuous's multi-head return so format_continuous_metrics
    works unchanged. headline_f1 is the BEAT head (depth=1) — the
    most v1-comparable signal.
    """
    if max_batches is not None and max_batches < len(batches):
        import random
        batches = random.sample(batches, max_batches)
    if head_thresholds is None:
        head_thresholds = MULTIDEPTH_HEAD_THRESHOLDS

    BT_full = model.batch_size * chunk_len

    losses = []
    per_head_tp = [0] * model.n_gru_layers
    per_head_fp = [0] * model.n_gru_layers
    per_head_fn = [0] * model.n_gru_layers
    per_head_pred = [0] * model.n_gru_layers
    per_head_true = [0] * model.n_gru_layers

    for batch_spec in batches:
        inputs, targets = materialize_batch_continuous(
            batch_spec, dataset, chunk_len, cache,
            n_classes=3, label_key='labels_hier', schema=schema)
        B = inputs.shape[0]
        T = inputs.shape[1]
        BT = B * T

        # (B, T, F) → (T, B, F) — matches train-time layout, lets the
        # model's pre-allocated seq buffers be re-used directly.
        inputs_TBF = np.ascontiguousarray(inputs.transpose(1, 0, 2))
        # (B, T, 3) → numpy reshape per-head; (T, B, C) layout matches
        # the (B*T,) flat-batch order the head logits download in.
        targets_TBC = np.ascontiguousarray(targets.transpose(1, 0, 2))

        gpu.upload(input_seq_buf, inputs_TBF.ravel())
        model.reset_hidden()
        head_outs = model.forward_multidepth(input_seq_buf, B=B, T=T)

        # (B, T) per head — same axes as targets_TBC[:, :, head_label_col]
        # after transpose-back.
        per_head_probs_BT = []
        for head_idx, head_buf in enumerate(head_outs):
            # head logits are (B*T, 1) flat — matches (T, B, 1) ravel.
            logits_flat = gpu.download(head_buf, np.float32, BT)
            logits_TB = logits_flat.reshape(T, B)
            logits_BT = logits_TB.transpose(1, 0)

            label_col = MULTIDEPTH_HEAD_TO_LABEL_COL[head_idx]
            tgt_BT = targets[:, :, label_col]  # (B, T)

            abs_z = np.abs(logits_BT)
            max_z = np.maximum(logits_BT, 0.0)
            loss = max_z - logits_BT * tgt_BT + np.log1p(np.exp(-abs_z))
            losses.append(float(loss.mean()))

            probs_BT = 1.0 / (1.0 + np.exp(-logits_BT))
            per_head_probs_BT.append(probs_BT)

        # Per-head F1@70ms.
        for head_idx in range(model.n_gru_layers):
            thr = head_thresholds.get(head_idx, 0.3)
            label_col = MULTIDEPTH_HEAD_TO_LABEL_COL[head_idx]
            m = _f1_for_head(per_head_probs_BT[head_idx],
                              targets[:, :, label_col],
                              pred_threshold=thr,
                              label_threshold=thr)
            per_head_tp[head_idx] += m['tp']
            per_head_fp[head_idx] += m['fp']
            per_head_fn[head_idx] += m['fn']
            per_head_pred[head_idx] += m['n_pred']
            per_head_true[head_idx] += m['n_true']

    avg_loss = float(np.mean(losses)) if losses else 0.0

    # Re-use the multi-head metrics-dict shape so format_continuous_metrics
    # works without a branch. Head names come from MULTIDEPTH_HEAD_NAMES
    # rather than HEAD_NAMES_3HEAD — caller is responsible for swapping
    # the lookup before printing (done in main()).
    per_head = {}
    headline_f1 = 0.0
    for head_idx in range(model.n_gru_layers):
        tp, fp, fn = (per_head_tp[head_idx], per_head_fp[head_idx],
                      per_head_fn[head_idx])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_head[head_idx] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'n_pred': per_head_pred[head_idx],
            'n_true': per_head_true[head_idx],
        }
        if MULTIDEPTH_HEAD_NAMES.get(head_idx) == 'beat':
            headline_f1 = f1

    agg_tp = sum(per_head_tp)
    agg_fp = sum(per_head_fp)
    agg_fn = sum(per_head_fn)
    agg_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) else 0.0
    agg_rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) else 0.0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) else 0.0
    metrics = {
        'tp': agg_tp, 'fp': agg_fp, 'fn': agg_fn,
        'precision': agg_prec, 'recall': agg_rec, 'f1': agg_f1,
        'n_pred': sum(per_head_pred), 'n_true': sum(per_head_true),
        'per_head': per_head,
    }
    return avg_loss, headline_f1, metrics


def _resolve_head_pos_weights(spec: str, data_dir: Path,
                                n_files: int, seed: int
                                ) -> list[float] | None:
    """Parse the --head-pos-weights CLI spec into a per-head list.
    Returns None for 'none' or '1' (uniform = no rebalancing).

    Spec values:
      'none' / '1' / '1.0'     -> None (unweighted)
      'auto'                    -> [n_neg/n_pos] per head, sampled
      'sqrt-auto'               -> [sqrt(n_neg/n_pos)] per head
      'w0,w1,w2'                -> explicit list in head-index order
    """
    s = spec.strip().lower()
    if s in ('none', '1', '1.0'):
        return None
    if s in ('auto', 'sqrt-auto'):
        # Sample positive rates from the corpus. Mirror the diagnostic
        # script's logic — keep it local rather than importing the
        # diag tool (which may not exist post-cleanup).
        rng = np.random.default_rng(seed)
        files = sorted(data_dir.glob('*.npz'))
        sample = rng.choice(files, size=min(n_files, len(files)),
                              replace=False)
        head_pos = {h: 0 for h in MULTIDEPTH_HEAD_TO_LABEL_COL}
        total_frames = 0
        for path in sample:
            try:
                d = np.load(path, allow_pickle=False)
                labels = d['labels_hier']
                if labels.ndim != 2 or labels.shape[1] < 3:
                    continue
                total_frames += labels.shape[0]
                for h, c in MULTIDEPTH_HEAD_TO_LABEL_COL.items():
                    head_pos[h] += int((labels[:, c] > 0.5).sum())
            except Exception:
                continue
        if total_frames == 0:
            print('WARN: head-pos-weights auto failed; falling back to '
                  'uniform.', file=sys.stderr)
            return None
        weights = []
        for h in sorted(head_pos):  # head 0..N-1 order
            rate = head_pos[h] / total_frames
            if rate <= 0:
                weights.append(1.0)
                continue
            inv = (1.0 - rate) / rate  # n_neg/n_pos
            weights.append(float(np.sqrt(inv)) if s == 'sqrt-auto' else float(inv))
        return weights
    # Explicit list.
    try:
        return [float(x) for x in spec.split(',')]
    except ValueError as e:
        raise ValueError(f'unparseable --head-pos-weights: {spec!r}; '
                          f'expected "auto"/"sqrt-auto"/"none" or '
                          f'comma-separated floats. ({e})')


def train_epoch_multidepth(model: MultiDepthBeatRNN, gpu, batches,
                           lr: float, chunk_len: int,
                           dataset, cache,
                           input_seq_buf,
                           per_head_target_bufs: list,
                           per_head_loss_bufs: list,
                           per_head_grad_bufs: list,
                           val_batches=None, val_data=None,
                           val_interval: float = 0.0,
                           val_sample_size: int = 50,
                           best_val_state=None,
                           save_path: Path | None = None,
                           epoch_num: int = 0,
                           prep_workers: int = 8,
                           schema: CorpusSchema | None = None,
                           ckpt_dims: dict | None = None,
                           head_pos_weights: list[float] | None = None):
    """Train one epoch of the multi-depth model.

    Per batch:
      1. forward_multidepth → list of (B*T, 1) head logits
      2. per head: bce_loss_dispatch against the head's label-column slice
      3. backward_multidepth(head_grads, B, T) — routes gradients through
         the stack (sum where head + upstream-GRU meet)
      4. sgd_step

    Mirrors train_epoch_continuous's mid-epoch validation behavior.
    """
    losses = []  # per-batch summed-over-heads loss for the train printout
    mid_epoch_val_active = (val_batches and val_data is not None
                            and val_interval > 0 and best_val_state is not None
                            and save_path is not None)
    last_val_time = time.monotonic()

    # n_classes=3 prep pipeline: targets_TB is (T, B, 3) with cols
    # 0=downbeat, 1=beat, 2=onset (labels_hier convention).
    prep = ContinuousPrepPipeline(batches, dataset, chunk_len, cache,
                                   max_queue=2, n_workers=prep_workers,
                                   n_classes=3, label_key='labels_hier',
                                   schema=schema)
    n_batches = len(batches)

    prep_iter = iter(prep)
    for batch_idx in range(n_batches):
        try:
            inputs_TBF, targets_TBC = next(prep_iter)
        except StopIteration:
            break

        B_actual = inputs_TBF.shape[1]
        T = inputs_TBF.shape[0]
        BT = B_actual * T

        model.zero_grad()
        model.reset_hidden()
        for lb, gb in zip(per_head_loss_bufs, per_head_grad_bufs):
            gpu.zero_buffer(lb)
            gpu.zero_buffer(gb)

        gpu.upload(input_seq_buf, inputs_TBF.ravel())
        # Per-head target uploads. targets_TBC is (T, B, 3); we slice each
        # head's label column and upload as flat (T*B,) which matches the
        # (B*T,) flat-batch order the head logits use.
        for head_idx in range(model.n_gru_layers):
            label_col = MULTIDEPTH_HEAD_TO_LABEL_COL[head_idx]
            tgt_TB = np.ascontiguousarray(targets_TBC[:, :, label_col])
            gpu.upload(per_head_target_bufs[head_idx], tgt_TB.ravel())

        head_outs = model.forward_multidepth(input_seq_buf, B=B_actual, T=T)

        # Per-head BCE — single-channel shader, three dispatches. The
        # multidepth corpus has very different positive rates per head
        # (downbeat ~1.5% vs onset ~12%), so per-head pos_weight is
        # essential to keep the downbeat head from being drowned out
        # by the negative-class contribution to its own BCE.
        for head_idx, head_buf in enumerate(head_outs):
            pw = (head_pos_weights[head_idx]
                  if head_pos_weights is not None else 1.0)
            bce_loss_dispatch(
                gpu, head_buf, per_head_target_bufs[head_idx],
                per_head_grad_bufs[head_idx],
                per_head_loss_bufs[head_idx],
                batch_size=BT, target_offset_floats=0, t_inv=1.0,
                pos_weight=pw,
            )

        # Summed loss across heads — comparable across batches and to
        # the multi-head n_classes=3 training-mode loss printout.
        batch_loss = 0.0
        for lb in per_head_loss_bufs:
            batch_loss += float(gpu.download(lb, np.float32, BT).mean())
        losses.append(batch_loss)

        model.backward_multidepth(per_head_grad_bufs, B=B_actual, T=T)
        model.sgd_step(lr)

        if (batch_idx + 1) % 10 == 0:
            print(f"    batch {batch_idx+1}/{n_batches} "
                  f"loss(sum-heads)={batch_loss:.4f}")

        if mid_epoch_val_active and (time.monotonic() - last_val_time) >= val_interval:
            elapsed_min = (time.monotonic() - last_val_time) / 60.0
            val_loss, val_f1, val_metrics = validate_multidepth(
                model, gpu, val_batches, chunk_len, input_seq_buf,
                None,  # per_head_logits_host unused
                val_data, cache, max_batches=val_sample_size,
                schema=schema)
            print(f"    [mid-epoch +{elapsed_min:.1f}min] "
                  f"batch {batch_idx+1}/{n_batches}  "
                  f"val_loss={val_loss:.4f}  F1[beat]={val_f1:.3f}  "
                  f"(sampled {val_sample_size}/{len(val_batches)} val batches)")
            print(format_continuous_metrics(
                val_metrics, head_names=MULTIDEPTH_HEAD_NAMES))
            if val_loss < best_val_state[0]:
                best_val_state[0] = val_loss
                ckpt_kwargs = dict(ckpt_dims) if ckpt_dims else {}
                save_checkpoint(save_path, model.save_weights(),
                                epoch_num, val_loss, **ckpt_kwargs)
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
    parser.add_argument('--n-classes', type=int, default=1, choices=[1, 3],
                        help='Output heads. 1 = single-channel BCE (default, '
                             'backward-compatible). 3 = multi-head sigmoid + '
                             'per-channel BCE for hierarchical (downbeat / '
                             'any-beat / any-onset) labels. Requires '
                             '--label-key labels_hier and .npz inputs with '
                             'the labels_hier field present.')
    parser.add_argument('--label-key', type=str, default='beat_score',
                        help='Key in the .npz to read targets from. Defaults '
                             'to beat_score (single-channel). Use '
                             'labels_hier for n_classes=3 hierarchical '
                             'targets (run tools/generate_beat_labels.py to '
                             'add it to your label corpus).')
    parser.add_argument('--proj-size', type=int, default=32,
                        help='Input projection dimension (linear: 216 → '
                             'proj_size → GRU). v1 default 32 stayed tight; '
                             'v2 (3-head) bumps to 64 to give each head '
                             'more raw spectrum signal. Must match the '
                             'runtime _PROJ_SIZE in beat_rnn.py.')
    parser.add_argument('--prep-workers', type=int, default=8,
                        help='Threads decompressing .npz files in parallel '
                             'inside the prep pipeline (default 8).')
    parser.add_argument('--architecture', type=str, default='single',
                        choices=['single', 'multidepth'],
                        help='Model architecture. "single" (default) = '
                             'one GRU with an N-head linear output (works '
                             'for both n_classes=1 v1 and n_classes=3 v3 '
                             'hierarchical layouts). "multidepth" = '
                             'stacked GRU, one head per layer; implies '
                             'n_classes=3 and labels_hier targets.')
    parser.add_argument('--n-gru-layers', type=int, default=3,
                        help='Stack depth for --architecture multidepth '
                             '(default 3, one per head). Ignored for '
                             '--architecture single.')
    parser.add_argument('--head-pos-weights', type=str, default='sqrt-auto',
                        help='Per-head BCE pos_weight (multidepth only). '
                             '"1" or "none" disables (uniform BCE). '
                             '"auto" = n_neg/n_pos sampled from data '
                             '(strongest rebalancing — risks overshoot). '
                             '"sqrt-auto" (default) = sqrt(n_neg/n_pos) '
                             '— gentler, pulls bias toward zero without '
                             'flipping it. Or pass three comma-separated '
                             'floats in head-index order (head 0=onset, '
                             '1=beat, 2=downbeat) e.g. "2.7,2.9,8.0".')
    parser.add_argument('--head-pos-weights-sample', type=int, default=200,
                        help='Number of files to sample when '
                             '--head-pos-weights auto/sqrt-auto (default 200).')
    args = parser.parse_args()

    if args.architecture == 'multidepth':
        if args.n_classes != 3:
            print(f'NOTE: --architecture multidepth implies n_classes=3 '
                  f'(was {args.n_classes}). Overriding.', file=sys.stderr)
            args.n_classes = 3
        if args.label_key != 'labels_hier':
            print(f'NOTE: --architecture multidepth requires '
                  f'--label-key labels_hier (was {args.label_key!r}). '
                  f'Overriding.', file=sys.stderr)
            args.label_key = 'labels_hier'
        if args.proj_size < 48:
            print(f'NOTE: --architecture multidepth typically uses '
                  f'proj_size>=48 (was {args.proj_size}). Continuing — '
                  f'pass --proj-size 64 for the intended default.',
                  file=sys.stderr)
    if args.file_cache_size < args.batch_size:
        print(f'WARN: --file-cache-size ({args.file_cache_size}) < '
              f'--batch-size ({args.batch_size}); bumping cache to batch size.',
              file=sys.stderr)
        args.file_cache_size = args.batch_size

    # Resolve per-head pos_weights from CLI. Only consulted by the
    # multidepth path; ignored for --architecture single.
    head_pos_weights = None
    if args.architecture == 'multidepth':
        head_pos_weights = _resolve_head_pos_weights(
            args.head_pos_weights, args.data_dir,
            args.head_pos_weights_sample, args.seed)
        if head_pos_weights is not None:
            names = [MULTIDEPTH_HEAD_NAMES[i]
                      for i in range(len(head_pos_weights))]
            paired = ', '.join(f'{n}={w:.3f}'
                                for n, w in zip(names, head_pos_weights))
            print(f'Per-head pos_weights: {paired}', file=sys.stderr)

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

    # Sanity check: multi-head mode requires hierarchical labels.
    if args.n_classes > 1 and args.label_key != 'labels_hier':
        print(f'WARN: --n-classes={args.n_classes} but '
              f'--label-key={args.label_key!r}. Multi-head mode is meant '
              f"for hierarchical labels (run tools/generate_beat_labels.py "
              f"to add labels_hier to your corpus). Continuing — you "
              f"probably want --label-key labels_hier.", file=sys.stderr)

    gpu = VkCompute()
    if args.architecture == 'multidepth':
        model = build_beat_crnn_multidepth(
            gpu, input_size=216, hidden_size=args.hidden,
            proj_size=args.proj_size, n_gru_layers=args.n_gru_layers,
            n_heads=args.n_gru_layers,
            batch_size=args.batch_size, max_seq_len=args.chunk_len)
    else:
        # seq_mode=True so linear layers' output buffers are sized for B*T
        # flat batches. n_classes drives the output dim of linear_out and
        # therefore the shape of grad_acc / target buffers below.
        model = build_beat_crnn(gpu, input_size=216, hidden_size=args.hidden,
                                 proj_size=args.proj_size,
                                 n_classes=args.n_classes,
                                 batch_size=args.batch_size,
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
          f'proj={args.proj_size}, output={args.n_classes})')
    print(f'Chunk length: {args.chunk_len} frames ({args.chunk_len / OUR_FPS:.2f}s)')
    if args.val_interval > 0:
        print(f'Mid-epoch validation: every {args.val_interval / 60:.0f} minutes')
    print()

    BT = args.batch_size * args.chunk_len
    C = args.n_classes
    in_buf = gpu.create_buffer(args.batch_size * 216 * 4)
    grad_buf = gpu.create_buffer(args.batch_size * 4)  # not used in seq path
    input_seq_buf = gpu.create_buffer(BT * 216 * 4)

    if args.architecture == 'multidepth':
        # Per-head buffers (one per GRU layer). Each head's BCE is a
        # single-channel dispatch so each gets its own (BT,) target /
        # loss / grad buffer. No combined (BT, n_classes) buffer
        # needed for this path.
        per_head_target_bufs = [gpu.create_buffer(BT * 4)
                                 for _ in range(args.n_gru_layers)]
        per_head_loss_bufs = [gpu.create_buffer(BT * 4)
                               for _ in range(args.n_gru_layers)]
        per_head_grad_bufs = [gpu.create_buffer(BT * 4)
                               for _ in range(args.n_gru_layers)]
        # Single-arch buffers below are unused in this branch; allocate
        # placeholders so the existing branch's variable names still
        # resolve (validate_continuous reads in_buf etc., but won't be
        # called for multidepth).
        target_seq_buf = gru_seq_out_buf = loss_acc_buf = grad_acc_buf = None
    else:
        # Target + grad buffers sized for n_classes channels. Loss is
        # summed over channels into a single (BT,) accumulator (the
        # multich BCE shader handles this; for n_classes=1 it's the
        # same as before).
        target_seq_buf = gpu.create_buffer(BT * C * 4)
        gru_seq_out_buf = gpu.create_buffer(BT * args.hidden * 4)
        loss_acc_buf = gpu.create_buffer(BT * 4)
        grad_acc_buf = gpu.create_buffer(BT * C * 4)

    file_cache = LazyFileCache(max_files=args.file_cache_size)

    # If the data dir is packed and has a schema manifest, load it and
    # pass to the prep pipeline so n_classes>1 can slice from packed
    # columns instead of re-opening source .npz files.
    schema = None
    if (args.data_dir / SCHEMA_FILENAME).exists():
        schema = load_schema(args.data_dir)
        print(f'Loaded packed-corpus schema: '
              f'{schema.total_columns} cols, channels={schema.names()}')

    best_val_state = [best_val_loss]
    for epoch in range(start_epoch, args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')

        train_batches = make_chunks(train_data, args.chunk_len,
                                     args.batch_size, rng)
        val_batches = make_chunks(val_data, args.chunk_len,
                                   args.batch_size, rng)

        # Architecture dims serialized into the checkpoint so the
        # runtime can reconstruct the model without hardcoded constants.
        # Built before the train call so train_epoch_multidepth's mid-
        # epoch saves use the same fields as the end-of-epoch save.
        ckpt_dims = dict(
            input_size=216, proj_size=args.proj_size,
            hidden_size=args.hidden, n_classes=args.n_classes,
        )
        if args.architecture == 'multidepth':
            ckpt_dims['architecture'] = 'multidepth'
            ckpt_dims['n_gru_layers'] = args.n_gru_layers

        if args.architecture == 'multidepth':
            train_loss = train_epoch_multidepth(
                model, gpu, train_batches, args.lr, args.chunk_len,
                train_data, file_cache, input_seq_buf,
                per_head_target_bufs, per_head_loss_bufs,
                per_head_grad_bufs,
                val_batches=val_batches, val_data=val_data,
                val_interval=args.val_interval,
                val_sample_size=args.val_sample_size,
                best_val_state=best_val_state,
                save_path=args.output, epoch_num=epoch,
                prep_workers=args.prep_workers,
                schema=schema, ckpt_dims=ckpt_dims,
                head_pos_weights=head_pos_weights,
            )
            val_loss, val_f1, val_metrics = validate_multidepth(
                model, gpu, val_batches, args.chunk_len, input_seq_buf,
                None, val_data, file_cache, schema=schema)
            f1_label = 'F1[beat]'
        else:
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
                n_classes=args.n_classes,
                label_key=args.label_key,
                schema=schema,
            )
            # Validation: per-head F1 in n_classes>1 mode (each head has
            # its own threshold — see DEFAULT_HEAD_THRESHOLDS_3HEAD).
            # Headline F1 returned is the col-1 (any-beat) one since
            # that's what's directly comparable to v1's F1=0.544 (v1's
            # beat_score was 1 - non_beat_prob = col 0 + col 1).
            val_loss, val_f1, val_metrics = validate_continuous(
                model, gpu, val_batches, args.chunk_len, in_buf,
                val_data, file_cache, peak_threshold=args.peak_threshold,
                target_col=args.target_col,
                n_classes=args.n_classes, label_key=args.label_key,
                schema=schema,
            )
            f1_label = ('F1[any-beat]' if args.n_classes > 1 else 'F1')

        print(f'  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'{f1_label}={val_f1:.3f}')
        # Multi-depth uses depth-ordered head names (onset/beat/downbeat
        # at head index 0/1/2) — different from the single-arch
        # labels_hier column order (downbeat/any-beat/any-onset).
        names = (MULTIDEPTH_HEAD_NAMES
                 if args.architecture == 'multidepth' else None)
        print(format_continuous_metrics(val_metrics, head_names=names))
        if val_loss < best_val_state[0]:
            best_val_state[0] = val_loss
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, val_loss, **ckpt_dims)
            print(f'  -> saved best weights (val_loss={val_loss:.4f})')
        else:
            save_checkpoint(args.output, model.save_weights(),
                            epoch + 1, best_val_state[0], **ckpt_dims)

        print()

    print(f'Training complete. Best val_loss={best_val_state[0]:.4f}')
    print(f'Weights saved to: {args.output}')


if __name__ == '__main__':
    main()
