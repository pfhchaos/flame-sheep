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
import time
from pathlib import Path

import numpy as np

from wallpaper_ml.vk_compute import VkCompute
from wallpaper_ml import build_beat_crnn, VkGRU


def _read_npy_header_shape(path: Path) -> tuple[int, ...] | None:
    """Read the shape of a packed .npy file from its header only."""
    from numpy.lib import format as npy_fmt
    try:
        with open(path, 'rb') as f:
            major, minor = npy_fmt.read_magic(f)
            if (major, minor) == (1, 0):
                shape, _, _ = npy_fmt.read_array_header_1_0(f)
            elif (major, minor) == (2, 0):
                shape, _, _ = npy_fmt.read_array_header_2_0(f)
            else:
                return None
            return shape
    except (OSError, ValueError):
        return None


def _scan_one_npz(path: Path) -> tuple[Path, int] | None:
    """Read spectrum length from a label .npz without decompressing the
    array. The .npy header inside the zip carries shape in its first
    ~128 bytes; DEFLATE stops there. ~50× faster than np.load()['spectrum'].
    Returns None for files that aren't label files (no 'spectrum.npy')
    or that fail to parse.
    """
    import zipfile
    from numpy.lib import format as npy_fmt
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if 'spectrum.npy' not in names:
                return None
            with zf.open('spectrum.npy') as f:
                major, minor = npy_fmt.read_magic(f)
                if (major, minor) == (1, 0):
                    shape, _, _ = npy_fmt.read_array_header_1_0(f)
                elif (major, minor) == (2, 0):
                    shape, _, _ = npy_fmt.read_array_header_2_0(f)
                else:
                    return None
                return (path, int(shape[0]))
    except (zipfile.BadZipFile, OSError, ValueError, KeyError):
        return None


def _scan_one_npy(path: Path) -> tuple[Path, int] | None:
    """Read frame count from a packed (T, 217) .npy file's header."""
    shape = _read_npy_header_shape(path)
    if shape is None or len(shape) < 1:
        return None
    return (path, int(shape[0]))


# Back-compat alias — early callers grep this name; keep it working.
_scan_one = _scan_one_npz


def load_dataset(data_dir: Path, max_files: int | None = None):
    """Scan label files and return list of (path, n_frames) tuples.

    Detects format automatically:
    - If data_dir contains *.npy files, treat them as packed mmap files
      (shape (T, 217), cols [spec(108), diff(108), beat_score(1)]).
    - Else fall back to legacy *.npz scanning.

    Lazy: array data is NOT loaded here.
    """
    import multiprocessing as mp
    npy_files = sorted(data_dir.glob('*.npy'))
    if npy_files:
        files = npy_files
        scanner = _scan_one_npy
        ext = 'npy'
    else:
        files = sorted(data_dir.glob('*.npz'))
        scanner = _scan_one_npz
        ext = 'npz'

    if max_files:
        files = files[:max_files]

    n_total = len(files)
    print(f"Scanning {n_total} .{ext} files from {data_dir}...")
    report_every = max(1, n_total // 20)

    dataset = []
    total_frames = 0
    skipped = 0
    # imap (ordered) so the returned dataset is reproducible across runs.
    with mp.Pool() as pool:
        for i, result in enumerate(
                pool.imap(scanner, files, chunksize=32)):
            if result is None:
                skipped += 1
            else:
                dataset.append(result)
                total_frames += result[1]
            if (i + 1) % report_every == 0 or i + 1 == n_total:
                print(f"  scanned {i + 1}/{n_total} "
                      f"({total_frames} frames so far)")
    if skipped:
        print(f"  (skipped {skipped} non-label .{ext} files)")

    print(f"Scanned {len(dataset)} files, {total_frames} total frames "
          f"({total_frames / 93.75 / 60:.1f} minutes)")
    return dataset


class LazyFileCache:
    """LRU cache of file handles keyed by path. Handles two formats:

    - Packed mmap .npy (shape (T, 217), cols [spec(108), diff(108),
      beat_score(1)]): cached entry is a single np.memmap. Slicing
      returns a view (~free), the OS page cache holds hot pages. Each
      handle is one file descriptor — bound the LRU at max_files to
      respect ulimit -n.
    - Legacy .npz: cached entry is a dict of decompressed arrays
      ({'spectrum', 'diff', 'labels', 'beat_score', ...}). Decompressing
      is expensive so the cache earns its keep here.

    Thread-safe.
    """

    def __init__(self, max_files: int = 64):
        import threading
        self.max_files = max_files
        self._cache: dict = {}
        self._order: list = []
        self._lock = threading.Lock()

    def get(self, path):
        """Returns either a {key: ndarray} dict (legacy .npz) or a
        single np.memmap (packed .npy). The materialize step branches
        on which it got."""
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
        # Outside the lock — the load (decompress for .npz, mmap+header
        # for .npy) is the slow part. Re-check inside the lock after
        # to avoid two threads loading the same file twice.
        if path.suffix == '.npy':
            entry = np.load(path, mmap_mode='r')
        else:
            d = np.load(path)
            entry = {k: np.array(d[k]) for k in d.files}
        with self._lock:
            if key in self._cache:
                # Another thread beat us to it; reuse theirs.
                self._order.remove(key)
                self._order.append(key)
                return self._cache[key]
            self._cache[key] = entry
            self._order.append(key)
            while len(self._cache) > self.max_files:
                oldest = self._order.pop(0)
                del self._cache[oldest]
        return entry


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


class BatchPrepPipeline:
    """Background thread that materializes + transposes batches ahead of GPU.

    materialize_batch (load files via cache, slice chunks, concat
    spec+diff) plus the (B, T, F) → (T, B, F) transpose is pure CPU work
    that doesn't block on GPU compute. Running it on a background thread
    lets the prep for batch N+1 overlap with the GPU work for batch N,
    cutting the per-batch wall time by whatever the prep cost was.

    Bounded queue (size 2 by default) so prep stays at most one batch
    ahead — keeps memory pressure bounded and ensures the consumer
    doesn't outrun the producer. Lock-free: just a queue.Queue under
    the hood.

    None sentinel signals end-of-iterator.
    """

    def __init__(self, batches, dataset, chunk_len: int,
                 cache: LazyFileCache, max_queue: int = 2):
        import queue, threading
        self.batches = batches
        self.dataset = dataset
        self.chunk_len = chunk_len
        self.cache = cache
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        for batch_spec in self.batches:
            if self._stop.is_set():
                break
            inputs, labels = materialize_batch(
                batch_spec, self.dataset, self.chunk_len, self.cache)
            inputs_TBF = np.ascontiguousarray(inputs.transpose(1, 0, 2))
            labels_TBF = np.ascontiguousarray(labels.transpose(1, 0, 2))
            self.queue.put((inputs_TBF, labels_TBF))
        self.queue.put(None)  # end sentinel

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item

    def stop(self):
        """Signal the background thread to stop early."""
        self._stop.set()
        # Drain queue so producer can put the None sentinel and exit.
        try:
            while True:
                self.queue.get_nowait()
        except Exception:
            pass


def materialize_batch(batch_spec, dataset, chunk_len: int, cache: LazyFileCache):
    """Load files for a single batch via cache, return (inputs, labels) arrays.

    inputs: [B, T, 216] float32 (spectrum + diff concatenated)
    labels: [B, T, 3]   float32 (soft probabilities)
    """
    batch_inputs = []
    batch_labels = []
    for file_idx, start in batch_spec:
        path, _ = dataset[file_idx]
        arrs = cache.get(path)
        spec, diff, labels = arrs['spectrum'], arrs['diff'], arrs['labels']
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
                    best_val_loss: float,
                    *,
                    input_size: int | None = None,
                    proj_size: int | None = None,
                    hidden_size: int | None = None,
                    n_classes: int | None = None) -> None:
    """Save weights + training metadata in .npz format.

    Layout: {weights, next_epoch, best_val_loss,
             input_size?, proj_size?, hidden_size?, n_classes?}.

    `next_epoch` semantically means "first epoch to do on resume" — so an
    end-of-epoch save (just finished epoch N) stores next_epoch=N+1, but
    a mid-epoch save (still doing epoch N) stores next_epoch=N. Resume
    sets start_epoch = next_epoch directly. This way mid-epoch
    checkpoints correctly cause the resume to re-do that epoch's
    remaining data with the saved (mid-epoch-improved) weights, rather
    than skipping ahead.

    Architecture dims (input_size, proj_size, hidden_size, n_classes)
    are optional but should always be passed by new callers — they let
    BeatRNNDetector reconstruct the model without hardcoded constants.
    Legacy files (pre-dim-serialization) have these omitted; the
    runtime falls back to a known-sizes lookup table.

    Atomic write via tmp + rename so partial writes don't corrupt a
    good checkpoint.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    extras = {}
    for name, value in (('input_size', input_size),
                        ('proj_size', proj_size),
                        ('hidden_size', hidden_size),
                        ('n_classes', n_classes)):
        if value is not None:
            extras[name] = int(value)
    np.savez(tmp, weights=weights, next_epoch=next_epoch,
             best_val_loss=best_val_loss, **extras)
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
                label_seq_buf, loss_acc_buf, grad_acc_buf,
                input_seq_buf, gru_seq_out_buf, gru_upstream_buf,
                class_weights_buf, focal_gamma: float = 0.0,
                val_batches=None, val_data=None, val_interval: float = 0.0,
                val_sample_size: int = 200,
                best_val_state=None, save_path: Path | None = None,
                epoch_num: int = 0, profile: bool = False):
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

    from wallpaper_ml import cross_entropy_dispatch
    N_CLASSES = 3

    # Unpack model layers — train loop drives them directly in seq mode
    # because VkModel.forward chains all layers with a single batch_size,
    # but seq mode has linears at B*T flat batch while GRU operates on B
    # sequences of T steps. Direct layer calls let us pass the right
    # batch_size to each.
    linear_in, gru, linear_out = model.layers

    # Spin up a background thread that materializes + transposes batches
    # ahead of the GPU. With queue size 2, prep can stay one batch ahead
    # so the GPU never waits for CPU file-loading + numpy work.
    prep = BatchPrepPipeline(batches, dataset, chunk_len, cache, max_queue=2)
    n_batches = len(batches)

    # Profiling timers — accumulate per-phase wall time, print every 10 batches.
    # Each phase boundary uses time.monotonic() which is cheap (~30ns/call).
    # Most VkCompute calls block until the GPU finishes (submit + wait), so
    # the timer for each section actually reflects GPU work for that phase.
    if profile:
        phase_total = {
            'wait_prep': 0.0,  # block waiting for background prep thread
            'reset':     0.0,  # zero_grad, reset_hidden, zero accumulators
            'upload':    0.0,  # input_seq + label_seq → GPU
            'forward':   0.0,  # linear_in + gru_seq + linear_out
            'loss':      0.0,  # cross_entropy dispatch + loss download
            'backward':  0.0,  # linear_out.backward + slice slip + gru.backward + linear_in.backward
            'sgd':       0.0,  # weight update
        }
        last_log_time = time.monotonic()

    prep_iter = iter(prep)
    for batch_idx in range(n_batches):
        if profile:
            t0 = time.monotonic()
        try:
            inputs_TBF, labels_TBF = next(prep_iter)
        except StopIteration:
            break
        if profile:
            phase_total['wait_prep'] += time.monotonic() - t0; t0 = time.monotonic()

        B_actual = inputs_TBF.shape[1]
        T = inputs_TBF.shape[0]
        BT = B_actual * T

        # Zero per-layer gradient accumulators and the loss/grad acc.
        model.zero_grad()
        gru.reset_hidden()
        gpu.zero_buffer(loss_acc_buf)
        gpu.zero_buffer(grad_acc_buf)
        if profile:
            phase_total['reset'] += time.monotonic() - t0; t0 = time.monotonic()

        # Single per-batch upload of inputs and labels
        gpu.upload(input_seq_buf, inputs_TBF.ravel())
        gpu.upload(label_seq_buf, labels_TBF.ravel())
        if profile:
            phase_total['upload'] += time.monotonic() - t0; t0 = time.monotonic()

        # --- Forward: 4 dispatches total ---
        linear_in.forward(input_seq_buf, BT, (linear_in.in_features,))
        gru.forward_sequence(linear_in.output_buf, gru_seq_out_buf,
                             B_actual, T)
        linear_out.forward(gru_seq_out_buf, BT, (gru.hidden_size,))
        if profile:
            phase_total['forward'] += time.monotonic() - t0; t0 = time.monotonic()

        # cross_entropy: per-(t,b) loss + grad in one dispatch, with
        # per-class weighting and focal loss to fight majority-class
        # collapse on imbalanced beat-detection labels.
        cross_entropy_dispatch(
            gpu, linear_out.output_buf, label_seq_buf,
            grad_acc_buf, loss_acc_buf,
            class_weights_buf,
            batch_size=BT, n_classes=N_CLASSES,
            target_offset_floats=0, t_inv=1.0,
            focal_gamma=focal_gamma,
        )
        loss_per = gpu.download(loss_acc_buf, np.float32, BT)
        chunk_loss = float(loss_per.mean())
        losses.append(chunk_loss)
        if profile:
            phase_total['loss'] += time.monotonic() - t0; t0 = time.monotonic()

        # --- Backward ---
        linear_out.backward(grad_acc_buf, BT)

        # GRU backward: single dispatch via gru_seq_backward.comp.
        # linear_out._grad_input_buf has shape (T, B, hidden) — every
        # timestep's upstream gradient — and the seq backward shader
        # adds each timestep's contribution to the running recurrent
        # grad. Real per-timestep upstream now, not the last-step-only
        # approximation that was the limit of the per-step backward.
        gru.backward_sequence(linear_out._grad_input_buf, B_actual, T)

        linear_in.backward(gru._grad_input_buf, BT)
        if profile:
            phase_total['backward'] += time.monotonic() - t0; t0 = time.monotonic()

        model.sgd_step(lr)
        if profile:
            phase_total['sgd'] += time.monotonic() - t0

        if (batch_idx + 1) % 10 == 0:
            if profile:
                window_total = sum(phase_total.values())
                window_wall = time.monotonic() - last_log_time
                breakdown = '  '.join(f'{k}={v*1000/10:.0f}ms({v/window_total*100:.0f}%)'
                                       for k, v in phase_total.items())
                print(f"    batch {batch_idx+1}/{n_batches} loss={chunk_loss:.4f}  "
                      f"[wall {window_wall*1000/10:.0f}ms/batch  {breakdown}]")
                phase_total = {k: 0.0 for k in phase_total}
                last_log_time = time.monotonic()
            else:
                print(f"    batch {batch_idx+1}/{n_batches} loss={chunk_loss:.4f}")

        # Mid-epoch validation: runs whenever val_interval seconds have
        # elapsed since the last validation. Wall-time-based instead of
        # batch-count-based so the user gets a checkpoint within a known
        # time budget regardless of throughput swings.
        if mid_epoch_val_active and (time.monotonic() - last_val_time) >= val_interval:
            elapsed_min = (time.monotonic() - last_val_time) / 60.0
            # Sample a small subset of val for speed; end-of-epoch will run
            # the full val set for a high-confidence accuracy reading.
            val_loss, val_acc, val_metrics = validate(
                model, gpu, val_batches, chunk_len, in_buf,
                val_data, cache, max_batches=val_sample_size)
            print(f"    [mid-epoch +{elapsed_min:.1f}min] batch {batch_idx+1}/{n_batches}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                  f"(sampled {val_sample_size}/{len(val_batches)} val batches)")
            # Per-class metrics so we can catch class-imbalance illusions
            # (e.g., 90% acc that's just "always predict non-beat").
            print(format_val_metrics(val_metrics))
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

    Returns (avg_loss, accuracy, metrics_dict). metrics_dict has per-class
    precision/recall/F1 and the 3x3 confusion matrix — beat detection is
    heavily class-imbalanced (~90% non-beat frames), so plain accuracy
    is misleading. A model that always predicts non-beat scores ~0.90
    accuracy but has zero recall on the actually-useful classes.
    """
    if max_batches is not None and max_batches < len(batches):
        import random
        batches = random.sample(batches, max_batches)
    losses = []
    # 3x3 confusion matrix: rows = true class, cols = predicted class.
    # confusion[t][p] = count of frames where true=t and pred=p.
    confusion = np.zeros((3, 3), dtype=np.int64)

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
            # Vectorized confusion-matrix update.
            for tc, pc in zip(true_class, pred_class):
                confusion[tc, pc] += 1

    avg_loss = float(np.mean(losses)) if losses else 0.0
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / max(total, 1))

    # Per-class precision, recall, F1.
    # BeatNet output order is (downbeat, beat, non-beat) — verified
    # empirically: class 2 is the dominant 77-91% support class in
    # every file, which has to be non-beat in any music dataset.
    # The generate_beat_labels.py docstring was reversed.
    class_names = ('downbeat', 'beat', 'non-beat')
    per_class = {}
    for c, name in enumerate(class_names):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            'precision': precision, 'recall': recall, 'f1': f1,
            'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
        }
    # Baseline: what would "always predict the majority class" score?
    class_supports = [int(confusion[c, :].sum()) for c in range(3)]
    majority_class = int(np.argmax(class_supports))
    majority_baseline = class_supports[majority_class] / max(total, 1)
    metrics = {
        'confusion': confusion.tolist(),
        'per_class': per_class,
        'class_names': class_names,
        'majority_baseline': majority_baseline,
        'majority_class': class_names[majority_class],
    }
    return avg_loss, accuracy, metrics


def format_val_metrics(metrics: dict) -> str:
    """One-line + per-class precision/recall summary suitable for logging."""
    cm = metrics['confusion']
    names = metrics['class_names']
    lines = []
    lines.append(f"  baseline (always {metrics['majority_class']}): "
                 f"{metrics['majority_baseline']:.3f}")
    lines.append(f"  {'class':<10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for name in names:
        p = metrics['per_class'][name]
        lines.append(f"  {name:<10} {p['precision']:>10.3f} {p['recall']:>10.3f} "
                     f"{p['f1']:>10.3f} {p['support']:>10}")
    lines.append(f"  confusion (rows=true, cols=pred):")
    lines.append(f"             {' '.join(f'{n:>10}' for n in names)}")
    for i, name in enumerate(names):
        row = ' '.join(f'{cm[i][j]:>10}' for j in range(3))
        lines.append(f"  {name:<10} {row}")
    return '\n'.join(lines)


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
    parser.add_argument('--profile', action='store_true',
                        help='Print per-phase timing breakdown every 10 batches: '
                             'wait_prep | reset | upload | forward | loss | '
                             'backward | sgd. Shows where wall-clock time is '
                             'actually going so you can target the biggest knob.')
    parser.add_argument('--class-weights', type=str, default='auto',
                        help='Per-class loss weights. "auto" (default) computes '
                             'sqrt-inverse-frequency from a 50-file sample of '
                             'the training data (normalized so majority class '
                             '= 1.0). Or pass three comma-separated floats '
                             '"6.8,3.9,1.0". Combats majority-class collapse '
                             'on heavily-imbalanced beat-detection labels.')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Focal-loss exponent. 0 disables (standard '
                             'weighted CE). 2.0 (Lin 2017 default) modulates '
                             'each example by (1-p_true)^gamma, so confident-'
                             'correct predictions barely contribute. Pairs '
                             'with class weights to address both '
                             'under-representation and over-confidence at once.')
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

    # Build model in seq mode — linear layers' output buffers sized for
    # (B * T) flat batch so we can run one dispatch per linear per batch
    # instead of T dispatches each. GRU keeps batch=B and handles the
    # sequence dimension internally via gru_seq_forward.
    gpu = VkCompute()
    model = build_beat_crnn(gpu, input_size=216, hidden_size=args.hidden,
                            n_classes=3, batch_size=args.batch_size,
                            max_seq_len=args.chunk_len, seq_mode=True)

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

    # Pre-allocate GPU buffers once (reused across all batches/epochs).
    # in_buf still holds the current timestep's input frame (uploaded each
    # timestep from CPU); grad_buf is legacy and unused by the new GPU-loss
    # path (kept so the existing layer .backward() helpers stay compatible
    # if grad_buf gets passed elsewhere). The new path uses:
    #   - label_seq_buf: full (B, T, 3) labels uploaded once per batch
    #   - loss_acc_buf:  (B,) per-batch loss accumulator (1/T weighted)
    #   - grad_acc_buf:  (B, 3) per-batch gradient accumulator (1/T weighted)
    # Buffers for the seq-mode pipeline. T = args.chunk_len, B = args.batch_size.
    # Layout convention: (T, B, F) — timestep outer, batch inner. Matches
    # gru_seq_forward's (t*B + b) indexing and lets linear layers treat
    # the whole sequence as a flat (B*T) batch.
    BT = args.batch_size * args.chunk_len
    in_buf = gpu.create_buffer(args.batch_size * 216 * 4)  # legacy, unused in seq path
    grad_buf = gpu.create_buffer(args.batch_size * 3 * 4)  # legacy, unused in seq path
    input_seq_buf = gpu.create_buffer(BT * 216 * 4)             # (T, B, 216)
    label_seq_buf = gpu.create_buffer(BT * 3 * 4)               # (T, B, 3)
    gru_seq_out_buf = gpu.create_buffer(BT * args.hidden * 4)   # (T, B, hidden)
    loss_acc_buf = gpu.create_buffer(BT * 4)                    # (B*T,) loss per (t, b)
    grad_acc_buf = gpu.create_buffer(BT * 3 * 4)                # (T, B, 3) grad on logits
    gru_upstream_buf = gpu.create_buffer(args.batch_size * args.hidden * 4)  # (B, H) — last-step slice
    class_weights_buf = gpu.create_buffer(3 * 4)                # (3,) per-class loss weights

    # Lazy file cache shared across train + validate. Holds the N most
    # recently used files; cap at ~64 files (~640 MB) which is enough
    # working set for typical chunk shuffles without thrashing.
    file_cache = LazyFileCache(max_files=getattr(args, 'file_cache_size', 64))

    # Compute per-class loss weights. With class imbalance like beat
    # detection (~91% non-beat, ~7% beat, ~2% downbeat), uniform weights
    # let the model collapse to "always predict majority class" and
    # never learn the minority signal. sqrt(inv_freq) is the standard
    # softened alternative to pure inverse-frequency (40x, 14x) — it
    # corrects the imbalance without the gradient instability of
    # extreme per-element weights.
    if args.class_weights == 'auto':
        # Sample frame-class distribution from the training data.
        # 50 random files is plenty — beat distribution is roughly
        # stable across the corpus.
        sample_size = min(50, len(train_data))
        sample_files = rng.choice(len(train_data), sample_size, replace=False)
        class_counts = np.zeros(3, dtype=np.int64)
        for idx in sample_files:
            path, _ = train_data[idx]
            with np.load(path) as d:
                lbls = d['labels']  # (T, 3) soft
                argmax = lbls.argmax(axis=1)
                for c in range(3):
                    class_counts[c] += int((argmax == c).sum())
        freq = class_counts / max(class_counts.sum(), 1)
        weights = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
        # Normalize so the most frequent class has weight = 1.
        weights = (weights / weights.min()).astype(np.float32)
        log_msg = (f'auto class weights from {sample_size}-file sample: '
                   f'counts={class_counts.tolist()} freq={freq.round(4).tolist()} '
                   f'weights={weights.round(3).tolist()}')
    else:
        weights = np.array(
            [float(x) for x in args.class_weights.split(',')],
            dtype=np.float32)
        if len(weights) != 3:
            print(f'--class-weights must be 3 comma-separated floats '
                  f'or "auto", got {args.class_weights}', file=sys.stderr)
            sys.exit(1)
        log_msg = f'manual class weights: {weights.tolist()}'
    print(log_msg)
    if args.focal_gamma > 0:
        print(f'focal loss enabled: gamma={args.focal_gamma}')
    gpu.upload(class_weights_buf, weights)

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
                                 label_seq_buf, loss_acc_buf, grad_acc_buf,
                                 input_seq_buf, gru_seq_out_buf, gru_upstream_buf,
                                 class_weights_buf, args.focal_gamma,
                                 val_batches=val_batches, val_data=val_data,
                                 val_interval=args.val_interval,
                                 val_sample_size=args.val_sample_size,
                                 best_val_state=best_val_state,
                                 save_path=args.output, epoch_num=epoch,
                                 profile=args.profile)
        val_loss, val_acc, val_metrics = validate(
            model, gpu, val_batches, args.chunk_len, in_buf,
            val_data, file_cache)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_acc={val_acc:.3f}")
        # End-of-epoch full-val pass — print per-class breakdown for an
        # honest read of how the model is actually performing relative
        # to the majority-class baseline.
        print(format_val_metrics(val_metrics))

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
