"""Tests for the in-place BeatNet channel-order fix script.

Recovery formula correctness is the critical property: applying the
fix to a packed/npz file should leave it byte-equivalent to what
would have been produced by a corrected `pack_beat_labels.py` from
the same source BeatNet output.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_FIX_SCRIPT = (Path(__file__).resolve().parents[2]
               / 'tools' / 'fix_beat_label_channels.py')


def _load_fix_module():
    import sys
    name = 'fix_beat_label_channels'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _FIX_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


def _make_synthetic_bn(T: int, seed: int = 0) -> np.ndarray:
    """Random BeatNet-shaped (T, 3) softmax output: rows sum to 1."""
    rng = np.random.default_rng(seed)
    raw = rng.random((T, 3)).astype(np.float32)
    raw = raw / raw.sum(axis=1, keepdims=True)
    return raw


def _build_old_labels_hier(bn: np.ndarray) -> np.ndarray:
    """Reproduce the OLD (buggy) build_hierarchical_labels output:
    col 0 = bn[:, 0]  (was wrongly attributed to downbeat)
    col 1 = bn[:, 0] + bn[:, 1]  (correct any-beat)
    col 2 = max(any_beat, 0)  (any-onset; no separate onset here)
    """
    any_beat = np.clip(bn[:, 0] + bn[:, 1], 0.0, 1.0)
    return np.stack([bn[:, 0], any_beat, any_beat], axis=1).astype(np.float32)


def _build_new_labels_hier(bn: np.ndarray) -> np.ndarray:
    """Reproduce the FIXED build_hierarchical_labels output:
    col 0 = bn[:, 1]  (true downbeat activation)
    cols 1 and 2 unchanged.
    """
    any_beat = np.clip(bn[:, 0] + bn[:, 1], 0.0, 1.0)
    return np.stack([bn[:, 1], any_beat, any_beat], axis=1).astype(np.float32)


# --- Recovery math (the load-bearing correctness property) ---

def test_recovery_matches_corrected_build():
    """In-place subtraction must produce the same labels_hier you'd
    get from regenerating with the corrected build function."""
    bn = _make_synthetic_bn(T=512)
    old = _build_old_labels_hier(bn)
    expected = _build_new_labels_hier(bn)

    fixed = old.copy()
    fixed[:, 0] = np.clip(fixed[:, 1] - fixed[:, 0], 0.0, 1.0)

    np.testing.assert_allclose(fixed[:, 0], expected[:, 0], atol=1e-6)
    # cols 1 and 2 must be unchanged.
    np.testing.assert_array_equal(fixed[:, 1], expected[:, 1])
    np.testing.assert_array_equal(fixed[:, 2], expected[:, 2])


def test_recovery_idempotent_when_sentinel_present(tmp_path):
    """Re-running the script after a sentinel exists is a no-op."""
    mod = _load_fix_module()
    # Build a one-file packed dir.
    arr = np.zeros((10, 220), dtype=np.float32)
    bn = _make_synthetic_bn(T=10, seed=1)
    arr[:, 216] = _build_old_labels_hier(bn)[:, 0]
    arr[:, 217] = _build_old_labels_hier(bn)[:, 1]
    arr[:, 218] = _build_old_labels_hier(bn)[:, 2]
    np.save(tmp_path / 'sample.npy', arr)

    # First run fixes; second run sees sentinel and skips.
    s1 = mod.fix_packed_dir(tmp_path, dry_run=False)
    assert s1.fixed == 1
    assert (tmp_path / mod.PACKED_SENTINEL).exists()

    s2 = mod.fix_packed_dir(tmp_path, dry_run=False)
    assert s2.fixed == 0
    assert s2.inspected == 0  # short-circuited entirely


def test_packed_fix_writes_correct_values(tmp_path):
    """After fix, col 216 of the packed file holds bn[:, 1]."""
    mod = _load_fix_module()
    bn = _make_synthetic_bn(T=20, seed=2)
    old_lh = _build_old_labels_hier(bn)
    arr = np.zeros((20, 219), dtype=np.float32)
    arr[:, 216:219] = old_lh
    np.save(tmp_path / 'sample.npy', arr)

    s = mod.fix_packed_dir(tmp_path, dry_run=False)
    assert s.fixed == 1
    fixed = np.load(tmp_path / 'sample.npy')

    np.testing.assert_allclose(fixed[:, 216], bn[:, 1], atol=1e-6)
    np.testing.assert_allclose(fixed[:, 217], old_lh[:, 1])  # unchanged


def test_dry_run_does_not_modify(tmp_path):
    mod = _load_fix_module()
    bn = _make_synthetic_bn(T=10)
    arr = np.zeros((10, 219), dtype=np.float32)
    arr[:, 216:219] = _build_old_labels_hier(bn)
    np.save(tmp_path / 'sample.npy', arr)
    on_disk_before = np.load(tmp_path / 'sample.npy').copy()

    s = mod.fix_packed_dir(tmp_path, dry_run=True)
    assert s.fixed == 1
    on_disk_after = np.load(tmp_path / 'sample.npy')
    np.testing.assert_array_equal(on_disk_before, on_disk_after)
    assert not (tmp_path / mod.PACKED_SENTINEL).exists()


def test_npz_fix_writes_correct_values_and_stamps(tmp_path):
    mod = _load_fix_module()
    bn = _make_synthetic_bn(T=15, seed=3)
    old_lh = _build_old_labels_hier(bn)
    # Build a label-shaped npz with the auxiliary arrays the loader
    # expects so we round-trip them too.
    np.savez_compressed(
        tmp_path / 'track.npz',
        spectrum=np.zeros((15, 108), dtype=np.float32),
        diff=np.zeros((15, 108), dtype=np.float32),
        labels=bn,
        labels_hier=old_lh,
        onsets=np.zeros(15, dtype=np.float32),
    )

    s = mod.fix_npz_dir(tmp_path, dry_run=False)
    assert s.fixed == 1

    d = np.load(tmp_path / 'track.npz')
    np.testing.assert_allclose(d['labels_hier'][:, 0], bn[:, 1], atol=1e-6)
    np.testing.assert_array_equal(d['labels_hier'][:, 1], old_lh[:, 1])
    # Stamp recorded.
    assert mod.NPZ_FIX_KEY in d.files
    assert bool(d[mod.NPZ_FIX_KEY]) is True

    # Re-running skips the already-fixed file.
    s2 = mod.fix_npz_dir(tmp_path, dry_run=False)
    assert s2.fixed == 0
    assert s2.already_fixed == 1


def test_subset_inequality_holds_after_fix():
    """The fix-script's spot-check property: db_col_sum <= beat_col_sum.
    Locks in the inequality that ground-truth alignment leans on."""
    bn = _make_synthetic_bn(T=1000)
    fixed = _build_new_labels_hier(bn)
    assert fixed[:, 0].sum() <= fixed[:, 1].sum() + 1e-3
