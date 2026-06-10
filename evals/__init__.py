"""Unified evaluation framework.

A small, opinionated framework for running flame-sheep's audio/visual
evals as a single command and tracking results over time alongside the
git history.

Design rationale:

  * One JSONL file (`evals/results.jsonl`) is the source of truth for
    historical runs. Append-only; never edited in place. Checked into
    git so the eval history evolves with the code.

  * Each eval is a registered function returning `dict[str, float]`
    of metric_name → value. Names are dot-namespaced
    (`tempo.osu.baseline.exact_pct`) so the report can group by
    prefix.

  * One run captures: timestamp, git commit + dirty flag, branch,
    free-form notes, and ALL registered metrics that were requested.
    Partial runs (with `--metrics tempo.osu`) are recorded as such
    so the report doesn't conflate "this metric was 0%" with "this
    metric wasn't measured."

  * No subprocess parsing. Existing eval scripts (eval_tempo_reconcile,
    eval_beat_detection, ...) get thin wrapper functions in this
    package that call them programmatically and return structured
    results. The CLI scripts stay usable on their own.

Per `feedback_signal_balance_audits.md` and `feedback_val_acc_bias.md`:
single-number wins on one metric routinely mask regressions on
another. The point of the unified eval is to make the broader
scorecard visible every time.
"""
from __future__ import annotations

from .registry import register, meta_eval, REGISTRY, EvalSpec
from .storage import (append_run, read_runs, latest_metric_values,
                       RunEntry, RESULTS_PATH)
from .runner import run, run_report

__all__ = [
    'register', 'meta_eval', 'REGISTRY', 'EvalSpec',
    'append_run', 'read_runs', 'latest_metric_values',
    'RunEntry', 'RESULTS_PATH',
    'run', 'run_report',
]
