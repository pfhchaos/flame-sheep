"""Run registered evals and record results."""
from __future__ import annotations

from typing import Iterable

from .registry import REGISTRY, EvalSpec
from .storage import (RunEntry, append_run, git_snapshot, now_iso,
                       latest_metric_values)


def _flatten_metrics(eval_name: str, sub_metrics: dict) -> dict[str, float]:
    """Convert {'baseline.exact_pct': 21} from an eval into
    {'tempo.osu.baseline.exact_pct': 21} for the run entry."""
    return {f'{eval_name}.{k}': v for k, v in sub_metrics.items()}


def run(notes: str = '', only: Iterable[str] | None = None,
        skip_slow: bool = False, dry_run: bool = False
        ) -> RunEntry:
    """Run all registered evals (or the subset in `only`).

    `notes` is free-form text recorded with the run (e.g. what was
    changed since the last run). `skip_slow` skips evals marked
    `slow=True` in their registration. `dry_run` runs everything but
    doesn't append to results.jsonl.

    Meta-evals (registry.meta_eval) always run after regular evals so
    they see today's freshly computed values, not just stale history.
    """
    selected: list[EvalSpec] = []
    for name, spec in REGISTRY.items():
        if only is not None and name not in only:
            continue
        if skip_slow and spec.slow:
            continue
        selected.append(spec)

    if not selected:
        raise SystemExit('No evals selected. Check --only filter or registry.')

    regulars = [s for s in selected if not s.is_meta]
    metas = [s for s in selected if s.is_meta]

    metrics: dict[str, float] = {}
    contexts: dict[str, dict] = {}

    for spec in regulars:
        print(f'>>> {spec.name}: {spec.description or "(no description)"}')
        result = spec.fn()
        if isinstance(result, dict) and '__context__' in result:
            ctx = result.pop('__context__')
            contexts[spec.name] = ctx
        metrics.update(_flatten_metrics(spec.name, result))
        print(f'    {len(result)} metric(s) recorded')

    if metas:
        # Build the latest-metric view meta-evals see: history ∪ today,
        # with today taking precedence.
        latest = latest_metric_values()
        latest.update(metrics)
        for spec in metas:
            print(f'>>> {spec.name} [meta]: {spec.description or "(no description)"}')
            result = spec.fn(latest)
            if isinstance(result, dict) and '__context__' in result:
                ctx = result.pop('__context__')
                contexts[spec.name] = ctx
            metrics.update(_flatten_metrics(spec.name, result))
            print(f'    {len(result)} metric(s) recorded')

    sha, dirty, branch = git_snapshot()
    entry = RunEntry(
        timestamp=now_iso(),
        commit=sha,
        commit_dirty=dirty,
        branch=branch,
        notes=notes,
        metrics=metrics,
        contexts=contexts,
    )
    if not dry_run:
        append_run(entry)
        print(f'\nRecorded run with {len(metrics)} metric(s).')
    else:
        print(f'\nDRY RUN — would record {len(metrics)} metric(s).')
    return entry


def run_report(last_n: int = 5) -> str:
    """Return a markdown table of the last N runs, columns = metrics.

    Side-effect-free — pure read of results.jsonl.
    """
    from .storage import read_runs
    runs = read_runs()
    if not runs:
        return 'No eval runs recorded yet.'
    runs = runs[-last_n:]

    # Collect all metric names that appeared in ANY of the displayed runs.
    all_metrics: set[str] = set()
    for r in runs:
        all_metrics.update(r.metrics.keys())
    metric_names = sorted(all_metrics)

    # Header row: timestamp + commit + notes preview
    lines = []
    header = ['metric'] + [
        f'{r.commit}{"*" if r.commit_dirty else ""} ({r.timestamp[5:10]})'
        for r in runs
    ]
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('| ' + ' | '.join('---' for _ in header) + ' |')

    # Notes row, one per column under the header
    notes_row = ['_notes_'] + [
        (r.notes[:30] + '…') if len(r.notes) > 30 else (r.notes or '_(none)_')
        for r in runs
    ]
    lines.append('| ' + ' | '.join(notes_row) + ' |')

    for m in metric_names:
        cells = [m]
        for r in runs:
            v = r.metrics.get(m)
            cells.append('—' if v is None else f'{v:.3f}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
