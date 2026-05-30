"""Drift-from-baseline regression tests for flame_sheep_audio behavioral evals.

One test per registered eval category. Each loads the baseline JSON,
runs the eval, and asserts every metric is within its declared
tolerance. When you intentionally change the algorithm:

    tools/refresh_eval_baselines.py <category> --note "what changed"

then commit the regenerated baseline JSON alongside the algorithm
change. The diff in the baseline IS the documentation of the behavior
shift.

If the regression fails unexpectedly: read the drift report (printed
on failure), figure out whether the drift is a real regression or
an acceptable shift, and either fix the algorithm or refresh the
baseline. Never refresh without understanding the change.
"""
from __future__ import annotations

import pytest

from flame_sheep_audio.eval.common import (
    REGISTRY, load_baseline, compare,
)


@pytest.mark.parametrize(
    'category',
    REGISTRY,
    ids=[c.name for c in REGISTRY],
)
def test_eval_within_tolerance(category):
    """Run the category's eval and assert no metric has drifted from
    baseline beyond the category-declared tolerance."""
    baseline = load_baseline(category.name)
    if baseline is None:
        pytest.skip(
            f'No baseline for {category.name!r}. Run '
            f'tools/refresh_eval_baselines.py {category.name} '
            f'to capture one.')

    current_metrics, _per_stim = category.run()
    report = compare(category.name, current_metrics, baseline,
                     category.tolerances)

    if not report.ok:
        pytest.fail(report.summary())
