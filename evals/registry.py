"""Decorator-based registration of evaluation functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class EvalSpec:
    """One registered eval — metadata + the callable."""
    name: str                       # dot-namespaced, e.g. 'tempo.osu'
    description: str                # one-liner for --list output
    fn: Callable[..., dict]         # returns dict[str, float] of metric → value
    slow: bool = False              # True if eval takes > ~1 min
    requires: tuple[str, ...] = ()  # corpus/data dependencies, for --list and graceful skip
    is_meta: bool = False           # True for cross-eval consistency checks
                                    # whose `fn` takes `latest_metrics`


REGISTRY: dict[str, EvalSpec] = {}


def register(name: str, *, description: str = '',
             slow: bool = False, requires: tuple[str, ...] = ()):
    """Decorator: register an eval function under the given name.

    The function must return a `dict[str, float]` where keys are
    sub-metric names. The full metric name in results.jsonl will be
    `<name>.<sub-metric>` (e.g. `tempo.osu.baseline.exact_pct`).
    """
    def decorate(fn: Callable[..., dict]) -> Callable[..., dict]:
        if name in REGISTRY:
            raise ValueError(f'Eval {name!r} already registered')
        REGISTRY[name] = EvalSpec(name=name, description=description,
                                    fn=fn, slow=slow, requires=requires)
        return fn
    return decorate


def meta_eval(name: str, *, description: str = '',
              requires: tuple[str, ...] = ()):
    """Decorator: register a meta-eval that derives metrics from the
    latest values of other evals — no subprocess, no corpus access.

    The decorated function signature is
        fn(latest_metrics: dict[str, float]) -> dict[str, float]
    where `latest_metrics` is a flat dict keyed by the full metric
    name (e.g. `beat.osu.percentile.pooled.f1@70ms`). It combines:
      * the most recent value for each metric across `results.jsonl`, and
      * any metrics already computed in the CURRENT scorecard run
        (those take precedence over historical rows).

    Meta-evals run AFTER all regular evals in `runner.run`, so a
    co-scheduled `beat.osu.percentile` + `beat.gtzan.percentile` pair
    will feed today's consistency check with today's freshly computed
    numbers — not stale rows from prior commits.

    Meta-evals are not marked slow — they don't do real work.
    """
    def decorate(fn: Callable[..., dict]) -> Callable[..., dict]:
        if name in REGISTRY:
            raise ValueError(f'Eval {name!r} already registered')
        REGISTRY[name] = EvalSpec(name=name, description=description,
                                    fn=fn, slow=False, requires=requires,
                                    is_meta=True)
        return fn
    return decorate
