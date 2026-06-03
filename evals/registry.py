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
