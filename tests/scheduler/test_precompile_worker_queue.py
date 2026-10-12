"""Step-2 shader-warming redesign — the scheduling that MOVED into the worker.

Step 2's core change: the ``queue.PriorityQueue`` + ``(priority, counter,
tuple)`` tie-break + ``_seen``/``is_warm`` dedup left ``PrecompileDriver`` and
now live in ``precompile_worker_gl`` (``CompileQueue``), and the worker's
stdin routing was factored out of ``main()`` into the GL-free ``dispatch_line``
(closing step 1's notated "routing is GL-gated" skip). This file specs + tests
both — the durable, high-value core of the step.

All tests here are GPU-FREE: ``CompileQueue`` and ``dispatch_line`` touch no GL
context (that's the whole point of factoring them out), so they run in the fast
suite. ``is_warm`` (a filesystem marker stat) is monkeypatched so the dedup
tests are deterministic regardless of what's in the real Mesa warm-marker dir.
"""
from __future__ import annotations

import queue as _queue

import pytest

from flame_sheep.scheduler.precompile_worker_gl import (
    CompileQueue, dispatch_line, _FLOOD_TASK, _FLOOD_RESULT_PRIORITY,
)


@pytest.fixture
def no_warm(monkeypatch):
    """Force pipeline_warm.is_warm -> False so the is_warm dedup layer never
    skips (so the _seen / priority behavior under test is isolated). Patched at
    the source module, which CompileQueue.enqueue imports by attribute."""
    from flame_sheep.rendering.vk import pipeline_warm
    monkeypatch.setattr(pipeline_warm, 'is_warm', lambda *a, **k: False)
    return pipeline_warm


def _drain(cq: CompileQueue) -> list:
    """Pop everything, return the payloads in pop order."""
    out = []
    while True:
        try:
            _p, _s, payload = cq.get(timeout=0)
        except _queue.Empty:
            break
        out.append(payload)
    return out


# ================================================================
# 1. CompileQueue — dedup  [home: precompile_worker_gl, was precompile_driver]
# ----------------------------------------------------------------
# BEHAVIOR SPEC. enqueue(tuples, priority) MUST:
#   (a) push each NEW tuple once and return the count actually added;
#   (b) drop a tuple already seen this run (the _seen layer) — a re-enqueue of
#       an already-queued key adds nothing;
#   (c) drop a tuple whose Mesa disk cache is already warm (the is_warm layer),
#       counting it as not-added.
# These are the exact two dedup layers PrecompileDriver.enqueue owned in step
# 1, relocated verbatim. Decomposition: unit (CompileQueue + patched is_warm).
# ================================================================

def test_enqueue_dedups_seen(no_warm):
    """(a)(b) MUTATION GUARD for dedup: break the _seen check and the second
    enqueue adds a duplicate -> both asserts go red."""
    cq = CompileQueue()
    t = (2, 0, frozenset({0, 1}))

    assert cq.enqueue([t], priority=0) == 1
    assert cq.enqueue([t], priority=0) == 0          # already seen -> 0 added
    # ...also deduped WITHIN a single batch.
    assert cq.enqueue([(1, 0, frozenset({9})),
                       (1, 0, frozenset({9}))], priority=0) == 1

    payloads = _drain(cq)
    assert payloads == [t, (1, 0, frozenset({9}))]   # exactly one of each


def test_enqueue_skips_is_warm(monkeypatch):
    """(c) MUTATION GUARD for the is_warm layer: a tuple reported warm is not
    enqueued. Remove the is_warm skip and the warm tuple leaks onto the queue."""
    from flame_sheep.rendering.vk import pipeline_warm
    warm_keep = frozenset({7, 8})
    monkeypatch.setattr(
        pipeline_warm, 'is_warm',
        lambda n, hf, kv: kv == warm_keep)   # only this key is "warm"

    cq = CompileQueue()
    cold = (1, 0, frozenset({1}))
    warm = (3, 0, warm_keep)
    added = cq.enqueue([cold, warm], priority=0)

    assert added == 1
    assert _drain(cq) == [cold]                       # warm one skipped


# ================================================================
# 2. CompileQueue — priority + FIFO tie-break
# ----------------------------------------------------------------
# BEHAVIOR SPEC. get() MUST pop in ascending priority (lower = more urgent),
# and ties MUST break FIFO via the monotonic counter — NEVER by comparing the
# tuple payloads (which would TypeError on frozensets). This is the reactive
# -100 cold-miss overtaking the MED(0) loop members overtaking the LOW(10)
# flood, exactly as in step 1 — now resolved inside the worker.
# Decomposition: unit (CompileQueue + patched is_warm).
# ================================================================

def test_priority_then_fifo_order(no_warm):
    """MUTATION GUARD for ordering: if the queue key drops priority (or puts
    the counter first), the pop order changes and this goes red."""
    cq = CompileQueue()
    low = (0, 0, frozenset({10}))      # LOW 10
    med_a = (0, 0, frozenset({1}))     # MED 0, enqueued first
    med_b = (0, 0, frozenset({2}))     # MED 0, enqueued second
    urgent = (0, 0, frozenset({99}))   # -100

    cq.enqueue([low], priority=10)
    cq.enqueue([med_a], priority=0)
    cq.enqueue([med_b], priority=0)
    cq.enqueue([urgent], priority=-100)

    # urgent first; then MED in FIFO order; then LOW last.
    assert _drain(cq) == [urgent, med_a, med_b, low]


def test_mixed_priority_same_as_distinct(no_warm):
    """The counter makes the key totally ordered even when two payloads are
    equal-priority frozensets — pop must not raise (would TypeError if it fell
    through to comparing frozensets)."""
    cq = CompileQueue()
    cq.enqueue([(0, 0, frozenset({1, 2}))], priority=0)
    cq.enqueue([(0, 0, frozenset({3, 4}))], priority=0)
    # No exception, FIFO order.
    assert _drain(cq) == [(0, 0, frozenset({1, 2})),
                          (0, 0, frozenset({3, 4}))]


# ================================================================
# 3. CompileQueue — flood-compute task scheduling
# ----------------------------------------------------------------
# BEHAVIOR SPEC. enqueue_flood(spec) MUST schedule ONE flood-COMPUTE task
# (payload (_FLOOD_TASK, spec)) at the spec's priority (default 0), and MUST
# NOT dedup (a flood is a one-shot computation; its product tuples dedup when
# they re-enter via enqueue). The task sorts correctly against compile tasks.
# Decomposition: unit (CompileQueue).
# ================================================================

def test_enqueue_flood_schedules_compute_task(no_warm):
    cq = CompileQueue()
    spec = {'seed_ids': [1, 2], 'hops': 2, 'fan_per_node': 5, 'priority': 0}
    cq.enqueue_flood(spec)
    cq.enqueue_flood(spec)                     # NOT deduped -> two tasks

    payloads = _drain(cq)
    assert payloads == [(_FLOOD_TASK, spec), (_FLOOD_TASK, spec)]


def test_flood_task_priority_orders_against_compiles(no_warm):
    cq = CompileQueue()
    cq.enqueue([(0, 0, frozenset({1}))], priority=-100)   # urgent compile
    cq.enqueue_flood({'seed_ids': [9], 'priority': _FLOOD_RESULT_PRIORITY})  # LOW
    cq.enqueue([(0, 0, frozenset({2}))], priority=0)      # MED compile

    payloads = _drain(cq)
    assert payloads[0] == (0, 0, frozenset({1}))          # urgent first
    assert payloads[1] == (0, 0, frozenset({2}))          # MED next
    assert payloads[2][0] == _FLOOD_TASK                  # LOW flood last


# ================================================================
# 4. dispatch_line — GL-free stdin routing
# ----------------------------------------------------------------
# BEHAVIOR SPEC. dispatch_line(line, cq) classifies ONE request line and
# applies it to the queue, returning a branch tag:
#   {"flood": {...}}                  -> 'flood',   cq.enqueue_flood(spec)
#   {"enqueue": {priority, tuples}}   -> 'enqueue', cq.enqueue(tuples, priority)
#                                        with tuples deserialized from the wire
#   ''                                -> 'empty'    (no queue mutation)
#   non-JSON                          -> 'bad-json' (no queue mutation)
#   valid JSON but not a dict / no
#     recognized key (any control)    -> 'unknown'  (no queue mutation)
# This is step 1's notated GL-gated routing skip, now closed: the routing is a
# standalone GL-free function. Decomposition: unit with a recording fake queue
# (routing is about WHICH method is called with WHAT, independent of queue
# internals). Mutation-provable: swap the flood/enqueue branches and the two
# routing asserts flip.
# ================================================================

class _RecordingQueue:
    def __init__(self):
        self.enqueued: list[tuple[list, int]] = []
        self.floods: list[dict] = []

    def enqueue(self, tuples, priority=0):
        self.enqueued.append((list(tuples), priority))
        return len(list(tuples))

    def enqueue_flood(self, spec):
        self.floods.append(spec)


def test_dispatch_routes_flood():
    cq = _RecordingQueue()
    spec = {'seed_ids': [1, 2], 'hops': 2, 'fan_per_node': 5, 'priority': 0}
    tag = dispatch_line('{"flood": %s}' % _json(spec), cq)
    assert tag == 'flood'
    assert cq.floods == [spec]
    assert cq.enqueued == []            # routed to flood, NOT enqueue


def test_dispatch_routes_enqueue_and_deserializes():
    cq = _RecordingQueue()
    line = ('{"enqueue": {"priority": -100, '
            '"tuples": [[3, 1, [2, 1]], [0, 0, [5]]]}}')
    tag = dispatch_line(line, cq)
    assert tag == 'enqueue'
    assert cq.floods == []
    # Wire lists -> canonical tuples (has_final 0/1, vars -> frozenset), and
    # the priority is carried through.
    (tuples, priority), = cq.enqueued
    assert priority == -100
    assert tuples == [(3, 1, frozenset({1, 2})), (0, 0, frozenset({5}))]


def test_dispatch_ignores_noise():
    cq = _RecordingQueue()
    assert dispatch_line('', cq) == 'empty'
    assert dispatch_line('   ', cq) == 'empty'
    assert dispatch_line('not json at all', cq) == 'bad-json'
    assert dispatch_line('[1, 2, 3]', cq) == 'unknown'       # JSON, not a dict
    assert dispatch_line('{"wat": 1}', cq) == 'unknown'      # unknown control
    # None of the noise touched the queue.
    assert cq.enqueued == [] and cq.floods == []


def _json(obj):
    import json
    return json.dumps(obj)
