"""LoopPlayer sequencing contract -- the `_start_loop` resume position.

REGRESSION (shader-warming.md step 4a). `_start_loop` must resume ONE PAST the
entry genome's position IN THE PLAYBACK SEQUENCE, for every structure, so the
first `next_in_sequence()` after a loop load yields the entry's *successor* --
never the entry genome again, and never a wrong-structure genome.

Two pre-existing bugs this pins:
  * dead swap (cyclic / palindrome): the old code advanced the generator `start`
    times and set `_loop_step = start`, so the first yield re-yielded the entry
    genome -- one wasted morph G->G at every loop entry.
  * index/step conflation (rondo, palindrome back half): `start` is a GENOME
    INDEX but was used as a SEQUENCE STEP COUNT. They coincide only for cyclic
    and palindrome's forward half; for rondo the resume point was simply wrong.

Both were invisible to the step-3 PLAN A/B check -- `peek_next_in_sequence` and
the worker planner both call `follow_after`, so they agreed with each OTHER on
the same wrong genome; nothing compared against the intended sequence. Step 4
makes `follow_after` authoritative, so the contract is pinned here instead.

MUTATION GUARD: revert `_start_loop` to `for _ in range(start)` / `_loop_step =
start` and the cyclic+palindrome cases re-yield the entry (first assertion
reddens); rondo yields home instead of the successor (reddens).

NOT tested here (notated): the GL compile of these shaders (GPU-gated, on-box);
branch policy (`should_exit_loop`) and the readiness gate (own tests).
"""
import numpy as np
import pytest

from flame_sheep.axes._loop_player import LoopPlayer
from flame_sheep.loops.sequence import loop_sequence, cycle_length
from tests.scheduler.fake_library import FakeLibrary

STRUCTURES = ["cyclic", "palindrome", "rondo"]


class _G:
    """Minimal genome: `_start_loop` touches only `.db_id` + `.distance()`;
    `next_in_sequence` just yields the object."""

    def __init__(self, db_id):
        self.db_id = db_id
        self.transforms = []  # _describe_genome (debug log) iterates this

    def distance(self, other):
        return 0.0


def _player(structure, n):
    lib = FakeLibrary()
    ids = list(range(100, 100 + n))
    for gid in ids:
        lib.add_genome(gid, _G(gid))
    lib.add_loop(1, ids, structure=structure)
    player = LoopPlayer(lib, lambda: _G(-1), _G(999),
                        np.random.default_rng(0))
    return player, ids


def _pure_order(n, structure):
    """The genome-INDEX playback order for one cycle, straight from the pure
    sequence layer -- the oracle, independent of `_start_loop`."""
    seq = loop_sequence(list(range(n)), structure)
    return [next(seq) for _ in range(cycle_length(n, structure))]


def _expected_successor_id(ids, structure, entry_index):
    order = _pure_order(len(ids), structure)
    pos = order.index(entry_index)
    return ids[order[(pos + 1) % len(order)]]


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("entry_index", [0, 2, 4])
def test_first_follow_is_entry_successor(structure, entry_index):
    player, ids = _player(structure, 5)
    entry_id = ids[entry_index]
    player._start_loop(1, _G(999), entry_genome_id=entry_id)

    expected = _expected_successor_id(ids, structure, entry_index)

    # the arrival peek must predict the first move (the A/B invariant), ...
    assert player.peek_next_in_sequence().db_id == expected
    # ... and the actual first move must be the entry's SUCCESSOR, not the entry.
    first = player.next_in_sequence()
    assert first.db_id != entry_id, "re-yielded the entry genome (dead swap)"
    assert first.db_id == expected


@pytest.mark.parametrize("structure", STRUCTURES)
def test_full_cycle_matches_pure_sequence_from_successor(structure):
    """Beyond the first step: the whole cycle after entry must track the pure
    sequence (guards a wrap / drift regression, not just the off-by-one)."""
    player, ids = _player(structure, 5)
    entry_index = 2
    player._start_loop(1, _G(999), entry_genome_id=ids[entry_index])

    order = _pure_order(len(ids), structure)
    start_pos = (order.index(entry_index) + 1) % len(order)
    steps = len(order) + 2  # past the wrap
    expected = [ids[order[(start_pos + k) % len(order)]] for k in range(steps)]
    actual = [player.next_in_sequence().db_id for _ in range(steps)]
    assert actual == expected
