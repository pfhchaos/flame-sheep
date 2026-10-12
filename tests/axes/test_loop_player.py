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


def run_until_exit(player) -> int:
    """Simulate ONE loop lifetime: step the loaded loop via the real
    ``next_in_sequence`` path until ``should_exit_loop`` fires, and return the
    exit POSITION (loop_step). Opens the MIN_EXIT_CYCLES gate first.

    The reusable "simulate a loop" primitive — use this instead of re-inlining the
    step loop. (Promote to a shared tests module when a second file needs it.)"""
    player._cycle_count = player.MIN_EXIT_CYCLES          # open the exit gate
    bound = player._loop_cycle_len * (player.MIN_EXIT_CYCLES + 3)
    for _ in range(bound):
        if player.should_exit_loop():
            return player.loop_step
        player.next_in_sequence()
    raise AssertionError("should_exit_loop never fired within the step bound")


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


def test_advance_step_mirrors_next_in_sequence_counter():
    """advance_step (step-4 S3: worker supplies the FOLLOW genome, render keeps
    the counter) moves loop_step/cycle exactly like next_in_sequence, minus the
    yield — so the two paths stay consistent."""
    p1, ids = _player('cyclic', 5)
    p1._start_loop(1, _G(999), entry_genome_id=ids[0])
    p2, _ = _player('cyclic', 5)
    p2._start_loop(1, _G(999), entry_genome_id=ids[0])
    for _ in range(7):                       # past a wrap (cycle_len 5)
        p1.next_in_sequence()
        p2.advance_step()
    assert p2.loop_step == p1.loop_step
    assert p2.cycle_count == p1.cycle_count


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


# ----------------------------------------------------------------
# should_exit_loop — PICK-THE-EXIT-NODE (uniform exit POSITION)
#
# The exit model is: _start_loop draws one exit sequence-position _exit_at in
# [0, cycle_len); after MIN_EXIT_CYCLES, should_exit_loop fires the FIRST time
# loop_step lands on it — a bounded hold (MIN..+1 cycles) and a uniform exit
# POSITION, replacing the old `pos / n` ramp that clustered exits late.
#
# Uniformity is guaranteed BY CONSTRUCTION (integers(0, cycle_len)) — not tested
# statistically. Instead we pin, deterministically, that the exit can land at ANY
# position (0 / middle / last) and fires at exactly that position, nowhere else.
#
# MUTATION GUARD: if should_exit_loop ignored _exit_at and returned True for
# every position past MIN_EXIT_CYCLES, test_exit_fires_only_at_exit_at reddens
# (it would fire at all positions, not just _exit_at). Revert the pos/n ramp and
# the gate tests go non-deterministic / red.
# ----------------------------------------------------------------

MIN = LoopPlayer.MIN_EXIT_CYCLES


@pytest.mark.parametrize("exit_at", [0, 2, 4])
def test_exit_fires_only_at_exit_at(exit_at):
    """After MIN_EXIT_CYCLES, should_exit_loop is True iff loop_step == _exit_at
    — proving the exit can land ANYWHERE (start / middle / last), not clustered
    late. exit_at is set directly, so this is fully deterministic."""
    player, ids = _player('cyclic', 5)      # cyclic 5 -> cycle_len 5
    player._start_loop(1, _G(999), entry_genome_id=ids[0])
    player._cycle_count = MIN
    player._exit_at = exit_at
    fired = []
    for step in range(player._loop_cycle_len):
        player._loop_step = step
        if player.should_exit_loop():
            fired.append(step)
    assert fired == [exit_at]


def test_no_exit_before_min_cycles():
    """The MIN_EXIT_CYCLES gate holds even sitting on the exit node."""
    player, ids = _player('cyclic', 5)
    player._start_loop(1, _G(999), entry_genome_id=ids[0])
    player._cycle_count = MIN - 1
    player._loop_step = player._exit_at         # on the exit node, but too early
    assert player.should_exit_loop() is False


def test_exit_at_wrap_position_zero():
    """_exit_at == 0 (the wrap point) is a valid exit position: fires at step 0,
    not elsewhere."""
    player, ids = _player('cyclic', 5)
    player._start_loop(1, _G(999), entry_genome_id=ids[0])
    player._cycle_count = MIN
    player._exit_at = 0
    player._loop_step = 0
    assert player.should_exit_loop() is True
    player._loop_step = 1
    assert player.should_exit_loop() is False


def test_exit_at_drawn_in_range_on_every_entry():
    """_start_loop draws _exit_at within [0, cycle_len) on each entry (the source
    of the uniform exit position). Deterministic bound check, not a distribution
    test."""
    player, ids = _player('cyclic', 5)
    for _ in range(50):
        player._start_loop(1, _G(999), entry_genome_id=ids[0])
        assert 0 <= player._exit_at < player._loop_cycle_len


def test_single_genome_loop_exits_after_min_cycles():
    """Edge: cycle_len 1 (n<=1 guard). Exits unconditionally once past the gate,
    independent of _exit_at / loop_step."""
    player, ids = _player('cyclic', 1)          # cycle_len 1
    player._start_loop(1, _G(999), entry_genome_id=ids[0])
    assert player._loop_cycle_len == 1
    player._cycle_count = MIN - 1
    assert player.should_exit_loop() is False   # gate still holds
    player._cycle_count = MIN
    assert player.should_exit_loop() is True


def test_empty_loop_never_exits():
    """No loop loaded -> never exits (empty-loop guard)."""
    player, ids = _player('cyclic', 5)
    player._loop_genomes = []
    player._cycle_count = MIN + 5
    assert player.should_exit_loop() is False


def test_realized_exit_positions_are_uniform():
    """PRIMARY behavioral invariant: across many loop lifetimes — each drawing
    its OWN _exit_at and stepping the REAL sequence (next_in_sequence) until
    should_exit_loop fires — the realized exit POSITIONS are ~uniform over
    [0, cycle_len). This is the spec pick-the-exit-node restores and the old
    `exit_prob = pos/n` ramp violated (it clustered exits LATE, starving low
    positions). Implementation-agnostic: it pins the distribution, not _exit_at.

    Seeded rng -> fully deterministic, NOT flaky.

    SELF-VALIDATION (characterization): reverting should_exit_loop to the old
    `rng.random() < pos/n` makes this REDDEN — position 0 (prob 0/n) never
    exits, so its bucket is empty and chi-square blows past the band. Verified by
    temporary revert during development; source restored.
    """
    lib = FakeLibrary()
    n = 5
    ids = list(range(100, 100 + n))
    for gid in ids:
        lib.add_genome(gid, _G(gid))
    lib.add_loop(1, ids, structure='cyclic')
    player = LoopPlayer(lib, lambda: _G(-1), _G(999),
                        np.random.default_rng(20251012))
    cyc = cycle_length(n, 'cyclic')          # 5
    counts = [0] * cyc
    N = 5000
    for _ in range(N):
        player._start_loop(1, _G(999), entry_genome_id=ids[0])
        counts[run_until_exit(player)] += 1  # the loop-sim primitive

    expected = N / cyc
    assert all(c > 0 for c in counts), f"a position never exited: {counts}"
    # Chi-square vs uniform; df = cyc-1 = 4. Critical ~9.49 (p=.05), ~18.5
    # (p=.001). Seeded uniform sits well under 15; the pos/n ramp blows past it
    # (and trips the all>0 guard first).
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    assert chi2 < 15.0, f"exit positions not uniform (chi2={chi2:.1f}): {counts}"
