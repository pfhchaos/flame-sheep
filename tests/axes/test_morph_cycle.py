"""MorphCycle — the pure morph state machine (UNIT spec).

Covers `axes/_morph_cycle.py` in isolation. This was the test gap flagged in the
2026-10-12 spec audit (skill `flame-sheep/references/transition-fsm.md`):
`MorphCycle` was only exercised *transitively* through `GenomeAxis`, so its own
transition contract had no isolated test. Step-4 S2 restructures the
orchestration AROUND this FSM (the pick moves to dwell-release), so a
timing-independent unit contract here is the net that catches a MorphCycle
regression without the noise of the full axis.

Behavior spec (the contract):
  * boots in DWELL, t = 0
  * state predicates (`morphing`/`in_dwell`/`in_swap_ready`) mirror `state` exactly
  * `advance()` adds `cfg.drift.morph_speed`, CLAMPED at 1.0 (never exceeds)
  * `is_complete()` iff t >= 1.0
  * `enter_swap_ready()` -> SWAP_READY
  * `commit_swap(clock)` -> DWELL, t reset to 0, dwell_start = clock
  * `release_dwell()` -> MORPHING (t untouched)
  * `start_morph()` -> MORPHING, t reset to 0
  * `nudge(amount)` adds `amount` ONLY while MORPHING (no-op otherwise), clamp 1.0

MUTATION GUARDS (each reddens a named test):
  * drop `advance()`'s `min(1.0, …)` clamp -> test_advance_clamps_at_one
  * drop `nudge`'s `if morphing` guard   -> test_nudge_is_noop_unless_morphing
  * drop `nudge`'s clamp                  -> test_nudge_clamps_at_one
  * `commit_swap` not resetting t          -> test_commit_swap_resets_t_sets_dwell

NOT tested here (notated): the ORCHESTRATION that drives these transitions (which
trigger fires when, per audio mode) is `GenomeAxis`'s job — `test_genome_axis.py`
+ (step-4) the relocated-pick specs. This file is the pure FSM only: no audio, no
genomes, no loop.
"""
import pytest

from flame_sheep.axes._morph_cycle import MorphCycle, MorphState
from flame_sheep.config import cfg

SPEED = cfg.drift.morph_speed


def _mc(state=None, t=0.0):
    mc = MorphCycle()
    if state is not None:
        mc.state = state
    mc.t = t
    return mc


# --- boot + predicates -----------------------------------------------------

def test_boots_in_dwell_at_zero():
    mc = MorphCycle()
    assert mc.state == MorphState.DWELL
    assert mc.t == 0.0
    assert mc.dwell_start == 0.0
    assert mc.in_dwell and not mc.morphing and not mc.in_swap_ready


@pytest.mark.parametrize("state,attr", [
    (MorphState.MORPHING, "morphing"),
    (MorphState.DWELL, "in_dwell"),
    (MorphState.SWAP_READY, "in_swap_ready"),
])
def test_predicates_mirror_state(state, attr):
    mc = _mc(state=state)
    preds = {"morphing": mc.morphing,
             "in_dwell": mc.in_dwell,
             "in_swap_ready": mc.in_swap_ready}
    assert preds[attr] is True
    assert sum(preds.values()) == 1          # exactly one predicate true


# --- advance + is_complete -------------------------------------------------

def test_advance_adds_morph_speed():
    mc = _mc(state=MorphState.MORPHING, t=0.0)
    mc.advance()
    assert mc.t == pytest.approx(SPEED)


def test_advance_clamps_at_one():
    """t must saturate at 1.0, never exceed it — however many advances."""
    mc = _mc(state=MorphState.MORPHING, t=0.0)
    seen_max = 0.0
    for _ in range(int(1.0 / SPEED) + 5):
        mc.advance()
        seen_max = max(seen_max, mc.t)
    assert mc.t == pytest.approx(1.0)
    assert seen_max <= 1.0                     # clamp held throughout


def test_is_complete_only_at_one():
    mc = _mc(state=MorphState.MORPHING, t=1.0 - SPEED / 2)
    assert not mc.is_complete()
    mc.t = 1.0
    assert mc.is_complete()


# --- transitions -----------------------------------------------------------

def test_enter_swap_ready():
    mc = _mc(state=MorphState.MORPHING, t=1.0)
    mc.enter_swap_ready()
    assert mc.state == MorphState.SWAP_READY
    assert mc.t == 1.0                         # does NOT touch t


def test_commit_swap_resets_t_sets_dwell():
    mc = _mc(state=MorphState.SWAP_READY, t=1.0)
    mc.commit_swap(clock=42.0)
    assert mc.state == MorphState.DWELL
    assert mc.t == 0.0
    assert mc.dwell_start == 42.0


def test_release_dwell_goes_morphing_keeps_t():
    mc = _mc(state=MorphState.DWELL, t=0.0)
    mc.release_dwell()
    assert mc.state == MorphState.MORPHING     # t untouched (was 0)


def test_start_morph_resets_t():
    mc = _mc(state=MorphState.DWELL, t=0.7)
    mc.start_morph()
    assert mc.state == MorphState.MORPHING
    assert mc.t == 0.0


# --- nudge -----------------------------------------------------------------

def test_nudge_adds_while_morphing():
    mc = _mc(state=MorphState.MORPHING, t=0.2)
    mc.nudge(0.3)
    assert mc.t == pytest.approx(0.5)


@pytest.mark.parametrize("state", [MorphState.DWELL, MorphState.SWAP_READY])
def test_nudge_is_noop_unless_morphing(state):
    mc = _mc(state=state, t=0.4)
    mc.nudge(0.3)
    assert mc.t == pytest.approx(0.4)          # unchanged outside MORPHING


def test_nudge_clamps_at_one():
    mc = _mc(state=MorphState.MORPHING, t=0.9)
    mc.nudge(0.5)
    assert mc.t == pytest.approx(1.0)
