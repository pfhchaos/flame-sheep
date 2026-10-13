"""Shared stepping-simulation primitives for the test suite.

Axis integration tests and the LoopPlayer tests repeatedly hand-roll the same
stepping loops: tick an axis N frames while advancing a clock, tick until some
state flips, or step a loaded loop until it signals an exit. These are the
consolidated versions — import them instead of re-inlining the loop:

    from tests.sim import run_ticks, tick_until, run_until_exit

Keep this module tiny: only patterns with real duplication live here. A loop
with a special per-iteration condition (e.g. the interleaved
next_in_sequence/advance_step comparison in test_loop_player) stays inline,
where its meaning is visible.
"""


def run_ticks(axis, audio, n, dt=1 / 60, clock0=0.0):
    """Tick ``axis`` ``n`` times, advancing a frame clock by ``dt`` before each tick.

    Mirrors the ``for _ in range(n): clock += dt; axis.tick(audio, dt, clock)``
    idiom that pervades the axis integration tests. The clock is advanced BEFORE
    each tick, so the first tick sees ``clock0 + dt`` and the last sees
    ``clock0 + n*dt``. Returns the final clock, so a caller can keep stepping::

        clock = run_ticks(axis, idle, 2000)
        clock = run_ticks(axis, idle, 2000, clock0=clock)

    ``audio`` is one AudioState reused across every tick. Axes treat the audio
    frame as read-only input (no axis assigns to it), so this is identical to
    constructing a fresh equal-valued frame each iteration — just cheaper.
    """
    clock = clock0
    for _ in range(n):
        clock += dt
        axis.tick(audio, dt, clock)
    return clock


def tick_until(axis, predicate, audio, max_ticks, dt=1 / 60, clock0=0.0):
    """Tick ``axis`` (advancing the clock like :func:`run_ticks`) until
    ``predicate(axis)`` is True, returning the tick count it took.

    ``predicate`` is a callable taking the axis, evaluated AFTER each tick. If it
    never holds within ``max_ticks`` ticks, raise AssertionError — so a test that
    expects a state transition still fails loudly if that transition stops
    happening, instead of silently running the loop out.
    """
    clock = clock0
    for i in range(1, max_ticks + 1):
        clock += dt
        axis.tick(audio, dt, clock)
        if predicate(axis):
            return i
    raise AssertionError(
        f"predicate never satisfied within {max_ticks} ticks")


def run_until_exit(player) -> int:
    """Simulate ONE loop lifetime: step the loaded loop via the real
    ``next_in_sequence`` path until ``should_exit_loop`` fires, and return the
    exit POSITION (loop_step). Opens the MIN_EXIT_CYCLES gate first.

    The reusable "simulate a loop" primitive — use this instead of re-inlining
    the step loop.
    """
    player._cycle_count = player.MIN_EXIT_CYCLES          # open the exit gate
    bound = player._loop_cycle_len * (player.MIN_EXIT_CYCLES + 3)
    for _ in range(bound):
        if player.should_exit_loop():
            return player.loop_step
        player.next_in_sequence()
    raise AssertionError("should_exit_loop never fired within the step bound")
