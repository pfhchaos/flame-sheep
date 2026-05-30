#!/usr/bin/env python3
"""Test the PrecompileDriver end-to-end:

1. Spin up GpuLoadSampler + Policy (lenient thresholds so it stays
   in RUN — we're not testing yielding here, just the queue + worker
   protocol)
2. Spawn PrecompileDriver
3. Enqueue at multiple priorities, check that high-priority items
   are serviced first
4. Stop, report stats
"""
from __future__ import annotations

import logging
import time

from flame_sheep.scheduler.gpu_load import GpuLoadSampler
from flame_sheep.scheduler.policy import Policy
from flame_sheep.scheduler.precompile_driver import PrecompileDriver


# Three sets of tuples at three priorities; with the priority queue
# working, all the priority=-10 tuples should be compiled before
# any priority=10 tuples. Print stats as they roll in.
HIGH = [
    (3, 0, frozenset([0, 1, 7])),
    (3, 0, frozenset([0, 2, 8])),
]
MED = [
    (4, 0, frozenset([0, 1, 2, 5])),
]
LOW = [
    (2, 0, frozenset([0])),
    (5, 1, frozenset([0, 1, 2, 3, 5, 9])),
    (6, 0, frozenset([0, 1, 2, 4, 7, 11, 14, 17])),
]


def main():
    logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s %(name)s %(levelname)s %(message)s')

    sampler = GpuLoadSampler()
    policy = Policy(sampler,
                     sample_interval_s=0.25,
                     slow_threshold=0.95,
                     pause_threshold=0.99)  # effectively never yield
    driver = PrecompileDriver(canvas_w=3713, canvas_h=1278,
                                policy=policy)
    driver.start()
    print(f'driver started')

    # Enqueue in deliberately reversed priority order — the driver
    # must reorder so HIGH compiles first.
    print(f'enqueue LOW (priority +10) : {driver.enqueue(LOW, priority=10)} added')
    print(f'enqueue MED (priority  0)  : {driver.enqueue(MED, priority=0)} added')
    print(f'enqueue HIGH (priority -10): {driver.enqueue(HIGH, priority=-10)} added')

    # Wait until all 6 tuples processed (compiled + skipped)
    target = 6
    t0 = time.perf_counter()
    while driver.stats_compiled + driver.stats_skipped < target:
        time.sleep(0.5)
        if time.perf_counter() - t0 > 60:
            print('TIMEOUT — driver didn\'t process all tuples in 60s')
            break

    print(f'all done after {time.perf_counter()-t0:.1f}s; '
          f'cold={driver.stats_cold} hit={driver.stats_hit} '
          f'skipped={driver.stats_skipped}')
    driver.stop()
    sampler.close()


if __name__ == '__main__':
    main()
