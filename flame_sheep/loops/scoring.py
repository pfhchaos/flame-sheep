"""Loop fitness scoring — produces the composite fitness + components
that the library persists alongside a loop. Stateless function; depends
only on motion-field utilities and genome distance.

Lives here (not in storage) because it's a loop-level operation that
happens to be CALLED from `Library.save_loop`. The reverse import
direction would create a cycle (loops already imports storage for the
Library type), so `Library.save_loop` imports this lazily.
"""

from __future__ import annotations

import numpy as np

from ..genome import Genome
from ..genome.motion_field import motion_field_coherence


def score_loop(genomes: list[Genome],
               motion_fields: list[np.ndarray],
               structure: str = 'cyclic') -> dict[str, float]:
    """
    Compute automated fitness metrics for a loop.

    Parameters
    ----------
    genomes : list[Genome]
        The genomes in loop order.
    motion_fields : list[np.ndarray]
        Motion field for each transition (len == len(genomes), wrapping).
    structure : str
        'cyclic', 'palindrome', or 'rondo'.

    Returns
    -------
    dict with keys:
      mean_coherence -- average motion coherence between consecutive transitions
      min_coherence  -- worst-case coherence (one bad jerk tanks this)
      diversity      -- visual variety across genomes in the loop
      palette_flow   -- smoothness of color transitions
      fitness        -- weighted composite of the above
    """
    n = len(genomes)

    # -- motion coherence --
    coherences = []
    for i in range(len(motion_fields)):
        mf_a = motion_fields[i]
        mf_b = motion_fields[(i + 1) % len(motion_fields)]
        coherences.append(motion_field_coherence(mf_a, mf_b))
    mean_coh = float(np.mean(coherences)) if coherences else 0.0
    min_coh = float(np.min(coherences)) if coherences else 0.0

    # -- diversity: average pairwise distance between genomes --
    # High = the loop covers interesting visual ground
    # Low = all genomes look similar (boring)
    if n >= 2:
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(genomes[i].distance(genomes[j]))
        diversity = float(np.mean(dists))
    else:
        diversity = 0.0

    # -- palette flow: how smoothly colors transition around the loop --
    # Compare average palette color between consecutive genomes.
    # Small steps = smooth flow, big jumps = jarring.
    palette_diffs = []
    for i in range(n):
        p_a = genomes[i].palette.mean(axis=0)       # average RGB
        p_b = genomes[(i + 1) % n].palette.mean(axis=0)
        palette_diffs.append(float(np.linalg.norm(p_a - p_b)))
    if palette_diffs:
        # Low variance in step size = smooth; normalize by mean to get consistency
        mean_diff = np.mean(palette_diffs)
        if mean_diff > 1e-6:
            # Consistency: 1 = perfectly even steps, 0 = erratic
            consistency = 1.0 - float(np.std(palette_diffs) / mean_diff)
            consistency = max(0.0, consistency)
            # Moderate step size is best — too small = no change, too big = jarring
            # Peak at ~0.3 palette distance per step
            step_quality = 1.0 - abs(mean_diff - 0.3) / 0.3
            step_quality = max(0.0, min(1.0, step_quality))
            palette_flow = (consistency + step_quality) / 2.0
        else:
            palette_flow = 0.0  # no color change at all
    else:
        palette_flow = 0.0

    # -- smoothness: evenness of consecutive genome distances --
    # Penalizes loops where one transition is a huge jump (the snap problem).
    # Measures coefficient of variation of step distances — 0 = all equal, high = uneven.
    step_dists = []
    for i in range(n):
        step_dists.append(genomes[i].distance(genomes[(i + 1) % n]))
    if step_dists:
        mean_step = np.mean(step_dists)
        max_step = max(step_dists)
        if mean_step > 1e-6:
            # Ratio of worst step to mean — 1.0 = perfectly even, 0 = one step dominates
            smoothness = 1.0 - (max_step - mean_step) / max_step
            smoothness = max(0.0, smoothness)
        else:
            smoothness = 1.0
    else:
        smoothness = 0.0

    # -- composite fitness (structure-dependent weights) --
    if structure == 'palindrome':
        # Palindrome: wrap-around less important (reversal is smooth),
        # bidirectional coherence matters, smoothness still king
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.30   # higher — coherence in both directions
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.05    # lower — wrap transition less critical
        )
    elif structure == 'rondo':
        # Rondo: home genome quality matters, diversity of episodes matters
        # min_coherence less important (transitions to/from home vary)
        fitness = (
            smoothness * 0.25
            + mean_coh * 0.20
            + diversity * 0.30   # higher — episodes should be distinct
            + palette_flow * 0.15
            + min_coh * 0.10
        )
    else:  # cyclic
        fitness = (
            smoothness * 0.30
            + mean_coh * 0.25
            + diversity * 0.20
            + palette_flow * 0.15
            + min_coh * 0.10
        )

    return dict(
        mean_coherence=mean_coh,
        min_coherence=min_coh,
        diversity=diversity,
        palette_flow=palette_flow,
        smoothness=smoothness,
        fitness=fitness,
    )
