#!/usr/bin/env python3
"""Survey genome viability metrics across random genomes.

Generates random genomes, runs survey_attractor at multiple rotation angles,
and reports statistics on coverage, bbox area, and variance. Used to find
thresholds for rejecting flying dots and pulsars.

Usage:
    python tools/survey_viability.py --count 1000
    python tools/survey_viability.py --count 500 --angles 12
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flame_sheep.genome import Genome

log = logging.getLogger(__name__)


def survey_at_rotation(genome: Genome, rotation: float,
                       n_test: int = 5000) -> dict:
    """Run survey_attractor at a specific rotation angle."""
    original_rotation = genome.rotation
    genome.rotation = rotation
    result = genome.survey_attractor(n_test=n_test)
    genome.rotation = original_rotation
    return result


def survey_multi_angle(genome: Genome, n_angles: int = 8,
                       n_test: int = 3000) -> dict:
    """Survey a genome at multiple rotation angles.

    Returns per-angle data plus aggregate metrics.
    """
    base_rotation = genome.rotation
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False) + base_rotation

    coverages = []
    bbox_areas = []
    in_viewport_count = 0

    for angle in angles:
        survey = survey_at_rotation(genome, angle, n_test=n_test)
        coverages.append(survey['coverage'])
        if survey['in_viewport'] and 'bbox_min_x' in survey:
            in_viewport_count += 1
            extent_x = survey['bbox_max_x'] - survey['bbox_min_x']
            extent_y = survey['bbox_max_y'] - survey['bbox_min_y']
            bbox_areas.append(extent_x * extent_y)
        else:
            bbox_areas.append(0.0)

    coverages = np.array(coverages)
    bbox_areas = np.array(bbox_areas)

    # Bbox ratio: max/min area across angles (strobe/pulsar detection)
    min_area = float(bbox_areas.min())
    max_area = float(bbox_areas.max())
    bbox_ratio = max_area / max(min_area, 0.001)

    return {
        'coverages': coverages,
        'bbox_areas': bbox_areas,
        'min_coverage': float(coverages.min()),
        'max_coverage': float(coverages.max()),
        'mean_coverage': float(coverages.mean()),
        'std_coverage': float(coverages.std()),
        'cv_coverage': float(coverages.std() / coverages.mean()) if coverages.mean() > 0 else 0,
        'min_bbox_area': min_area,
        'max_bbox_area': max_area,
        'mean_bbox_area': float(bbox_areas.mean()),
        'std_bbox_area': float(bbox_areas.std()),
        'bbox_ratio': bbox_ratio,
        'in_viewport_fraction': in_viewport_count / n_angles,
        'any_zero_coverage': bool(np.any(coverages == 0)),
        'n_zero_angles': int(np.sum(coverages == 0)),
    }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser(description='Survey genome viability metrics')
    parser.add_argument('--count', type=int, default=1000,
                        help='Number of genomes to generate')
    parser.add_argument('--angles', type=int, default=8,
                        help='Rotation angles to test per genome')
    parser.add_argument('--test-points', type=int, default=3000,
                        help='Chaos game iterations per angle')
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # Stats accumulators
    viable_count = 0
    survey_pass_count = 0
    metrics = []
    flying_dots = []
    pulsars = []

    t0 = time.time()

    for i in range(args.count):
        g = Genome.random(rng)
        viable_count += 1  # Genome.random only returns viable genomes

        result = survey_multi_angle(g, n_angles=args.angles,
                                    n_test=args.test_points)
        metrics.append(result)

        # Classify
        is_flying_dot = result['min_bbox_area'] < 0.5  # very small bbox at any angle
        is_pulsar = (result['cv_coverage'] > 1.0 or  # high coefficient of variation
                     result['n_zero_angles'] > 0)  # any angle with zero coverage

        if is_flying_dot:
            flying_dots.append(i)
        if is_pulsar:
            pulsars.append(i)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            log.info('%d/%d genomes (%.1f/s) — %d dots, %d pulsars so far',
                     i + 1, args.count, rate, len(flying_dots), len(pulsars))

    elapsed = time.time() - t0

    # Report
    print(f'\n=== Viability Survey: {args.count} genomes, {args.angles} angles ===')
    print(f'Time: {elapsed:.1f}s ({args.count / elapsed:.1f} genomes/s)')
    print(f'Per genome: {elapsed / args.count * 1000:.1f}ms')
    print()

    # Coverage stats
    min_covs = [m['min_coverage'] for m in metrics]
    mean_covs = [m['mean_coverage'] for m in metrics]
    cv_covs = [m['cv_coverage'] for m in metrics]
    min_bboxes = [m['min_bbox_area'] for m in metrics]
    mean_bboxes = [m['mean_bbox_area'] for m in metrics]

    print('Coverage (min across angles):')
    for p in [1, 5, 10, 25, 50, 75, 90]:
        print(f'  P{p}: {np.percentile(min_covs, p):.4f}')
    print()

    print('Coverage coefficient of variation:')
    for p in [50, 75, 90, 95, 99]:
        print(f'  P{p}: {np.percentile(cv_covs, p):.3f}')
    print()

    print('Min bbox area (across angles):')
    for p in [1, 5, 10, 25, 50, 75, 90]:
        print(f'  P{p}: {np.percentile(min_bboxes, p):.3f}')
    print()

    print('Mean bbox area:')
    for p in [1, 5, 10, 25, 50]:
        print(f'  P{p}: {np.percentile(mean_bboxes, p):.3f}')
    print()

    # Zero-coverage angles
    zero_angle_counts = [m['n_zero_angles'] for m in metrics]
    any_zero = sum(1 for z in zero_angle_counts if z > 0)
    print(f'Genomes with any zero-coverage angle: {any_zero}/{args.count} ({100*any_zero/args.count:.1f}%)')
    print()

    # Classification
    print(f'Flying dots (min_bbox < 0.5): {len(flying_dots)}/{args.count} ({100*len(flying_dots)/args.count:.1f}%)')
    print(f'Pulsars (CV > 1.0 or zero angles): {len(pulsars)}/{args.count} ({100*len(pulsars)/args.count:.1f}%)')
    both = set(flying_dots) & set(pulsars)
    print(f'Both: {len(both)}')
    either = set(flying_dots) | set(pulsars)
    print(f'Either (would reject): {len(either)}/{args.count} ({100*len(either)/args.count:.1f}%)')
    print()

    # Threshold sensitivity
    print('=== Threshold sensitivity (% rejected) ===')
    print('Min bbox area threshold:')
    for thresh in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        rejected = sum(1 for m in metrics if m['min_bbox_area'] < thresh)
        print(f'  < {thresh}: {rejected}/{args.count} ({100*rejected/args.count:.1f}%)')
    print()

    print('Min coverage threshold:')
    for thresh in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]:
        rejected = sum(1 for m in metrics if m['min_coverage'] < thresh)
        print(f'  < {thresh}: {rejected}/{args.count} ({100*rejected/args.count:.1f}%)')
    print()

    bbox_ratios = [m['bbox_ratio'] for m in metrics]
    print('Bbox ratio (max/min area):')
    for p in [50, 75, 90, 95, 99, 100]:
        print(f'  P{p}: {np.percentile(bbox_ratios, p):.1f}')
    print()

    print('Bbox ratio threshold (% rejected):')
    for thresh in [5, 10, 20, 50, 100, 500]:
        rejected = sum(1 for m in metrics if m['bbox_ratio'] > thresh)
        print(f'  > {thresh}: {rejected}/{args.count} ({100*rejected/args.count:.1f}%)')


if __name__ == '__main__':
    main()
