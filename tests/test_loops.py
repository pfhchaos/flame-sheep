"""
Tests for loop composition.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flame_sheep.genome import Genome
from flame_sheep.storage import Library, compute_motion_field, MOTION_GRID
from flame_sheep.loops import (
    compose_loops, save_best_loops, LoopCandidate, _build_one_loop,
    crossover, mutate_loop, evolve_loops,
    loop_overlap, _too_similar,
    loop_sequence, cycle_length, STRUCTURES,
)


# ----------------------------------------------------------------
# Shared fixtures — class-scoped to avoid redundant work
# ----------------------------------------------------------------

@pytest.fixture(scope="class")
def seeded_lib(tmp_path_factory):
    """Library pre-loaded with enough genomes to form loops."""
    d = tmp_path_factory.mktemp("seeded")
    lib = Library(data_dir=d)
    rng = np.random.default_rng(100)
    for _ in range(30):
        lib.save_genome(Genome.random(rng))
    yield lib
    lib.close()


@pytest.fixture(scope="class")
def composed_results(seeded_lib):
    """Run compose_loops once and share across all tests in the class."""
    return compose_loops(seeded_lib, pool_size=15, loop_length=4,
                         n_attempts=10, min_coherence=-1.0)


@pytest.fixture(scope="class")
def lib_with_loops(tmp_path_factory):
    """Library pre-loaded with genomes and a few loops."""
    d = tmp_path_factory.mktemp("with_loops")
    lib = Library(data_dir=d)
    rng = np.random.default_rng(300)
    for _ in range(40):
        lib.save_genome(Genome.random(rng))
    for _ in range(4):
        gids = [int(rng.integers(1, 41)) for _ in range(5)]
        gids = list(dict.fromkeys(gids))
        if len(gids) >= 4:
            lib.save_loop(gids[:5] if len(gids) >= 5 else gids)
    yield lib
    lib.close()


# ----------------------------------------------------------------
# LoopCandidate unit tests (instant)
# ----------------------------------------------------------------

class TestLoopCandidate:

    def test_mean_coherence(self):
        c = LoopCandidate(
            genome_ids=[1, 2, 3],
            motion_fields=[np.zeros((3, 3, 2))] * 3,
            coherences=[0.5, 0.8, 0.2],
        )
        assert abs(c.mean_coherence - 0.5) < 1e-6

    def test_min_coherence(self):
        c = LoopCandidate(
            genome_ids=[1, 2, 3],
            motion_fields=[np.zeros((3, 3, 2))] * 3,
            coherences=[0.5, 0.8, 0.2],
        )
        assert abs(c.min_coherence - 0.2) < 1e-6

    def test_length(self):
        c = LoopCandidate(
            genome_ids=[1, 2, 3, 4],
            motion_fields=[np.zeros((3, 3, 2))] * 4,
            coherences=[0.5] * 4,
        )
        assert c.length == 4

    def test_empty_coherences(self):
        c = LoopCandidate(genome_ids=[], motion_fields=[], coherences=[])
        assert c.mean_coherence == 0.0
        assert c.min_coherence == 0.0


# ----------------------------------------------------------------
# compose_loops — shared result, one call for all tests
# ----------------------------------------------------------------

@pytest.mark.slow
class TestComposeLoops:

    def test_returns_list(self, composed_results):
        assert isinstance(composed_results, list)

    def test_loop_has_correct_length(self, composed_results):
        for loop in composed_results:
            assert loop.length == 4

    def test_no_duplicate_genomes_in_loop(self, composed_results):
        for loop in composed_results:
            assert len(set(loop.genome_ids)) == len(loop.genome_ids)

    def test_sorted_by_coherence(self, composed_results):
        for i in range(len(composed_results) - 1):
            assert composed_results[i].mean_coherence >= composed_results[i + 1].mean_coherence

    def test_motion_fields_present(self, composed_results):
        for loop in composed_results:
            assert len(loop.motion_fields) == loop.length
            for mf in loop.motion_fields:
                assert mf.shape == (MOTION_GRID, MOTION_GRID, 2)

    def test_coherences_match_motion_fields(self, composed_results):
        """Coherences should be recomputable from the stored motion fields."""
        from flame_sheep.storage import motion_field_coherence
        for loop in composed_results:
            for i, coh in enumerate(loop.coherences):
                mf_a = loop.motion_fields[i]
                mf_b = loop.motion_fields[(i + 1) % len(loop.motion_fields)]
                expected = motion_field_coherence(mf_a, mf_b)
                assert abs(coh - expected) < 1e-5


class TestComposeLoopsEdgeCase:

    def test_too_few_genomes_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            lib = Library(data_dir=Path(d))
            rng = np.random.default_rng(200)
            for _ in range(3):
                lib.save_genome(Genome.random(rng))
            results = compose_loops(lib, pool_size=10, loop_length=4,
                                    n_attempts=5)
            assert results == []
            lib.close()


# ----------------------------------------------------------------
# save_best_loops — reuses composed_results, fresh lib for writes
# ----------------------------------------------------------------

@pytest.mark.slow
class TestSaveBestLoops:

    def test_saves_to_library(self, seeded_lib, composed_results):
        if not composed_results:
            pytest.skip('No loops generated')
        n_before = seeded_lib.loop_count()
        ids = save_best_loops(seeded_lib, composed_results, n_keep=2)
        assert len(ids) <= 2
        assert seeded_lib.loop_count() == n_before + len(ids)

    def test_saved_loops_loadable(self, seeded_lib, composed_results):
        if not composed_results:
            pytest.skip('No loops generated')
        # Check any previously saved loop is loadable
        if seeded_lib.loop_count() == 0:
            ids = save_best_loops(seeded_lib, composed_results, n_keep=1)
        else:
            top = seeded_lib.top_loops(n=1)
            ids = [top[0][0]]
        loaded = seeded_lib.load_loop(ids[0])
        assert len(loaded) >= 1


# ----------------------------------------------------------------
# Loop breeding
# ----------------------------------------------------------------

@pytest.mark.slow
class TestCrossover:

    def test_produces_new_loop(self, lib_with_loops):
        loops = lib_with_loops.top_loops(n=2)
        if len(loops) < 2:
            pytest.skip('Need at least 2 loops')
        rng = np.random.default_rng(310)
        child_id = None
        for _ in range(10):
            child_id = crossover(lib_with_loops, loops[0][0], loops[1][0], rng)
            if child_id is not None:
                break
        if child_id is None:
            pytest.skip('Could not produce viable crossover')
        child_ids = lib_with_loops.loop_genome_ids(child_id)
        assert len(set(child_ids)) == len(child_ids)  # no duplicates

    def test_child_has_fitness(self, lib_with_loops):
        loops = lib_with_loops.top_loops(n=2)
        if len(loops) < 2:
            pytest.skip('Need at least 2 loops')
        rng = np.random.default_rng(311)
        child_id = None
        for _ in range(10):
            child_id = crossover(lib_with_loops, loops[0][0], loops[1][0], rng)
            if child_id is not None:
                break
        if child_id is None:
            pytest.skip('Could not produce viable crossover')
        fitness = lib_with_loops.loop_fitness(child_id)
        assert np.isfinite(fitness['fitness'])


@pytest.mark.slow
class TestMutateLoop:

    def test_produces_new_loop(self, lib_with_loops):
        loops = lib_with_loops.top_loops(n=1)
        if not loops:
            pytest.skip('No loops')
        rng = np.random.default_rng(320)
        child_id = None
        for _ in range(10):
            child_id = mutate_loop(lib_with_loops, loops[0][0], rng)
            if child_id is not None:
                break
        if child_id is None:
            pytest.skip('Could not produce viable mutation')
        parent_ids = lib_with_loops.loop_genome_ids(loops[0][0])
        child_ids = lib_with_loops.loop_genome_ids(child_id)
        assert len(child_ids) == len(parent_ids)
        # Exactly one genome should differ
        diffs = sum(1 for a, b in zip(parent_ids, child_ids) if a != b)
        assert diffs == 1


# ----------------------------------------------------------------
# Diversity / overlap
# ----------------------------------------------------------------

class TestLoopOverlap:

    def test_identical_is_one(self):
        ids = [1, 2, 3, 4]
        assert loop_overlap(None, ids, ids) == 1.0

    def test_disjoint_is_zero(self):
        assert loop_overlap(None, [1, 2, 3], [4, 5, 6]) == 0.0

    def test_partial_overlap(self):
        # 2 shared out of 4
        overlap = loop_overlap(None, [1, 2, 3, 4], [3, 4, 5, 6])
        assert 0.4 <= overlap <= 0.6

    def test_empty_is_zero(self):
        assert loop_overlap(None, [], [1, 2]) == 0.0

    def test_too_similar_catches_high_overlap(self, lib_with_loops):
        loops = lib_with_loops.top_loops(n=1)
        if not loops:
            pytest.skip('No loops')
        loop_id = loops[0][0]
        ids = lib_with_loops.loop_genome_ids(loop_id)
        # Same IDs should be too similar
        assert _too_similar(lib_with_loops, ids, [loop_id], max_overlap=0.5)


# ----------------------------------------------------------------
# Loop sequence generator
# ----------------------------------------------------------------

class TestLoopSequence:

    def _take(self, gen, n):
        """Take n items from a generator."""
        return [next(gen) for _ in range(n)]

    def test_cyclic_repeats(self):
        seq = loop_sequence(['A', 'B', 'C'], 'cyclic')
        assert self._take(seq, 9) == ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']

    def test_palindrome_reverses(self):
        seq = loop_sequence(['A', 'B', 'C', 'D'], 'palindrome')
        # A B C D C B | A B C D C B | ...
        result = self._take(seq, 12)
        assert result == ['A', 'B', 'C', 'D', 'C', 'B', 'A', 'B', 'C', 'D', 'C', 'B']

    def test_palindrome_no_doubled_endpoints(self):
        seq = loop_sequence(['A', 'B', 'C'], 'palindrome')
        result = self._take(seq, 8)
        # A B C B | A B C B — never AA or CC
        assert result == ['A', 'B', 'C', 'B', 'A', 'B', 'C', 'B']

    def test_palindrome_two_items(self):
        seq = loop_sequence(['A', 'B'], 'palindrome')
        result = self._take(seq, 6)
        assert result == ['A', 'B', 'A', 'B', 'A', 'B']

    def test_rondo_alternates_with_home(self):
        seq = loop_sequence(['A', 'B', 'C', 'D'], 'rondo')
        result = self._take(seq, 12)
        # A B A C A D | A B A C A D
        assert result == ['A', 'B', 'A', 'C', 'A', 'D', 'A', 'B', 'A', 'C', 'A', 'D']

    def test_rondo_home_is_first(self):
        seq = loop_sequence(['X', 'Y', 'Z'], 'rondo')
        result = self._take(seq, 8)
        assert result[0] == 'X'
        # Every even index should be home
        for i in range(0, len(result), 2):
            assert result[i] == 'X'

    def test_rondo_single_item(self):
        seq = loop_sequence(['A'], 'rondo')
        result = self._take(seq, 4)
        assert result == ['A', 'A', 'A', 'A']

    def test_empty_produces_nothing(self):
        seq = loop_sequence([], 'cyclic')
        # Should not yield anything
        result = []
        for _ in range(3):
            try:
                result.append(next(seq))
            except StopIteration:
                break
        assert result == []

    def test_works_with_integers(self):
        seq = loop_sequence([1, 2, 3], 'palindrome')
        result = self._take(seq, 8)
        assert result == [1, 2, 3, 2, 1, 2, 3, 2]


class TestCycleLength:

    def test_cyclic(self):
        assert cycle_length(4, 'cyclic') == 4

    def test_palindrome(self):
        assert cycle_length(4, 'palindrome') == 6  # A B C D C B

    def test_rondo(self):
        assert cycle_length(4, 'rondo') == 6  # A B A C A D

    def test_single_item(self):
        assert cycle_length(1, 'cyclic') == 1
        assert cycle_length(1, 'palindrome') == 1
        assert cycle_length(1, 'rondo') == 1

    def test_two_items(self):
        assert cycle_length(2, 'cyclic') == 2
        assert cycle_length(2, 'palindrome') == 2
        assert cycle_length(2, 'rondo') == 2
