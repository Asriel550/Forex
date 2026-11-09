"""Tests del experimento de saturación Gen398–497."""
import numpy as np
import pytest

from scripts.run_m5_upstream_plateau import (
    FIRST_GENERATION, LAST_GENERATION, GENERATIONS, SEEDS,
    POPULATION, PARENT_GENOME_SHA, genome_sha, trajectory,
)


def record(*, generation=398, score=-.5, frequency=.1):
    return dict(
        generation=generation,
        qualifies=False,
        score=score,
        worst_quarter_return=-.05,
        quarter_return_std=.1,
        min_gate_only_frequency=frequency,
        median_score=score-.1,
    )


def test_exact_100_new_generations_with_128_models():
    assert (FIRST_GENERATION, LAST_GENERATION, GENERATIONS)==(398,497,100)
    assert POPULATION==128
    assert SEEDS==(42,43,44)
    assert POPULATION*GENERATIONS*len(SEEDS)==38400
    assert all(len(PARENT_GENOME_SHA[s])==64 for s in SEEDS)


def test_best_fitness_does_not_confuse_stagnation_with_absence_of_candidates():
    prior=record(generation=392,score=-.6)
    history=[record(generation=g,score=-.65) for g in range(398,498)]
    history[39]=record(generation=437,score=-.58)
    history[49]=record(generation=447,score=-.6)
    stats=trajectory(history,prior)
    assert stats["improving_generations"]==[437]
    assert stats["improvements_count"]==1
    assert stats["last_improvement_generation"]==437
    assert stats["generations_since_last_improvement"]==60
    assert stats["best_fitness_by_generation"][0]==-.6
    assert stats["best_fitness_by_generation"][-1]==-.58
    assert stats["fitness_gain_last_20"]==0.
    assert len(stats["median_fitness_by_generation"])==100


def test_no_improvements_detects_full_plateau():
    prior=record(generation=386,score=-.3)
    history=[record(generation=g,score=-.4) for g in range(398,498)]
    stats=trajectory(history,prior)
    assert stats["improvements_count"]==0
    assert stats["generations_since_last_improvement"]==100
    assert stats["last_improvement_generation"] is None


def test_missing_generation_rejected():
    history=[record(generation=g,score=-.4) for g in range(398,498)]
    history[-1]["generation"]=496
    with pytest.raises(ValueError,match="Faltan generaciones"):
        trajectory(history, record(generation=392,score=-.5))


def test_genome_sha_is_stable_for_f64_values():
    genome=np.array([.05,-.1,1.25],dtype=np.float64)
    assert genome_sha(genome.tolist())==genome_sha(genome)
    assert genome_sha([.05,-.1,1.26]) != genome_sha(genome)
