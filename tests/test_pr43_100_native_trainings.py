"""Contract tests for 100 per-asset PR43 genetic trainings (synthetic only)."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_pr43_100_native_trainings as train


def test_exactly_one_hundred_unique_seeds_per_asset_with_disjoint_shards():
    seeds = [n for shard in range(train.SHARDS) for n in train.seeds_for_shard(shard)]
    assert train.SHARDS == 20
    assert train.SEEDS_PER_SHARD == 5
    assert seeds == list(range(1000, 1100))
    assert len(set(seeds)) == 100
    assert 4 * 100 * train.EVALUATIONS_PER_SEED == 1_024_000
    assert train.FIRST_GENERATION == 1 and train.LAST_GENERATION == 20
    assert train.POPULATION == 128


@pytest.mark.parametrize("bad_shard", [-1, 20, 1.5, "3"])
def test_shard_rejects_unknown_ids(bad_shard):
    with pytest.raises(ValueError):
        train.seeds_for_shard(bad_shard)


def test_none_profit_factor_is_not_promoted_to_finite_pf():
    rows = train.evaluate_batch(np.zeros((2, 22)), object()) if False else None
    assert train.transport(None) != train.transport(None)
    assert train.transport([None, 1.2])[1] == 1.2
    assert np.isnan(train.transport([None, 1.2])[0])


def test_seed_training_starts_from_zero_and_uses_symbol_specific_rng(monkeypatch, tmp_path):
    fake_problem = {"symbol": "AUDUSD",
                    "is_snapshot_sha256": train.SOURCE_SHA["AUDUSD"]}
    seen = {}

    def fake_evolve(evaluate, founder, label, rng_seed, *, first_generation,
                    last_generation, ranking):
        seen["founder"] = founder.copy()
        seen["rng_seed"] = rng_seed
        seen["label"] = label
        seen["generation_span"] = (first_generation, last_generation)
        seen["ranking"] = ranking
        return dict(evaluations=2560, genome=np.full(22, .1),
                    champion={"score": 3.5, "qualifies": True, "generation": 8},
                    generations=[{"generation": i, "candidate": 0, "score": 3.5,
                                  "median_score": 2.3, "quarter_pf": [1.6, 1.4],
                                  "semester_dd": 4.5, "semester_rate": .05,
                                  "mutation_sigma": .12,
                                  "qualifies": True} for i in range(1, 21)])

    def fake_evaluate(genes, problem):
        assert problem is fake_problem
        return {"score": 3.5, "qualifies": True}

    def fake_summary(report):
        return {"qualified": True, "pf_q1": 1.6, "pf_q2": 1.4,
                "executed_hourly": .05, "dd_semester_pct": 4.5}

    monkeypatch.setattr(train, "evolve_population", fake_evolve)
    monkeypatch.setattr(train, "evaluate_local", fake_evaluate)
    monkeypatch.setattr(train, "result_summary", fake_summary)
    monkeypatch.setattr(train, "strict_asset_pass", lambda r: True)

    result = train.train_seed("AUDUSD", 1000, fake_problem, output=tmp_path)
    assert np.array_equal(seen["founder"], np.zeros(22))
    assert seen["rng_seed"] == 21000
    assert seen["generation_span"] == (1, 20)
    assert result["genetic_evaluations"] == 2560
    assert result["q3_accessed"] is False
    assert result["old_pr41_genome_used"] is False
    assert (tmp_path / "audusd" / "seed-1000.json").exists()
    assert len(result["history"]) == 20


def test_per_symbol_batch_loads_one_universe_and_never_eur_for_cad(monkeypatch, tmp_path):
    calls = []

    def fake_native(symbol, universe, manifest, *, golden=None, golden_manifest=None):
        assert symbol == "USDCAD"
        assert golden is None and golden_manifest is None
        calls.append((symbol, universe, manifest))
        return {"symbol": symbol, "is_snapshot_sha256": train.SOURCE_SHA[symbol]}

    def fake_seed(symbol, seed, problem, *, output):
        assert symbol == problem["symbol"] == "USDCAD"
        return {"champion_sha256": str(seed), "champion": {"qualified": False}}

    monkeypatch.setattr(train, "native_problem", fake_native)
    monkeypatch.setattr(train, "train_seed", fake_seed)
    result = train.train_shard(symbol="USDCAD", shard=19,
        universe=Path("cad.npz"), manifest=Path("cad.json"), output=tmp_path)
    assert calls == [("USDCAD", Path("cad.npz"), Path("cad.json"))]
    assert result["seed_range"] == list(range(1095, 1100))
    assert result["genetic_evaluations"] == 12800
    assert result["q3_accessed"] is False


def test_reject_different_market_inputs_and_seed():
    with pytest.raises(ValueError, match="Invasión"):
        train.train_seed("EURUSD", 1000, {
            "symbol": "GBPUSD", "is_snapshot_sha256": train.SOURCE_SHA["GBPUSD"]})
    with pytest.raises(ValueError, match="Semilla"):
        train.train_seed("EURUSD", 42, {
            "symbol": "EURUSD", "is_snapshot_sha256": train.SOURCE_SHA["EURUSD"]})
