"""Synthetic PR43 continuation checks; DO NOT open real Q3 data."""
import json

import numpy as np
import pytest

from scripts import run_pr43_100_native_trainings as train
from scripts.run_pr42_isolated_silos import SOURCE_SHA


def test_budget_and_parent_instead_of_100_fresh_starts():
    assert train.ADDITIONAL_GENERATIONS == 100
    assert train.FIRST_GENERATION == 21 and train.LAST_GENERATION == 120
    assert train.NEW_EVALUATIONS == 12_800
    assert train.NEW_EVALUATIONS * len(train.SYMBOLS) * len(train.SEEDS) == 153_600
    assert train.POPULATION == 128


@pytest.mark.parametrize("symbol", ["EURUSD", "GBPUSD", "AUDUSD", "USDCAD"])
def test_four_independent_pairs(symbol):
    assert symbol in SOURCE_SHA


def test_undefined_pf_cannot_be_treated_as_real():
    assert np.isnan(train._transport(None))
    assert np.isnan(train._transport([None, 1.2])[0])


def test_founder_metric_drift_rejected(monkeypatch):
    frozen = {"input_universe_sha256": SOURCE_SHA["GBPUSD"],
              "champion": {k: 1. for k in (
                  "pf_q1", "pf_q2", "dd_semester_pct", "executed_hourly",
                  "pnl_q1", "pnl_q2", "pnl_semester", "garch_lower", "garch_upper",
                  "margin_threshold")}}
    actual = dict(qualifies=True, quarter_pf=[1.1, 1.2], semester_dd=4.,
                  semester_rate=.01, quarter_pnl_usd=[10., 20.],
                  semester_pnl_usd=30., regime_lower=.5, regime_upper=2.,
                  direction_margin_threshold=1.)
    monkeypatch.setattr(train, "evaluate_local", lambda genome, problem: actual)
    monkeypatch.setattr(train, "strict_asset_pass", lambda r: True)
    with pytest.raises(ValueError, match="Fundador no reproduce"):
        train.check_founder(frozen, np.zeros(22), {"is_snapshot_sha256": SOURCE_SHA["GBPUSD"]})


def test_continuation_uses_frozen_22_genes_and_generations_21_120(monkeypatch, tmp_path):
    symbol, seed = "AUDUSD", 43
    parent = np.linspace(-.3, .4, 22, dtype="<f8")
    signed = {"champion_generation": 19,
              "champion_genome_sha256": train.genome_sha(parent)}
    fake_problem = {"symbol": symbol, "is_snapshot_sha256": SOURCE_SHA[symbol]}
    champion_metrics = {"score": 8., "qualifies": True}
    seen = {}

    monkeypatch.setattr(train, "frozen_candidate", lambda *args: (signed, parent.copy()))
    monkeypatch.setattr(train, "native_problem", lambda *args, **kw: fake_problem)
    monkeypatch.setattr(train, "check_founder", lambda f, g, p: dict(champion_metrics))
    monkeypatch.setattr(train, "evaluate_local", lambda genome, problem: dict(champion_metrics))
    monkeypatch.setattr(train, "result_summary", lambda r: {
        "pf_q1": 1.5, "pf_q2": 1.4, "dd_semester_pct": 5.,
        "executed_hourly": .05, "qualified": True})

    def evolve(evaluate, founder, arm, rng, *, first_generation, last_generation, ranking):
        seen.update(founder=founder.copy(), arm=arm, rng=rng,
                    start=first_generation, end=last_generation)
        return {"evaluations": 12800, "genome": founder.tolist(),
                "champion": {"score": 8., "qualifies": True, "generation": 25},
                "generations": [
                    {"generation": k, "candidate": 0, "score": 8.,
                     "median_score": 7., "mutation_sigma": .12,
                     "quarter_pf": [1.5, 1.4], "semester_rate": .05,
                     "semester_dd": 5., "qualifies": True}
                    for k in range(21, 121)]}
    monkeypatch.setattr(train, "evolve_population", evolve)
    result = train.continue_champion(
        symbol=symbol, seed=seed, report="dummy.json",
        universe="aud.npz", manifest="aud.json", output=tmp_path)
    np.testing.assert_array_equal(seen["founder"], parent)
    assert seen["rng"] == 2043
    assert seen["start"] == 21 and seen["end"] == 120
    assert result["source_pr42_genome_sha256"] == signed["champion_genome_sha256"]
    assert result["zero_initialized"] is False
    assert result["stochastic_resume_exact"] is False
    assert result["q3_accessed"] is False and result["auto_promoted"] is False
    assert len(result["history"]) == 100
    assert (tmp_path / "audusd" / "seed-43.json").exists()


def test_reject_other_seed_or_pair():
    with pytest.raises(ValueError):
        train.continue_champion(symbol="EURUSD", seed=1000,
                                report="x", universe="x", manifest="x")
