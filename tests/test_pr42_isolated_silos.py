"""No-Q3, frozen-genome and no-cross-asset contract for PR42 research silos."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_pr42_isolated_silos as silo
from scripts.run_pr41_native_asset_evolution import genome_sha


def _report(symbol="GBPUSD", seed=42):
    genes = np.zeros(22, dtype="<f8")
    champion = {k: 1.0 for k in silo.FIELDS}
    champion.update(pf_q1=1.6, pf_q2=1.4, dd_semester_pct=5.0,
                    executed_hourly=.05, qualified=True)
    return dict(protocol=silo.PROTOCOL, symbol=symbol, seed=seed,
                evaluations=2560, generations=20, population=128,
                champion_generation=silo.CHAMPION_GENERATION[symbol][seed],
                oos_q3_accessed=False, input_timeframes=["M5", "M15"],
                raw_m1_signal_features=False, qualified_standalone=True,
                independent_from_other_pairs=True, euro_legacy_entry_weights_used=False,
                euro_legacy_direction_weights_used=False,
                input_universe_sha256=silo.SOURCE_SHA[symbol],
                champion_genome=genes.tolist(), champion_genome_sha256=genome_sha(genes),
                champion=champion,
                history=[dict(generation=g) for g in range(1, 21)])


def _write(tmp_path, record):
    path = tmp_path / "champion.json"
    path.write_text(json.dumps(record))
    return path


def test_frozen_champion_is_exactly_pr41_h1_and_keeps_all_20_generations(tmp_path):
    report, genes = silo.frozen_candidate(_write(tmp_path, _report()), "GBPUSD", 42)
    assert report["champion_generation"] == 20
    assert len(genes) == 22


@pytest.mark.parametrize("change", [
    lambda r: r.update(oos_q3_accessed=True),
    lambda r: r.update(champion_generation=19),
    lambda r: r.update(evaluations=2559),
    lambda r: r.update(input_universe_sha256="0" * 64),
    lambda r: r.update(qualified_standalone=False),
    lambda r: r["champion_genome"].__setitem__(3, .5),
    lambda r: r["history"].pop(),
])
def test_rejects_any_champion_or_protocol_tampering(tmp_path, change):
    candidate = _report()
    change(candidate)
    with pytest.raises(ValueError):
        silo.frozen_candidate(_write(tmp_path, candidate), "GBPUSD", 42)


def test_no_lookahead_in_native_problem_and_separate_accounts(monkeypatch):
    observed = {f"2026{m:02d}": 100. for m in range(1, 7)}
    timestamps = pd.date_range("2026-01-02", "2026-06-29",
                               freq="5min", tz="UTC")
    times = timestamps.asi8
    source_calls = []

    def verify(symbol, archive, manifest):
        source_calls.append((symbol, archive, manifest))
        return dict(row_times_ns=times), {
            "universe_sha256": silo.SOURCE_SHA[symbol],
            "months_observed_hours": observed,
        }

    def forbidden_eur(*args):
        raise AssertionError("El silo GBPUSD no debe cargar EURUSD")

    monkeypatch.setattr(silo, "verify_new_asset", verify)
    monkeypatch.setattr(silo, "load_universe", forbidden_eur)
    a = silo.native_problem("GBPUSD", "gbp-only.npz", "gbp-only.json")
    b = silo.native_problem("GBPUSD", "gbp-only.npz", "gbp-only.json")
    assert source_calls == [("GBPUSD", "gbp-only.npz", "gbp-only.json")] * 2
    assert a["account"] is not b["account"]
    assert a["risk"] is not b["risk"]
    assert a["cap"] == b["cap"] == 2
    assert a["hours"] == {"Q1": 300., "Q2": 300., "semester": 600.}
    assert a["times"][-1] < silo.END


def test_reject_q3_even_if_manifest_is_mislabeled(monkeypatch):
    observed = {f"2026{m:02d}": 100. for m in range(1, 7)}
    timestamps = pd.date_range("2026-01-02", "2026-07-02",
                               freq="5min", tz="UTC")
    monkeypatch.setattr(silo, "verify_new_asset",
                        lambda symbol, universe, manifest: (
                            dict(row_times_ns=timestamps.asi8),
                            dict(universe_sha256=silo.SOURCE_SHA[symbol],
                                 months_observed_hours=observed)))
    with pytest.raises(ValueError, match="Q3"):
        silo.native_problem("USDCAD", "usdcad.npz", "usdcad.json")


def test_no_cross_symbol_portfolio_import_or_shared_cap():
    assert silo.CAP_PER_SYMBOL_HOUR == 2
    assert silo.PR41_RUN_ID == 42576852134
    assert set(silo.SOURCE_SHA) == set(silo.SYMBOLS)
    assert all(set(v) == set(silo.SEEDS) for v in silo.CHAMPION_GENERATION.values())
