"""PR42: sealed PR41 champions in 12 isolated H1 research instances.

Each process reads ONE symbol's sealed Q1/Q2 M5+M15/GARCH dataset and
ONE frozen PR41 seed report. No evolution, shared ledger, shared cap,
cross-asset feature, Q3 data, broker orders or Telegram sends.
The 3 seeds are three alternative four-silo research configurations,
NOT three simultaneous trading mandates per symbol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_m5_full_is_universe import load_universe
from scripts.run_pr41_filtered_portfolio import make_stream
from scripts.run_pr41_native_asset_evolution import (
    GENE_COUNT_LOCAL, PROTOCOL, SEEDS, SYMBOLS, asset_head,
    evaluate_local, genome_sha, strict_asset_pass,
)
from src.decision import RiskConfig
from src.environment import AccountConfig
from src.m5_multiasset_inputs import verify_new_asset
from src.m5_multiasset_portfolio import START, Q2, END, check_streams
from src.replay_engine import replay_contracts
from src.artifacts import write_json


PR41_RUN_ID = 42576852134
SOURCE_SHA = {
    "EURUSD": "05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9",
    "GBPUSD": "2cc9ab95266cb3a87c2988ffd8006c79e9db68765d5ab54ded129c161fff4a0b",
    "AUDUSD": "8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee",
    "USDCAD": "bf866689bb549bf2d647d8600bd343148dfb44040fb51c9cbb2baa7738f4f8b9",
}
CHAMPION_GENERATION = {
    "EURUSD": {42: 19, 43: 4, 44: 17},
    "GBPUSD": {42: 20, 43: 7, 44: 18},
    "AUDUSD": {42: 8, 43: 19, 44: 18},
    "USDCAD": {42: 18, 43: 17, 44: 20},
}
CAP_PER_SYMBOL_HOUR = 2
OUTPUT = Path("reports/pr42-isolated-silos")
FIELDS = ("pf_q1", "pf_q2", "dd_semester_pct", "executed_hourly",
          "pnl_q1", "pnl_q2", "pnl_semester", "garch_lower",
          "garch_upper", "margin_threshold")


def frozen_candidate(path, symbol, seed):
    """Accept exactly one PR41-winning, three-seed-qualified IS artefact."""
    if symbol not in SYMBOLS or seed not in SEEDS:
        raise ValueError("Símbolo o semilla fuera del protocolo congelado")
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if (r.get("protocol") != PROTOCOL or r.get("symbol") != symbol
            or r.get("seed") != seed or r.get("evaluations") != 2560
            or r.get("generations") != 20 or r.get("population") != 128
            or r.get("champion_generation") != CHAMPION_GENERATION[symbol][seed]
            or r.get("oos_q3_accessed") is not False
            or r.get("input_timeframes") != ["M5", "M15"]
            or r.get("raw_m1_signal_features") is not False
            or r.get("qualified_standalone") is not True
            or r.get("independent_from_other_pairs") is not True
            or r.get("euro_legacy_entry_weights_used") is not False
            or r.get("euro_legacy_direction_weights_used") is not False
            or r.get("input_universe_sha256") != SOURCE_SHA[symbol]
            or len(r.get("history", [])) != 20
            or [h.get("generation") for h in r["history"]] != list(range(1, 21))):
        raise ValueError("Reporte no corresponde al campeón IS sellado de PR41")
    genes = np.asarray(r.get("champion_genome"), dtype="<f8")
    if (genes.shape != (GENE_COUNT_LOCAL,) or not np.isfinite(genes).all()
            or genome_sha(genes) != r.get("champion_genome_sha256")):
        raise ValueError("Los 22 genes no coinciden con su SHA256")
    champion = r.get("champion", {})
    if (champion.get("qualified") is not True
            or not all(isinstance(champion.get(k), (int, float)) and
                       np.isfinite(champion[k]) for k in FIELDS)
            or champion["pf_q1"] <= 1.20 or champion["pf_q2"] <= 1.20
            or champion["dd_semester_pct"] >= 15.):
        raise ValueError("El campeón no superó el filtro estricto IS")
    return r, genes


def native_problem(symbol, universe, manifest, *, golden=None, golden_manifest=None):
    """Load ONE source. Non-EUR never opens EUR data nor shared portfolio."""
    if symbol not in SYMBOLS:
        raise ValueError("Par no autorizado")
    if symbol == "EURUSD":
        if golden is None or golden_manifest is None:
            raise ValueError("EUR requiere sello histórico EUR")
        data, info = load_universe(universe, manifest, golden, golden_manifest)
    else:
        data, info = verify_new_asset(symbol, universe, manifest)
    if info.get("universe_sha256") != SOURCE_SHA[symbol]:
        raise ValueError("No coincide la fuente M5/M15/GARCH nativa de PR41")
    times = pd.to_datetime(data["row_times_ns"], unit="ns", utc=True)
    required_months = [f"2026{month:02d}" for month in range(1, 7)]
    months = info.get("months_observed_hours", {})
    if (len(times) < 25000 or not times.is_monotonic_increasing or
            not times.is_unique or times[0] < START or times[-1] >= END
            or not (times < Q2).any() or not (times >= Q2).any()
            or set(months) != set(required_months)
            or any(not np.isfinite(months[m]) or months[m] <= 0 for m in required_months)):
        raise ValueError("Q3 o fecha fuera de H1, o calendario IS incompleto")
    # New account & stop objects for each process: no bankroll, cap, gate,
    # adaptive state, or trading ledger is supplied by any other pair.
    q1 = float(sum(months[k] for k in required_months[:3]))
    q2 = float(sum(months[k] for k in required_months[3:]))
    return dict(symbol=symbol, data=data, info=info, times=times,
                scores=np.zeros(len(times), dtype=float),
                side=np.ones(len(times), dtype=np.int8),
                account=AccountConfig(initial_balance=1000., stake=10., payout=.80),
                risk=RiskConfig(drawdown_limit=.20),
                cap=CAP_PER_SYMBOL_HOUR, legacy_threshold=1.,
                hours=dict(Q1=q1, Q2=q2, semester=q1 + q2),
                is_snapshot_sha256=info["universe_sha256"])


def _same(label, expected, actual):
    if (expected is None or actual is None
            or not np.isfinite(expected) or not np.isfinite(actual)
            or not np.isclose(expected, actual, atol=1e-7, rtol=0.)):
        raise ValueError(f"No reproduce el campeón PR41: {label} ({expected}, {actual})")


def replay_one(problem, genes, champion):
    """Independent local risk, local p85 state and local 2/h cap."""
    expected = evaluate_local(genes, problem)
    if not strict_asset_pass(expected):
        raise ValueError("Los genes congelados no reproducen el filtro individual")
    checks = {
        "pf_q1": expected["quarter_pf"][0],
        "pf_q2": expected["quarter_pf"][1],
        "dd_semester_pct": expected["semester_dd"],
        "executed_hourly": expected["semester_rate"],
        "pnl_q1": expected["quarter_pnl_usd"][0],
        "pnl_q2": expected["quarter_pnl_usd"][1],
        "pnl_semester": expected["semester_pnl_usd"],
        "garch_lower": expected["regime_lower"],
        "garch_upper": expected["regime_upper"],
        "margin_threshold": expected["direction_margin_threshold"],
    }
    for name, value in checks.items():
        _same(name, champion[name], value)
    stream, band = make_stream(problem, genes)
    check_streams({problem["symbol"]: stream})
    if band["band_lower"] != expected["regime_lower"]:
        raise ValueError("Se cambió la banda local GARCH")
    parts = {}
    traces = {}
    for name, lower, upper in (("Q1", START, Q2), ("Q2", Q2, END), ("semester", START, END)):
        rows = stream.loc[(stream.index >= lower) & (stream.index < upper)]
        frame = pd.DataFrame({"outcome": rows["outcome"]}, index=rows.index)
        result, trace = replay_contracts(
            frame, rows["side"].to_numpy(), problem["account"], problem["risk"],
            observed_hours=problem["hours"][name],
            hourly_cap=problem["cap"], candidate_reasons=rows["gate_reason"].to_numpy(),
        )
        op = result["operational"]
        parts[name] = dict(pf=op["profit_factor"], dd_pct=op["max_drawdown_pct"],
                           executed_hourly=op["contracts_per_observed_hour"],
                           pnl_usd=op["pnl_usd"], contracts=op["trades"],
                           stopped=op["account_stop"] is not None,
                           start_balance_usd=problem["account"].initial_balance,
                           end_balance_usd=problem["account"].initial_balance + op["pnl_usd"])
        traces[name] = trace
    for name, key in (("Q1", "pf_q1"), ("Q2", "pf_q2")):
        _same(name + " PF", champion[key], parts[name]["pf"])
    _same("H1 DD", champion["dd_semester_pct"], parts["semester"]["dd_pct"])
    _same("H1 frequency", champion["executed_hourly"], parts["semester"]["executed_hourly"])
    if any(p["stopped"] for p in parts.values()):
        raise ValueError("Replay de silo tuvo stop inesperado")
    return parts, traces["semester"]


def run(*, symbol, seed, report, universe, manifest, golden=None,
        golden_manifest=None, output=OUTPUT):
    frozen, genes = frozen_candidate(report, symbol, seed)
    problem = native_problem(symbol, universe, manifest,
                             golden=golden, golden_manifest=golden_manifest)
    if frozen["input_universe_sha256"] != problem["is_snapshot_sha256"]:
        raise ValueError("Se mezclaron universos entre pares")
    metrics, audit = replay_one(problem, genes, frozen["champion"])
    destination = Path(output) / symbol.lower() / f"seed-{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    result = dict(
        protocol="PR42_H1_four_airtight_FX_silos_v1",
        pr41_source_run_id=PR41_RUN_ID, symbol=symbol, seed=seed,
        champion_generation=frozen["champion_generation"],
        champion_genome_sha256=genome_sha(genes),
        input_universe_sha256=problem["is_snapshot_sha256"],
        features=["M5", "M15"], native_garch=True, gate="p85",
        independent_bankroll=True, independent_hourly_cap=problem["cap"],
        shared_account=False, shared_cap=False, cross_asset_features=False,
        no_genetic_retraining=True, no_oos_access=True, q3_accessed=False,
        strictly_replayed_is=True, execution_mode="research_only",
        initial_balance_usd=problem["account"].initial_balance,
        stake_usd=problem["account"].stake, payout=problem["account"].payout,
        max_drawdown_stop_pct=20., results=metrics,
        note="Q1/Q2 already selected in PR41. This is reproducibility only, not independent OOS evidence.",
    )
    write_json(destination / "silo.json", result)
    audit.to_csv(destination / "decisions.csv", index=False)
    print("PR42_ISOLATED_SILO " + json.dumps({
        "symbol": symbol, "seed": seed,
        "pf_q1": metrics["Q1"]["pf"], "pf_q2": metrics["Q2"]["pf"],
        "executed_hourly": metrics["semester"]["executed_hourly"],
        "dd_h1": metrics["semester"]["dd_pct"],
        "q3_accessed": False, "no_orders": True,
    }, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", required=True, choices=SYMBOLS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    for name in ("report", "universe", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--golden", type=Path)
    p.add_argument("--golden-manifest", type=Path)
    p.add_argument("--output", type=Path, default=OUTPUT)
    options = vars(p.parse_args())
    run(**{k.replace("-", "_"): v for k, v in options.items()})
