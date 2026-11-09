"""PR40: frozen PR38 GARCH=0/p85 M5+M15 signals on independent FX majors.

No refitting, no mutation, no asset-specific gene optimization.
Requires *separately verified* H1 2026 M5/M15/GARCH archives for
GBPUSD/AUDUSD/USDCAD plus the original EURUSD Release v2.
If any is absent the run MUST FAIL: never extrapolate 0.2/h from EURUSD.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.check_pr40_data_readiness import validate_inventory
from scripts.run_m5_dynamic_upstream import load_problem
from scripts.run_m5_garch_sweetspot import verify_pr38_parent
from scripts.run_m5_m15_target_experiments import score_and_mask
from scripts.run_m5_temporal_evolution import frozen_gen375
from scripts.run_m5_adaptive_gate import GEN354_SHA256
from scripts.run_m5_conviction_evolution import OUTPUT as EVOLUTION_OUTPUT, setup
from src.conviction_regime_gate import gate_reasons
from src.evolution_multiscale import direction_scores
from src.m5_multiasset_inputs import verify_new_asset, PERIODS
from src.m5_multiasset_portfolio import (
    SYMBOLS,replay_portfolio,meets_portfolio_targets,START,END,
)
from src.upstream_filters import dynamic_upstream, GENE_COUNT
from src.train import write_json
from src.artifacts import file_sha256


SEEDS=(42,43,44)
OUTPUT=Path("reports/pr40-multiasset")


def frozen_direction():
    archive=json.loads((EVOLUTION_OUTPUT/"evolution.json").read_text(encoding="utf-8"))
    return frozen_gen375(GEN354_SHA256,archive)


def make_signals(symbol, data, genome, frozen_parent, legacy_threshold):
    """Original PR38 genome and original p85; no EUR-only fitted derivatives."""
    direction=direction_scores(data["direction_inputs"],[frozen_parent])[0]
    problem=dict(data=data,scores=direction)
    modified,regime=score_and_mask(problem,genome,"garch_regime")
    up,threshold=dynamic_upstream(
        data["entry"],data["margin"],data["timeframe"],data["volatility"],
        np.asarray(genome,dtype="<f8")[:GENE_COUNT],legacy_threshold,
    )
    idx=np.flatnonzero(up&regime)
    times=pd.to_datetime(data["row_times_ns"][idx],unit="ns",utc=True)
    rows=pd.DataFrame({"outcome":data["outcome"][idx]},index=times)
    if len(rows)<=500:
        raise ValueError(symbol+" tiene <=500 propuestas GARCH y no puede activar p85")
    reasons=gate_reasons(rows,modified[idx],data["volatility"][idx],"p85")
    signals=pd.DataFrame(
        dict(side=np.where(modified[idx]>=0,1,-1).astype("i1"),
             outcome=np.asarray(data["outcome"][idx],dtype="i1"),
             gate_reason=reasons),
        index=times,
    )
    audit=dict(
        symbol=symbol,eligible_m5=len(data["outcome"]),
        proposed_after_original_garch=int(len(idx)),
        accepted_by_local_p85_before_global_cap=int((reasons=="accepted").sum()),
        gate_warmup=int((reasons=="conviction_warmup").sum()),
        account_guard="shared bankroll, portfolio-global cap and stop",
        frozen_gen375_sha256=sha256(
            np.asarray(frozen_parent,dtype="<f8").tobytes()
        ).hexdigest(),
    )
    return signals,audit


def _compact(ledger):
    return dict(
        pf_q1=ledger["Q1"]["pf"],
        pf_q2=ledger["Q2"]["pf"],
        dd_semester_pct=ledger["semester"]["drawdown_pct"],
        executed_contracts_per_hour=ledger["semester"]["executed_hourly"],
        pnl_q1=ledger["Q1"]["pnl_usd"],
        pnl_q2=ledger["Q2"]["pnl_usd"],
        pnl_semester_continuous=ledger["semester"]["pnl_usd"],
        stopped=ledger["semester"]["stopped"],
        trades=ledger["semester"]["executed"],
    )


def run(*,seed,founder_report,original_universe,original_manifest,
        golden,golden_manifest,new_assets,inventory,output=OUTPUT):
    if seed not in SEEDS:
        raise ValueError("Semilla PR38 ajena al protocolo")
    readiness=validate_inventory(inventory)
    if not readiness["ready_for_real_backtest"]:
        raise RuntimeError(
            "PR40 BLOCKED: faltan snapshots H1 de "+", ".join(readiness["missing_assets"])
        )
    inventory=json.loads(Path(inventory).read_text(encoding="utf-8"))
    eur=load_problem(original_universe,original_manifest,golden,golden_manifest)
    genes,source_gen,base,_=verify_pr38_parent(founder_report,eur,seed)
    parent=frozen_direction()
    # EUR alone MUST be equivalent to the exact PR38 audited replay.
    streams={"EURUSD":None}
    eur_stream,eur_audit=make_signals(
        "EURUSD",eur["data"],genes,parent,eur["legacy_threshold"]
    )
    streams["EURUSD"]=eur_stream
    provenance=dict(EURUSD=eur["is_snapshot_sha256"])
    qc=dict(EURUSD=eur_audit)
    _,_,_,_,_,_,_,mean,std=setup()
    normalizer_hash=sha256(
        np.asarray(mean,dtype="<f8").tobytes()+
        np.asarray(std,dtype="<f8").tobytes()
    ).hexdigest()
    for symbol in SYMBOLS[1:]:
        directory=Path(new_assets)/symbol.lower()
        archive=directory/"universe.npz"
        manifest=directory/"manifest.json"
        if not archive.is_file() or not manifest.is_file():
            raise RuntimeError("PR40 BLOCKED: no hay archivo M5/M15 para "+symbol)
        want=inventory["assets"][symbol]["universe_sha256"]
        if want!=file_sha256(archive):
            raise ValueError("SHA de Release del activo diferente: "+symbol)
        data,meta=verify_new_asset(symbol,archive,manifest)
        if meta["gen375_normalizer_sha256"]!=normalizer_hash:
            raise ValueError("Input neuronal normalizado distinto de Gen375: "+symbol)
        # Coverage control: compare each pair's observed monthly trading
        # hours against original EURUSD denominator, no imputed market times.
        for month in PERIODS:
            a=float(meta["months_observed_hours"][month])
            b=float(eur["info"]["months_observed_hours"][month])
            if not .90<=a/b<=1.10:
                raise ValueError(f"{symbol} tiene cobertura H1 incompatible en {month}")
        symbol_stream,audit=make_signals(
            symbol,data,genes,parent,eur["legacy_threshold"]
        )
        streams[symbol]=symbol_stream
        provenance[symbol]=meta["universe_sha256"]
        qc[symbol]=audit
    account,risk=eur["account"],eur["risk"]
    individual={}
    for symbol in SYMBOLS:
        individual[symbol]=_compact(replay_portfolio(
            {symbol:streams[symbol]},account,risk,
            hours=eur["hours"],global_hourly_cap=eur["cap"],
        ))
    for item,expected in (
        ("pf_q1",base["quarter_pf"][0]),
        ("pf_q2",base["quarter_pf"][1]),
        ("dd_semester_pct",base["semester_dd"]),
        ("executed_contracts_per_hour",base["semester_rate"]),
    ):
        if not np.isclose(individual["EURUSD"][item],expected,atol=1e-9,rtol=0):
            raise ValueError("El replay solo EURUSD no reproduce el PR38 original: "+item)
    portfolio=replay_portfolio(
        streams,account,risk,hours=eur["hours"],global_hourly_cap=eur["cap"],
    )
    permitted=meets_portfolio_targets(
        portfolio,min_symbols=4,min_pf=1.20,
        min_executed_hourly=1.00,max_dd=12.0
    )
    result=dict(
        protocol="PR40_horizontal_M5_M15_IS_v1",
        seed=seed,pr38_parent_generation=source_gen,
        pr38_frozen_genome_sha256=sha256(genes.astype("<f8").tobytes()).hexdigest(),
        original_universe_sha256=eur["is_snapshot_sha256"],
        universe_sha256_by_symbol=provenance,
        frozen_gen375_sha256=qc["EURUSD"]["frozen_gen375_sha256"],
        normalizer_sha256=normalizer_hash,
        signal_timeframes=["M5","M15"],q1_q2_in_sample=True,q3_accessed=False,
        no_retraining=True,no_model_promotion=True,
        portfolio_global_hourly_cap=eur["cap"],
        per_symbol_signal_audit=qc,
        standalone_individual_accounts=individual,
        portfolio_shared_account=portfolio,
        portfolio_compact=_compact(portfolio),
        portfolio_qualified=permitted,
        success_definition=dict(
            pf_q1_q2_gt=1.2,portfolio_executed_hourly_ge=1.0,
            portfolio_dd_semester_lt=12.,
            all_three_original_pr38_seeds_required=True,
        ),
    )
    Path(output).mkdir(parents=True,exist_ok=True)
    write_json(Path(output)/f"portfolio-seed-{seed}.json",result)
    print("PR40_PORTFOLIO "+json.dumps({
        "seed":seed,"baseline_eur":individual["EURUSD"],
        "portfolio":result["portfolio_compact"],
        "joint_targets_met":permitted,"q3_accessed":False,
    },allow_nan=False),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,required=True,choices=SEEDS)
    for field in ("founder_report","original_universe","original_manifest",
                  "golden","golden_manifest","new_assets","inventory"):
        p.add_argument("--"+field.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(seed=a.seed,founder_report=a.founder_report,
        original_universe=a.original_universe,
        original_manifest=a.original_manifest,
        golden=a.golden,golden_manifest=a.golden_manifest,
        new_assets=a.new_assets,inventory=a.inventory)
