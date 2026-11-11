"""Conditional shared-account replay ONLY of 3-seed-qualified independent assets.

Never build a portfolio from an asset that merely performed well in one
seed, and never use the H1 portfolio outcome to choose assets post hoc.
Uses original globally shared M5 2/h cap, original stop, no Q3.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_pr41_native_asset_evolution import (
    GENE_COUNT_LOCAL,asset_head,genome_sha,local_problem,
    SEEDS,SYMBOLS,PROTOCOL,strict_asset_pass,evaluate_local,
)
from src.conviction_regime_gate import gate_reasons
from src.m5_multiasset_portfolio import replay_portfolio,meets_portfolio_targets
from src.artifacts import write_json


OUTPUT=Path("reports/pr41-filtered-portfolio")


def make_stream(problem,genes):
    score,mask,diag=asset_head(problem["data"],genes)
    idx=np.flatnonzero(mask)
    if len(idx)<=500:raise ValueError("No hay suficiente historial p85")
    ts=problem["times"][idx]
    rows=pd.DataFrame({"outcome":problem["data"]["outcome"][idx]},index=ts)
    reasons=gate_reasons(rows,score[idx],problem["data"]["volatility"][idx],"p85")
    stream=pd.DataFrame(
        dict(side=np.where(score[idx]>=0,1,-1).astype("i1"),
             outcome=np.asarray(problem["data"]["outcome"][idx],dtype="i1"),
             gate_reason=reasons),
        index=ts,
    )
    return stream,diag


def run(*,seed,selection,individual_reports,asset_data,eur_universe,
        eur_manifest,golden,golden_manifest):
    if seed not in SEEDS:raise ValueError("Semilla no autorizada")
    selection_data=json.loads(Path(selection).read_text())
    accepted=selection_data.get("eligible_assets_3seeds")
    if (selection_data.get("protocol")!="PR41_per_asset_native_20x128_three_seeds_v1"
            or selection_data.get("oos_q3_accessed") is not False
            or not isinstance(accepted,list) or len(accepted)<2
            or not set(accepted)<=set(SYMBOLS)):
        raise ValueError("PR41 no autorizó una cartera multiasset")
    eur=local_problem(
        "EURUSD",eur_universe,eur_manifest,eur_universe,eur_manifest,golden,golden_manifest
    )
    streams={};checks={}
    for symbol in accepted:
        path=Path(individual_reports)/f"{symbol.lower()}-seed-{seed}.json"
        report=json.loads(path.read_text())
        if (report.get("protocol")!=PROTOCOL or report["symbol"]!=symbol
                or report["seed"]!=seed or report["evaluations"]!=2560
                or report["oos_q3_accessed"] is not False
                or report["qualified_standalone"] is not True):
            raise ValueError("El asset seleccionado no está autorizado por tres seeds")
        genes=np.asarray(report["champion_genome"],dtype="<f8")
        if genes.shape!=(GENE_COUNT_LOCAL,) or genome_sha(genes)!=report["champion_genome_sha256"]:
            raise ValueError("Genoma independiente mal firmado")
        if symbol=="EURUSD":problem=eur
        else:
            slug=symbol.lower()
            problem=local_problem(symbol,Path(asset_data)/slug/"universe.npz",
                Path(asset_data)/slug/"manifest.json",eur_universe,eur_manifest,golden,golden_manifest)
        if problem["is_snapshot_sha256"]!=report["input_universe_sha256"]:
            raise ValueError("Fuente distinta de la evolución del activo")
        actual=evaluate_local(genes,problem)
        stored=report["champion"]
        for field,new in (
            ("pf_q1",actual["quarter_pf"][0]),("pf_q2",actual["quarter_pf"][1]),
            ("dd_semester_pct",actual["semester_dd"]),
            ("executed_hourly",actual["semester_rate"])
        ):
            if not np.isclose(stored[field],new,atol=1e-10,rtol=0):
                raise ValueError("Modelo no reproduce su filtro individual: "+field)
        if not strict_asset_pass(actual):
            raise ValueError("Asset sin pase standalone original")
        stream,band=make_stream(problem,genes)
        streams[symbol]=stream
        checks[symbol]=dict(genome_sha256=genome_sha(genes),
                            dataset_sha256=problem["is_snapshot_sha256"],
                            standalone_pf=actual["quarter_pf"],
                            standalone_dd=actual["semester_dd"],
                            signal_count=len(stream),
                            garch_lower=band["band_lower"],garch_upper=band["band_upper"])
    # This common denominator and shared bankroll are the original EUR
    # environment, not a per-asset sum or multiplied market capacity.
    portfolio=replay_portfolio(
        streams,eur["account"],eur["risk"],hours=eur["hours"],
        global_hourly_cap=eur["cap"]
    )
    accepted_portfolio=meets_portfolio_targets(
        portfolio,min_symbols=2,min_pf=1.20,min_executed_hourly=1.0,
        max_dd=15.
    )
    result=dict(
        protocol="PR41_selected_native_assets_shared_account_v1",
        seed=seed,selected_assets=accepted,asset_provenance=checks,
        portfolio=portfolio,
        pfs=[portfolio["Q1"]["pf"],portfolio["Q2"]["pf"]],
        h1_executed_hourly=portfolio["semester"]["executed_hourly"],
        h1_dd_pct=portfolio["semester"]["drawdown_pct"],
        qualifies=accepted_portfolio,
        asset_selection_frozen_before_portfolio_replay=True,
        timeframes=["M5","M15"],q3_accessed=False,
        no_automatic_promotion=True,
        note="IS selection and IS portfolio replay are not independent OOS evaluation.",
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write_json(OUTPUT/f"seed-{seed}.json",result)
    print("PR41_FILTERED_PORTFOLIO "+json.dumps({
        "seed":seed,"symbols":accepted,"pf_q1":result["pfs"][0],
        "pf_q2":result["pfs"][1],"executed_h":result["h1_executed_hourly"],
        "dd_h1":result["h1_dd_pct"],"qualified":accepted_portfolio,
        "q3_accessed":False,
    },allow_nan=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    for key in ("selection","individual_reports","asset_data","eur_universe",
                "eur_manifest","golden","golden_manifest"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    d=p.parse_args()
    run(seed=d.seed,selection=d.selection,individual_reports=d.individual_reports,
        asset_data=d.asset_data,eur_universe=d.eur_universe,
        eur_manifest=d.eur_manifest,golden=d.golden,golden_manifest=d.golden_manifest)
