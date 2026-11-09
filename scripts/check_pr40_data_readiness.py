"""PR40 fail-closed data readiness: no invented per-asset PF or volume.

The only currently verified data Release in this repository is EURUSD.
The portfolio workflow must not enter a 'real IS validation' stage until
all four instrument manifests are independently pinned and downloaded.
"""
import argparse
import json
import os
from pathlib import Path
import re

from src.m5_multiasset_portfolio import SYMBOLS
from src.train import write_json


INVENTORY=Path("datasets/multiasset/pr40_asset_inventory.json")
OUTPUT=Path("reports/pr40-multiasset/readiness.json")
SHA=re.compile(r"^[a-f0-9]{64}$")
EXPECTED_MONTHS=[f"2026{m:02d}" for m in range(1,7)]


def validate_inventory(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if (data.get("protocol")!="PR40_four_major_pairs_IS_2026H1_v1"
            or data.get("symbols")!=list(SYMBOLS)
            or set(data.get("assets",{}))!=set(SYMBOLS)
            or data.get("signal_timeframes")!=["M5","M15"]
            or data.get("q3_accessed") is not False
            or data.get("train_periods")!=EXPECTED_MONTHS
            or data.get("no_model_refit") is not True):
        raise ValueError("Inventario no coincide con el protocolo PR40 sellado")
    verified=[]
    candles_ready=[]
    missing=[]
    for sym in SYMBOLS:
        asset=data["assets"][sym]
        if asset.get("status")=="verified":
            if (not asset.get("release_tag")
                    or not SHA.fullmatch(str(asset.get("universe_sha256","")))):
                raise ValueError("Firma inválida para "+sym)
            verified.append(sym)
        elif asset.get("status")=="candles_ready":
            if (not asset.get("candle_release_tag")
                    or not SHA.fullmatch(str(asset.get("candle_ohlcv_sha256","")))
                    or asset.get("verified_source_months")!=EXPECTED_MONTHS):
                raise ValueError("Velas HistData incompletas o sin sello: "+sym)
            candles_ready.append(sym)
            missing.append(sym)  # Still lacks GARCH and frozen-model inputs.
        elif asset.get("status")=="missing":
            missing.append(sym)
        else:
            raise ValueError("Un activo no puede inventar estado: "+sym)
    if "EURUSD" not in verified:
        raise ValueError("La fuente EURUSD canónica desapareció")
    return dict(
        protocol="PR40_multiasset_readiness_v1",
        signal_timeframes=["M5","M15"],seeds=[42,43,44],
        asset_count_required=len(SYMBOLS),verified_assets=verified,
        candle_assets_ready=candles_ready,
        missing_assets=missing,ready_for_real_backtest=not missing,
        portfolio_pf_q1=None,portfolio_pf_q2=None,
        portfolio_executed_contracts_per_hour=None,
        portfolio_max_dd_pct=None,
        q3_accessed=False,
        explanation=(
            "HistData M5/M15 candles are acquired, but GARCH and the frozen "
            "neural input universe must be built and independently sealed. "
            "No extrapolation from EURUSD allowed."
        ),
    )


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",type=Path,default=INVENTORY)
    parser.add_argument("--out",type=Path,default=OUTPUT)
    a=parser.parse_args()
    report=validate_inventory(a.inventory)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    write_json(a.out,report)
    print("PR40_DATA_READINESS "+json.dumps(report,allow_nan=False),flush=True)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"],"a",encoding="utf-8") as file:
            file.write(
                "ready="+("true" if report["ready_for_real_backtest"] else "false")+"\n"
            )
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"],"a",encoding="utf-8") as file:
            file.write("## PR40 data readiness / OOS quarantine\n\n")
            file.write(
                "Verified: "+", ".join(report["verified_assets"])+"; Missing: "+
                (", ".join(report["missing_assets"]) or "none")+"\n\n"
            )
            file.write(
                "**Portfolio PF/frequency/DD are not measured** if missing; "
                "Q3 July-September is unaccessed.\n"
            )
