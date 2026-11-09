"""Build PR40 per-symbol M5+M15 IS inference universe from VERIFIED HistData bars.

This pipeline does not train the original PR38 direction/upstream genome.
It runs the original causal M5 GARCH/MC recipe on the target PAIR,
the FROZEN founder Gen268 trigger and Gen375/43 M15-arm normalizer,
and outputs all eligible M5 decision rows for Jan-Jun 2026 only.

M1 is NOT loaded or used by this code. It consumes ONLY HistData-derived,
already closed M5 and M15 OHLCV Release assets with strict SHA256.
"""
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_conviction_evolution import setup
from scripts.run_m5_direction_ablation import founder_observations
from scripts.run_m5_semester_replay import MONTHS,PERIODS,END,START,assert_isolated
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv,prepare_market_data
from src.evolution_context import (
    _closed_bar_index,m15_realized_volatility,RANGE_SPAN,
    M15_TACTICAL,H1_STRUCTURAL,
)
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET,MARKET_FEATURES
from src.evolution_multiscale import (
    FEATURE_NAMES,M5_TACTICAL,ARM_WIDTHS,arm_inputs,scale_inputs,
)
from src.market import MarketConfig
from src.m5_multiasset_inputs import FORMAT,FIELDS
from src.m5_multiasset_portfolio import SYMBOLS,Q2
from src.quant_engine import enrich_features
from src.reproducibility import array_sha256,market_frame_fingerprint
from src.upstream_filters import SIGNAL_NAMES,timeframe_features
from src.train import write_json


def read_candles(asset_path,manifest_path,symbol):
    if symbol not in SYMBOLS[1:]:
        raise ValueError("El EURUSD original debe permanecer en su Release v2")
    d=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if (d.get("schema")!="pr40_histdata_m5_m15_h1_ohlcv_v1"
            or d.get("symbol")!=symbol
            or d.get("months")!=list(PERIODS)
            or d.get("output_timeframes")!=["M5","M15"]
            or d.get("m1_trading_features") is not False
            or d.get("q3_accessed") is not False
            or d.get("ohlcv_sha256")!=file_sha256(asset_path)):
        raise ValueError("Velas HistData sin SHA/periodo/símbolo válido")
    with np.load(asset_path,allow_pickle=False) as f:
        required={f"{tf}_{field}" for tf in ("m5","m15")
                  for field in ("timestamp_ns","open","high","low","close","volume")}
        if set(f.files)!=required:
            raise ValueError("OHLCV contiene campos desconocidos")
        data={key:f[key].copy() for key in f.files}
    candles={}
    for tf in ("m5","m15"):
        ts=pd.to_datetime(data[tf+"_timestamp_ns"],unit="ns",utc=True)
        frame=pd.DataFrame({"timestamp":ts,**{
            key:data[tf+"_"+key] for key in ("open","high","low","close","volume")
        }})
        fingerprint=market_frame_fingerprint(
            frame,columns=("open","high","low","close","volume")
        )
        if (fingerprint!=d[tf+"_ohlcv_fingerprint"]
                or len(frame)!=d[tf+"_rows"]
                or frame.timestamp.min()<START
                or (frame.timestamp+pd.Timedelta(minutes=5 if tf=="m5" else 15)>END).any()):
            raise ValueError("HistData OHLCV no coincide con sello para "+tf)
        frame.attrs["timeframe"]=tf.upper()
        candles[tf]=frame
    return candles["m5"],candles["m15"],d


def m15_closed_context(candles,decisions):
    """Same M15 columns/formulas as native_direction_context, H1 arm ZERO."""
    t=candles.timestamp
    ready=t.diff().eq(MarketConfig("M15").step)
    previous=candles.close.shift().where(ready)
    tr=pd.concat((
        candles.high-candles.low,
        (candles.high-previous).abs(),
        (candles.low-previous).abs()
    ),axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/RANGE_SPAN,min_periods=RANGE_SPAN,
               adjust=False).mean()/candles.close
    rv=m15_realized_volatility(candles)
    ix=_closed_bar_index(t,MarketConfig("M15"),pd.DatetimeIndex(decisions))
    aligned=np.full((len(ix),len(M15_TACTICAL)),np.nan)
    okay=ix>=0
    aligned[okay]=np.column_stack((rv,atr))[ix[okay]]
    return pd.DataFrame(aligned,index=decisions,columns=M15_TACTICAL)


def build(symbol,ohlcv,ohlcv_manifest,output,manifest):
    assert_isolated(PERIODS)
    m5,m15,source=read_candles(ohlcv,ohlcv_manifest,symbol)
    reference,founder,quant,risk,_,_,_,mean,std=setup()
    # Original EUR Gen268 account/normalizer/entry weights retained.
    enriched=enrich_features(m5,quant,timeframe="M5",expiry_candles=1)
    prepared=prepare_market_data(enriched,market=MARKET)
    context,normalized,market_normalizer=founder_observations(prepared,founder)
    m5time=timeframe_features(m5,m15)
    rows=[]
    hours={}
    for period,start,end in MONTHS:
        frame=prepared.loc[
            (prepared.timestamp>=start)&
            (prepared.timestamp+MARKET.step<=end)
        ]
        env=BinaryOptionsEnv(frame,market=MARKET,prepared=True,
                             account=founder.account,normalizer=market_normalizer,
                             risk=RiskConfig(enabled=False,
                               drawdown_limit=founder.drawdown_limit))
        sim=M5Simulation(env,normalized.loc[env._times])
        entry,direction=sim.decision_scores(founder.genome)
        opens=env._times[sim.entries]
        times=opens+MARKET.step
        admitted=frame.iloc[sim.entries].set_index("timestamp")
        raw=admitted.loc[:,list(MARKET_FEATURES)].join(
            context.loc[opens,[
                name for name in M5_TACTICAL if name not in MARKET_FEATURES
            ]]
        )
        raw.index=times
        raw=raw.join(m15_closed_context(m15,times))
        # The frozen Gen375 M15 arm has zero effective H1 input coefficients.
        # Supply explicit zero placeholders before arm_inputs zeroes them.
        for name in H1_STRUCTURAL:
            raw[name]=0.
        raw=raw.loc[:,list(FEATURE_NAMES)]
        if (len(raw)!=len(opens) or not raw.index.equals(times)):
            raise ValueError("Contexto de M5/M15 sin orden o universo alineado")
        direction_inputs=arm_inputs(
            scale_inputs(raw.to_numpy(dtype=float),mean,std),"m15"
        )
        timeframe=m5time.reindex(opens)
        if timeframe.isna().any().any():
            raise ValueError("Velas M5/M15 faltantes en entradas elegibles")
        margin=np.abs(direction)
        mask=(entry>0)&(margin>0)&(margin>float(founder.genome[-1]))
        rows.append(dict(
            row_times_ns=times.as_unit("ns").asi8.astype("<i8"),
            outcome=np.asarray(sim.change,dtype="<i1"),
            volatility=np.asarray(env._sigma[sim.entries],dtype="<f8"),
            entry=np.asarray(entry,dtype="<f8"),
            margin=np.asarray(margin,dtype="<f8"),
            timeframe=timeframe.to_numpy(dtype="<f8"),
            direction_inputs=direction_inputs.astype("<f8"),
            legacy_mask=mask.astype("?"),
        ))
        hours[period]=float(sim.observed_hours)
    arrays={k:np.concatenate([row[k] for row in rows]) for k in FIELDS}
    times=pd.to_datetime(arrays["row_times_ns"],unit="ns",utc=True)
    if (len(times)<25000 or times[0]<START or times[-1]>=END
            or not times.is_unique or not times.is_monotonic_increasing
            or not np.isfinite(arrays["direction_inputs"]).all()
            or not np.isfinite(arrays["timeframe"]).all()
            or not np.isfinite(arrays["volatility"]).all()
            or not np.isin(arrays["outcome"],(-1,0,1)).all()):
        raise ValueError("Universo IS/índice M5 incompleto")
    dest=Path(output);out_meta=Path(manifest)
    if dest.exists() or out_meta.exists():
        raise ValueError("Dataset experimental sellado no se sobrescribe")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("xb") as file:
        np.savez_compressed(file,**arrays)
    meta=dict(
        schema=FORMAT,symbol=symbol,
        signal_timeframes=["M5","M15"],
        source_timeframes=["M5","M15"],
        raw_acquisition="HistData M1 for OHLCV aggregation ONLY; never a model feature",
        source_ohlcv_sha256=source["ohlcv_sha256"],
        source_m5_ohlcv_sha256=source["m5_ohlcv_fingerprint"]["values_sha256"],
        source_m15_ohlcv_sha256=source["m15_ohlcv_fingerprint"]["values_sha256"],
        garch_m5_recipe_sha256=sha256(
            Path("src/quant_engine.py").read_bytes()+
            json.dumps(asdict(quant),sort_keys=True).encode()
        ).hexdigest(),
        gen375_normalizer_sha256=sha256(
            np.asarray(mean,dtype="<f8").tobytes()+
            np.asarray(std,dtype="<f8").tobytes()
        ).hexdigest(),
        garch_m5_causal_refit=True,
        garch_retained_refits=enriched.attrs.get("garch_retained_refits",[]),
        garch_config=asdict(quant),
        gen375_normalizer_frozen=True,
        gen375_direction_input_schema=list(FEATURE_NAMES),
        timeframe_signal_names=list(SIGNAL_NAMES),
        original_eurusd_frozen_entry_genome=True,
        start_utc=START.isoformat(),end_exclusive_utc=END.isoformat(),
        is_months=list(PERIODS),q3_accessed=False,
        months_observed_hours=hours,
        universe_sha256=file_sha256(dest),
        arrays_sha256={
            k:array_sha256(v,dtype=(
                "<i8" if k=="row_times_ns" else
                "<i1" if k=="outcome" else
                "?" if k=="legacy_mask" else "<f8"
            )) for k,v in arrays.items()
        },
        rows=len(times),original_proposals=int(arrays["legacy_mask"].sum()),
        caveats="Per-asset GARCH is a fresh causal fit. The frozen EURUSD model has NOT been retrained; generalization is unproven.",
    )
    write_json(out_meta,meta)
    print("PR40_GARCH_UNIVERSE "+json.dumps({
        "symbol":symbol,"rows":len(times),
        "original_proposals":meta["original_proposals"],
        "sha256":meta["universe_sha256"],
        "m1_signals":False,"q3_used":False
    },allow_nan=False),flush=True)
    return meta


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol",required=True,choices=SYMBOLS[1:])
    for key in ("ohlcv","ohlcv_manifest","output","manifest"):
        p.add_argument("--"+key.replace("_","-"),required=True,type=Path)
    a=p.parse_args()
    build(a.symbol,a.ohlcv,a.ohlcv_manifest,a.output,a.manifest)
