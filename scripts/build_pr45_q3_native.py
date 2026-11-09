"""PR45: acquire real HistData Q3, join sealed H1 and build causal M5/M15/GARCH.

DATA PREPARATION ONLY. No PR42/PR43 genomes, scores, replay, PF, DD or
model selection are touched. Builds 4 genuine Q3 Parquet datasets and
PR44-compatible native NPZ universes from upstream historical sources.
"""
from __future__ import annotations
from dataclasses import asdict
from hashlib import sha256
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from scripts.acquire_pr40_histdata import BASE, TIMEZONE, verify_zip
from scripts.build_histdata_dataset import read_native_minutes
from scripts.build_pr40_garch_universe import m15_closed_context, read_candles
from scripts.run_m5_conviction_evolution import setup
from scripts.run_m5_direction_ablation import founder_observations
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_context import H1_STRUCTURAL, M5_TACTICAL
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET, MARKET_FEATURES
from src.evolution_multiscale import FEATURE_NAMES, arm_inputs, scale_inputs
from src.market import MarketConfig
from src.m5_multiasset_inputs import FIELDS
from src.quant_engine import enrich_features
from src.reproducibility import array_sha256, market_frame_fingerprint
from src.resampling import aggregate
from src.upstream_filters import SIGNAL_NAMES, timeframe_features

SYMBOLS=("EURUSD","GBPUSD","AUDUSD","USDCAD")
H1_START=pd.Timestamp("2026-01-01T00:00:00Z")
Q3_START=pd.Timestamp("2026-07-01T00:00:00Z")
Q3_END=pd.Timestamp("2026-10-01T00:00:00Z")
Q3_MONTHS=("202607","202608","202609")
Q3_BOUNDARIES=(Q3_START,pd.Timestamp("2026-08-01T00:00:00Z"),
               pd.Timestamp("2026-09-01T00:00:00Z"),Q3_END)
PAIRS_H1_SCHEMA="pr40_histdata_m5_m15_h1_ohlcv_v1"
EXPECTED_H1_SHA={
    "GBPUSD":"10a5d1aeecf9019134a6ed2b6c3131c6f48ca56dcb583df3dfca441690807b45",
    "AUDUSD":"0c728f5602bc622fca9f98553e80aaf9bde4c50f1ca9746bc04d4fddc3083db0",
    "USDCAD":"c1f0b62d0b05cff5841185fe7803a7aa39ae9bea2287cbf1a6f51c1b21c6241c",
}
# New historical source must be the ACTUAL publisher. Whitelist excludes Oct.
ALLOWED={symbol:tuple(f"2026{m:02d}" for m in range(1,10))
         if symbol=="EURUSD" else ("202606",*Q3_MONTHS)
         for symbol in SYMBOLS}
OUTPUT=Path("reports/pr45-q3-native")
Q3_SCHEMA="PR44_Q3_native_M5_M15_GARCH_universe_v1"
UTC_TAG="UTC, candle open"
TYPES=dict(row_times_ns="<i8",outcome="<i1",volatility="<f8",
           entry="<f8",margin="<f8",timeframe="<f8",
           direction_inputs="<f8",legacy_mask="?")


def write_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n",
                          encoding="utf-8")


def form_for_q3(html,symbol,period):
    # PR40 form parser has H1-only deliberate quarantine; this is the
    # separately authorized Q3 DATA PREPARATION path (no scoring).
    from scripts.acquire_histdata import Forms
    forms=Forms();forms.feed(html)
    possible=[form for form in forms.forms
        if form["fields"].get("timeframe","").upper()=="M1"
        and form["fields"].get("fxpair","").upper()==symbol
        and form["fields"].get("tk")
        and form.get("action","").split("?")[0].rstrip("/")=="/get.php"]
    if len(possible)!=1:
        raise ValueError("HistData no sirve formulario autenticado exacto "+symbol+" "+period)
    fields=possible[0]["fields"]
    if (fields.get("date")!="2026" or fields.get("datemonth")!=period
        or fields.get("platform")!="ASCII"):
        raise ValueError("Formulario equivocado de proveedor "+symbol+" "+period)
    return fields


def get_month(symbol,period,folder,client):
    if symbol not in SYMBOLS or period not in ALLOWED[symbol]:
        raise ValueError("Mes no autorizado para adquisición PR45")
    url=(f"{BASE}/download-free-forex-historical-data/?/"
         f"ascii/1-minute-bar-quotes/{symbol.lower()}/2026/{int(period[-2:])}")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"HISTDATA_COM_ASCII_{symbol}_M1_{period}.zip"
    if not path.exists():
        html=client.get(url,timeout=(35,100));html.raise_for_status()
        form=form_for_q3(html.text,symbol,period)
        payload=client.post(f"{BASE}/get.php",data=form,
            headers={"Referer":url,"Origin":BASE},timeout=(25,150))
        payload.raise_for_status()
        if len(payload.content)>30*1024*1024:
            raise ValueError("Descarga HistData demasiado grande")
        tmp=path.with_suffix(".part")
        tmp.write_bytes(payload.content)
        try:
            verify_zip(tmp,symbol,period)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    members=verify_zip(path,symbol,period)
    return dict(symbol=symbol,period=period,provider="HistData.com",
                format="Generic ASCII M1 bid OHLCV",
                source_url=url,zip_file=path.name,sha256=file_sha256(path),
                bytes=path.stat().st_size,entries=members,
                raw_role="M1 ONLY resampled into closed M5/M15 candles")


def acquire(*,symbol,raw_dir,output):
    if symbol not in SYMBOLS:raise ValueError("Par fuera de los cuatro fijados")
    folder=Path(raw_dir)/symbol.lower()
    sources=[]
    with requests.Session() as client:
        client.headers.update({"User-Agent":"Forex-PR45-Q3-source-audit/1.0"})
        for month in ALLOWED[symbol]:
            sources.append(get_month(symbol,month,folder,client))
            time.sleep(.7)
    result=dict(protocol="PR45_H1_to_Q3_official_raw_sources_v1",
                symbol=symbol,months=list(ALLOWED[symbol]),
                sources=sources,source_timezone=TIMEZONE,
                start_utc=H1_START.isoformat(),q3_start_utc=Q3_START.isoformat(),
                q3_end_exclusive_utc=Q3_END.isoformat(),
                m1_signals=False,zero_model_access=True,no_alpha_evaluation=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    write_json(output,result)
    print("PR45_SOURCE_ACQUIRED "+json.dumps({
        "symbol":symbol,"months":result["months"],
        "hashes":{s["period"]:s["sha256"] for s in sources},
    },allow_nan=False),flush=True)
    return result


def _check_frame(frame,step,start,end):
    t=pd.DatetimeIndex(frame.timestamp)
    m=int(step.total_seconds()/60)
    if (len(frame)<(10000 if m==5 else 3500) or t.tz is None or not t.is_monotonic_increasing or
        not t.is_unique or t.min()<start or t.max()+step>end or
        not t.equals(t.floor(f"{m}min"))):
        raise ValueError("Velas OOS no ordenadas, incompletas o fuera de Q3")
    v=frame[["open","high","low","close","volume"]].to_numpy(float)
    if (not np.isfinite(v).all() or (v[:,:4]<=0).any()
        or (v[:,1]<np.maximum(v[:,0],v[:,3])).any()
        or (v[:,2]>np.minimum(v[:,0],v[:,3])).any()):
        raise ValueError("Velas del proveedor inválidas")
    for a,b in zip(Q3_BOUNDARIES[:-1],Q3_BOUNDARIES[1:]):
        if ((t>=a)&(t<b)).sum()<2500:
            raise ValueError("Mes Q3 sin historial suficiente")
    if t.min()>Q3_START+pd.Timedelta(hours=10) or t.max()<Q3_END-pd.Timedelta(days=2):
        raise ValueError("Falta inicio/final del trimestre Q3")


def _unpack_h1(pair,ohlcv,manifest):
    if pair=="EURUSD":raise ValueError("EUR reconstruye H1 desde fuente oficial M1")
    if file_sha256(ohlcv)!=EXPECTED_H1_SHA[pair]:
        raise ValueError("El H1 histórico no coincide con la release PR40 sellada")
    a,b,info=read_candles(ohlcv,manifest,pair)
    for df in (a,b):
        df.attrs["source"]="PR40_H1_original"
    return a,b,info


def _continuous_h1_join(h1_m5,h1_m15,raw_q3_m5,raw_q3_m15,minute_frame):
    """Q3 hours from June EST source plus original H1 candles, without splicing future bars."""
    overlap=minute_frame.loc[(minute_frame.timestamp>=pd.Timestamp("2026-06-03T00:00Z"))&
                             (minute_frame.timestamp<Q3_START)]
    for old,step in ((h1_m5,"M5"),(h1_m15,"M15")):
        full=aggregate(overlap,1,step,now=Q3_START)
        minimum=2000 if step=="M5" else 700
        if len(full)<minimum:raise ValueError("June Q2->Q3 source overlap missing")
        matched=old.set_index("timestamp").join(full.set_index("timestamp"),
                   how="inner",lsuffix="_old",rsuffix="_fresh")
        if len(matched)<minimum:
            raise ValueError("Fuente June no coincide en sellos de tiempo H1")
        for name in ("open","high","low","close","volume"):
            if not np.allclose(matched[name+"_old"],matched[name+"_fresh"],
                               rtol=0.,atol=1e-9):
                raise ValueError("OHLCV H1/Q3 incompatible, no se permite unión arbitraria")
    def join(old,new,minutes):
        frame=pd.concat([old.loc[old.timestamp<Q3_START],
                         new.loc[new.timestamp>=Q3_START]],ignore_index=True)
        if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError("Cronología H1/Q3 no causal")
        frame.attrs["timeframe"]=f"M{minutes}"
        return frame
    return join(h1_m5,raw_q3_m5,5),join(h1_m15,raw_q3_m15,15)


def build(*,symbol,raw_dir,source_manifest,h1_ohlcv=None,
          h1_manifest=None,h1_universe=None,output=OUTPUT):
    if symbol not in SYMBOLS:raise ValueError("Símbolo no preinscrito")
    if h1_universe is None or not Path(h1_universe).is_file():
        raise ValueError("Se exige el universo H1 sellado para continuidad GARCH Q2→Q3")
    source=json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    expected=ALLOWED[symbol]
    entries=source.get("sources",[])
    if (source.get("protocol")!="PR45_H1_to_Q3_official_raw_sources_v1"
        or source.get("symbol")!=symbol or source.get("months")!=list(expected)
        or [s.get("period") for s in entries]!=list(expected)
        or source.get("m1_signals") is not False):
        raise ValueError("Fuente Q3 no certificada")
    minutes,source_audit=read_native_minutes(
        Path(raw_dir)/symbol.lower(),entries)
    # June's fixed UTC-5 ZIP contains Jul 1, 00:00..04:59 UTC.
    # Without June, the first Q3 bars would be silently missing.
    minute_q3=minutes.loc[(minutes.timestamp>=Q3_START)&
                          (minutes.timestamp<Q3_END)].reset_index(drop=True)
    if minute_q3.empty:raise ValueError("No se adquirió Q3")
    m5_q3=aggregate(minute_q3,1,"M5",now=Q3_END)
    m15_q3=aggregate(minute_q3,1,"M15",now=Q3_END)
    _check_frame(m5_q3,pd.Timedelta(minutes=5),Q3_START,Q3_END)
    _check_frame(m15_q3,pd.Timedelta(minutes=15),Q3_START,Q3_END)
    if symbol=="EURUSD":
        # EURUSD has no verified H1 OHLCV release M5; rebuild it from Jan-Jun
        # same official source and concatenate BEFORE causal feature engineering.
        m5_all=aggregate(minutes.loc[minutes.timestamp<Q3_END],1,"M5",now=Q3_END)
        m15_all=aggregate(minutes.loc[minutes.timestamp<Q3_END],1,"M15",now=Q3_END)
        m5_all.attrs["timeframe"]="M5"
        m15_all.attrs["timeframe"]="M15"
        if m5_all.timestamp.min()>pd.Timestamp("2026-01-06T00:00Z"):
            raise ValueError("El M5 EUR carece de warmup Q1/Q2")
        hist_info=dict(universe_sha256=None,source="HistData EURUSD Jan-Sep 2026")
    else:
        if h1_ohlcv is None or h1_manifest is None:
            raise ValueError("GBP/AUD/CAD requieren su H1 OHLCV PR40 original")
        a,b,hist_info=_unpack_h1(symbol,h1_ohlcv,h1_manifest)
        m5_all,m15_all=_continuous_h1_join(
            a,b,m5_q3,m15_q3,minutes)
    if (len(m5_all)<40000 or m5_all.timestamp.min()>pd.Timestamp("2026-01-07T00:00Z")
        or m5_all.timestamp.max()<Q3_END-pd.Timedelta(days=2)):
        raise ValueError("No existe historial M5 causal continuo H1→Q3")
    reference,founder,quant,risk,_,_,_,mean,std=setup()
    enriched=enrich_features(m5_all,quant,timeframe="M5",expiry_candles=1)
    prepared=prepare_market_data(enriched,market=MARKET)
    context,normalized,market_normalizer=founder_observations(prepared,founder)
    tf=timeframe_features(m5_all,m15_all)
    rows=[];hours={}
    for i,month in enumerate(Q3_MONTHS):
        start,end=Q3_BOUNDARIES[i:i+2]
        # Avoid any trade requiring Q4 outcomes to settle.
        frame=prepared.loc[(prepared.timestamp>=start)&
                           (prepared.timestamp+MARKET.step<end)]
        if len(frame)<2500:raise ValueError("Falta mes Q3 en simulación causal")
        env=BinaryOptionsEnv(frame,market=MARKET,prepared=True,
                             account=founder.account,normalizer=market_normalizer,
                             risk=RiskConfig(enabled=False,
                                  drawdown_limit=founder.drawdown_limit))
        sim=M5Simulation(env,normalized.loc[env._times])
        entry,direction=sim.decision_scores(founder.genome)
        opens=env._times[sim.entries]
        closes=opens+MARKET.step
        admission=frame.iloc[sim.entries].set_index("timestamp")
        raw=admission.loc[:,list(MARKET_FEATURES)].join(
            context.loc[opens,[k for k in M5_TACTICAL if k not in MARKET_FEATURES]])
        raw.index=closes
        raw=raw.join(m15_closed_context(m15_all,closes))
        for name in H1_STRUCTURAL:raw[name]=0.
        raw=raw.loc[:,list(FEATURE_NAMES)]
        directions=arm_inputs(
            scale_inputs(raw.to_numpy(dtype=float),mean,std),"m15")
        timeframe=tf.reindex(opens)
        if timeframe.isna().any().any() or len(timeframe)!=len(closes):
            raise ValueError("M15 no cerrada o M5 sin features Q3")
        margin=np.abs(direction)
        original_mask=(entry>0)&(margin>0)&(margin>float(founder.genome[-1]))
        rows.append(dict(row_times_ns=closes.as_unit("ns").asi8.astype("<i8"),
            outcome=np.asarray(sim.change,dtype="<i1"),
            volatility=np.asarray(env._sigma[sim.entries],dtype="<f8"),
            entry=np.asarray(entry,dtype="<f8"),margin=np.asarray(margin,dtype="<f8"),
            timeframe=timeframe.to_numpy(dtype="<f8"),
            direction_inputs=directions.astype("<f8"),legacy_mask=original_mask.astype("?")))
        hours[month]=float(sim.observed_hours)
    arrays={key:np.concatenate([row[key] for row in rows]) for key in FIELDS}
    t=pd.to_datetime(arrays["row_times_ns"],unit="ns",utc=True)
    if (len(t)<10000 or t[0]<Q3_START or t[-1]>=Q3_END
        or not t.is_monotonic_increasing or not t.is_unique
        or any(not (t.month==m).any() for m in (7,8,9))
        or arrays["timeframe"].shape!=(len(t),len(SIGNAL_NAMES))
        or arrays["direction_inputs"].shape!=(len(t),len(FEATURE_NAMES))
        or not all(np.isfinite(arrays[key]).all() for key in
                   ("volatility","entry","margin","timeframe","direction_inputs"))):
        raise ValueError("Modelo Q3 no derivado del mercado con features H1 consistentes")

    # Parquet user-facing: actual completed M5 OHLCV and causal GARCH/M15
    # features. Unlabeled price/timeframe data are never scored in this job.
    q3_bars=m5_all.loc[(m5_all.timestamp>=Q3_START)&
                       (m5_all.timestamp+pd.Timedelta(minutes=5)<=Q3_END)].copy()
    enriched_values=enriched.set_index("timestamp")
    indicators=tf.reindex(pd.DatetimeIndex(q3_bars.timestamp))
    if indicators.isna().any().any():raise ValueError("Contexto M15 incompleto")
    q3_bars["m5_close_utc"]=q3_bars.timestamp+pd.Timedelta(minutes=5)
    q3_bars["garch_sigma_m5"]=enriched_values["cond_volatility"].reindex(
        pd.DatetimeIndex(q3_bars.timestamp)).to_numpy(dtype=float)
    if (not np.isfinite(q3_bars["garch_sigma_m5"]).all()
            or (q3_bars["garch_sigma_m5"]<0).any()):
        raise ValueError("GARCH Q3 sin continuidad H1 válida")
    for label in SIGNAL_NAMES:
        q3_bars[label]=indicators[label].to_numpy(dtype=float)
    q3_bars=q3_bars.rename(columns={"timestamp":"m5_open_utc"})
    out=Path(output)/symbol.lower()
    out.mkdir(parents=True,exist_ok=True)
    parquet=out/f"{symbol}_M5_Q3_2026.parquet"
    archive=out/"universe.npz"
    manifest=out/"manifest.json"
    if any(path.exists() for path in (parquet,archive,manifest)):
        raise ValueError("No se puede sobrescribir Q3 sellado")
    q3_bars.to_parquet(parquet,index=False,compression="zstd")
    with archive.open("xb") as f:
        np.savez_compressed(f,**arrays)
    norm_hash=sha256(
        np.asarray(mean,dtype="<f8").tobytes()+
        np.asarray(std,dtype="<f8").tobytes()).hexdigest()
    if (symbol!="EURUSD" and
        norm_hash!=hist_info.get("gen375_normalizer_sha256",norm_hash)):
        # H1 OHLCV contains no normalizer metadata: the reference is the
        # frozen Gen375 setup() / same original code as PR40.
        raise ValueError("Normalizador de referencia modificado")
    source_hashes={s["period"]:s["sha256"] for s in entries}
    source_m5_fp=market_frame_fingerprint(m5_q3,columns=("open","high","low","close","volume"))
    source_m15_fp=market_frame_fingerprint(m15_q3,columns=("open","high","low","close","volume"))
    original_h1_sha={
        "EURUSD":"05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9",
        "GBPUSD":"2cc9ab95266cb3a87c2988ffd8006c79e9db68765d5ab54ded129c161fff4a0b",
        "AUDUSD":"8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee",
        "USDCAD":"bf866689bb549bf2d647d8600bd343148dfb44040fb51c9cbb2baa7738f4f8b9",
    }
    # The PR41 relative-GARCH band uses the previous 48 eligible M5
    # proposals; Q2 must seed the first Q3 decisions, never cold-start.
    original_h1_archive={
        "EURUSD":"05487661977ea0a612f7e717b81e84aadbb5e2071238ca6835d9944a06bc9be9",
        "GBPUSD":"2cc9ab95266cb3a87c2988ffd8006c79e9db68765d5ab54ded129c161fff4a0b",
        "AUDUSD":"8d1f34d5bf38ce1c4beeb5079d374a7af1a9d07a8161d07fc010d56b251b0eee",
        "USDCAD":"bf866689bb549bf2d647d8600bd343148dfb44040fb51c9cbb2baa7738f4f8b9",
    }
    if file_sha256(h1_universe)!=original_h1_archive[symbol]:
        raise ValueError("Universo H1/GARCH PR41 no firmado")
    with np.load(h1_universe,allow_pickle=False) as frozen:
        old_sigma=np.asarray(frozen["volatility"],dtype="<f8")
        old_times=np.asarray(frozen["row_times_ns"],dtype="<i8")
    if (len(old_sigma)<48 or len(old_times)!=len(old_sigma) or
        pd.Timestamp(old_times[-1],unit="ns",tz="UTC")>=Q3_START or
        not np.isfinite(old_sigma).all() or (old_sigma<0).any()):
        raise ValueError("Sin los 48 estados finales GARCH válidos de Q2")
    q2_tail=np.ascontiguousarray(old_sigma[-48:],dtype="<f8")
    meta=dict(schema=Q3_SCHEMA,symbol=symbol,
        provider="HistData.com",
        signal_timeframes=["M5","M15"],source_timeframes=["M5","M15"],
        raw_m1_signal_features=False,garch_native_causal_refit=True,
        frozen_h1_normalizer=True,m15_closed_no_lookahead=True,
        q3_full_source_coverage=True,
        start_utc=Q3_START.isoformat(),end_exclusive_utc=Q3_END.isoformat(),
        months=list(Q3_MONTHS),months_observed_hours=hours,
        source_zip_sha256={k:source_hashes[k] for k in Q3_MONTHS},
        h1_q3_warmup_source_zip_sha256={
            k:v for k,v in source_hashes.items() if k not in Q3_MONTHS},
        h1_causal_context="Jan-Jun sealed PR40 M5/M15" if symbol!="EURUSD"
            else "HistData EUR Jan-Jun M1->closed M5/M15",
        frozen_h1_input_sha256=original_h1_sha[symbol],
        source_m5_ohlcv_sha256=source_m5_fp["values_sha256"],
        source_m15_ohlcv_sha256=source_m15_fp["values_sha256"],
        source_m5_ohlcv_fingerprint=source_m5_fp,
        source_m15_ohlcv_fingerprint=source_m15_fp,
        garch_m5_recipe_sha256=sha256(
            Path("src/quant_engine.py").read_bytes()+
            json.dumps(asdict(quant),sort_keys=True).encode()).hexdigest(),
        frozen_h1_normalizer_sha256=norm_hash,
        source_evidence_months=list(ALLOWED[symbol]),
        original_h1_source_sha256=EXPECTED_H1_SHA.get(symbol),
        q2_eligible_volatility_tail=q2_tail.tolist(),
        q2_eligible_volatility_tail_sha256=array_sha256(q2_tail),
        garch_retained_refits=enriched.attrs.get("garch_retained_refits",[]),
        universe_sha256=file_sha256(archive),
        m5_parquet_sha256=file_sha256(parquet),
        m5_parquet_filename=parquet.name,
        arrays_sha256={key:array_sha256(arrays[key],dtype=TYPES[key])
            for key in FIELDS},
        original_founder_only_for_universe_features=True,
        model_genomes_loaded=False,model_pfs_evaluated=False,
        replay_performed=False,
        output_rows=len(t),source_audit=source_audit,
        note="Only historical native Q3 features. PR44 OOS is separately authorized; no PF/DD or model selection here.")
    write_json(manifest,meta)
    print("PR45_Q3_DATA_READY "+json.dumps({
        "symbol":symbol,"q3_m5_rows":len(q3_bars),
        "q3_m5_sha256":meta["m5_parquet_sha256"],
        "q3_universe_rows":len(t),"q3_universe_sha256":meta["universe_sha256"],
        "q3_months":meta["months"],"model_evaluations":0,
    },allow_nan=False),flush=True)
    return meta


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["acquire","build"])
    p.add_argument("--symbol",required=True,choices=SYMBOLS)
    p.add_argument("--raw-dir",type=Path,default=Path("reports/pr45-raw"))
    p.add_argument("--source-manifest",type=Path)
    p.add_argument("--h1-ohlcv",type=Path)
    p.add_argument("--h1-manifest",type=Path)
    p.add_argument("--h1-universe",type=Path)
    p.add_argument("--output",type=Path,default=OUTPUT)
    a=p.parse_args()
    if a.phase=="acquire":
        target=a.source_manifest or (a.output/a.symbol.lower()/"source-manifest.json")
        acquire(symbol=a.symbol,raw_dir=a.raw_dir,output=target)
    else:
        source=a.source_manifest or (a.output/a.symbol.lower()/"source-manifest.json")
        build(symbol=a.symbol,raw_dir=a.raw_dir,source_manifest=source,
              h1_ohlcv=a.h1_ohlcv,h1_manifest=a.h1_manifest,
              h1_universe=a.h1_universe,output=a.output)
