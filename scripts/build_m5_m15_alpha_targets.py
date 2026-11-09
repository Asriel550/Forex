"""Targets supervisados alternativos sobre OHLC M5 y ATR de M15 CERRADA.

Las etiquetas consultan un horizonte futuro de 6 velas M5 (30 min) SOLO
durante entrenamiento IS Q1/Q2. JAMÁS entran en las features de decisión.
M1 solo es insumo técnico de agregación OHLCV, nunca un rasgo del modelo.
Si TP y SL aparecen en la misma M5, se presupone SL primero (conservador).
El replay operativo sigue siendo el contrato binario M5 de +80%/-100%;
estas etiquetas NO son un backtest de órdenes R:R con fills reales.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_adaptive_gate import RAW
from scripts.run_m5_direction_ablation import load_minutes
from scripts.run_m5_semester_replay import END, PERIODS, START, SOURCE_MANIFEST, assert_isolated, verify_source_periods
from scripts.build_m5_full_is_universe import load_universe
from scripts.verify_m5_durable_snapshot import verify as verify_golden
from src.artifacts import file_sha256
from src.reproducibility import market_frame_fingerprint, array_sha256
from src.resampling import aggregate
from src.train import write_json


SCHEMA="pr38_m5_m15_future_train_labels_v1"
M5_NS=5*60*10**9
M15_NS=15*60*10**9
FORWARD_STEPS=6
EXPANSION_ATR_MULTIPLIER=1.0
STOP_ATR_MULTIPLIER=.50
TAKE_ATR_MULTIPLIER=.75
OUTPUT=Path("reports/m5-alpha-targets")
FIELDS=("atr_m15","atr_expansion_label","rr15_label","valid_label","low_momentum")


def atr_on_closed_m15(m15, times):
    """ATR14 conocida al cierre de la M15, nunca en su formación."""
    high=m15.high.to_numpy(dtype=float)
    low=m15.low.to_numpy(dtype=float)
    close=m15.close.to_numpy(dtype=float)
    prev=np.r_[close[0], close[:-1]]
    true_range=np.maximum(high-low,np.maximum(np.abs(high-prev),np.abs(low-prev)))
    tr=pd.Series(true_range).rolling(14,min_periods=14).mean().to_numpy()
    closes=pd.DatetimeIndex(m15.timestamp).as_unit("ns").asi8+M15_NS
    idx=np.searchsorted(closes,np.asarray(times,dtype="<i8"),side="right")-1
    value=np.full(len(times),np.nan,dtype="<f8")
    valid=idx>=0
    value[valid]=tr[idx[valid]]
    return value


def targets_from_m5(m5, m15, decision_times, timeframe):
    """Expansión y primero tocar TP/SL con reglas de ambigüedad conservadoras."""
    times=np.asarray(decision_times,dtype="<i8")
    tf=np.asarray(timeframe,dtype=float)
    if (tf.shape != (len(times),8) or len(times)<1
            or not np.isfinite(tf).all() or
            not (np.diff(times)>0).all()):
        raise ValueError("Índice y features M5/M15 inválidas")
    o=pd.DatetimeIndex(pd.to_datetime(m5.timestamp,utc=True)).as_unit("ns").asi8
    high=m5.high.to_numpy(dtype=float)
    low=m5.low.to_numpy(dtype=float)
    close=m5.close.to_numpy(dtype=float)
    a=atr_on_closed_m15(m15,times)
    expansion=np.zeros(len(times),dtype=np.int8)
    rr=np.zeros(len(times),dtype=np.int8)
    valid=np.zeros(len(times),dtype=bool)
    # Momentum es una feature contemporánea permitida, no etiqueta futura.
    low_momentum=((np.abs(tf[:,0])<.18) & (np.abs(tf[:,4])<.18))
    for i,t in enumerate(times):
        if not np.isfinite(a[i]) or a[i]<=0 or t+FORWARD_STEPS*M5_NS>END.value:
            continue
        prev=np.searchsorted(o,t-M5_NS)
        start=np.searchsorted(o,t)
        if (prev>=len(o) or o[prev]!=t-M5_NS
                or start+FORWARD_STEPS>len(o) or o[start]!=t):
            continue
        future_opens=o[start:start+FORWARD_STEPS]
        if not np.array_equal(future_opens,t+np.arange(FORWARD_STEPS)*M5_NS):
            continue
        anchor=close[prev]
        highs=high[start:start+FORWARD_STEPS]
        lows=low[start:start+FORWARD_STEPS]
        if not np.isfinite(anchor) or anchor<=0:
            continue
        valid[i]=True
        upper=max(0.,float(highs.max()-anchor))
        lower=max(0.,float(anchor-lows.min()))
        if upper>=EXPANSION_ATR_MULTIPLIER*a[i] and upper>1.2*lower:
            expansion[i]=1
        elif lower>=EXPANSION_ATR_MULTIPLIER*a[i] and lower>1.2*upper:
            expansion[i]=-1

        stop=STOP_ATR_MULTIPLIER*a[i]
        take=TAKE_ATR_MULTIPLIER*a[i]
        # True if TP happened BEFORE SL on a later full candle.
        def wins(direction):
            for hh,ll in zip(highs,lows,strict=True):
                if direction==1:
                    if ll<=anchor-stop: return False   # SL first if both inside same M5
                    if hh>=anchor+take: return True
                else:
                    if hh>=anchor+stop: return False
                    if ll<=anchor-take: return True
            return False
        long_win,short_win=wins(1),wins(-1)
        if long_win != short_win:
            rr[i]=1 if long_win else -1
    return dict(
        atr_m15=np.nan_to_num(a,nan=0.).astype("<f8"),
        atr_expansion_label=expansion,
        rr15_label=rr,
        valid_label=valid,
        low_momentum=low_momentum,
    )


def build(*,universe,manifest,golden,golden_manifest,output=OUTPUT/"targets.npz",
          output_manifest=OUTPUT/"manifest.json"):
    assert_isolated(PERIODS)
    verify_golden(golden,golden_manifest)
    d,meta=load_universe(universe,manifest,golden,golden_manifest)
    sources=json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    verify_source_periods(sources)
    minutes=load_minutes(RAW,PERIODS,end=END)  # ingest only
    m5=aggregate(minutes,1,"M5",now=END)
    m15=aggregate(minutes,1,"M15",now=END)
    golden_meta=json.loads(Path(golden_manifest).read_text(encoding="utf-8"))
    fp=market_frame_fingerprint(m5,columns=("open","high","low","close","volume"))
    if fp!=golden_meta["fingerprints"]["m5"]:
        raise ValueError("M5 OHLCV fuera del snapshot canónico GARCH")
    data=targets_from_m5(m5,m15,d["row_times_ns"],d["timeframe"])
    if (len(data["valid_label"])!=len(d["outcome"])
            or data["valid_label"].sum()<10000
            or not set(np.unique(data["atr_expansion_label"])).issubset({-1,0,1})
            or not set(np.unique(data["rr15_label"])).issubset({-1,0,1})
            or np.any(data["atr_expansion_label"][~data["valid_label"]])
            or np.any(data["rr15_label"][~data["valid_label"]])):
        raise ValueError("Etiquetas IS incompletas o inválidas")
    target=Path(output); info=Path(output_manifest)
    if target.exists() or info.exists(): raise ValueError("No se sobrescribe dataset sellado")
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as f:
        np.savez_compressed(f,**data)
    fields={
        name:array_sha256(value,dtype=(
            "<f8" if name=="atr_m15" else "<i1" if name.endswith("_label") else "?"
        )) for name,value in data.items()
    }
    manifest_out=dict(
        schema=SCHEMA,is_universe_sha256=meta["universe_sha256"],
        original_garch_sha256=golden_meta["snapshot_sha256"],
        source_m5_sha256=fp["values_sha256"],labels_sha256=file_sha256(target),
        array_sha256=fields,signal_timeframes=["M5","M15"],
        horizon_minutes=30,atr_lookback_closed_m15=14,
        rr_stop_atr=.5,rr_take_atr=.75,
        same_m5_tp_sl_rule="assume_stop_first",
        available_training_rows=int(data["valid_label"].sum()),
        counts={
            name:{"up":int(np.count_nonzero(data[name]==1)),
                  "down":int(np.count_nonzero(data[name]==-1)),
                  "neutral":int(np.count_nonzero(data[name]==0))}
            for name in ("atr_expansion_label","rr15_label")
        },
        q3_accessed=False,train_start=START.isoformat(),train_end_exclusive=END.isoformat(),
        warning="Future labels for IS only; NEVER deploy as realtime indicators or infer R:R fills.",
    )
    write_json(info,manifest_out)
    print("PR38_TARGET_SEAL "+json.dumps({
        "sha256":manifest_out["labels_sha256"],"valid":manifest_out["available_training_rows"],
        "counts":manifest_out["counts"],"q3_accessed":False,
    },allow_nan=False),flush=True)


def load_targets(path,manifest,universe_sha):
    info=json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (info.get("schema")!=SCHEMA or info.get("is_universe_sha256")!=universe_sha
            or info.get("q3_accessed") is not False
            or info.get("signal_timeframes")!=["M5","M15"]
            or info.get("train_end_exclusive")!=END.isoformat()
            or info.get("horizon_minutes")!=30
            or info.get("labels_sha256")!=file_sha256(path)):
        raise ValueError("Dataset de etiquetas incompatible con IS sellado")
    with np.load(path,allow_pickle=False) as src:
        if set(src.files)!=set(FIELDS):
            raise ValueError("Etiquetas tienen campos desconocidos")
        data={name:src[name].copy() for name in FIELDS}
    for key,value in data.items():
        dtype="<f8" if key=="atr_m15" else "<i1" if key.endswith("_label") else "?"
        if array_sha256(value,dtype=dtype)!=info["array_sha256"][key]:
            raise ValueError("Etiqueta alterada: "+key)
    if (not set(np.unique(data["atr_expansion_label"])).issubset({-1,0,1})
            or not set(np.unique(data["rr15_label"])).issubset({-1,0,1})
            or len(data["valid_label"])!=len(data["atr_m15"])
            or not np.isfinite(data["atr_m15"]).all()
            or np.any(data["rr15_label"][~data["valid_label"]])):
        raise ValueError("Etiqueta dañada o contaminada")
    return data,info


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("universe","manifest","golden","golden_manifest"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--target",type=Path,default=OUTPUT/"targets.npz")
    p.add_argument("--target-manifest",type=Path,default=OUTPUT/"manifest.json")
    p.add_argument("--verify-only",action="store_true")
    a=p.parse_args()
    if a.verify_only:
        _,info=load_targets(a.target,a.target_manifest,
            json.loads(a.manifest.read_text(encoding="utf-8"))["universe_sha256"])
        print("PR38_TARGET_VERIFIED "+info["labels_sha256"],flush=True)
    else:
        build(universe=a.universe,manifest=a.manifest,golden=a.golden,
              golden_manifest=a.golden_manifest,output=a.target,output_manifest=a.target_manifest)
