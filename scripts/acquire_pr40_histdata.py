"""PR40: acquire GBPUSD/AUDUSD/USDCAD HistData Jan-Jun 2026, make M5/M15.

HistData distributes Generic ASCII M1 rather than native M5/M15 OHLCV.
The M1 files are RAW INPUT TO RESAMPLING ONLY, never features/signals.
No July/September files; any UTC >= 2026-07-01 data from a June local
archive is filtered out *before* candle engineering, not assessed.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import requests

from scripts.acquire_histdata import Forms
from scripts.build_histdata_dataset import read_native_minutes
from src.artifacts import file_sha256
from src.resampling import aggregate
from src.reproducibility import market_frame_fingerprint
from src.train import write_json

SYMBOLS=("GBPUSD","AUDUSD","USDCAD")
MONTHS=tuple(f"2026{m:02d}" for m in range(1,7))
START=pd.Timestamp("2026-01-01T00:00:00Z")
Q2=pd.Timestamp("2026-04-01T00:00:00Z")
END=pd.Timestamp("2026-07-01T00:00:00Z")
TIMEZONE="EST fixed UTC-05:00 without DST"
BASE="https://www.histdata.com"
SCHEMA="pr40_histdata_m5_m15_h1_ohlcv_v1"
COLUMNS=("timestamp","open","high","low","close","volume")
NPZ_COLUMNS=("timestamp_ns","open","high","low","close","volume")


def url_for(symbol,period):
    if symbol not in SYMBOLS or period not in MONTHS:
        raise ValueError("Q3 y símbolos no autorizados para HistData PR40")
    return (f"{BASE}/download-free-forex-historical-data/?/"
            f"ascii/1-minute-bar-quotes/{symbol.lower()}/2026/{int(period[-2:])}")


def parse_source_form(html,symbol,period):
    if symbol not in SYMBOLS or period not in MONTHS:
        raise ValueError("Fuente inválida")
    parser=Forms()
    parser.feed(html)
    matches=[
        x for x in parser.forms
        if x["fields"].get("timeframe","").upper()=="M1"
        and x["fields"].get("fxpair","").upper()==symbol
        and x["fields"].get("tk")
    ]
    if len(matches)!=1:
        raise ValueError(f"HistData no publicó un formulario M1 único para {symbol} {period}")
    fields=matches[0]["fields"]
    if any(k.lower() in {"202607","202608","202609"} for k in fields.values()):
        raise ValueError("Formulario fuera del periodo IS")
    return fields


def verify_zip(path,symbol,period):
    path=Path(path)
    if not path.is_file() or path.stat().st_size<10000 or path.stat().st_size>30*1024*1024:
        raise ValueError(f"HistData ZIP inválido o tamaño anómalo: {symbol} {period}")
    try:
        with ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError("CRC inválido: "+str(path))
            names=[name for name in z.namelist() if name.lower().endswith(".csv")]
            if len(names)!=1 or names[0].split("/")[-1].upper()!=(
                f"DAT_ASCII_{symbol}_M1_{period}.CSV"
            ):
                raise ValueError(f"HistData CSV no corresponde a {symbol}/{period}: {names}")
            if z.getinfo(names[0]).file_size>80*1024*1024:
                raise ValueError("Entrada CSV sobredimensionada")
            meta=[dict(name=x.filename,uncompressed_bytes=x.file_size)
                  for x in z.infolist()]
    except BadZipFile as exc:
        raise ValueError("Descarga no-ZIP de HistData") from exc
    return meta


def acquire_month(raw_dir,symbol,period,*,session=None):
    """Fetch source with explicit ISO 2026 H1 whitelist and content integrity."""
    address=url_for(symbol,period)
    raw_dir=Path(raw_dir)
    raw_dir.mkdir(parents=True,exist_ok=True)
    target=raw_dir/f"HISTDATA_COM_ASCII_{symbol}_M1_{period}.zip"
    if target.is_file():
        entries=verify_zip(target,symbol,period)
    else:
        client=session or requests.Session()
        client.headers.update({"User-Agent":"Forex-PR40-historical-data-research/1.0"})
        result=client.get(address,timeout=(35,100))
        result.raise_for_status()
        form=parse_source_form(result.text,symbol,period)
        payload=client.post(
            f"{BASE}/get.php",data=form,
            headers={"Referer":address,"Origin":BASE},timeout=(25,150)
        )
        payload.raise_for_status()
        if len(payload.content)>30*1024*1024:
            raise ValueError("Respuesta ZIP demasiado grande")
        temp=target.with_suffix(".part")
        temp.write_bytes(payload.content)
        try:
            entries=verify_zip(temp,symbol,period)
            temp.replace(target)
        finally:
            temp.unlink(missing_ok=True)
    return dict(
        symbol=symbol,period=period,
        provider="HistData.com",format="Generic ASCII M1 bid OHLCV",
        source_url=address,raw_role="aggregator input only, not an M1 model signal",
        zip_file=target.name,sha256=file_sha256(target),
        bytes=target.stat().st_size,entries=entries,
    )


def serialize_bars(frame):
    return {
        "timestamp_ns":pd.DatetimeIndex(frame.timestamp).as_unit("ns").asi8.astype("<i8"),
        **{key:frame[key].to_numpy(dtype="<f8") for key in COLUMNS[1:]}
    }


def validate_ohlc(frame,timeframe):
    minutes=5 if timeframe=="M5" else 15
    if len(frame)<5000:
        raise ValueError(f"Demasiado pocas velas {timeframe} para H1")
    t=pd.DatetimeIndex(frame.timestamp)
    if (not t.is_unique or not t.is_monotonic_increasing
            or t.tz is None or t.min()<START or t.max()+pd.Timedelta(minutes=minutes)>END
            or not t.equals(t.floor(f"{minutes}min"))):
        raise ValueError(f"Velas {timeframe} fuera del IS o mal alineadas")
    values=frame[list(COLUMNS[1:])].to_numpy(dtype=float)
    o,h,l,c=(frame[k].to_numpy(dtype=float) for k in ("open","high","low","close"))
    if (not np.isfinite(values).all() or (o<=0).any() or (h<np.maximum(o,c)).any()
            or (l>np.minimum(o,c)).any() or (h<l).any()):
        raise ValueError(f"OHLC {timeframe} inválido")
    for left,right in ((START,Q2),(Q2,END)):
        if not ((t>=left)&(t<right)).sum()>2500:
            raise ValueError(f"Q1/Q2 incompleto en velas {timeframe}")


def build(symbol,sources,raw_dir,output_dir):
    if symbol not in SYMBOLS:
        raise ValueError("Solo nuevos pares PR40; no tocar EURUSD original")
    if [x["period"] for x in sources]!=list(MONTHS) or any(
        x["symbol"]!=symbol or len(x["sha256"])!=64 for x in sources
    ):
        raise ValueError("Seis fuentes IS por par obligatorias")
    minutes,audit=read_native_minutes(Path(raw_dir),sources)
    # June is in fixed EST, so the provider's June ZIP can include July
    # UTC timestamps; exclude them BEFORE making even a single M5/M15 bar.
    outside=int((minutes.timestamp>=END).sum())
    minutes=minutes.loc[
        (minutes.timestamp>=START)&(minutes.timestamp<END)
    ].reset_index(drop=True)
    if minutes.empty or minutes.timestamp.min()>pd.Timestamp("2026-01-06T00:00Z"):
        raise ValueError("Falta inicio del semestre")
    m5=aggregate(minutes,1,"M5",now=END)
    m15=aggregate(minutes,1,"M15",now=END)
    for timeframe,bars in (("M5",m5),("M15",m15)):
        validate_ohlc(bars,timeframe)
    dest=Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    ohlc=dest/"ohlcv.npz"
    if ohlc.exists():raise ValueError("No sobrescribir dataset sellado")
    with ohlc.open("xb") as file:
        np.savez_compressed(
            file,**{f"{tf.lower()}_{field}":value
                    for tf,frame in (("M5",m5),("M15",m15))
                    for field,value in serialize_bars(frame).items()}
        )
    m5fp=market_frame_fingerprint(
        m5,columns=("open","high","low","close","volume")
    )
    m15fp=market_frame_fingerprint(
        m15,columns=("open","high","low","close","volume")
    )
    source_manifest=dict(
        protocol="PR40_histdata_raw_is_sources_v1",
        symbol=symbol,periods=list(MONTHS),sources=sources,
        raw_m1_used_for_aggregation_only=True,
        original_source_timezone=TIMEZONE,
        start_utc=START.isoformat(),end_exclusive_utc=END.isoformat(),
        q3_rows_excluded_before_aggregation=outside,q3_used_for_training_or_signals=False,
    )
    write_json(dest/"source-manifest.json",source_manifest)
    manifest=dict(
        schema=SCHEMA,symbol=symbol,
        provider="HistData.com",source_timeframe="M1_raw_aggregation_only",
        output_timeframes=["M5","M15"],signal_timeframes=["M5","M15"],
        months=list(MONTHS),q3_accessed=False,
        start_utc=START.isoformat(),end_exclusive_utc=END.isoformat(),
        utc_timestamp_convention="candle_open",
        original_source_timezone=TIMEZONE,
        q3_source_rows_rejected=int(outside),
        m1_trading_features=False,
        m5_rows=int(len(m5)),m15_rows=int(len(m15)),
        m5_start_utc=m5.timestamp.iloc[0].isoformat(),
        m5_last_open_utc=m5.timestamp.iloc[-1].isoformat(),
        m15_start_utc=m15.timestamp.iloc[0].isoformat(),
        m15_last_open_utc=m15.timestamp.iloc[-1].isoformat(),
        m5_ohlcv_fingerprint=m5fp,m15_ohlcv_fingerprint=m15fp,
        native_source_files=[x["zip_file"] for x in sources],
        native_source_sha256={x["period"]:x["sha256"] for x in sources},
        input_audit=audit,
        ohlcv_sha256=file_sha256(ohlc),
        limitations=(
            "This is a real HistData candle dataset only. Does NOT yet "
            "include EURUSD-trained neural inputs, per-pair causal GARCH, "
            "inference signals or measured portfolio returns."
        ),
    )
    write_json(dest/"manifest.json",manifest)
    print("PR40_HISTDATA_CANDLES "+json.dumps({
        "symbol":symbol,"m5":len(m5),"m15":len(m15),
        "sha256":manifest["ohlcv_sha256"],"utc_q3_rows_excluded":outside,
        "q3_used":False,
    },allow_nan=False),flush=True)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol",required=True,choices=SYMBOLS)
    parser.add_argument("--raw-dir",type=Path,default=Path("reports/pr40-histdata-raw"))
    parser.add_argument("--output-dir",type=Path,default=None)
    a=parser.parse_args()
    out=a.output_dir or Path("reports/pr40-histdata")/a.symbol.lower()
    archives=[]
    with requests.Session() as client:
        for month in MONTHS:
            archives.append(acquire_month(a.raw_dir/a.symbol.lower(),a.symbol,month,session=client))
            time.sleep(1.)
    build(a.symbol,archives,a.raw_dir/a.symbol.lower(),out)
