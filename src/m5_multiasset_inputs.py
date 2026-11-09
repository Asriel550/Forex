"""Validate per-symbol 2026 H1 M5/M15+GARCH inference universes.

Untrusted manifests are not proof of historical provenance by themselves:
every new pair needs independently reviewed, pinned native M5/M15 data,
causal GARCH refits, model inputs with the ORIGINAL EURUSD normalizer, and
an immutable Release SHA. There are currently no GBP/AUD/CAD archives.
No M1 model features; refuse Q3 and dataset substitutions.
"""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.artifacts import file_sha256
from src.evolution_multiscale import FEATURE_NAMES
from src.reproducibility import array_sha256
from src.upstream_filters import SIGNAL_NAMES
from src.m5_multiasset_portfolio import SYMBOLS,START,END


FORMAT="pr40_m5_m15_symbol_universe_v1"
FIELDS=("row_times_ns","outcome","volatility","entry","margin",
        "timeframe","direction_inputs","legacy_mask")
PERIODS=[f"2026{month:02d}" for month in range(1,7)]
HEX=re.compile(r"^[0-9a-f]{64}$")


def verify_new_asset(symbol,archive_path,manifest_path):
    if symbol not in SYMBOLS or symbol=="EURUSD":
        raise ValueError("El universo EUR/USD se verifica mediante su Release v2 PR36")
    meta=json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    checks=(
        meta.get("schema")==FORMAT,
        meta.get("symbol")==symbol,
        meta.get("signal_timeframes")==["M5","M15"],
        meta.get("source_timeframes")==["M5","M15"],
        meta.get("start_utc")==START.isoformat(),
        meta.get("end_exclusive_utc")==END.isoformat(),
        meta.get("is_months")==PERIODS,
        meta.get("q3_accessed") is False,
        meta.get("garch_m5_causal_refit") is True,
        meta.get("gen375_normalizer_frozen") is True,
        meta.get("gen375_direction_input_schema")==list(FEATURE_NAMES),
        meta.get("timeframe_signal_names")==list(SIGNAL_NAMES),
        meta.get("universe_sha256")==file_sha256(archive_path),
    )
    required_hashes=("source_m5_ohlcv_sha256","source_m15_ohlcv_sha256",
                     "garch_m5_recipe_sha256","gen375_normalizer_sha256")
    if not all(checks) or any(not HEX.fullmatch(str(meta.get(k,""))) for k in required_hashes):
        raise ValueError("Procedencia M5/M15 o esquema GARCH sin verificar para "+symbol)
    with np.load(archive_path,allow_pickle=False) as f:
        if set(f.files)!=set(FIELDS):
            raise ValueError("Array extra o ausente en símbolo "+symbol)
        arrays={k:f[k].copy() for k in FIELDS}
    n=len(arrays["row_times_ns"])
    if (n<10000 or arrays["direction_inputs"].shape!=(n,len(FEATURE_NAMES))
            or arrays["timeframe"].shape!=(n,len(SIGNAL_NAMES))
            or any(arrays[k].shape!=(n,) for k in
                   ("outcome","volatility","entry","margin","legacy_mask"))):
        raise ValueError("El archivo no cubre un semestre M5 completo "+symbol)
    dates=pd.to_datetime(arrays["row_times_ns"],utc=True,unit="ns")
    if (dates.min()<START or dates.max()>=END or not dates.is_monotonic_increasing
            or not dates.is_unique or not dates.equals(dates.floor("5min"))
            or not np.isin(arrays["outcome"],[-1,0,1]).all()
            or not np.isfinite(arrays["direction_inputs"]).all()
            or not np.isfinite(arrays["timeframe"]).all()
            or any(not np.isfinite(arrays[k]).all()
                   for k in ("volatility","entry","margin"))
            or (arrays["volatility"]<0).any()
            or arrays["legacy_mask"].dtype!=np.dtype("?")):
        raise ValueError("Causalidad H1/fin de M5 o features alteradas "+symbol)
    if (not (dates< pd.Timestamp("2026-04-01T00:00:00Z")).any()
            or not (dates>=pd.Timestamp("2026-04-01T00:00:00Z")).any()):
        raise ValueError("Falta trimestre Q1 o Q2 "+symbol)
    sizes={"row_times_ns":"<i8","outcome":"<i1","volatility":"<f8",
           "entry":"<f8","margin":"<f8","timeframe":"<f8",
           "direction_inputs":"<f8","legacy_mask":"?"}
    for key,dtype in sizes.items():
        if array_sha256(arrays[key],dtype=dtype)!=meta.get("arrays_sha256",{}).get(key):
            raise ValueError("Huella alterada "+key+" de "+symbol)
    hours=meta.get("months_observed_hours",{})
    if set(hours)!=set(PERIODS) or any(
        not isinstance(hours[k],(float,int)) or
        not np.isfinite(hours[k]) or hours[k]<=0 for k in PERIODS
    ):
        raise ValueError("Horas observadas H1 no certificadas")
    # These hashes authenticate bytes relative to a manifest. They still need
    # an independent code/data-lineage review before the research is valid.
    return arrays,meta
