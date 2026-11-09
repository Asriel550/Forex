"""Cabeza primaria evolutiva sobre señales M5 + M15 exclusivamente.

M1 puede alimentar la agregación de precios del dataset, pero NO aporta
features directas ni decisiones. Los valores M5/M15 usan solo velas cerradas
al instante de la decisión (fin de la M5).
"""
import numpy as np
import pandas as pd

SIGNAL_NAMES = (
    "m5_body", "m5_close_location", "m5_range", "m5_previous_body",
    "m15_body", "m15_close_location", "m15_range", "m15_previous_body",
)
GENE_COUNT = len(SIGNAL_NAMES) + 3
ENTRY_SCALE = 4.0
MAX_MARGIN_RELAXATION = 0.95
M5_STEP = pd.Timedelta(minutes=5)
M15_STEP = pd.Timedelta(minutes=15)


def _ohlc_features(frame, *, step):
    """Cuerpo, cierre dentro de rango, rango normalizado y cuerpo previo causal."""
    names = ("timestamp", "open", "high", "low", "close")
    if frame.empty or not set(names).issubset(frame.columns):
        raise ValueError("Se requieren velas OHLC completas")
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True))
    if (times.hasnans or not times.is_unique or not times.is_monotonic_increasing
            or not times.equals(times.floor(f"{int(step.total_seconds() // 60)}min"))):
        raise ValueError("Barras desordenadas o no alineadas")
    o, h, l, c = (frame[col].to_numpy(dtype=float) for col in names[1:])
    if (not np.isfinite(np.column_stack((o,h,l,c))).all()
            or (np.minimum.reduce((o,h,l,c)) <= 0).any()
            or (h < np.maximum.reduce((o,c,l))).any()
            or (l > np.minimum.reduce((o,c,h))).any()):
        raise ValueError("OHLC incoherente")
    extent = np.maximum(h-l, 1e-12)
    body = (c-o) / extent
    location = np.clip(2*(c-l)/extent - 1., -1., 1.)
    range_scaled = np.tanh((h-l)/c * 1000.)
    consecutive = np.r_[False, np.diff(times.as_unit("ns").asi8) == step.value]
    previous_body = np.r_[0., body[:-1]]
    previous_body[~consecutive] = 0.
    matrix = np.column_stack((body, location, range_scaled, previous_body))
    if not np.isfinite(matrix).all() or (np.abs(matrix) > 1.000001).any():
        raise ValueError("Features OHLC no finitos")
    return times, matrix


def timeframe_features(m5, m15):
    """M5 cerrada + última M15 CERRADA, sin mirar la M15 en formación.

    M5 abierta en t se decide en t+5. Solo se incluye una M15 cuando su
    apertura+15 <= t+5. Una M15 demasiado antigua se trata como neutral.
    """
    t5, f5 = _ohlc_features(m5, step=M5_STEP)
    t15, f15 = _ohlc_features(m15, step=M15_STEP)
    decision = t5 + M5_STEP
    closes = t15 + M15_STEP
    match = np.searchsorted(closes.as_unit("ns").asi8, decision.as_unit("ns").asi8, side="right") - 1
    valid = match >= 0
    valid[valid] &= (decision[valid] - closes[match[valid]] <= M15_STEP)
    aligned = np.zeros((len(t5), 4), dtype=float)
    aligned[valid] = f15[match[valid]]
    result = pd.DataFrame(np.column_stack((f5, aligned)),
                          index=t5.as_unit("ns"), columns=SIGNAL_NAMES)
    return result


def dynamic_upstream(entry, margin, timeframe, volatility, genes, legacy_threshold):
    """Cero genes reproduce exactamente la admisión heredada.

    Solo ajusta la cabeza de entrada y el umbral de margen; no toca
    la dirección, el gate p85, cupo horario ni límites de riesgo.
    """
    entry = np.asarray(entry, dtype=float)
    margin = np.asarray(margin, dtype=float)
    timeframe = np.asarray(timeframe, dtype=float)
    sigma = np.asarray(volatility, dtype=float)
    genes = np.asarray(genes, dtype=float)
    n = len(entry)
    if (entry.shape != (n,) or margin.shape != (n,) or sigma.shape != (n,)
            or timeframe.shape != (n, len(SIGNAL_NAMES)) or genes.shape != (GENE_COUNT,)
            or not np.isfinite(entry).all() or not np.isfinite(margin).all()
            or not np.isfinite(timeframe).all() or not np.isfinite(sigma).all()
            or not np.isfinite(genes).all() or not np.isfinite(legacy_threshold)
            or legacy_threshold <= 0 or (sigma < 0).any()):
        raise ValueError("La cabeza requiere features M5+M15 y genes finitos")
    adjustment = ENTRY_SCALE * np.tanh(timeframe @ genes[:len(SIGNAL_NAMES)] + genes[-3])
    past_median = pd.Series(np.maximum(sigma, 1e-12)).shift().rolling(
        48, min_periods=1
    ).median().fillna(float(max(sigma[0], 1e-12))).to_numpy()
    relative_vol = np.clip(np.log(np.maximum(sigma, 1e-12) / past_median), -4., 4.)
    relaxation = MAX_MARGIN_RELAXATION * np.maximum(
        0., np.tanh(genes[-2] + genes[-1]*relative_vol)
    )
    cutoff = legacy_threshold * (1. - relaxation)
    return (entry + adjustment > 0.) & (margin > 0.) & (margin > cutoff), cutoff


def legacy_upstream(entry, margin, threshold):
    return (np.asarray(entry) > 0.) & (np.asarray(margin) > 0.) & (np.asarray(margin) > threshold)
