"""Cabeza de admisión M5 evolutiva, separada del Gate de Convicción Pura.

Proxies de la microestructura M1/M5 (NO order book ni bid/ask reales).
Cada proxy utiliza exclusivamente minutos cerrados antes de decidir en t+5m.
La cabeza genética no conoce outcomes ni modifica dirección, cupo o DD.
"""
import numpy as np
import pandas as pd

PROXY_NAMES = (
    "m1_pressure", "m1_last_move", "m1_up_fraction", "m1_volume_pressure",
    "m5_body", "m5_close_location", "m5_range_pips", "m5_previous_body",
)
GENE_COUNT = len(PROXY_NAMES) + 3
ENTRY_SCALE = 4.0
MAX_MARGIN_RELAXATION = 0.95


def microstructure_proxies(minutes, m5):
    """Devuelve proxies alineados con apertura M5: incluyen únicamente 5 M1 ya cerradas."""
    if (minutes.empty or m5.empty or not minutes.timestamp.is_monotonic_increasing
            or minutes.timestamp.duplicated().any()
            or not m5.timestamp.is_monotonic_increasing or m5.timestamp.duplicated().any()):
        raise ValueError("M1/M5 requieren índices UTC ordenados y únicos")
    minute_time = pd.DatetimeIndex(pd.to_datetime(minutes.timestamp, utc=True))
    bucket = minute_time.floor("5min")
    movement = (minutes.close.to_numpy(float) - minutes.open.to_numpy(float))
    volume = minutes.volume.to_numpy(float)
    if not np.isfinite(movement).all() or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("M1 inválido")
    one = pd.DataFrame(dict(bucket=bucket, movement=movement, absolute=np.abs(movement),
                            up=(movement > 0).astype(float), volume=volume,
                            signed_volume=np.sign(movement) * volume))
    g = one.groupby("bucket", sort=True)
    summary = g.agg(count=("movement", "size"), signed=("movement", "sum"),
                    absolute=("absolute", "sum"), last=("movement", "last"),
                    up=("up", "mean"), volume=("volume", "sum"),
                    signed_volume=("signed_volume", "sum"))
    opens = pd.DatetimeIndex(pd.to_datetime(m5.timestamp, utc=True))
    complete = summary.reindex(opens)
    if complete["count"].isna().any() or (complete["count"] != 5).any():
        raise ValueError("La microestructura exige cinco M1 completas por cada M5")
    gross = complete["absolute"].to_numpy(float)
    pressure = np.divide(complete["signed"].to_numpy(float), gross, out=np.zeros(len(gross)),
                         where=gross > 0)
    last = np.divide(complete["last"].to_numpy(float), gross, out=np.zeros(len(gross)),
                     where=gross > 0)
    volumes = complete["volume"].to_numpy(float)
    flow = np.divide(complete["signed_volume"].to_numpy(float), volumes,
                     out=np.zeros(len(volumes)), where=volumes > 0)
    o,h,l,c = (m5[name].to_numpy(float) for name in ("open","high","low","close"))
    span = np.maximum(h - l, 1e-10)
    body = (c-o)/span
    location = np.clip(2*(c-l)/span - 1, -1, 1)
    candle_range = np.tanh((h-l) / c * 10000. / 10.)
    prev_body = pd.Series(body).shift(1).fillna(0.).to_numpy()
    data = np.column_stack((
        pressure, last, 2*complete["up"].to_numpy(float)-1, flow, body,
        location, candle_range, prev_body,
    ))
    if not np.isfinite(data).all() or (np.abs(data) > 1.000001).any():
        raise ValueError("Proxies M1/M5 inválidos")
    return pd.DataFrame(data, index=opens, columns=PROXY_NAMES)


def dynamic_upstream(entry, margin, proxies, volatility, genes, legacy_threshold):
    """Zero genome reproduce EXACTAMENTE el filtro legacy congelado.

    La desviación entry es una cabeza causal tanh; el umbral de margen puede
    reducirse según la volatilidad relativa, sin añadir veto de volatilidad.
    """
    entry = np.asarray(entry, dtype=float)
    margin = np.asarray(margin, dtype=float)
    proxies = np.asarray(proxies, dtype=float)
    sigma = np.asarray(volatility, dtype=float)
    genes = np.asarray(genes, dtype=float)
    n = len(entry)
    if (entry.shape != (n,) or margin.shape != (n,) or sigma.shape != (n,)
            or proxies.shape != (n, len(PROXY_NAMES)) or genes.shape != (GENE_COUNT,)
            or not np.isfinite(entry).all() or not np.isfinite(margin).all()
            or not np.isfinite(proxies).all() or not np.isfinite(sigma).all()
            or not np.isfinite(genes).all() or not np.isfinite(legacy_threshold)
            or legacy_threshold <= 0 or (sigma < 0).any()):
        raise ValueError("La cabeza dinámica requiere scores causales y genes válidos")
    entry_adjustment = ENTRY_SCALE * np.tanh(proxies @ genes[:len(PROXY_NAMES)] + genes[-3])
    # Se fija una referencia histórica local, exclusiva de las últimas 48 velas.
    past_median = pd.Series(np.maximum(sigma, 1e-12)).shift().rolling(
        48, min_periods=1
    ).median().fillna(float(max(sigma[0], 1e-12))).to_numpy()
    relative_vol = np.clip(np.log(np.maximum(sigma, 1e-12) / past_median), -4., 4.)
    margin_relaxation = MAX_MARGIN_RELAXATION * np.maximum(
        0., np.tanh(genes[-2] + genes[-1]*relative_vol)
    )
    cutoff = legacy_threshold * (1. - margin_relaxation)
    accepts = (entry + entry_adjustment > 0.) & (margin > 0.) & (margin > cutoff)
    return accepts, cutoff


def legacy_upstream(entry, margin, threshold):
    return (np.asarray(entry)>0.) & (np.asarray(margin)>0.) & (np.asarray(margin)>threshold)
