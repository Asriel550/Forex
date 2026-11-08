"""Huellas binarias reproducibles de feeds M1/M5 y features cuantitativas.

No depende del CSV formateado ni del orden de las claves JSON; detecta un cambio
de un timestamp, un precio, una volatilidad o una probabilidad MC.
"""
from hashlib import sha256

import numpy as np
import pandas as pd


def array_sha256(values, *, dtype="<f8"):
    data = np.asarray(values, dtype=dtype)
    if data.dtype.kind == "f" and np.isinf(data).any():
        raise ValueError("Infinito en un snapshot de mercado")
    return sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def market_frame_fingerprint(frame, *, columns, timestamp="timestamp"):
    if (frame.empty or timestamp not in frame or not frame[timestamp].is_unique
            or not frame[timestamp].is_monotonic_increasing):
        raise ValueError("La huella requiere velas ordenadas y timestamps únicos")
    times = pd.DatetimeIndex(pd.to_datetime(frame[timestamp], utc=True)).as_unit("ns")
    values = frame.loc[:, list(columns)].to_numpy(dtype="<f8")
    return dict(
        rows=len(frame), first_utc=times[0].isoformat(), last_utc=times[-1].isoformat(),
        timestamps_sha256=array_sha256(times.asi8, dtype="<i8"),
        values_sha256=array_sha256(values),
        columns=list(columns),
    )


def admissions_fingerprint(rows, *, step):
    if rows.empty or not rows.index.is_unique or not rows.index.is_monotonic_increasing:
        raise ValueError("Admisiones no ordenadas o vacías")
    opens = (rows.index - step).as_unit("ns")
    return dict(
        proposals=len(rows), opens_sha256=array_sha256(opens.asi8, dtype="<i8"),
        outcome_sha256=array_sha256(rows.outcome.to_numpy(), dtype="<i1"),
    )


def canonical_open_diff(opens, canonical):
    got = pd.DatetimeIndex(opens).as_unit("ns").asi8
    expected = pd.DatetimeIndex(canonical).as_unit("ns").asi8
    missing = np.setdiff1d(expected, got)
    extra = np.setdiff1d(got, expected)
    return dict(
        missing_count=len(missing), unexpected_count=len(extra),
        missing_open_utc=[pd.Timestamp(t, unit="ns", tz="UTC").isoformat() for t in missing[:20]],
        unexpected_open_utc=[pd.Timestamp(t, unit="ns", tz="UTC").isoformat() for t in extra[:20]],
    )
