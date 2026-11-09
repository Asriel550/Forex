"""Regímenes causales del percentil del Gate de Convicción (solo M5/M15).

El GARCH de M5 está congelado; el percentil dinámico usa exclusivamente
volatilidad de la vela ya cerrada y hasta 48 observaciones PREVIAS.
No utiliza resultados futuros para decidir ni filtra operaciones por
volatilidad. Mantiene warmup de 500, liquidaciones maduras y elasticidad.
"""
from collections import deque

import numpy as np
import pandas as pd

from src.adaptive_gate import (
    AdaptiveGate, CONVICTION_WINDOW, MASTERY_ACCURACY,
)
from src.evolution_lineage import MARKET


MODES = ("p85", "p80", "p75", "volatility_dynamic")
PERCENTILES = {"p85": .85, "p80": .80, "p75": .75}


def causal_volatility_percentiles(volatility):
    sigma = np.asarray(volatility, dtype=float)
    if (sigma.ndim != 1 or len(sigma) < 1 or
            not np.isfinite(sigma).all() or (sigma < 0).any()):
        raise ValueError("Volatilidad M5 inválida")
    past = pd.Series(np.maximum(sigma, 1e-12)).shift(1).rolling(
        48, min_periods=24
    ).median().to_numpy()
    ratio = sigma / np.where(np.isfinite(past), past, np.maximum(sigma, 1e-12))
    # Relajar solo en volatilidad moderada/baja; mayor severidad si hay expansión.
    return np.where(ratio >= 1.20, .85, np.where(ratio < .80, .75, .80))


def _possible_mastery(scores, outcomes):
    signed = np.where(scores >= 0., 1, -1)
    settled = np.asarray(outcomes) != 0
    wins = (signed[settled] == np.asarray(outcomes)[settled]).astype(np.int64)
    if len(wins) < CONVICTION_WINDOW:
        return False
    cumulative = np.r_[0, np.cumsum(wins)]
    return bool(np.any(
        cumulative[CONVICTION_WINDOW:] - cumulative[:-CONVICTION_WINDOW]
        >= CONVICTION_WINDOW * MASTERY_ACCURACY
    ))


def _exact_adaptive_reasons(rows, scores, selected_percentile):
    """Fallback completo con masterización, asentamientos y etapas elásticas."""
    gate = AdaptiveGate(percentile=.85, elastic=True)
    pending = deque()
    reasons = []
    for time, score, outcome, percentile in zip(
        rows.index, scores, rows.outcome, selected_percentile, strict=True
    ):
        while pending and pending[0][0] <= time:
            _, won = pending.popleft()
            gate.observe_settlement(won)
        gate._stages = (float(percentile), .60, .30, 0.)
        reasons.append(gate.decide(score).reason)
        won = None if outcome == 0 else bool((1 if score >= 0 else -1) == outcome)
        pending.append((time + MARKET.step, won))
    return np.asarray(reasons, dtype="<U24")


def gate_reasons(rows, scores, volatility, mode):
    """Razones del gate en cada timestamp, con cuantiles shift(1) causales."""
    if mode not in MODES:
        raise ValueError("Modo gate inválido")
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(rows),) or not np.isfinite(scores).all()
            or not rows.index.is_monotonic_increasing or not rows.index.is_unique
            or not rows.outcome.isin([-1, 0, 1]).all()):
        raise ValueError("Filas del gate inválidas")
    if mode != "volatility_dynamic":
        from scripts.run_m5_temporal_evolution import causal_reasons
        return causal_reasons(rows, scores, PERCENTILES[mode])
    percentiles = causal_volatility_percentiles(volatility)
    if percentiles.shape != scores.shape:
        raise ValueError("Volatilidad no alineada con scores de dirección")
    if _possible_mastery(scores, rows.outcome.to_numpy()):
        return _exact_adaptive_reasons(rows, scores, percentiles)
    conv = np.abs(np.tanh(scores))
    roll = pd.Series(conv).rolling(CONVICTION_WINDOW)
    cuts = np.where(
        percentiles == .75, roll.quantile(.75).shift(1).to_numpy(),
        np.where(
            percentiles == .80, roll.quantile(.80).shift(1).to_numpy(),
            roll.quantile(.85).shift(1).to_numpy(),
        ),
    )
    out = np.where(conv > cuts, "accepted", "conviction").astype("<U24")
    out[:CONVICTION_WINDOW] = "conviction_warmup"
    return out
