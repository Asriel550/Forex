"""Admisión causal por convicción relativa y madurez observada."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import MarketConfig


CONVICTION_WINDOW = 500
PERCENTILE_STAGES = (.85, .60, .30, 0.)
MASTERY_CONVICTION = .80
MASTERY_ACCURACY = .95
MASTERY_BLOCKS = 3
MASTERY_REGIME_MINIMUM = 40
MASTERY_STEP_SETTLEMENTS = 100
VOLATILITY_HOURS = 48
VOLATILITY_PERCENTILE = .80
VOLATILITY_MINIMUM_BARS = 48


@dataclass(frozen=True)
class AdaptiveDecision:
    reason: str
    conviction: float
    mean_conviction: float | None
    cutoff: float | None
    percentile: float
    mastery_evidence: bool
    extreme_volatility: bool | None


def conviction(score):
    if not np.isfinite(score):
        raise ValueError('El puntaje de dirección debe ser finito')
    return float(abs(np.tanh(score)))


class AdaptiveGate:
    """Cada decisión consulta solo propuestas y liquidaciones anteriores."""

    def __init__(self, *, use_volatility=True, elastic=True, warmup_scores=()):
        self.use_volatility = use_volatility
        self.elastic = elastic
        self._convictions = deque((conviction(s) for s in warmup_scores), maxlen=CONVICTION_WINDOW)
        self._settled = deque(maxlen=CONVICTION_WINDOW)
        self._resolved = 0
        self._mastery_since = None

    def observe_settlement(self, won, extreme_volatility):
        """Registra una propuesta ya vencida, incluso si el gate la rechazó."""
        if won is not None:
            self._settled.append((bool(won), extreme_volatility))
            self._resolved += 1

    def _mastered(self, mean_conviction):
        if (not self.elastic or mean_conviction is None or mean_conviction <= MASTERY_CONVICTION
                or len(self._settled) < CONVICTION_WINDOW):
            return False
        hits = np.fromiter((hit for hit, _ in self._settled), dtype=bool)
        if any(block.mean() < MASTERY_ACCURACY for block in np.array_split(hits, MASTERY_BLOCKS)):
            return False
        for regime in (False, True):
            sample = [hit for hit, extreme in self._settled if extreme == regime]
            if len(sample) < MASTERY_REGIME_MINIMUM or np.mean(sample) < MASTERY_ACCURACY:
                return False
        return True

    def decide(self, score, volatility=np.nan, ceiling=np.nan):
        current = conviction(score)
        ready = len(self._convictions) == CONVICTION_WINDOW
        mean = float(np.mean(self._convictions)) if ready else None
        extreme = bool(volatility > ceiling) if np.isfinite([volatility, ceiling]).all() else None
        mastered = self._mastered(mean)
        if mastered and self._mastery_since is None:
            self._mastery_since = self._resolved
        elif not mastered:
            self._mastery_since = None
        confirmed = 0 if self._mastery_since is None else self._resolved - self._mastery_since
        stage = min(confirmed // MASTERY_STEP_SETTLEMENTS, len(PERCENTILE_STAGES) - 1)
        percentile = PERCENTILE_STAGES[stage]
        cutoff = (float(np.quantile(self._convictions, percentile)) if percentile else 0.) if ready else None
        reason = 'accepted'
        if not ready:
            reason = 'conviction_warmup'
        elif percentile and current <= cutoff:
            reason = 'conviction'
        elif self.use_volatility and extreme is None:
            reason = 'volatility_warmup'
        elif self.use_volatility and percentile and extreme:
            reason = 'volatility'
        self._convictions.append(current)
        return AdaptiveDecision(reason, current, mean, cutoff, percentile, mastered, extreme)


def rolling_m15_regime(closed_volatility, decisions):
    """Alinea la última vela M15 cerrada con el p80 de sus 48 h anteriores."""
    m15, m5 = MarketConfig('M15'), MarketConfig('M5')
    index = closed_volatility.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or (index != index.floor(m15.frequency)).any()
            or np.isinf(closed_volatility).any() or (closed_volatility.dropna() < 0.).any()):
        raise ValueError('La volatilidad M15 requiere cierres ordenados y válidos')
    history = closed_volatility.rolling(pd.Timedelta(hours=VOLATILITY_HOURS), closed='left',
                                       min_periods=VOLATILITY_MINIMUM_BARS)
    p80 = history.quantile(VOLATILITY_PERCENTILE)
    last = index.get_indexer(decisions, method='pad', tolerance=m15.step - m5.step)
    aligned = np.full((len(last), 2), np.nan)
    ready = last >= 0
    aligned[ready] = np.column_stack((closed_volatility, p80))[last[ready]]
    return pd.DataFrame(aligned, index=decisions, columns=['volatility', 'ceiling'])
