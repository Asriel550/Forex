"""Filtros causales de entrada para la evolución de binarias M5."""

import numpy as np
import pandas as pd

from .market import MarketConfig
from .resampling import aggregate

M5_STEP = MarketConfig('M5').step
TREND_TIMEFRAMES = ('H1', 'H4')
FAST_SPAN, SLOW_SPAN, RANGE_SPAN = 8, 24, 14
NEUTRAL_RANGE_FRACTION = .10
SESSION_BLACKOUTS = (
    ('America/New_York', 16 * 60 + 30, 18 * 60),
    ('Asia/Tokyo', 8 * 60, 10 * 60),
)
VOLATILITY_WINDOW = 12
VOLATILITY_QUANTILE = .20


def closed_trend_direction(frame, timeframe):
    """Tendencia al cierre M5 a partir de velas completas H1 o H4."""
    market = MarketConfig(timeframe)
    timestamp = frame.timestamp
    result = np.full(len(frame), np.nan)
    if not timestamp.dt.floor(market.frequency).value_counts().eq(market.step / M5_STEP).any():
        return result
    decisions = pd.DatetimeIndex(timestamp + M5_STEP)
    ohlc = frame[['timestamp', 'open', 'high', 'low', 'close']].assign(volume=0.)
    candles = aggregate(ohlc, 5, timeframe, now=decisions[-1])
    fast = candles.close.ewm(span=FAST_SPAN, adjust=False, min_periods=FAST_SPAN).mean()
    slow = candles.close.ewm(span=SLOW_SPAN, adjust=False, min_periods=SLOW_SPAN).mean()
    band = NEUTRAL_RANGE_FRACTION * (candles.high - candles.low).ewm(
        span=RANGE_SPAN, adjust=False, min_periods=RANGE_SPAN,
    ).mean()
    gap = fast - slow
    direction = np.select([gap > band, gap < -band], [1., -1.], default=0.)
    direction[gap.isna() | band.isna()] = np.nan
    closes = pd.DatetimeIndex(candles.timestamp + market.step)
    last = closes.get_indexer(decisions, method='pad', tolerance=market.step - M5_STEP)
    available = last >= 0
    result[available] = direction[last[available]]
    return result


def session_allowed(bar_opens):
    """Comprueba entrada y vencimiento contra ventanas locales [inicio, fin)."""
    entry = pd.DatetimeIndex(bar_opens) + M5_STEP
    expiry = entry + M5_STEP
    allowed = np.ones(len(entry), dtype=bool)
    for zone, start, end in SESSION_BLACKOUTS:
        for instant in (entry, expiry):
            local = instant.tz_convert(zone)
            minutes = local.hour * 60 + local.minute
            allowed &= ~((minutes >= start) & (minutes < end))
    return allowed


def realized_volatility(frame):
    """Raíz de la suma de retornos log M5 al cuadrado durante la última hora."""
    consecutive = frame.timestamp.diff().eq(M5_STEP)
    squares = np.log(frame.close).diff().pow(2).where(consecutive)
    return np.sqrt(squares.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).sum()).to_numpy()


def build_entry_context(frame, train_end):
    """Fija el piso de volatilidad con velas de train en sesiones permitidas."""
    timestamps = pd.DatetimeIndex(frame.timestamp).as_unit('ns')
    sessions = session_allowed(timestamps)
    volatility = realized_volatility(frame)
    fit = (timestamps <= train_end) & sessions & np.isfinite(volatility)
    if not fit.any():
        raise ValueError('Train carece de ventanas M5 para ajustar el piso de volatilidad')
    floor = float(np.quantile(volatility[fit], VOLATILITY_QUANTILE))
    context = pd.DataFrame({
        **{tf: closed_trend_direction(frame, tf) for tf in TREND_TIMEFRAMES},
        'session_allowed': sessions, 'realized_volatility': volatility,
        'volatility_allowed': (volatility > 0.) & (volatility >= floor),
    }, index=timestamps)
    settings = dict(
        trend=dict(timeframes=list(TREND_TIMEFRAMES), fast_span=FAST_SPAN, slow_span=SLOW_SPAN,
                   range_span=RANGE_SPAN, neutral_range_fraction=NEUTRAL_RANGE_FRACTION,
                   max_age_minutes_exclusive={'H1': 60, 'H4': 240}),
        session_blackouts=[dict(timezone=zone, start_local=f'{start // 60:02d}:{start % 60:02d}',
                                end_local=f'{end // 60:02d}:{end % 60:02d}')
                           for zone, start, end in SESSION_BLACKOUTS],
        realized_volatility=dict(window_m5=VOLATILITY_WINDOW, train_quantile=VOLATILITY_QUANTILE,
                                 floor=floor, fit_rows=int(fit.sum()), fit_end=train_end.isoformat()),
    )
    return context, settings


def filter_setups(context, entries, actions, eligible):
    """Devuelve máscara y conteos secuenciales sobre entradas ya válidas para MC."""
    side = np.where(actions == 2, 1, -1)
    allowed = np.ones(len(entries), dtype=bool)
    stages, rejected, regimes = {'MC': len(entries)}, {}, {}
    for timeframe in TREND_TIMEFRAMES:
        directions = context[timeframe].to_numpy(dtype=float)
        trend = directions[entries]
        rejected[timeframe] = dict(opposite=int((allowed & (trend == -side)).sum()),
                                   lateral=int((allowed & (trend == 0)).sum()),
                                   unavailable=int((allowed & np.isnan(trend)).sum()))
        allowed &= trend == side
        stages[timeframe] = int(allowed.sum())
        observed = directions[eligible]
        regimes[timeframe] = dict(up=int((observed == 1).sum()), down=int((observed == -1).sum()),
                                  lateral=int((observed == 0).sum()), unavailable=int(np.isnan(observed).sum()))
    for stage in ('session', 'volatility'):
        passes = context[f'{stage}_allowed'].to_numpy(dtype=bool)[entries]
        rejected[stage] = int((allowed & ~passes).sum())
        allowed &= passes
        stages[stage] = int(allowed.sum())
    return allowed, dict(stages=stages, rejected=rejected, regimes=regimes)
