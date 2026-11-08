"""Observaciones causales de contexto para la política evolutiva M5."""

import numpy as np
import pandas as pd

from .market import MarketConfig
from .resampling import aggregate

M5_STEP = MarketConfig('M5').step
TREND_TIMEFRAMES = ('H1', 'H4')
FAST_SPAN, SLOW_SPAN, RANGE_SPAN = 8, 24, 14
SESSION_WINDOWS = (
    ('ny_rollover', 'America/New_York', 16 * 60 + 30, 18 * 60),
    ('tokyo_early', 'Asia/Tokyo', 8 * 60, 10 * 60),
)
VOLATILITY_WINDOW = 12
M15_DIRECTION_CLIP = 3.
CONTEXT_FEATURES = (
    'h1_bias', 'h1_available', 'h4_bias', 'h4_available',
    'm5_realized_volatility', 'm5_volatility_available', 'm5_range_fraction',
    'ny_rollover', 'tokyo_early',
)
M15_TACTICAL = ('m15_realized_volatility', 'm15_atr_fraction')
H1_STRUCTURAL = ('h1_ema_slope', 'h1_range_fraction')
CONTEXT_SETTINGS = dict(
    trend=dict(fast_span=FAST_SPAN, slow_span=SLOW_SPAN, range_span=RANGE_SPAN,
               max_age_minutes_exclusive={'H1': 60, 'H4': 240}),
    realized_volatility_window_m5=VOLATILITY_WINDOW,
    session_windows=[dict(feature=name, timezone=zone, start_local=f'{start // 60:02d}:{start % 60:02d}',
                          end_local=f'{end // 60:02d}:{end % 60:02d}')
                     for name, zone, start, end in SESSION_WINDOWS],
)


def _closed_bar_index(opens, market, decisions):
    closes = pd.DatetimeIndex(opens + market.step)
    return closes.get_indexer(decisions, method='pad', tolerance=market.step - M5_STEP)


def closed_trend_bias(frame, timeframe):
    """Diferencia de EMA en unidades del rango de una vela completa."""
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
    average_range = (candles.high - candles.low).ewm(
        span=RANGE_SPAN, adjust=False, min_periods=RANGE_SPAN,
    ).mean()
    gap = fast - slow
    scale = average_range.to_numpy()
    bias = np.divide(gap.to_numpy(), scale, out=np.zeros(len(scale)), where=scale > 0.)
    bias[gap.isna() | average_range.isna()] = np.nan
    last = _closed_bar_index(candles.timestamp, market, decisions)
    available = last >= 0
    result[available] = bias[last[available]]
    return result


def m15_direction_bias(frame):
    """Sesgo M15 causal, en sus propias unidades y neutral cuando no está disponible."""
    bias = np.nan_to_num(closed_trend_bias(frame, 'M15'), nan=0.)
    return pd.Series(np.clip(bias, -M15_DIRECTION_CLIP, M15_DIRECTION_CLIP), index=frame.timestamp)


def native_direction_context(minutes, decisions, *, end):
    """Alinea indicadores de M15 y H1 completos, agregados desde el M1 nativo."""
    m15 = aggregate(minutes, 1, 'M15', now=end)
    h1 = aggregate(minutes, 1, 'H1', now=end)
    consecutive = m15.timestamp.diff().eq(MarketConfig('M15').step)
    returns = np.log(m15.close).diff().where(consecutive)
    realized = np.sqrt(returns.pow(2).rolling(8, min_periods=8).sum())
    previous = m15.close.shift().where(consecutive)
    true_range = pd.concat((m15.high - m15.low, (m15.high - previous).abs(),
                            (m15.low - previous).abs()), axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / RANGE_SPAN, min_periods=RANGE_SPAN, adjust=False).mean()
    ema = h1.close.ewm(span=SLOW_SPAN, min_periods=SLOW_SPAN, adjust=False).mean()
    hourly_range = (h1.high - h1.low).ewm(span=RANGE_SPAN, min_periods=RANGE_SPAN, adjust=False).mean()

    result = pd.DataFrame(index=decisions)
    for candles, timeframe, names, values in (
        (m15, 'M15', M15_TACTICAL, (realized, atr / m15.close)),
        (h1, 'H1', H1_STRUCTURAL, (ema.diff() / h1.close, hourly_range / h1.close)),
    ):
        last = _closed_bar_index(candles.timestamp, MarketConfig(timeframe), decisions)
        aligned = np.full((len(last), len(names)), np.nan)
        ready = last >= 0
        aligned[ready] = np.column_stack(values)[last[ready]]
        result[list(names)] = aligned
    return result


def session_observations(bar_opens):
    """Marca si la entrada o el vencimiento tocan una de las ventanas locales."""
    entry = pd.DatetimeIndex(bar_opens) + M5_STEP
    expiry = entry + M5_STEP
    observations = {}
    for name, zone, start, end in SESSION_WINDOWS:
        inside = np.zeros(len(entry), dtype=bool)
        for instant in (entry, expiry):
            local = instant.tz_convert(zone)
            minutes = local.hour * 60 + local.minute
            inside |= (minutes >= start) & (minutes < end)
        observations[name] = inside.astype(float)
    return observations


def realized_volatility(frame):
    """Raíz de la suma de retornos log M5 al cuadrado durante la última hora."""
    consecutive = frame.timestamp.diff().eq(M5_STEP)
    squares = np.log(frame.close).diff().pow(2).where(consecutive)
    return np.sqrt(squares.rolling(VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).sum()).to_numpy()


def build_observations(frame):
    """Conserva todas las filas y distingue ausencia de una observación neutral."""
    columns = {}
    for timeframe in TREND_TIMEFRAMES:
        bias = closed_trend_bias(frame, timeframe)
        columns[f'{timeframe.lower()}_bias'] = np.nan_to_num(bias, nan=0.)
        columns[f'{timeframe.lower()}_available'] = np.isfinite(bias).astype(float)
    volatility = realized_volatility(frame)
    columns['m5_realized_volatility'] = np.nan_to_num(volatility, nan=0.)
    columns['m5_volatility_available'] = np.isfinite(volatility).astype(float)
    columns['m5_range_fraction'] = ((frame.high - frame.low) / frame.close).to_numpy()
    columns.update(session_observations(frame.timestamp))
    return pd.DataFrame(columns, columns=CONTEXT_FEATURES,
                        index=pd.DatetimeIndex(frame.timestamp).as_unit('ns'))
