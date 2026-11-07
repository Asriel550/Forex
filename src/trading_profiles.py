"""Resolve profile presets once and pass the same contract to each research stage."""
from dataclasses import dataclass
import math
import os
from pathlib import Path

from .market import MarketConfig, MINUTES


@dataclass(frozen=True)
class Profile:
    timeframes: tuple[str, ...]
    expiry: int
    confidence: float
    entropy: float
    neutral_cost: float
    trend_window: int


PROFILES = {
    'scalping': Profile(('M5', 'M1'), 1, .54, .03, .20, 8),
    'intraday': Profile(('M15',), 1, .60, .02, .05, 8),
    'daily': Profile(('H1',), 2, .63, .02, .05, 12),
    'swing': Profile(('H4',), 2, .65, .02, .05, 16),
}
PROFILE_FOR_TIMEFRAME = {
    timeframe: name for name, profile in PROFILES.items() for timeframe in profile.timeframes
}
SCALPING_TARGET_PER_HOUR = (5, 10)

# Settings expressed in bars and units of conditional volatility.
# Shorter windows respond to micro-trends. The higher cap permits more
# displacement of the GBM while the percentile definitions stay fixed.
SCALPING_MOMENTUM = {
    'M1': dict(window=6, scale=1.8, noise_z=.50, cap_sigma=.90),
    'M5': dict(window=4, scale=2.2, noise_z=.50, cap_sigma=.90),
}


def resolve_profile(
    profile='auto', timeframe='auto', trading_mode='binary', expiry_candles=0,
    confidence='auto', entropy='auto', neutral_cost='auto', drift_scale='auto',
):
    """An explicit timeframe must agree with an explicit profile."""
    if profile == 'auto':
        profile = PROFILE_FOR_TIMEFRAME.get(timeframe) if timeframe != 'auto' else 'intraday'
    if profile not in PROFILES:
        raise ValueError('Perfil o temporalidad desconocidos')
    preset = PROFILES[profile]
    if timeframe == 'auto':
        timeframe = preset.timeframes[0]
    if timeframe not in preset.timeframes:
        raise ValueError(f'{profile} admite {", ".join(preset.timeframes)}, no {timeframe}')

    if isinstance(expiry_candles, bool) or not isinstance(expiry_candles, int):
        raise ValueError('expiry_candles debe ser entero')
    if expiry_candles == 0:
        expiry_candles = preset.expiry if trading_mode == 'binary' else 1
    market = MarketConfig(timeframe, trading_mode, expiry_candles)
    if profile == 'scalping' and market.expiry_candles > 3:
        raise ValueError('Scalping admite vencimiento de 1 a 3 velas')

    threshold = preset.confidence if confidence == 'auto' else float(confidence)
    ent_coef = preset.entropy if entropy == 'auto' else float(entropy)
    neutral_penalty = preset.neutral_cost if neutral_cost == 'auto' else float(neutral_cost)
    if not math.isfinite(threshold) or not .50 <= threshold < 1:
        raise ValueError('Umbral de confianza inválido')
    if profile == 'scalping' and not .52 <= threshold <= .54:
        raise ValueError('Scalping admite un umbral nominal entre 0.52 y 0.54')
    if not math.isfinite(ent_coef) or not 0 <= ent_coef <= .1:
        raise ValueError('ent_coef fuera del rango admitido por PPO')
    if not math.isfinite(neutral_penalty) or neutral_penalty < 0:
        raise ValueError('Penalización de inacción inválida')
    momentum = SCALPING_MOMENTUM.get(timeframe, dict(
        window=preset.trend_window, scale=1.5, noise_z=1.96, cap_sigma=.75,
    ))
    scale = momentum['scale'] if drift_scale == 'auto' else float(drift_scale)
    if not math.isfinite(scale) or not 0 < scale <= 3:
        raise ValueError('Escala de momentum fuera del rango admitido')

    target_activity = .10
    if profile == 'scalping' and trading_mode == 'binary':
        target_activity = min(1., SCALPING_TARGET_PER_HOUR[1] * MINUTES[timeframe] / 60)

    return dict(
        FOREX_TRADING_PROFILE=profile,
        FOREX_TIMEFRAME=market.timeframe,
        FOREX_TRADING_MODE=market.trading_mode,
        FOREX_EXPIRY_CANDLES=market.expiry_candles,
        FOREX_MIN_CONFIDENCE=threshold,
        FOREX_MIN_EXPECTED_EDGE=0. if profile == 'scalping' else .02,
        FOREX_ENT_COEF=ent_coef,
        FOREX_NEUTRAL_OPPORTUNITY_PENALTY=neutral_penalty,
        FOREX_TREND_WINDOW=momentum['window'],
        FOREX_DRIFT_SCALE=scale,
        FOREX_DRIFT_NOISE_Z=momentum['noise_z'],
        FOREX_DRIFT_CAP_SIGMA=momentum['cap_sigma'],
        FOREX_GARCH_WINDOW=3072 if timeframe == 'M1' else 2016,
        FOREX_REFIT_EVERY=480 if timeframe == 'M1' else 96,
        FOREX_TARGET_ACTIVITY=target_activity,
    )


def main():
    resolved = resolve_profile(
        profile=os.environ.get('FOREX_TRADING_PROFILE', 'auto'),
        timeframe=os.environ.get('FOREX_TIMEFRAME', 'auto'),
        trading_mode=os.environ.get('FOREX_TRADING_MODE', 'binary'),
        expiry_candles=int(os.environ.get('FOREX_EXPIRY_CANDLES', '0')),
        confidence=os.environ.get('FOREX_MIN_CONFIDENCE', 'auto'),
        entropy=os.environ.get('FOREX_ENT_COEF', 'auto'),
        neutral_cost=os.environ.get('FOREX_NEUTRAL_OPPORTUNITY_PENALTY', 'auto'),
        drift_scale=os.environ.get('FOREX_DRIFT_SCALE', 'auto'),
    )
    if os.environ.get('RESEARCH_MODE') == 'diagnostics':
        if (resolved['FOREX_TRADING_PROFILE'], resolved['FOREX_EXPIRY_CANDLES'],
                resolved['FOREX_TRADING_MODE']) != ('intraday', 1, 'binary'):
            raise ValueError('diagnostics conserva M15/binary/1; usar mode=train para otro perfil')
    lines = ''.join(f'{key}={value}\n' for key, value in resolved.items())
    print(lines, end='')
    if 'GITHUB_ENV' in os.environ:
        with Path(os.environ['GITHUB_ENV']).open('a', encoding='utf-8') as destination:
            destination.write(lines)


if __name__ == '__main__':
    main()
