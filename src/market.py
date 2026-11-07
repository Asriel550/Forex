"""Contrato explícito de mercado; una configuración requiere pesos propios."""
from dataclasses import dataclass
import pandas as pd

# Frontera del snapshot protegido en PR #9; nunca se desplaza hacia adelante
# al reagrupar las velas o incorporar nuevos datos. Timestamp de apertura UTC.
PROTECTED_HOLDOUT_START = pd.Timestamp('2026-04-07T15:15:00Z')

MINUTES = {'M1': 1, 'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240}

@dataclass(frozen=True)
class MarketConfig:
    timeframe: str = 'M15'
    trading_mode: str = 'binary'
    expiry_candles: int = 1

    def __post_init__(self):
        if self.timeframe not in MINUTES or self.trading_mode not in ('binary', 'forex_spot'):
            raise ValueError('timeframe/trading_mode desconocido')
        if isinstance(self.expiry_candles, bool) or not isinstance(self.expiry_candles, int) or not 1 <= self.expiry_candles <= 32:
            raise ValueError('expiry_candles debe ser entero entre 1 y 32')
        if self.trading_mode == 'forex_spot' and self.expiry_candles != 1:
            raise ValueError('Forex spot no tiene vencimiento binario: usar expiry_candles=1')

    @property
    def step(self): return pd.Timedelta(minutes=MINUTES[self.timeframe])

    @property
    def frequency(self): return f'{MINUTES[self.timeframe]}min'

    def check_frame(self, frame):
        if frame.attrs.get('timeframe', 'M15') != self.timeframe:
            raise ValueError('Temporalidad de datos incompatible con el entorno')
        if frame.attrs.get('direction_probability_horizon', 1) != self.expiry_candles:
            raise ValueError('Horizonte de probabilidad MC distinto a expiry_candles')
