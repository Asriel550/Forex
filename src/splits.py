"""Particiones cronológicas disjuntas y protección del holdout histórico."""
from __future__ import annotations

import numpy as np

from .market import PROTECTED_HOLDOUT_START


def chronological_split(prepared, *, train_fraction=.70, validation_fraction=.20, purge_bars=1):
    """Subconjuntos disjuntos. Ningún step accede al close de otro subconjunto.

    purge_bars descarta filas iniciales de validation y test como separación
    adicional. El forecast MC de 32 barras es una feature, no una etiqueta
    futura; no requiere purgar 32 barras para una expiración binaria de una.
    Los indicadores de validación pueden usar pasado de train legítimamente.
    """
    if (not np.isfinite([train_fraction, validation_fraction]).all()
            or min(train_fraction, validation_fraction) <= 0
            or train_fraction+validation_fraction >= 1):
        raise ValueError("Fracciones positivas cuya suma sea < 1; el resto se reserva para test")
    if not isinstance(purge_bars, int) or purge_bars < 0:
        raise ValueError("purge_bars debe ser entero no negativo")
    n = len(prepared)
    train_end = int(n*train_fraction)
    validation_end = int(n*(train_fraction+validation_fraction))
    pieces = dict(train=prepared.iloc[:train_end].copy().reset_index(drop=True),
                  validation=prepared.iloc[train_end+purge_bars:validation_end].copy().reset_index(drop=True),
                  test=prepared.iloc[validation_end+purge_bars:].copy().reset_index(drop=True))
    if any(len(frame) < 2 for frame in pieces.values()):
        raise ValueError("Histórico insuficiente para train/validation/test y purga")
    return pieces


def chronological_market_split(prepared, market, *, train_fraction=.70,
                               validation_fraction=.20, purge_bars=1):
    """Protege también el holdout histórico ABSOLUTO al cambiar timeframe.

    La agrupación/warmup pueden mover el percentil 90 de fechas. Si lo mueven
    hacia adelante se particiona solo el prefijo anterior a la frontera PR9.
    Ninguna OHLC cuya barra cierre después de esa frontera entra a train/val.
    """
    pieces = chronological_split(prepared, train_fraction=train_fraction,
                                 validation_fraction=validation_fraction, purge_bars=purge_bars)
    if pieces['test'].timestamp.iloc[0] > PROTECTED_HOLDOUT_START:
        prefix = prepared.loc[prepared.timestamp+market.step <= PROTECTED_HOLDOUT_START]
        end = int(len(prefix)*train_fraction/(train_fraction+validation_fraction))
        pieces = dict(train=prefix.iloc[:end].reset_index(drop=True),
                      validation=prefix.iloc[end+purge_bars:].reset_index(drop=True),
                      test=prepared.loc[prepared.timestamp >= PROTECTED_HOLDOUT_START].reset_index(drop=True))
    for phase in ('train', 'validation'):
        pieces[phase] = pieces[phase].loc[pieces[phase].timestamp+market.step <= PROTECTED_HOLDOUT_START].reset_index(drop=True)
    if any(len(f) < 2 for f in pieces.values()):
        raise ValueError('Datos insuficientes antes de la frontera protegida del holdout PR9')
    return pieces
