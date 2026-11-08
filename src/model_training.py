"""Pérdida direccional auxiliar para neuroevolución M5.

La red tanh se optimiza con mutación/selección, no por retropropagación.
Esta pérdida se incorpora a la aptitud genética y no modifica el gate en vivo.
"""
import numpy as np

HIGH_VOL_FALSE_POSITIVE_WEIGHT = .40


def directional_tanh_loss(scores, outcomes, high_volatility, *,
                          false_positive_weight=HIGH_VOL_FALSE_POSITIVE_WEIGHT):
    """Logística acotada + confianza errónea en regímenes de alta volatilidad.

    El target es PUT=-1 / CALL=1; los reembolsos (0) no supervisan dirección.
    Los errores en alta volatilidad reciben penalización adicional proporcional
    a la convicción |tanh(score)|². No se filtra ninguna señal en inferencia.
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes)
    high = np.asarray(high_volatility, dtype=bool)
    if (scores.ndim != 1 or outcomes.shape != scores.shape or high.shape != scores.shape
            or not np.isfinite(scores).all() or not np.isin(outcomes, [-1, 0, 1]).all()
            or not np.isfinite(false_positive_weight) or false_positive_weight < 0):
        raise ValueError('Pérdida direccional requiere scores y etiquetas válidos')
    settled = outcomes != 0
    if not settled.any():
        raise ValueError('No hay liquidaciones supervisadas')
    direction = np.where(scores >= 0., 1, -1)
    activated = np.tanh(scores)
    base = float(np.logaddexp(0., -2. * outcomes[settled] * activated[settled]).mean())
    wrong_high = settled & high & (direction != outcomes)
    # Denominador fijo: todas las propuestas resueltas, no solo las elegidas.
    false_positive = float((wrong_high * activated ** 2).sum() / settled.sum())
    return dict(
        total=base + false_positive_weight * false_positive,
        direction=base,
        high_vol_false_positive=false_positive,
        high_vol_wrong_count=int(wrong_high.sum()),
        settled_count=int(settled.sum()),
    )
