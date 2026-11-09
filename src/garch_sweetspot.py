"""PR39: bandeo GARCH sellado y gate causal condicionado al régimen M5.

No usa ninguna feature M1 ni lee Q3. Las bandas se obtienen de la
volatilidad M5 GARCH del snapshot y de medianas desplazadas de sus
48 observaciones anteriores. +15% significa multiplicar el suelo por
0.85 y el techo por 1.15, sin refit ni sustitución del feed congelado.
"""
import numpy as np
import pandas as pd

from src.adaptive_gate import CONVICTION_WINDOW
from src.conviction_regime_gate import _possible_mastery, _exact_adaptive_reasons
from src.upstream_filters import GENE_COUNT


BAND_RELAXATIONS = (0.0, 0.075, 0.15, 0.225, 0.30)
GATE_MODES = ("p85", "p80", "p75", "regime_dynamic")
TARGET_MIN_HOURLY = .50
TARGET_PF = 1.20
MAX_SEMESTER_DD = 12.0


def sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x,-20.,20.)))


def normalized_garch_ratio(volatility):
    sigma = np.asarray(volatility, dtype=float)
    if sigma.ndim != 1 or not len(sigma) or not np.isfinite(sigma).all() or np.any(sigma<0):
        raise ValueError("GARCH M5 fuera de contrato")
    # Same reference formula used by PR38 structural garch_regime.
    history = pd.Series(np.maximum(sigma,1e-12)).shift(1).rolling(
        48,min_periods=24
    ).median().to_numpy()
    return sigma/np.where(np.isfinite(history), history, np.maximum(sigma,1e-12))


def garch_band(genome, volatility, relaxation=0.0):
    genes=np.asarray(genome,dtype=float)
    if (genes.ndim!=1 or len(genes)<GENE_COUNT+13 or not np.isfinite(genes).all()
            or not np.isfinite(relaxation) or not 0.<=relaxation<=.50):
        raise ValueError("Genoma/banda GARCH inválidos")
    head=genes[GENE_COUNT:]
    original_low=.35+sigmoid(head[10])
    original_high=1.2+2.*sigmoid(head[11])
    low=original_low*(1.-relaxation)
    high=original_high*(1.+relaxation)
    ratio=normalized_garch_ratio(volatility)
    mask=(ratio>=low)&(ratio<=high)
    return mask,dict(
        original_low=float(original_low),original_high=float(original_high),
        applied_low=float(low),applied_high=float(high),
        relaxation=float(relaxation),selected_rows=int(mask.sum()),
        total_rows=int(len(mask)),
    ),ratio


def local_regime_percentiles(ratio, low, high):
    """p75 en el núcleo de la banda, p80 intermedio, p85 en sus bordes.

    No utiliza resultados futuros, percentiles condicionados a PnL ni Q3.
    La banda está fijada por genes seleccionados únicamente en Q1/Q2 IS.
    """
    values=np.asarray(ratio,dtype=float)
    if (not np.isfinite(values).all() or not np.isfinite(low) or not np.isfinite(high)
            or low<=0 or high<=low):
        raise ValueError("Umbrales del régimen inválidos")
    relative=(values-low)/(high-low)
    core=(relative>=.25)&(relative<=.75)
    middle=(relative>=.10)&(relative<=.90)
    return np.where(core,.75,np.where(middle,.80,.85)).astype(float)


def causal_custom_gate(rows,scores,percentiles):
    """Gate p75/p80/p85 por barra, con ventana de 500 estrictamente pasada."""
    s=np.asarray(scores,dtype=float)
    ps=np.asarray(percentiles,dtype=float)
    if (s.shape!=(len(rows),) or ps.shape!=s.shape
            or not np.isin(ps,(.75,.80,.85)).all()
            or not np.isfinite(s).all()
            or not rows.index.is_unique or not rows.index.is_monotonic_increasing
            or not rows.outcome.isin([-1,0,1]).all()):
        raise ValueError("Inputs gate no causales")
    if _possible_mastery(s,rows.outcome.to_numpy()):
        return _exact_adaptive_reasons(rows,s,ps)
    conviction=np.abs(np.tanh(s))
    rolling=pd.Series(conviction).rolling(CONVICTION_WINDOW)
    thresholds=np.where(
        ps==.75,rolling.quantile(.75).shift(1).to_numpy(),
        np.where(ps==.80,rolling.quantile(.80).shift(1).to_numpy(),
                 rolling.quantile(.85).shift(1).to_numpy()),
    )
    decision=np.where(conviction>thresholds,"accepted","conviction").astype("<U24")
    decision[:CONVICTION_WINDOW]="conviction_warmup"
    return decision


def accepted(report):
    pf=report["quarter_pf"]
    return bool(
        all(x is not None and np.isfinite(x) and x>TARGET_PF for x in pf)
        and all(p>0 for p in report["quarter_pnl_usd"])
        and report["semester_rate"]>=TARGET_MIN_HOURLY
        and report["semester_dd"]<MAX_SEMESTER_DD
        and not report["semester_stop"]
        and not any(report["quarter_stop"])
    )


def pf_frequency_objective(report):
    """PF geométrico Q1/Q2 × tasa ejecutada relativa a 0.50/h.

    Ambos factores aportan la misma elasticidad porcentual al producto.
    Se penaliza DD que excede 12% y trimestre de baja calidad, sin
    permitir que la fitness sustituya el criterio duro de aceptación.
    """
    pfs=np.asarray([v or 0 for v in report["quarter_pf"]],dtype=float)
    if not np.isfinite(pfs).all() or (pfs<0).any():
        raise ValueError("PF inválido")
    pf=np.sqrt(np.clip(pfs[0],0,3.)*np.clip(pfs[1],0,3.))
    rate=float(report["semester_rate"])
    dd=float(report["semester_dd"])
    if not np.isfinite(rate) or not np.isfinite(dd) or rate<0 or dd<0:
        raise ValueError("Frecuencia/DD no finitos")
    product=pf*(rate/TARGET_MIN_HOURLY)
    penalty=.55*max(0.,(dd-MAX_SEMESTER_DD)/MAX_SEMESTER_DD)
    penalty+=.12*sum(max(0.,TARGET_PF-v) for v in pfs)
    return float(product-penalty),float(product)
