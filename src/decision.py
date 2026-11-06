"""Filtro causal compartido por entrenamiento, backtest e inferencia.

Una fracción MC es probabilidad bajo GBM, no certeza ni confianza de PPO.
Los valores por defecto son una política previa, no una optimización del test.
"""
from dataclasses import dataclass
import numpy as np
from .quant_engine import MonteCarloGBM, PROBABILITY_FEATURES


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = True
    min_confidence: float = .70
    min_expected_edge: float = .02
    mc_z: float = 1.96
    trade_penalty: float = .05
    blocked_penalty: float = .25
    overtrade_penalty: float = 2.
    activity_decay: float = .95
    target_activity: float = .10
    drawdown_penalty: float = 10.
    drawdown_limit: float = .20
    stop_penalty: float = 5.
    risk_std_floor: float = .10

    def __post_init__(self):
        if not isinstance(self.enabled,bool):raise ValueError('enabled debe ser bool')
        values=[getattr(self,k) for k in self.__dataclass_fields__ if k!='enabled']
        if not np.isfinite(values).all() or min(values)<0:
            raise ValueError('Parámetros de riesgo finitos y no negativos')
        if not (.5<=self.min_confidence<1 and self.min_expected_edge<1
                and 0<=self.activity_decay<1 and 0<self.target_activity<=1
                and 0<self.drawdown_limit<=1 and self.risk_std_floor>0):
            raise ValueError('Umbral, actividad o límite de riesgo inválido')


def ensure_probabilities(frame):
    """Deriva columnas ausentes solo de sigma/config causal; nunca de etiquetas.

    Fuentes de producción usan las probabilidades del motor. Para importaciones
    antiguas sin esas columnas se aplica el GBM configurado en sus metadatos
    (defaults explícitos: 10k, 32, seed42, drift0). No acepta columnas parciales.
    """
    out=frame.copy()
    if out.attrs.get('direction_probability_horizon',1)!=1:
        raise ValueError('Las probabilidades de decisión deben corresponder a UNA vela M15')
    present=[c in out for c in PROBABILITY_FEATURES]
    if any(present) and not all(present):raise ValueError('Probabilidades MC incompletas')
    if not any(present):
        cfg=out.attrs.get('quant_config',{})
        mc=MonteCarloGBM(cfg.get('n_paths',10000),cfg.get('horizon',32),cfg.get('seed',42))
        sigma=out.cond_volatility.to_numpy(dtype=float)
        ready=np.isfinite(sigma)
        probs=mc.direction_probabilities(sigma[ready],cfg.get('drift_per_bar',0.))
        for name,values in zip(PROBABILITY_FEATURES,probs):
            out[name]=np.nan;out.loc[ready,name]=values
    matrix=out[PROBABILITY_FEATURES].to_numpy(dtype=float)
    ready=np.isfinite(matrix).all(axis=1)
    partial=np.isfinite(matrix).any(axis=1)&~ready
    if partial.any() or np.isinf(matrix).any() or ((matrix[ready]<0)|(matrix[ready]>1)).any() or not np.allclose(matrix[ready].sum(axis=1),1.,atol=1e-8,rtol=0):
        raise ValueError('Probabilidades MC deben estar en [0,1] y sumar 1')
    return out


def resolve_action(action, up, down, tie, *, payout, tie_policy, risk, n_paths=10000):
    """NEUTRO si p <= umbral o ventaja conservadora insuficiente.

    Wilson solo cuantifica ruido de muestreo MC, NO error del modelo financiero.
    El edge conservador trata incluso empates como pérdidas para no inflarlo.
    """
    p=np.asarray([up,down,tie],dtype=float)
    if (action not in (0,1,2) or not np.isfinite(p).all() or (p<0).any() or (p>1).any()
            or not np.isclose(p.sum(),1.,atol=1e-8,rtol=0) or n_paths<2):
        raise ValueError('Acción/probabilidades/n_paths inválidos')
    confidence=float(up if action==2 else down if action==0 else max(up,down))
    z2=risk.mc_z**2
    lower=(confidence+z2/(2*n_paths)-risk.mc_z*np.sqrt(confidence*(1-confidence)/n_paths+z2/(4*n_paths**2)))/(1+z2/n_paths)
    lower=float(np.clip(lower,0,1));edge=float(payout*lower-(1-lower))
    reason='agent_neutral' if action==1 else 'accepted'
    executed=int(action)
    if risk.enabled and action!=1:
        if confidence<=risk.min_confidence:executed,reason=1,'confidence'
        elif edge<=risk.min_expected_edge:executed,reason=1,'expected_edge'
    return dict(requested_action=int(action),action=executed,mc_confidence=confidence,
                mc_lower_bound=lower,expected_edge_lower=edge,filter_reason=reason,
                blocked=bool(executed!=action),mc_prob_up=float(up),mc_prob_down=float(down),mc_prob_tie=float(tie))
