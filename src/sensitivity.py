"""Sensibilidad train/validation de escala y umbral; diagnóstico MC, NO PPO.

No selecciona parámetros automáticamente ni evalúa el 10% final. Simula un
baseline que propone la dirección MC más probable; no promete que PPO opere.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import json,logging
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import file_sha256
from .quant_engine import QuantConfig,reproject_drift_scale
from .decision import RiskConfig,mc_probability_lower,ensure_probabilities
from .environment import AccountConfig,prepare_market_data,STEP
from .train import chronological_split
from .metrics import calculate_metrics,_atomic_write

ROOT=Path(__file__).resolve().parents[1]
SCALES=(1.,1.5,2.)
THRESHOLDS=(.58,.60,.62,.65)
LOG=logging.getLogger(__name__)


def evaluate_mc_baseline(frame,*,risk,account,n_paths):
    """Caja vectorizada equivalente al env, con propuesta greedy MC y stop 20%.

    Los cierres siguientes solo liquidan contratos ya elegidos con features t.
    No calcula reward ni entrena una política. Respeta huecos, reembolsos,
    saldo/pico acumulados y detención definitiva por riesgo/insuficiencia.
    """
    if len(frame)<2:raise ValueError('Se requieren dos filas')
    frame=ensure_probabilities(frame)
    times=pd.DatetimeIndex(pd.to_datetime(frame.timestamp,utc=True)).as_unit('ns')
    if not times.is_monotonic_increasing or not times.is_unique or times.hasnans or not times.equals(times.floor('15min')):raise ValueError('Timestamps inválidos')
    up=frame.mc_prob_up.to_numpy(dtype=float)[:-1];down=frame.mc_prob_down.to_numpy(dtype=float)[:-1]
    confidence=np.maximum(up,down)
    lower=mc_probability_lower(confidence,n_paths,risk.mc_z)
    direction=np.where(up>down,2,np.where(down>up,0,1))
    consecutive=np.diff(times.asi8)==STEP.value
    allowed=consecutive&(direction!=1)
    if risk.enabled:allowed &= (confidence>risk.min_confidence)&(account.payout*lower-(1-lower)>risk.min_expected_edge)
    close=frame.close.to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close<=0).any():raise ValueError('Precios inválidos')
    eligible_before_stop=int(allowed.sum())
    change=np.diff(close);tie=change==0
    win=((change>0)&(direction==2))|((change<0)&(direction==0))
    unit=np.where(tie,0. if account.tie_policy=='refund' else -1.,np.where(win,account.payout,-1.))
    profits=np.where(allowed,unit*account.stake,0.)
    balance=np.r_[account.initial_balance,account.initial_balance+np.cumsum(profits)]
    peak=np.maximum.accumulate(balance)
    limit=min(account.max_drawdown,risk.drawdown_limit) if risk.enabled else account.max_drawdown
    stops=np.flatnonzero((balance<account.stake)|((1-balance/peak)+1e-12>=limit))
    stopped=bool(len(stops));stop_row=int(stops[0]) if stopped else len(frame)-1
    # Contrato i liquida en fila i+1. Se conserva el contrato que causa el stop.
    allowed[stop_row:]=False;profits[stop_row:]=0.
    balance=np.r_[account.initial_balance,account.initial_balance+np.cumsum(profits)]
    outcome=np.where(tie,'tie',np.where(win,'win','loss'))
    trades=pd.DataFrame(dict(action=direction[allowed],profit=profits[allowed],outcome=outcome[allowed]))
    equity=pd.DataFrame(dict(timestamp=times+STEP,balance=balance))
    result=calculate_metrics(trades,equity,initial_balance=account.initial_balance)
    resolved=trades.profit.ne(0);wr=result['win_rate_excluding_refunds']
    target=win[allowed].astype(float)
    if account.tie_policy=='loss':target=np.where(tie[allowed],0.,target)
    # Para reembolsos, la calibración direccional se mide solo sin empate.
    mask=~tie[allowed] if account.tie_policy=='refund' else np.ones(len(target),dtype=bool)
    model_confidence=confidence[allowed]
    resolved_count=int(resolved.sum())
    empirical_lower=float(mc_probability_lower(wr,resolved_count,1.96)) if resolved_count>=2 else None
    result.update(decisions=int(consecutive.sum()),eligible_before_stop=eligible_before_stop,
        risk_stop=stopped,empirical_win_rate_lower_95=empirical_lower,
        lower_bound_exceeds_break_even=bool(empirical_lower is not None and empirical_lower>1/(1+account.payout)),
        break_even_win_rate_excluding_refunds=1/(1+account.payout),
        empirical_edge_per_resolved_stake=(account.payout*int((trades.profit>0).sum())-int((trades.profit<0).sum()))/int(resolved.sum()) if resolved.any() else None,
        mean_mc_confidence=float(model_confidence.mean()) if len(model_confidence) else None,
        brier_score=float(np.mean((model_confidence[mask]-target[mask])**2)) if mask.any() else None,
        put=int((trades.action==0).sum()),call=int((trades.action==2).sum()),
        neutral_before_stop=int(consecutive[:stop_row].sum()-len(trades)),
        decisions_skipped_after_risk_stop=int(consecutive[stop_row:].sum()) if stopped else 0,
        exceeds_payout_break_even=bool(wr is not None and wr>1/(1+account.payout)),
        policy='MC direction baseline; not trained PPO')
    return result


def run_sensitivity(input_path,*,report_dir=ROOT/'reports',scales=SCALES,thresholds=THRESHOLDS,account=None,risk=None,lookback=16,indicator_period=14,train_fraction=.70,validation_fraction=.20,purge_bars=1):
    account=account or AccountConfig();risk=risk if risk is not None else RiskConfig()
    if not scales or not thresholds or len(scales)*len(thresholds)>64:raise ValueError('Grid vacío o demasiado grande')
    input_path=Path(input_path);source_hash=file_sha256(input_path)
    raw=pd.read_parquet(input_path)
    if file_sha256(input_path)!=source_hash:raise ValueError('Entrada cambió durante lectura')
    cfg=QuantConfig.from_metadata(raw.attrs.get('quant_config',{}))
    # Misma preparación y mismos límites que train.py; test se reserva y no se evalúa.
    splits=chronological_split(prepare_market_data(raw,lookback,indicator_period),train_fraction=train_fraction,validation_fraction=validation_fraction,purge_bars=purge_bars)
    rows=[]
    for phase in ('train','validation'):
        part=splits[phase]
        for scale in scales:
            projected=reproject_drift_scale(part,float(scale))
            for threshold in thresholds:
                if not .58<=threshold<=.65:raise ValueError('Grid operativo requiere 58%..65%')
                candidate=replace(risk,min_confidence=float(threshold))
                metrics=evaluate_mc_baseline(projected,risk=candidate,account=account,n_paths=cfg.n_paths)
                rows.append(dict(phase=phase,drift_scale=float(scale),confidence_threshold=float(threshold),**metrics))
    if file_sha256(input_path)!=source_hash:raise ValueError('Entrada cambió durante evaluación')
    report=dict(source_sha256=source_hash,scope='train and validation only; no holdout outcomes',
        policy='diagnostic MC direction baseline, not PPO',automatic_selection=False,
        caveat='Scaling changes model probabilities, not observed win rates; validation comparison is research. Wilson bounds are nominal binomial diagnostics, not a multiple-testing correction or a guarantee of independent trades.',
        base_quant_config=asdict(cfg),base_risk=asdict(risk),account=asdict(account),
        split_config=dict(lookback=lookback,indicator_period=indicator_period,train_fraction=train_fraction,validation_fraction=validation_fraction,purge_bars=purge_bars),
        splits={k:dict(rows=len(splits[k]),start=splits[k].timestamp.iloc[0].isoformat(),end=splits[k].timestamp.iloc[-1].isoformat()) for k in ('train','validation')},results=rows)
    report_dir=Path(report_dir);report_dir.mkdir(parents=True,exist_ok=True)
    _atomic_write(report_dir/'sensitivity.json',lambda p:Path(p).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'))
    _atomic_write(report_dir/'sensitivity.csv',lambda p:pd.DataFrame(rows).to_csv(p,index=False))
    text=['# Sensibilidad de investigación (baseline MC, no PPO)','',
        f'Payout {account.payout:.2f}; acierto de equilibrio sin reembolsos {100/(1+account.payout):.2f}%.',
        'Solo entrenamiento/validación. No se selecciona un ganador automáticamente.','',
        '| Fase | Escala | Umbral | Contratos | PnL USD | PF | Acierto | DD | Stop |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        pf=f"{row['profit_factor']:.3f}" if row['profit_factor'] is not None else row['profit_factor_status']
        wr=f"{100*row['win_rate_excluding_refunds']:.2f}%" if row['win_rate_excluding_refunds'] is not None else 'N/D'
        text.append(f"| {row['phase']} | {row['drift_scale']:.2f} | {row['confidence_threshold']:.0%} | {row['total_trades']} | {row['net_profit']:.2f} | {pf} | {wr} | {row['max_drawdown_pct']:.2f}% | {row['risk_stop']} |")
    text += ['', 'No son resultados PPO ni una calibración de probabilidad. No garantizan rendimiento futuro. El stop se aplica al liquidar y puede exceder 20% con el último contrato.', '']
    _atomic_write(report_dir/'sensitivity.md',lambda p:Path(p).write_text('\n'.join(text)))
    LOG.info('Sensibilidad train/validation: %s',json.dumps(rows,allow_nan=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/eurusd_m15_features.parquet')
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports')
    parser.add_argument('--scales',type=float,nargs='+',default=list(SCALES))
    parser.add_argument('--thresholds',type=float,nargs='+',default=list(THRESHOLDS))
    parser.add_argument('--payout',type=float,default=.8)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    run_sensitivity(args.input,report_dir=args.report_dir,scales=args.scales,thresholds=args.thresholds,account=AccountConfig(payout=args.payout))

if __name__=='__main__':main()
