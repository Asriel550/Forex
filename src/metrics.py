"""Métricas monetarias y reportes para opciones binarias M15.

Los neutros no cuentan como operaciones. Sharpe incluye barras sin operar;
252*96 es una convención explícita de anualización, no horas de calendario.
JSON usa null y un estado para ratios indefinidos, nunca NaN/Infinity.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from .decision import mc_probability_lower


def calculate_metrics(trades, equity, *, initial_balance, periods_per_year=252*96,
                      annual_risk_free_rate=0.,allow_negative_equity=False):
    """Equity inicial y PnL de direcciones 0/2. Spot puede registrar deuda al liquidar."""
    if (not np.isfinite([initial_balance,periods_per_year,annual_risk_free_rate]).all()
            or initial_balance<=0 or periods_per_year<=0 or annual_risk_free_rate<=-1):
        raise ValueError('Capital, tasa o anualización inválidos')
    if not {'timestamp','balance'}.issubset(equity) or len(equity)<2:
        raise ValueError('Equity requiere timestamp,balance y al menos dos filas')
    times=pd.DatetimeIndex(pd.to_datetime(equity.timestamp,utc=True,errors='raise'))
    balances=equity.balance.to_numpy(dtype=float)
    if (times.hasnans or not times.is_unique or not times.is_monotonic_increasing
            or not np.isfinite(balances).all() or (not allow_negative_equity and (balances<0).any())
            or not np.isclose(balances[0],initial_balance,rtol=0,atol=1e-8)):
        raise ValueError('Equity inválida o primer saldo distinto del capital inicial')
    if 'profit' not in trades or (len(trades) and ('action' not in trades or not trades.action.isin([0,2]).all())):
        raise ValueError('Trades requiere profit y acciones PUT/CALL')
    profits=trades.profit.to_numpy(dtype=float)
    net=float(balances[-1]-initial_balance)
    if not np.isfinite(profits).all() or not np.isclose(profits.sum(),net,rtol=1e-10,atol=1e-8):
        raise ValueError('PnL no concilia con equity')
    wins,losses,flats=int((profits>0).sum()),int((profits<0).sum()),int((profits==0).sum())
    gross_profit,gross_loss=float(profits[profits>0].sum()),float(-profits[profits<0].sum())
    previous=balances[:-1]
    if ((previous==0)&(balances[1:]!=0)).any():
        raise ValueError('No se permite recapitalizar después de saldo cero')
    returns=np.divide(np.diff(balances),previous,out=np.zeros_like(previous),where=previous>0)
    excess=returns-np.expm1(np.log1p(annual_risk_free_rate)/periods_per_year)
    std=float(excess.std(ddof=1)) if len(excess)>1 else 0.
    downside=float(np.sqrt(np.mean(np.minimum(excess,0)**2)))
    sharpe=float(excess.mean()/std*np.sqrt(periods_per_year)) if std>1e-15 else None
    sortino=float(excess.mean()/downside*np.sqrt(periods_per_year)) if downside>1e-15 else None
    peaks=np.maximum.accumulate(balances)
    dd=peaks-balances
    return dict(initial_balance=float(initial_balance),final_balance=float(balances[-1]),net_profit=net,
                net_return_pct=100*net/initial_balance,total_trades=len(trades),winning_trades=wins,
                losing_trades=losses,breakeven_trades=flats,
                ties=int(trades.outcome.eq('tie').sum()) if 'outcome' in trades else 0,
                win_rate=wins/len(trades) if len(trades) else None,
                win_rate_pct=100*wins/len(trades) if len(trades) else None,
                win_rate_excluding_refunds=wins/(wins+losses) if wins+losses else None,
                gross_profit=gross_profit,gross_loss=gross_loss,
                profit_factor=gross_profit/gross_loss if gross_loss else None,
                profit_factor_status=('finite' if gross_loss else 'infinite_no_losses' if gross_profit
                                      else 'no_trades' if not len(trades) else 'undefined_no_profit_or_loss'),
                max_drawdown_usd=float(dd.max()),max_drawdown_pct=float(100*(dd/peaks).max()),
                sharpe_ratio=sharpe,sortino_ratio=sortino,
                sharpe_status='finite' if sharpe is not None else 'insufficient_or_zero_variance',
                sortino_status='finite' if sortino is not None else 'no_downside',
                periods_per_year=float(periods_per_year),annual_risk_free_rate=float(annual_risk_free_rate),
                return_observations=len(returns),return_basis='observed M15 equity bars including inactivity')


def _atomic_write(path, writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,suffix='.tmp');os.close(fd)
    try:
        writer(temporary);os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_reports(report_dir, metrics, trades, equity, signals):
    report_dir=Path(report_dir)
    payload=json.dumps(metrics,indent=2,allow_nan=False)+'\n'
    _atomic_write(report_dir/'metrics.json',lambda p:Path(p).write_text(payload,encoding='utf-8'))
    _atomic_write(report_dir/'summary.md',lambda p:Path(p).write_text(markdown_report(metrics),encoding='utf-8'))
    for name,frame in [('trades',trades),('equity',equity),('signals',signals)]:
        _atomic_write(report_dir/f'{name}.csv',lambda p,f=frame:f.to_csv(p,index=False))


def plot_reports(report_dir,equity,signals):
    """PNG compatibles con runners sin servidor gráfico."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report_dir=Path(report_dir)
    balances=equity.balance.to_numpy(dtype=float);peaks=np.maximum.accumulate(balances)
    times=pd.to_datetime(equity.timestamp,utc=True)
    figure,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    axes[0].plot(times,balances,color='#126b62');axes[0].set(ylabel='USD',title='Holdout: equity')
    axes[1].fill_between(times,100*(balances/peaks-1),0,color='#bc4b51',alpha=.6)
    axes[1].set(ylabel='Drawdown (%)',xlabel='UTC')
    _atomic_write(report_dir/'equity_drawdown.png',lambda p:figure.savefig(p,format='png',dpi=150));plt.close(figure)
    figure,axis=plt.subplots(figsize=(6,4),layout='constrained')
    counts=signals.action.value_counts() if len(signals) else pd.Series(dtype=int)
    axis.bar(['PUT','NEUTRO','CALL'],[int(counts.get(i,0)) for i in range(3)])
    axis.set(ylabel='Decisiones',title='Señales holdout')
    _atomic_write(report_dir/'signals.png',lambda p:figure.savefig(p,format='png',dpi=150));plt.close(figure)



CONFIDENCE_THRESHOLDS=(.58,.60,.62,.65,.70,.80)


def decision_metrics(signals, *, thresholds=CONFIDENCE_THRESHOLDS,payout=.8):
    """Desglose por decisión ejecutada y propuesta; NEUTRO no es un contrato.

    Para CALL/PUT, confianza=probabilidad de ganar según MC en UNA vela.
    Para NEUTRO, confianza=max(P_up,P_down): no existe 'probabilidad de ganar
    NEUTRO'. El corte retrospectivo no reejecuta la política y no selecciona
    umbrales con test; solo describe contratos que realmente se ejecutaron.
    """
    if not np.isfinite(payout) or not 0<payout<=1:raise ValueError('Payout inválido')
    required={'mc_confidence','action','requested_action','mc_prob_up','mc_prob_down','profit','blocked'}
    if not required.issubset(signals):raise ValueError('Se requieren señales auditables con probabilidades MC')
    if not signals.action.isin([0,1,2]).all() or not signals.requested_action.isin([0,1,2]).all():
        raise ValueError('Acciones inválidas')
    n=len(signals);out={}
    probs=signals[['mc_prob_up','mc_prob_down']].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any():raise ValueError('Confianzas inválidas')
    for column,key in [('action','executed'),('requested_action','requested')]:
        result={}
        for action,label in [(0,'PUT'),(1,'NEUTRO'),(2,'CALL')]:
            part=signals.loc[signals[column]==action]
            confidence=part.mc_prob_up if action==2 else part.mc_prob_down if action==0 else part[['mc_prob_up','mc_prob_down']].max(axis=1)
            result[label]=dict(count=len(part),percentage=100*len(part)/n if n else 0.,
                               mean_mc_confidence=float(confidence.mean()) if len(part) else None,
                               confidence_definition='max(up,down)' if action==1 else 'directional win probability')
        out[key]=result
    segmented=[]
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0<=threshold<1:raise ValueError('Umbral de reporte inválido')
        part=signals.loc[(signals.action!=1)&(signals.mc_confidence>threshold)]
        profits=part.profit.to_numpy(dtype=float)
        if not np.isfinite(profits).all():raise ValueError('PnL inválido')
        gain=float(profits[profits>0].sum());loss=float(-profits[profits<0].sum())
        wins=int((profits>0).sum());losses=int((profits<0).sum());resolved=wins+losses
        empirical_lower=float(mc_probability_lower(wins/resolved,resolved,1.96)) if resolved>=2 else None
        segmented.append(dict(empirical_win_rate_lower_95=empirical_lower,lower_bound_exceeds_break_even=bool(empirical_lower is not None and empirical_lower>1/(1+payout)),break_even_win_rate=1/(1+payout),
            empirical_win_rate_excluding_refunds=wins/resolved if resolved else None,
            empirical_edge_per_resolved_stake=(payout*wins-losses)/resolved if resolved else None,
            mean_mc_confidence=float(part.mc_confidence.mean()) if len(part) else None,threshold=float(threshold),comparison='strictly_greater',trades=len(part),
            net_profit=float(profits.sum()),wins=int((profits>0).sum()),losses=int((profits<0).sum()),
            profit_factor=gain/loss if loss else None,
            profit_factor_status='finite' if loss else 'infinite_no_losses' if gain else 'no_trades' if not len(part) else 'undefined_no_profit_or_loss'))
    return dict(missed_opportunities=int(signals['missed_opportunity'].sum()) if 'missed_opportunity' in signals else 0,payout=payout,break_even_win_rate_excluding_refunds=1/(1+payout),total_decisions=n,blocked_attempts=int(signals.blocked.sum()),decisions=out,
                confidence_segments=segmented,
                confidence_scope='uncalibrated GBM model probability at one M15 step; not PPO confidence',
                segment_scope='post-hoc subset of executed contracts, no policy rerun or threshold selection')


def markdown_report(metrics):
    text=['# Backtest de investigación', '',
          f"Beneficio neto: {metrics['net_profit']:.2f} USD; drawdown máximo: {metrics['max_drawdown_pct']:.2f}%.",
          '', '## Decisiones ejecutadas', '', '| Decisión | Cantidad | Porcentaje | Media probabilidad MC |',
          '| --- | ---: | ---: | ---: |']
    audit=metrics.get('decision_analysis',{})
    for name,row in audit.get('decisions',{}).get('executed',{}).items():
        p=row['mean_mc_confidence'];value=f'{p:.4f}' if p is not None else 'N/D'
        text.append(f"| {name} | {row['count']} | {row['percentage']:.2f}% | {value} |")
    text += ['', f"Intentos bloqueados: {audit.get('blocked_attempts',0)}; configuraciones elegibles omitidas: {audit.get('missed_opportunities',0)}.", '', (f"Payout: {audit['payout']:.2f}; acierto de equilibrio (sin reembolsos): {100*audit['break_even_win_rate_excluding_refunds']:.2f}%." if 'payout' in audit else ''), '', 'NEUTRO usa max(P_up,P_down); no es probabilidad de acierto de una operación neutra.',
             'Las probabilidades son del GBM a una vela, no certezas calibradas.', '',
             '## Subconjuntos por confianza (solo contratos ejecutados)', '',
             '| Umbral estricto | Contratos | PnL USD | Profit Factor | Acierto sin reembolsos |', '| --- | ---: | ---: | ---: | ---: |']
    for row in audit.get('confidence_segments',[]):
        pf=row['profit_factor'];value=f'{pf:.4f}' if pf is not None else row['profit_factor_status']
        rate=row.get('empirical_win_rate_excluding_refunds');wr=f'{100*rate:.2f}%' if rate is not None else 'N/D'
        text.append(f"| > {row['threshold']:.0%} | {row['trades']} | {row['net_profit']:.2f} | {value} | {wr} |")
    text += ['', 'Cortes descriptivos posteriores: no equivalen a reejecutar la política.',
             'El holdout histórico ya se examinó: esta comparación es investigación, no una nueva prueba independiente.', '']
    return '\n'.join(text)
