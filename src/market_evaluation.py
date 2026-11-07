"""Evaluación cronológica completa de contratos v4; nunca recibe el holdout."""
from pathlib import Path
import numpy as np
import pandas as pd
from .decision import mc_probability_lower
from .metrics import calculate_metrics,decision_metrics


def evaluate_market_policy(model,env,report_dir=None,*,trading_profile=None):
    balance=peak=env.account.initial_balance;activity=0.;rows=[];trades=[]
    equity=np.full(len(env._close),balance,dtype=float);filled=0
    total_reward=0.;max_dd=0.;stopped=False
    for start,_ in env.segments:
        equity[filled:start+1]=balance;filled=start+1
        obs,state=env.reset(options=dict(start_index=start,balance=balance,peak_balance=peak,trade_activity=activity))
        while True:
            entry=state['index'];action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=env.step(int(np.asarray(action).item()))
            total_reward+=reward;balance=info['balance'];peak=info['peak_balance'];activity=info['trade_activity']
            max_dd=max(max_dd,info['max_drawdown'])
            equity[filled:info['index']]=equity[entry];equity[info['index']]=balance;filled=info['index']+1
            record={k:info[k] for k in ('requested_action','action','blocked','missed_opportunity','mc_confidence','mc_lower_bound','expected_edge_lower','filter_reason','mc_prob_up','mc_prob_down','mc_prob_tie','profit','outcome','entry_index','expiry_time')}
            record['decision_time']=(env._times[entry]+env.bar_step).isoformat();rows.append(record)
            if env.market.trading_mode=='forex_spot': trades.extend(info['closed_trades'])
            elif info['action']!=1: trades.append(dict(action=info['action'],profit=info['profit'],outcome=info['outcome']))
            state=info
            if terminated or truncated: stopped=terminated;break
        if stopped:break
    equity[filled:]=balance
    signals=pd.DataFrame(rows)
    result=calculate_metrics(pd.DataFrame(trades,columns=['action','profit','outcome']),
        pd.DataFrame(dict(timestamp=env._times+env.bar_step,balance=equity)),
        initial_balance=env.account.initial_balance,allow_negative_equity=env.market.trading_mode=='forex_spot',periods_per_year=252*24*60/(env.bar_step.total_seconds()/60))
    labels=('PUT','NEUTRO','CALL') if env.market.trading_mode=='binary' else ('SHORT','FLAT','LONG')
    distribution={label:dict(count=int((signals.requested_action==i).sum()),
        percentage=float(100*(signals.requested_action==i).mean())) for i,label in enumerate(labels)}
    if env.market.trading_mode=='binary':
        analysis=decision_metrics(signals,payout=env.account.payout)
        analysis['confidence_scope']=f'Uncalibrated GBM at {env.market.expiry_candles} {env.market.timeframe} bars; not PPO confidence'
        confidence=np.maximum(signals.mc_prob_up,signals.mc_prob_down).to_numpy()
        lower=mc_probability_lower(confidence,env._n_paths,env.risk.mc_z)
        eligible_mask=(confidence>env.risk.min_confidence)&(env.account.payout*lower-(1-lower)>env.risk.min_expected_edge)
        eligible=int(eligible_mask.sum())
        missed=int(signals.missed_opportunity.sum());blocked=int((signals.blocked.to_numpy()&eligible_mask).sum())
        # Misma utilidad que aprende PPO, en reloj de oportunidades. El reward
        # YA incluye DD incremental, actividad, costes, omisiones y stop.
        # No se vuelve a restar 10*maxDD ni se diluye por esperas gratuitas.
        score=total_reward/max(eligible,1)
        if stopped:
            score=-abs(env.risk.stop_penalty)-max_dd-max(0.,-score)
    else:
        # No PF por payout ni etiqueta CALL/PUT para Forex spot.
        analysis=dict(scope='forex_spot; PnL de contratos cerrados y equity marcada',
            raw_policy_distribution=distribution,blocked_attempts=int(signals.blocked.sum()))
        eligible=missed=0;score=total_reward/max(len(rows),1)
        if stopped:score=-abs(env.risk.stop_penalty)-max_dd-max(0.,-score)
    result.update(return_basis=f'{env.market.timeframe} equity bars including inactivity',selection_score=float(score),decision_analysis=analysis,raw_policy_distribution=distribution,
        net_profit=balance-env.account.initial_balance,final_balance=balance,total_reward=total_reward,
        mean_reward=total_reward/max(len(rows),1),max_drawdown=max_dd,trades=len(trades),steps=len(rows),
        wins=result['winning_trades'],losses=result['losing_trades'],risk_stop=bool(stopped),
        eligible_setups_on_exposure=eligible,missed_opportunities=missed,
        signals={name:int((signals.action==i).sum()) for i,name in enumerate(tuple(x.lower() for x in labels))},
        evaluation='chronological full phase; no opportunity wrapper')
    # Cuenta el tiempo de todos los segmentos evaluables, incluso si el stop
    # impide operar su tramo final. Los huecos de mercado quedan fuera.
    observed_hours = sum(end - start for start, end in env.segments) * env.bar_step.total_seconds() / 3600
    result.update(observed_hours=observed_hours, trades_per_observed_hour=len(trades)/observed_hours)
    if trading_profile is not None:
        result['trading_profile'] = trading_profile
    if trading_profile == 'scalping' and env.market.trading_mode == 'binary':
        from .trading_profiles import SCALPING_TARGET_PER_HOUR
        minimum, maximum = SCALPING_TARGET_PER_HOUR
        expiry = env.market.expiry_candles
        setups = int(env._setup_mask[env._eligible].sum()) if env.risk.enabled else len(env._eligible)
        slots = sum((end - start) // expiry for start, end in env.segments)
        result['eligible_setups_before_policy'] = setups
        result['trades_per_hour_upper_bound'] = min(setups, slots) / observed_hours
        result['frequency_target_per_hour'] = list(SCALPING_TARGET_PER_HOUR)
        result['frequency_target_met'] = minimum <= result['trades_per_observed_hour'] <= maximum
    if report_dir is not None:
        from .train import write_json
        root=Path(report_dir);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'validation_metrics.json',result)
        signals.to_csv(root/'validation_decisions.csv',index=False)
        pd.DataFrame(trades,columns=['action','profit','outcome']).to_csv(root/'validation_trades.csv',index=False)
        (root/'validation.md').write_text(f"# Validación {env.market.trading_mode} {env.market.timeframe}\n\n"
            f"Contratos cerrados: {len(trades)}; PnL simulado: {result['net_profit']:.2f} USD; "
            f"PF: {result['profit_factor']} ({result['profit_factor_status']}); "
            f"DD: {100*max_dd:.2f}%.\n\n"
            f"Frecuencia media {result['trades_per_observed_hour']:.3f} operaciones por hora observable.\n\nHoldout congelado. Sin prueba de ventaja fuera de muestra.\n")
    return result
