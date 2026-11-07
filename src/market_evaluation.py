"""Evaluación cronológica completa de contratos v4; nunca recibe el holdout."""
from pathlib import Path
import numpy as np
import pandas as pd
from .decision import mc_probability_lower
from .metrics import calculate_metrics,decision_metrics


def stochastic_proposal_frequency(env, observed_hours):
    """Count the engine's best direction at each executable entry before PPO."""
    up, down, _ = env._probabilities[env._eligible].T
    directional = up != down
    confident = directional & (np.maximum(up, down) > env.risk.min_confidence)
    accepted = directional & env._setup_mask[env._eligible]
    if not env.risk.enabled:
        confident = accepted = directional

    def frequency(mask):
        count = int(mask.sum())
        return dict(count=count, per_hour=count / observed_hours)

    payout = env.account.payout
    return dict(
        observed_hours=observed_hours,
        before_filter=frequency(directional),
        after_confidence=frequency(confident),
        after_economic_filter=frequency(accepted),
        confidence_threshold=env.risk.min_confidence,
        payout_break_even_probability=1 / (1 + payout),
        min_wilson_probability_for_edge=(1 + env.risk.min_expected_edge) / (1 + payout),
        wilson_z=env.risk.mc_z,
        risk_filter_enabled=env.risk.enabled,
    )


def scalping_m5_quality(result, frequency_target):
    """Check realized outcomes and conversion over the full validation phase."""
    rate = result['win_rate_excluding_refunds']
    profit_factor = result['profit_factor']
    pf_status = result['profit_factor_status']
    frequency = result['trades_per_observed_hour']
    setups = result['eligible_setups_before_policy']
    conversion = result['total_trades'] / setups if setups else None
    minimum, maximum = frequency_target
    checks = {
        'win_rate_excluding_refunds': dict(
            value=rate, minimum=.56, passed=rate is not None and rate >= .56,
        ),
        'profit_factor': dict(
            value=profit_factor, status=pf_status, exclusive_minimum=1.2,
            passed=pf_status == 'infinite_no_losses' or (profit_factor is not None and profit_factor > 1.2),
        ),
        'trades_per_observed_hour': dict(
            value=frequency, minimum=minimum, maximum=maximum,
            passed=minimum <= frequency <= maximum,
        ),
        'eligible_setup_conversion': dict(
            value=conversion, minimum=.30, passed=conversion is not None and conversion >= .30,
        ),
    }
    return dict(passed=all(check['passed'] for check in checks.values()), checks=checks)


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
            if 'mc_setup_action' in info:
                record['mc_setup_action'] = info['mc_setup_action']
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
    if env.market.trading_mode == 'binary':
        result['stochastic_proposals'] = stochastic_proposal_frequency(env, observed_hours)
    if 'mc_setup_action' in signals:
        evaluated = signals.loc[signals.mc_setup_action != 1]
        aligned = int((evaluated.requested_action == evaluated.mc_setup_action).sum())
        neutral = int((evaluated.requested_action == 1).sum())
        opposite = len(evaluated) - aligned - neutral
        result['mc_alignment'] = dict(
            evaluated_setups=len(evaluated), aligned_requests=aligned,
            opposite_requests=opposite, neutral_requests=neutral,
            opposite_rate=opposite / len(evaluated) if len(evaluated) else None,
        )
    if trading_profile is not None:
        result['trading_profile'] = trading_profile
    if trading_profile == 'scalping' and env.market.trading_mode == 'binary':
        from .trading_profiles import SCALPING_TARGET_PER_HOUR
        frequency_target = SCALPING_TARGET_PER_HOUR[env.market.timeframe]
        minimum, maximum = frequency_target
        expiry = env.market.expiry_candles
        setups = result['stochastic_proposals']['after_economic_filter']['count'] if env.risk.enabled else len(env._eligible)
        slots = sum((end - start) // expiry for start, end in env.segments)
        result['eligible_setups_before_policy'] = setups
        result['trades_per_hour_upper_bound'] = min(setups, slots) / observed_hours
        result['frequency_target_per_hour'] = list(frequency_target)
        result['frequency_target_met'] = minimum <= result['trades_per_observed_hour'] <= maximum
        if env.market.timeframe == 'M5':
            result['validation_quality'] = scalping_m5_quality(result, frequency_target)
    if report_dir is not None:
        from .train import write_json
        root=Path(report_dir);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'validation_metrics.json',result)
        signals.to_csv(root/'validation_decisions.csv',index=False)
        pd.DataFrame(trades,columns=['action','profit','outcome']).to_csv(root/'validation_trades.csv',index=False)
        summary = (
            f"# Validación {env.market.trading_mode} {env.market.timeframe}\n\n"
            f"Contratos cerrados {len(trades)}. PnL simulado {result['net_profit']:.2f} USD. "
            f"PF {result['profit_factor']} ({result['profit_factor_status']}). "
            f"Drawdown máximo {100*max_dd:.2f}%.\n\n"
        )
        if 'stochastic_proposals' in result:
            proposals = result['stochastic_proposals']
            summary += (
                f"Frecuencia sobre {observed_hours:.2f} horas observables.\n\n"
                "| Etapa | Cantidad | Por hora |\n"
                "| --- | ---: | ---: |\n"
            )
            for key, label in (
                ('before_filter', 'Direcciones propuestas por MC'),
                ('after_confidence', 'Superan confianza nominal'),
                ('after_economic_filter', 'Superan confianza, Wilson y payout'),
            ):
                row = proposals[key]
                summary += f"| {label} | {row['count']} | {row['per_hour']:.4f} |\n"
            summary += f"| Contratos ejecutados por PPO | {len(trades)} | {result['trades_per_observed_hour']:.4f} |\n\n"
            summary += (
                f"Equilibrio del payout {proposals['payout_break_even_probability']:.2%}. "
                f"Límite inferior exigido por el margen {proposals['min_wilson_probability_for_edge']:.2%}.\n\n"
            )
        else:
            summary += f"Frecuencia media {result['trades_per_observed_hour']:.3f} operaciones por hora observable.\n\n"
        if 'mc_alignment' in result:
            alignment = result['mc_alignment']
            summary += (
                f"PPO ante {alignment['evaluated_setups']} setups evaluados. "
                f"Coincidencias {alignment['aligned_requests']}, "
                f"solicitudes contrarias {alignment['opposite_requests']}, "
                f"abstenciones {alignment['neutral_requests']}.\n\n"
            )
        if 'validation_quality' in result:
            quality = result['validation_quality']
            checks = quality['checks']
            win = checks['win_rate_excluding_refunds']
            pf = checks['profit_factor']
            rate = checks['trades_per_observed_hour']
            conversion = checks['eligible_setup_conversion']
            win_display = 'N/D' if win['value'] is None else f"{win['value']:.2%}"
            pf_display = '∞' if pf['status'] == 'infinite_no_losses' else (
                'N/D' if pf['value'] is None else f"{pf['value']:.3f}"
            )
            conversion_display = 'N/D' if conversion['value'] is None else f"{conversion['value']:.2%}"
            summary += (
                "## Objetivos de scalping M5\n\n"
                f"Gate de validación **{'Cumplido' if quality['passed'] else 'Rechazado'}**.\n\n"
                "| Métrica | Objetivo | Validación | Estado |\n"
                "| --- | --- | ---: | --- |\n"
            )
            for label, target, value, check in (
                ('Acierto sin reembolsos', '≥ 56%', win_display, win),
                ('Profit Factor', '> 1,2', pf_display, pf),
                ('Contratos ejecutados por hora', f"{rate['minimum']:g} a {rate['maximum']:g}", f"{rate['value']:.4f}", rate),
                ('Conversión sobre todos los setups', '≥ 30%', conversion_display, conversion),
            ):
                status = 'Cumplido' if check['passed'] else 'Pendiente'
                summary += f"| {label} | {target} | {value} | {status} |\n"
            summary += '\n'
        (root/'validation.md').write_text(summary + "Holdout congelado. Sin prueba de ventaja fuera de muestra.\n")
    return result
