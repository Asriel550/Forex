"""Desacopla oportunidades MC, propuestas PPO y ejecución, SOLO train/validation.

Las oportunidades se cuentan antes de liquidar y sin depender del saldo PPO.
La simulación PPO conserva su propio estado y stop; no simula órdenes brutas
sin filtro ni interpreta probabilidades GBM como probabilidades calibradas.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import importlib.metadata
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .agent import load_agent
from .artifacts import file_sha256
from .decision import RiskConfig, ensure_probabilities, mc_probability_lower
from .environment import AccountConfig, BinaryOptionsEnv, STEP, prepare_market_data
from .quant_engine import QuantConfig
from .metrics import calculate_metrics
from .sensitivity import evaluate_mc_baseline
from .train import chronological_split, train, write_json

ROOT = Path(__file__).resolve().parents[1]
LABELS = ('PUT', 'NEUTRO', 'CALL')


def distribution(actions):
    values = np.asarray(actions, dtype=int)
    return {name: dict(count=int((values == i).sum()),
                       percentage=float(100 * (values == i).mean()) if len(values) else 0.)
            for i, name in enumerate(LABELS)}


def mc_proposals(frame, *, risk, account, n_paths=10000):
    """Una fila por transición consecutiva; no lee precios de liquidación."""
    frame = ensure_probabilities(frame)
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).as_unit('ns')
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing or not times.equals(times.floor('15min')):
        raise ValueError('Timestamps inválidos')
    indices = np.flatnonzero(np.diff(times.asi8) == STEP.value)
    up = frame.mc_prob_up.to_numpy()[indices]
    down = frame.mc_prob_down.to_numpy()[indices]
    direction = np.where(up > down, 2, np.where(down > up, 0, 1))
    confidence = np.maximum(up, down)
    lower = mc_probability_lower(confidence, n_paths, risk.mc_z)
    edge = account.payout * lower - (1 - lower)
    passes_confidence = confidence > risk.min_confidence
    allowed = direction != 1
    if risk.enabled:
        allowed &= passes_confidence & (edge > risk.min_expected_edge)
    return pd.DataFrame(dict(entry_index=indices,
        decision_time=(times[indices] + STEP).astype(str), mc_raw_action=direction,
        mc_action=np.where(allowed, direction, 1), mc_confidence=confidence,
        mc_passes_confidence=passes_confidence, mc_expected_edge_lower=edge))


def diagnose_policy(model, frame, *, normalizer, risk, account, lookback=16):
    """Compara en las MISMAS velas; stops, huecos y estados son los del entorno.

    El motor se cuenta sobre todo el subconjunto y sobre la exposición efectiva
    de PPO por separado. Cambiar close[t+1] no cambia su propuesta en t.
    """
    n_paths = int(frame.attrs.get('quant_config', {}).get('n_paths', 10000))
    proposals = mc_proposals(frame, risk=risk, account=account, n_paths=n_paths)
    indexed = proposals.set_index('entry_index')
    env = BinaryOptionsEnv(frame, prepared=True, normalizer=normalizer,
                          lookback=lookback, risk=risk, account=account)
    rows = []
    balances = np.full(len(frame), np.nan)
    balance = peak = account.initial_balance
    activity = 0.
    filled = 0
    stopped = False
    try:
        for start, _ in env.segments:
            balances[filled:start+1] = balance
            filled = start+1
            obs, state = env.reset(options=dict(start_index=start, balance=balance,
                                    peak_balance=peak, trade_activity=activity))
            while True:
                index = state['index']
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                row = indexed.loc[index].to_dict()
                row.update(entry_index=index, requested_action=info['requested_action'],
                    action=info['action'], blocked=info['blocked'],
                    filter_reason=info['filter_reason'], profit=info['profit'],
                    outcome=info['outcome'], balance=info['balance'])
                rows.append(row)
                balance, peak, activity = info['balance'], info['peak_balance'], info['trade_activity']
                balances[info['index']] = balance
                filled = info['index']+1
                state = info
                if terminated or truncated:
                    stopped = terminated
                    break
            if stopped:
                break
    finally:
        env.close()
    balances[filled:] = balance
    trace = pd.DataFrame(rows)
    trades = trace.loc[trace.action != 1]
    metrics = calculate_metrics(trades, pd.DataFrame(dict(timestamp=frame.timestamp+STEP,
                    balance=balances)), initial_balance=account.initial_balance)
    eligible = trace.mc_action != 1
    agrees = eligible & (trace.requested_action == trace.mc_action)
    all_eligible = int((proposals.mc_action != 1).sum())
    metrics.update(mc_opportunities_full_phase=all_eligible,
        mc_full_phase_distribution=distribution(proposals.mc_action),
        mc_raw_full_phase_distribution=distribution(proposals.mc_raw_action),
        mc_confidence_pass_full_phase=int(proposals.mc_passes_confidence.sum()),
        mc_confidence_rejected_full_phase=int((~proposals.mc_passes_confidence).sum()),
        mc_edge_rejected_after_confidence_full_phase=int((proposals.mc_passes_confidence & (proposals.mc_expected_edge_lower<=risk.min_expected_edge)).sum()),
        mc_opportunity_rate_full_phase=all_eligible/len(proposals) if len(proposals) else None,
        consecutive_transitions=len(proposals), evaluated_decisions=len(trace),
        gap_transitions=len(frame)-1-len(proposals), risk_stop=bool(stopped),
        skipped_after_stop=len(proposals)-len(trace),
        raw_ppo_distribution=distribution(trace.requested_action),
        executed_distribution=distribution(trace.action),
        mc_opportunities_on_ppo_exposure=int(eligible.sum()),
        mc_opportunities_taken=int(agrees.sum()),
        mc_opportunities_skipped_neutral=int((eligible & (trace.requested_action == 1)).sum()),
        mc_opportunities_opposed=int((eligible & (trace.requested_action != 1) & ~agrees).sum()),
        opportunity_capture_rate=float(agrees.sum()/eligible.sum()) if eligible.any() else None,
        blocked_requests=int(trace.blocked.sum()),
        filter_reasons={str(k): int(v) for k, v in trace.filter_reason.value_counts().items()},
        cross_table=[dict(mc=LABELS[i], ppo_requested=LABELS[j],
                    count=int(((trace.mc_action == i) & (trace.requested_action == j)).sum()))
                    for i in range(3) for j in range(3)])
    return metrics, trace


def dispersion(rows):
    """Dispersión entre semillas; PF indefinidos/infinitos se cuentan aparte."""
    result = {}
    for phase in ('train', 'validation'):
        part = [r for r in rows if r['phase'] == phase]
        result[phase] = {}
        for key in ('total_trades', 'win_rate_excluding_refunds', 'profit_factor',
                    'net_profit', 'opportunity_capture_rate'):
            values = np.array([r[key] for r in part if r[key] is not None], dtype=float)
            finite = values[np.isfinite(values)]
            result[phase][key] = dict(finite_count=len(finite), missing_count=len(part)-len(finite),
                mean=float(finite.mean()) if len(finite) else None,
                median=float(np.median(finite)) if len(finite) else None,
                sample_std=float(finite.std(ddof=1)) if len(finite)>1 else None,
                minimum=float(finite.min()) if len(finite) else None,
                maximum=float(finite.max()) if len(finite) else None)
        result[phase]['profit_factor_status_counts'] = {
            status: sum(r['profit_factor_status'] == status for r in part)
            for status in sorted({r['profit_factor_status'] for r in part})}
    return result


def run_experiments(input_path, *, report_dir=ROOT/'reports/diagnostics',
                    seeds=(42, 43, 44), total_timesteps=100000, train_kwargs=None):
    """Hiperparámetros PR #8 fijos; varía solo seed PPO/reset, nunca seed MC.

    No importa backtest, no ejecuta test, no publica/promueve bundles. Los pesos
    quedan en un directorio exclusivo por seed, fuera de models de producción.
    """
    seeds = tuple(seeds)
    if not 1 <= len(seeds) <= 10 or len(set(seeds)) != len(seeds) or any(
            not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Usar 1..10 semillas enteras únicas entre 0 y 2**32-1')
    options = dict(train_kwargs or {})
    if set(options) - {'lookback','indicator_period','n_steps','batch_size','n_epochs','eval_freq','episode_length','verbose'}:
        raise ValueError('Solo opciones de ejecución/smoke; estrategia PR #8 fija')
    input_path = Path(input_path)
    source_hash = file_sha256(input_path)
    raw = pd.read_parquet(input_path)
    cfg = raw.attrs.get('quant_config', {})
    if asdict(QuantConfig.from_metadata(cfg)) != asdict(QuantConfig()):

        raise ValueError('Requiere configuración cuantitativa PR #8 completa y congelada')
    lookback = options.get('lookback', 16)
    period = options.get('indicator_period', 14)
    splits = chronological_split(prepare_market_data(raw, lookback, period))
    # Se descarta la referencia al holdout antes de cualquier evaluador.
    del splits['test']
    risk, account = RiskConfig(), AccountConfig()
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for phase, frame in splits.items():
        baselines[phase] = evaluate_mc_baseline(frame, risk=risk, account=account, n_paths=10000)
    rows = []
    for seed in seeds:
        model_dir = report_dir / f'seed_{seed}'
        # No se sobreescribe un experimento previo por accidente.
        if model_dir.exists():
            raise ValueError(f'Directorio de semilla ya existe: {model_dir.name}')
        manifest = train(input_path, output_dir=model_dir, seed=seed, total_timesteps=total_timesteps,
                         ent_coef=.02, risk=risk, account=account, **options)
        normalizer = json.loads((model_dir/'normalizer.json').read_text())
        model = load_agent(model_dir/'best_ppo_agent.zip')
        for phase, frame in splits.items():
            metrics, trace = diagnose_policy(model, frame, normalizer=normalizer,
                        risk=risk, account=account, lookback=lookback)
            trace.to_csv(model_dir/f'{phase}_decisions.csv', index=False)
            row = dict(seed=seed, phase=phase, actual_timesteps=manifest['actual_timesteps'], **metrics)
            rows.append(row)
            write_json(model_dir/f'{phase}_diagnostics.json', row)
        if file_sha256(input_path) != source_hash:
            raise ValueError('Input cambió durante experimento')
    report = dict(scope='train and validation only', holdout_evaluated=False,
        automatic_promotion=False, automatic_seed_selection=False, source_sha256=source_hash, software_versions=manifest['versions'],
        seeds=list(seeds), requested_timesteps=total_timesteps, ent_coef=.02,
        risk=asdict(risk), account=asdict(account), quant_config=cfg,
        splits={k: dict(rows=len(f), start=f.timestamp.iloc[0].isoformat(),
                       end=f.timestamp.iloc[-1].isoformat()) for k,f in splits.items()},
        mc_baselines=baselines, results=rows, dispersion=dispersion(rows),
        caveat='One frozen data snapshot; seeds are not independent market samples. '
            'Observational decomposition does not establish causal attribution to reward. '
            'MC probabilities are uncalibrated; no winner is chosen or promoted.')
    export_report(report, report_dir)
    logging.info('Diagnóstico completado: %d semillas; holdout no evaluado', len(seeds))
    return report


def export_report(report, report_dir):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = report['results']
    write_json(report_dir/'diagnostics.json', report)
    scalar = [{k:v for k,v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    pd.DataFrame(scalar).to_csv(report_dir/'diagnostics.csv', index=False)
    text = ['# Diagnóstico train/validation: filtro vs PPO', '',
        'Holdout congelado. Alcance: '+report['scope']+'.',
        f"Entropía {report['ent_coef']}; confianza {report['risk']['min_confidence']:.0%}; escala {report['quant_config'].get('drift_scale')}.", '',
        '| Seed | Fase | MC elegibles (fase) | MC elegibles (exposición PPO) | Capturadas | PPO NEUTRO % | Contratos | Acierto | PF | PnL USD |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |']
    for r in rows:
        wr = r['win_rate_excluding_refunds']
        text.append(f"| {r['seed']} | {r['phase']} | {r['mc_opportunities_full_phase']} | "
            f"{r['mc_opportunities_on_ppo_exposure']} | {r['mc_opportunities_taken']} | "
            f"{r['raw_ppo_distribution']['NEUTRO']['percentage']:.3f} | {r['total_trades']} | "
            f"{f'{wr:.2%}' if wr is not None else 'N/D'} | "
            f"{r['profit_factor'] if r['profit_factor'] is not None else r['profit_factor_status']} | {r['net_profit']:.2f} |")
    text += ['', '## Dispersión entre semillas', '',
        '| Fase | Métrica | N finitos | N sin valor finito | Media | Mediana | Desv. muestral | Mín | Máx |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for phase,metrics in report['dispersion'].items():
        for name,values in metrics.items():
            if name == 'profit_factor_status_counts':
                continue
            stats = ['N/D' if values[k] is None else f'{values[k]:.5f}'
                     for k in ('mean','median','sample_std','minimum','maximum')]
            text.append(f"| {phase} | {name} | {values['finite_count']} | {values['missing_count']} | "+' | '.join(stats)+' |')
    text += ['', 'La tabla usa el checkpoint elegido por validación de cada semilla; '
        'no selecciona una semilla ganadora. El JSON contiene distribución, cruce MC/PPO, '
        'motivos de bloqueo, baseline MC con stop y dispersión (media/mediana/desviación/mín/máx).', '',
        report['caveat'], '']
    (report_dir/'diagnostics.md').write_text('\n'.join(text))
    logging.info('Resumen MC/PPO: %s', json.dumps([{k:r[k] for k in
        ('seed','phase','mc_opportunities_full_phase','mc_opportunities_taken',
         'blocked_requests','total_trades','win_rate_excluding_refunds','profit_factor','profit_factor_status','net_profit')}
         for r in rows], allow_nan=False))
    return report


def diagnose_frozen_bundle(input_path, model_dir, *, report_dir):
    """Inspecciona PPO archivado antes del barrido, sin entrenar ni usar test.

    Verifica hashes y límites de train/validation del manifiesto. El Parquet
    debe ser el original del bundle, no una regeneración con otras versiones.
    """
    input_path, model_dir = Path(input_path), Path(model_dir)
    manifest = json.loads((model_dir/'training_manifest.json').read_text())
    if (manifest.get('manifest_version') != 3 or manifest.get('actual_timesteps',0) <= 0
            or manifest.get('reward_version') not in ('risk_adjusted_v1','risk_adjusted_v2')
            or manifest.get('normalizer_fit') != 'train only'
            or manifest.get('holdout_test_evaluated') is not False
            or not np.isclose(manifest['train_fraction']+manifest['validation_fraction'],.90)):
        raise ValueError('Bundle v3 con entrenamiento completado y split 70/20/10 requerido')
    for path,key in [(input_path,'input_sha256'),
                     (model_dir/'best_ppo_agent.zip','best_model_sha256'),
                     (model_dir/'normalizer.json','normalizer_sha256')]:
        if file_sha256(path) != manifest.get(key):
            raise ValueError(f'Hash inválido para {key}')
    raw = pd.read_parquet(input_path)
    lookback = manifest['lookback']
    splits = chronological_split(prepare_market_data(raw,lookback,manifest['indicator_period']),
        train_fraction=manifest['train_fraction'], validation_fraction=manifest['validation_fraction'],
        purge_bars=manifest['purge_bars'])
    del splits['test']
    for phase,frame in splits.items():
        expected = manifest['splits'][phase]
        actual = dict(rows=len(frame),start=frame.timestamp.iloc[0].isoformat(),end=frame.timestamp.iloc[-1].isoformat())
        if actual != expected:
            raise ValueError('Límites train/validation distintos al bundle')
    risk = RiskConfig.from_metadata(manifest['risk'])
    account = AccountConfig(**manifest['account'])
    model = load_agent(model_dir/'best_ppo_agent.zip')
    normalizer = json.loads((model_dir/'normalizer.json').read_text())
    if model.ent_coef != manifest.get('ent_coef',.01):
        raise ValueError('ent_coef del PPO distinto al manifiesto')
    rows = []
    baselines = {}
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    n_paths = int(raw.attrs['quant_config']['n_paths'])
    for phase,frame in splits.items():
        baselines[phase] = evaluate_mc_baseline(frame,risk=risk,account=account,n_paths=n_paths)
        metrics,trace = diagnose_policy(model,frame,normalizer=normalizer,
                        risk=risk,account=account,lookback=lookback)
        trace.to_csv(report_dir/f'{phase}_decisions.csv',index=False)
        rows.append(dict(seed=manifest['seed'],phase=phase,actual_timesteps=manifest['actual_timesteps'],**metrics))
    if file_sha256(input_path) != manifest['input_sha256']:
        raise ValueError('Input cambió durante diagnóstico')
    report = dict(scope='train and validation only; frozen archived PPO',holdout_evaluated=False,
        automatic_promotion=False,automatic_seed_selection=False,
        source_sha256=manifest['input_sha256'],software_versions=manifest['versions'],
        diagnostic_runtime_versions={k:importlib.metadata.version(k) for k in manifest['versions']},model_sha256=manifest['best_model_sha256'],
        ent_coef=manifest.get('ent_coef',.01),risk=asdict(risk),account=asdict(account),
        quant_config=raw.attrs['quant_config'],mc_baselines=baselines,results=rows,
        dispersion=dispersion(rows),caveat='Archived model, no training. Train metrics are in sample; '
            'validation previously used for checkpoint selection. Counts do not establish reward causality.')
    export_report(report,report_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/eurusd_m15_features.parquet')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/diagnostics')
    parser.add_argument('--model-dir', type=Path, help='Diagnosticar bundle archivado; no entrena')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    parser.add_argument('--total-timesteps', type=int, default=100000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.model_dir is not None:
        diagnose_frozen_bundle(args.input,args.model_dir,report_dir=args.report_dir)
        return
    run_experiments(args.input, report_dir=args.report_dir, seeds=args.seeds,
                    total_timesteps=args.total_timesteps)


if __name__ == '__main__':
    main()
