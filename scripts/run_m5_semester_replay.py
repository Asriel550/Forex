"""PR #32: replay diagnóstico enero-junio 2026 con OOS julio-septiembre aislado.

No entrena ni selecciona genomas. Utiliza únicamente pesos y normalizador
sellados en PR #29/#31; el corte histórico previo continúa siendo el default.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import load_minutes, m5_features
from scripts.run_m5_direction_benchmark import admission_blocks
from scripts.run_m5_adaptive_gate import GEN354_SHA256, RAW, simulate_admission
from scripts.run_m5_conviction_evolution import OUTPUT as EVOLUTION_OUTPUT, PERCENTILES, label, setup
from scripts.run_m5_gen333 import ARM
from scripts.run_m5_multiscale import MARCH_SIGNATURE, genome_hash
from src.artifacts import file_sha256
from src.evolution_context import native_direction_context
from src.evolution_lineage import MARKET
from src.evolution_multiscale import FEATURE_NAMES, M5_TACTICAL, arm_inputs, direction_scores, scale_inputs
from src.replay_engine import apply_hourly_cap, replay_contracts
from src.artifacts import write_json


OUTPUT = Path('reports/m5-semester-replay')
SOURCE_MANIFEST = Path('datasets/eurusd/manifest.json')
PERIODS = tuple(f'2026{month:02d}' for month in range(1, 7))
START = pd.Timestamp('2026-01-01T00:00:00Z')
Q2_START = pd.Timestamp('2026-04-01T00:00:00Z')
END = pd.Timestamp('2026-07-01T00:00:00Z')
OOS_END = pd.Timestamp('2026-10-01T00:00:00Z')
QUARTERS = (('Q1', START, Q2_START), ('Q2', Q2_START, END))
BOUNDARIES = pd.date_range(START, END, freq='MS')
MONTHS = tuple((PERIODS[i], BOUNDARIES[i], BOUNDARIES[i + 1]) for i in range(6))
MIN_PF, MIN_DENSITY, MAX_DRAWDOWN = 1.20, 1.20, 15.0


def assert_isolated(periods, *, start=START, end=END, oos_end=OOS_END):
    """No autoriza el acceso a julio-septiembre ni ventanas reetiquetadas."""
    if (tuple(periods) != PERIODS or start != START or end != END or oos_end != OOS_END
            or not START < Q2_START < END == OOS_END - pd.DateOffset(months=3)):
        raise ValueError('El replay semestral solo admite enero-junio; julio-septiembre es OOS cerrado')


def verify_source_periods(manifest):
    assert_isolated(PERIODS)
    by_period = {item['period']: item for item in manifest['sources']}
    if len(by_period) != len(manifest['sources']):
        raise ValueError('El manifiesto contiene periodos duplicados')
    for period in PERIODS:
        item = by_period.get(period)
        if not item or len(item['sha256']) != 64 or not item['zip_file'].endswith('.zip'):
            raise ValueError(f'Fuente M1 sin SHA256 para {period}')


def frozen_models(evolution, gen354):
    if (evolution['parent_sha256'] != GEN354_SHA256
            or evolution.get('protected_holdout_evaluated') is not False):
        raise ValueError('El linaje debe estar sellado antes de evaluar el semestre')
    result = {'gen354': dict(genome=gen354, generation=354, seed=43)}
    for percentile in (.80, .75):
        key = label(percentile)
        seed = evolution['winners'][key]
        item = next((r for r in evolution['replicas']
                     if r['percentile'] == percentile and r['seed'] == seed), None)
        if (item is None or item['champion']['generation'] != 375
                or genome_hash(item['genome']) != item['genome_sha256']):
            raise ValueError('Gen 375/43 incompatible con el sello de PR #31')
        result[f'gen375_{key}'] = dict(genome=item['genome'],
                                      generation=375, seed=seed)
    return result


def capacity(rows, hours, cap):
    accepted = apply_hourly_cap(rows.index, np.full(len(rows), 'accepted'), cap) == 'accepted'
    count = int(accepted.sum())
    return dict(contracts=count, contracts_per_observed_hour=count / hours)


def tested_metrics(result):
    metrics = result['operational']
    pf = metrics['profit_factor']
    return dict(
        **metrics,
        meets_pf=pf is not None and pf > MIN_PF,
        meets_density=metrics['contracts_per_observed_hour'] >= MIN_DENSITY,
        meets_drawdown=metrics['max_drawdown_pct'] < MAX_DRAWDOWN,
    )


def run():
    assert_isolated(PERIODS)
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    verify_source_periods(manifest)
    reference, founder, quant, risk, gen354, _, _, mean, std = setup()
    if genome_hash(gen354) != GEN354_SHA256:
        raise ValueError('El fundador no coincide con la Gen 354/43')
    evolution_path = EVOLUTION_OUTPUT / 'evolution.json'
    evolution = json.loads(evolution_path.read_text(encoding='utf-8'))
    models = frozen_models(evolution, gen354)

    # Únicamente seis ZIP M1 (enero-junio); julio-septiembre nunca se descarga.
    minutes = load_minutes(RAW, PERIODS, end=END)
    data = m5_features(minutes, quant, end=END)
    # Meses independientes para conservar la elegibilidad exacta de marzo:
    # una partición Q1 única pierde la primera propuesta del mes.
    windows, _ = admission_blocks(data, founder, MONTHS, evaluation_limit=END)
    quarter_rows = {
        'Q1': pd.concat([windows[period].rows for period in PERIODS[:3]]),
        'Q2': pd.concat([windows[period].rows for period in PERIODS[3:]]),
    }
    rows = pd.concat((quarter_rows['Q1'], quarter_rows['Q2']))
    if (rows.empty or not rows.index.is_monotonic_increasing or not rows.index.is_unique
            or rows.index.min() < START or rows.index.max() >= END):
        raise ValueError('Propuestas fuera de IS semestral o mal ordenadas')

    # Auditoría contra la misma máscara de marzo usada en la PR #30/#31.
    march = windows['202603'].rows
    march_opens = (march.index - MARKET.step).as_unit('ns')
    march_hash = sha256(march_opens.asi8.tobytes()).hexdigest()
    march_matches = len(march) == MARCH_SIGNATURE[1] and march_hash == MARCH_SIGNATURE[2]
    if not march_matches:
        raise ValueError(f'Máscara de marzo no reproducible: {len(march)} / {march_hash}')

    native = native_direction_context(minutes, rows.index, end=END)
    features = rows.loc[:, list(M5_TACTICAL)].join(native).loc[:, list(FEATURE_NAMES)]
    inputs = arm_inputs(scale_inputs(features.to_numpy(dtype=float), mean, std), ARM)
    hours = {period: windows[period].observed_hours for period in PERIODS}
    hours['Q1'] = sum(hours[period] for period in PERIODS[:3])
    hours['Q2'] = sum(hours[period] for period in PERIODS[3:])
    hours['semester'] = hours['Q1'] + hours['Q2']
    cap = reference['gate']['hourly_cap']
    reports, traces = {}, []
    capacities = {period: capacity(windows[period].rows, hours[period], cap) for period in PERIODS}
    capacities.update({name: capacity(quarter_rows[name], hours[name], cap) for name, _, _ in QUARTERS})
    capacities['semester'] = capacity(rows, hours['semester'], cap)

    for model_name, model in models.items():
        scores = direction_scores(inputs, [model['genome']])[0]
        side = np.where(scores >= 0., 1, -1)
        # Arranque frío con las primeras 500 propuestas de enero. La convicción
        # continúa entre trimestres; nunca recibe labels futuros antes de vencer.
        model_report = {}
        for percentile in PERCENTILES:
            gate = simulate_admission(rows, scores, (), percentile=percentile, elastic=True)
            results = {}
            for name, start, end in (*MONTHS, *QUARTERS, ('semester', START, END)):
                selected = (rows.index >= start) & (rows.index < end)
                window = rows.loc[selected]
                state = gate.loc[selected]
                outcome, trace = replay_contracts(
                    window, side[selected], founder.account, risk,
                    observed_hours=hours[name], hourly_cap=cap,
                    candidate_reasons=state.reason.to_numpy(),
                )
                results[name] = dict(
                    metrics=tested_metrics(outcome),
                    gate=dict(
                        distinct_cutoffs=int(state.cutoff.nunique()),
                        min_cutoff=float(state.cutoff.min()),
                        max_cutoff=float(state.cutoff.max()),
                        stages=sorted(state.percentile.unique().tolist(), reverse=True),
                        fully_open=int((state.percentile == 0.).sum()),
                        warmup_rejections=int((state.reason == 'conviction_warmup').sum()),
                    ),
                    filters=outcome['filters'],
                    gate_only_at_expiry=outcome['gate_only_at_expiry'],
                )
                if name == 'semester':
                    traces.append(trace.assign(model=model_name, percentile=label(percentile)))
            model_report[label(percentile)] = results
        reports[model_name] = dict(generation=model['generation'], seed=model['seed'],
                                   percentiles=model_report, sha256=genome_hash(model['genome']))

    report = dict(
        protocol='PR32_semester_diagnostic_v1',
        is_start=START.isoformat(), is_end_exclusive=END.isoformat(),
        new_oos_start=END.isoformat(), new_oos_end_exclusive=OOS_END.isoformat(),
        new_oos_accessed=False, old_2026_april_holdout_boundary_crossed=True,
        evolution_or_retraining_performed=False,
        source_manifest_sha256=file_sha256(SOURCE_MANIFEST),
        evolution_sha256=file_sha256(evolution_path),
        march_mask_reproduced=march_matches, march_mask_sha256=march_hash,
        cold_start_proposals=500, elapsed_hours=hours,
        targets=dict(profit_factor_gt=MIN_PF, contracts_per_hour_gte=MIN_DENSITY,
                     drawdown_pct_lt=MAX_DRAWDOWN),
        capacities=capacities, models=reports,
        monthly_risk_policy='account reset per month; exact March admission mask',
        quarter_risk_policy='account reset per quarter; gate state continuous',
        semester_risk_policy='single account throughout Jan-Jun; gate state continuous',
    )
    write_json(OUTPUT / 'semester.json', report)
    pd.concat(traces, ignore_index=True).to_csv(OUTPUT / 'admissions.csv', index=False)
    print('SEMESTER_REPLAY ' + json.dumps({
        'march_mask_reproduced': march_matches, 'hours': hours, 'capacity': capacities,
        'models': {name: {p: {w: value['metrics'] for w, value in periods.items()}
                          for p, periods in info['percentiles'].items()}
                   for name, info in reports.items()},
        'new_oos_accessed': False,
    }, allow_nan=False), flush=True)


if __name__ == '__main__':
    run()
