"""PR #33: evolución Gen 376+ optimizada por dos trimestres y loss tanh.

Mantiene el núcleo de población de 128, mutación, cruce, élites y semillas de
src.evolution_multiscale.evolve_population. No observa Q3 (OOS).
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import load_minutes, m5_features
from scripts.run_m5_direction_benchmark import admission_blocks
from scripts.run_m5_adaptive_gate import RAW, simulate_admission
from scripts.run_m5_conviction_evolution import OUTPUT as GEN375_OUTPUT, PERCENTILES, label, setup
from scripts.run_m5_gen333 import ARM
from scripts.run_m5_multiscale import MARCH_SIGNATURE, genome_hash
from scripts.run_m5_semester_replay import (
    END, MONTHS, PERIODS, Q2_START, START, SOURCE_MANIFEST,
    assert_isolated, verify_source_periods,
)
from src.adaptive_gate import CONVICTION_WINDOW, MASTERY_ACCURACY, rolling_conviction_admissions
from src.artifacts import file_sha256
from src.evolution_context import native_direction_context
from src.evolution_engine import assess_temporal_multiregime_fitness
from src.evolution_lineage import MARKET
from src.evolution_multiscale import (
    FEATURE_NAMES, M5_TACTICAL, POPULATION, SEEDS, arm_inputs, direction_scores,
    evolve_population, scale_inputs,
)
from src.model_training import directional_tanh_loss
from src.reproducibility import market_frame_fingerprint, admissions_fingerprint, canonical_open_diff, array_sha256
from src.resampling import aggregate
from src.replay_engine import replay_contracts
from src.train import write_json


OUTPUT = Path('reports/m5-temporal-evolution')
FIRST = 376
MAX_GENERATIONS = 21


def frozen_gen375(founder_sha, evolution):
    if (evolution['parent_sha256'] != founder_sha
            or evolution.get('protected_holdout_evaluated') is not False):
        raise ValueError('PR31 sin sello o ya evaluó un holdout')
    chosen = evolution['winners']['p80']
    replica = next((r for r in evolution['replicas']
                    if r['percentile'] == .80 and r['seed'] == chosen), None)
    if (replica is None or chosen != 43
            or replica['champion']['generation'] != 375
            or genome_hash(replica['genome']) != replica['genome_sha256']):
        raise ValueError('Se requiere el campeón Gen 375 / 43 / p80 sellado')
    return np.asarray(replica['genome'], dtype=float)


def causal_reasons(rows, scores, percentile):
    """Vectoriza el caso restrictivo; usa gate exacto si es posible maestría.

    La comprobación anticipada solo detecta si una ventana de 500 aciertos
    alcanzaría 95%; si no es posible, ningún gate podría abrirse. El resultado
    equivale a aplicar el gate causal en cada decisión.
    """
    outcomes = rows.outcome.to_numpy()
    direction = np.where(scores >= 0., 1, -1)
    settled = outcomes != 0
    correct = (direction[settled] == outcomes[settled]).astype(np.int64)
    if len(correct) >= CONVICTION_WINDOW:
        cumul = np.r_[0, np.cumsum(correct)]
        possible_mastery = bool(np.any(
            cumul[CONVICTION_WINDOW:] - cumul[:-CONVICTION_WINDOW]
            >= CONVICTION_WINDOW * MASTERY_ACCURACY
        ))
        if possible_mastery:
            return simulate_admission(rows, scores, (), percentile=percentile,
                                      elastic=True).reason.to_numpy()
    accept = rolling_conviction_admissions(scores, percentile)
    reasons = np.where(accept, 'accepted', 'conviction').astype('<U24')
    reasons[:CONVICTION_WINDOW] = 'conviction_warmup'
    return reasons


def prepare_is(*, audit_path=None, snapshot_path=None, snapshot_manifest=None):
    """Q1+Q2 en IS, normalizador original; jamás solicita Q3.

    Con snapshot sellado se omite completamente M1/M5/GARCH y se verifica el
    SHA256 antes de evaluar una sola propuesta.
    """
    if snapshot_path is not None:
        from scripts.m5_is_repro_audit import load_verified_snapshot
        return load_verified_snapshot(snapshot_path, snapshot_manifest)

    assert_isolated(PERIODS)
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding='utf-8'))
    verify_source_periods(manifest)
    reference, founder, quant, risk, gen354, _, _, mean, std = setup()
    archive_path = GEN375_OUTPUT / 'evolution.json'
    evolution = json.loads(archive_path.read_text(encoding='utf-8'))
    from scripts.run_m5_adaptive_gate import GEN354_SHA256
    parent = frozen_gen375(GEN354_SHA256, evolution)

    minutes = load_minutes(RAW, PERIODS, end=END)
    m5_candles = aggregate(minutes, 1, 'M5', now=END)
    fingerprints = dict(
        m1=market_frame_fingerprint(minutes, columns=('open', 'high', 'low', 'close', 'volume')),
        m5=market_frame_fingerprint(m5_candles, columns=('open', 'high', 'low', 'close', 'volume')),
        source_zip_sha256={source['period']: source['sha256']
                           for source in manifest['sources'] if source['period'] in PERIODS},
    )
    features = m5_features(minutes, quant, end=END)
    fingerprints['garch_and_mc'] = market_frame_fingerprint(
        features, columns=('cond_volatility', 'mc_prob_up', 'mc_prob_down', 'mc_prob_tie'))
    fingerprints['garch_retained_refits'] = features.attrs.get('garch_retained_refits', [])
    windows, _ = admission_blocks(features, founder, MONTHS, evaluation_limit=END)
    rows = pd.concat([windows[period].rows for period in PERIODS])
    if (rows.empty or not rows.index.is_monotonic_increasing or not rows.index.is_unique
            or rows.index.min() < START or rows.index.max() >= END):
        raise ValueError('Datos IS fuera de enero-junio o sin orden')
    march = windows['202603'].rows
    march_hash = sha256((march.index - MARKET.step).as_unit('ns').asi8.tobytes()).hexdigest()
    fingerprints['month_admissions'] = {
        period: admissions_fingerprint(windows[period].rows, step=MARKET.step)
        for period in PERIODS
    }
    fingerprints['march_audit'] = dict(
        canonical_proposals=MARCH_SIGNATURE[1], canonical_sha256=MARCH_SIGNATURE[2],
        matched=len(march) == MARCH_SIGNATURE[1] and march_hash == MARCH_SIGNATURE[2],
    )
    if not fingerprints['march_audit']['matched']:
        from scripts.run_m5_multiscale import archived_march_opens
        fingerprints['march_audit']['divergence'] = canonical_open_diff(
            march.index - MARKET.step, archived_march_opens()
        )
    if audit_path is not None:
        import importlib.metadata
        fingerprints['versions'] = {pkg: importlib.metadata.version(pkg)
                                    for pkg in ('numpy', 'pandas', 'scipy', 'arch')}
        dest = Path(audit_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(fingerprints, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not fingerprints['march_audit']['matched']:
        raise ValueError('Máscara de marzo incompatible con el linaje: '
                         + json.dumps(fingerprints['march_audit'], allow_nan=False))

    native = native_direction_context(minutes, rows.index, end=END)
    raw = rows.loc[:, list(M5_TACTICAL)].join(native).loc[:, list(FEATURE_NAMES)]
    inputs = arm_inputs(scale_inputs(raw.to_numpy(dtype=float), mean, std), ARM)
    fingerprints['model_inputs_sha256'] = array_sha256(inputs)
    fingerprints['outcomes_sha256'] = array_sha256(rows.outcome.to_numpy(), dtype='<i1')
    fingerprints['model_row_times_sha256'] = array_sha256(rows.index.as_unit('ns').asi8, dtype='<i8')
    q1 = rows.index < Q2_START
    quarters = (
        dict(name='Q1', rows=rows.loc[q1],
             observed_hours=sum(windows[period].observed_hours for period in PERIODS[:3]),
             mask=q1),
        dict(name='Q2', rows=rows.loc[~q1],
             observed_hours=sum(windows[period].observed_hours for period in PERIODS[3:]),
             mask=~q1),
    )
    if any(len(q['rows']) < CONVICTION_WINDOW or q['observed_hours'] <= 0 for q in quarters):
        raise ValueError('Q1 y Q2 requieren propuestas e historial suficientes')
    # Percentil de volatilidad fit en Q1 exclusivamente, nunca con etiquetas OOS.
    vols = rows.cond_volatility.to_numpy(dtype=float)
    if not np.isfinite(vols).all() or (vols < 0).any():
        raise ValueError('Volatilidad IS inválida')
    vol_cut = float(np.quantile(vols[q1], 2/3))
    high_vol = vols > vol_cut
    if audit_path is not None:
        Path(audit_path).write_text(json.dumps(fingerprints, indent=2, allow_nan=False) + '\n',
                                    encoding='utf-8')
    return dict(
        rows=rows, inputs=inputs, quarters=quarters, high_vol=high_vol,
        vol_cut=vol_cut, parent=parent, account=founder.account, risk=risk,
        cap=reference['gate']['hourly_cap'],
        hashes=dict(march_mask_sha256=march_hash, source_manifest_sha256=file_sha256(SOURCE_MANIFEST),
                    gen375_evolution_sha256=file_sha256(archive_path),
                    founding_genome_sha256=genome_hash(parent)),
        fingerprints=fingerprints,
    )


def score_individual(scores, sample, percentile):
    rows = sample['rows']
    side = np.where(scores >= 0., 1, -1)
    reasons = causal_reasons(rows, scores, percentile)
    quarterly = []
    for q in sample['quarters']:
        m = q['mask']
        replay, _ = replay_contracts(
            q['rows'], side[m], sample['account'], sample['risk'],
            observed_hours=q['observed_hours'], hourly_cap=sample['cap'],
            candidate_reasons=reasons[m],
        )
        quarterly.append(dict(**replay['operational'],
                              admissions=len(q['rows']), observed_hours=q['observed_hours']))
    loss = directional_tanh_loss(scores, rows.outcome.to_numpy(), sample['high_vol'])
    fitness = assess_temporal_multiregime_fitness(
        quarterly, stake=sample['account'].stake, direction_loss=loss['total']
    )
    fitness.update(loss_direction=loss['direction'],
                   loss_high_vol_false_positive=loss['high_vol_false_positive'],
                   high_vol_wrong_count=loss['high_vol_wrong_count'],
                   quarter_pnl_usd=[q['pnl_usd'] for q in quarterly],
                   quarter_pf=[q['profit_factor'] for q in quarterly],
                   quarter_trades=[q['trades'] for q in quarterly])
    return fitness


def rank(fitness):
    return (fitness['qualifies'], fitness['score'], fitness['worst_quarter_return'],
            -fitness['quarter_return_std'], fitness['min_quarter_frequency'])


def run(*, generations=MAX_GENERATIONS, seeds=SEEDS, percentiles=PERCENTILES):
    if (type(generations) is not int or not 1 <= generations <= MAX_GENERATIONS
            or not seeds or not set(seeds).issubset(SEEDS)
            or len(seeds) != len(set(seeds))
            or not percentiles or not set(percentiles).issubset(PERCENTILES)
            or len(percentiles) != len(set(percentiles))):
        raise ValueError('Parámetros fuera del protocolo Gen 376+')
    output = OUTPUT / ('evolution-full.json' if generations == MAX_GENERATIONS and
                       set(seeds) == set(SEEDS) and set(percentiles) == set(PERCENTILES)
                       else f'evolution-smoke-{generations}g-{len(seeds)}s-{len(percentiles)}p.json')
    if output.exists():
        raise ValueError('El archivo de la ejecución ya existe; no sobrescribir el sello')
    sample = prepare_is()
    replicas = []
    for seed in seeds:
        for percentile in percentiles:
            counts = dict(qualifies=0)

            def evaluate(population):
                records = [score_individual(scores, sample, percentile)
                           for scores in direction_scores(sample['inputs'], population)]
                counts['qualifies'] += sum(int(rec['qualifies']) for rec in records)
                return {key: np.asarray([rec[key] for rec in records])
                        for key in records[0]}

            result = evolve_population(
                evaluate, sample['parent'].copy(), ARM, seed,
                first_generation=FIRST, last_generation=FIRST + generations - 1, ranking=rank,
            )
            result.update(
                seed=seed, percentile=percentile,
                genome_sha256=genome_hash(result['genome']),
                candidates_qualifying=counts['qualifies'],
            )
            # Verifica repetición con el mismo estado y datos, sin consultar OOS.
            expected = score_individual(
                direction_scores(sample['inputs'], [result['genome']])[0], sample, percentile
            )
            for name in ('score', 'worst_quarter_return', 'quarter_return_std'):
                if not np.isclose(expected[name], result['champion'][name], atol=1e-11, rtol=0):
                    raise ValueError('El campeón no reproduce su fitness temporal: ' + name)
            replicas.append(result)
            print(f'Gen {FIRST + generations - 1}, seed {seed}, {label(percentile)}: '
                  f'score={expected["score"]:.4f}; PF_Q1_Q2={expected["quarter_pf"]}', flush=True)

    report = dict(
        protocol='PR33_temporal_IS_v1',
        first_generation=FIRST, last_generation=FIRST + generations - 1,
        population=POPULATION, seeds=list(seeds), percentiles=list(percentiles),
        evaluations=sum(r['evaluations'] for r in replicas),
        is_start=START.isoformat(), is_end_exclusive=END.isoformat(),
        oos_start=END.isoformat(), oos_end_exclusive='2026-10-01T00:00:00+00:00',
        oos_accessed=False, q1_q2_both_in_sample=True,
        selection_requires_both_quarters=True,
        high_volatility_training_cutoff_from_q1=sample['vol_cut'],
        hash_audit=sample['hashes'],
        interpretation='research: never promote a non-qualifying champion',
        replicas=replicas,
    )
    write_json(output, report)
    print('TEMPORAL_FITNESS ' + json.dumps({
        'output': str(output), 'evaluations': report['evaluations'],
        'winner_candidates': [{
            'seed': r['seed'], 'percentile': label(r['percentile']),
            'generation': r['champion']['generation'],
            'qualifies': r['champion']['qualifies'],
            'score': r['champion']['score'],
            'quarter_pf': r['champion']['quarter_pf'],
            'quarter_pnl_usd': r['champion']['quarter_pnl_usd'],
        } for r in replicas], 'oos_accessed': False,
    }, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations', type=int, default=MAX_GENERATIONS)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--percentiles', nargs='+', type=float, default=list(PERCENTILES))
    args = parser.parse_args()
    run(generations=args.generations, seeds=tuple(args.seeds), percentiles=tuple(args.percentiles))
