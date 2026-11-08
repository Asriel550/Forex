"""Evoluciona la dirección en marzo y reproduce abril tras sellar los pesos."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from scripts.run_m5_direction_ablation import (
    FROZEN_MARCH_ADMISSIONS_SHA256, FROZEN_SEED_SHA256, load_minutes, m5_features,
)
from scripts.run_m5_direction_benchmark import admission_blocks, load_references
from src.artifacts import file_sha256
from src.evolution_context import native_direction_context
from src.evolution_lineage import load_founder
from src.evolution_multiscale import (
    ARM_WIDTHS, FEATURE_NAMES, FIRST_GENERATION, GENOME_WIDTH, HIDDEN, LAST_GENERATION,
    M5_TACTICAL, MIN_SUBBLOCK, MultiregimeFitness, PF_DISPERSION_WEIGHT, PF_PRIOR, POPULATION,
    SCALE_CLIP, SEEDS, WORST_WEIGHT, arm_inputs, champion_key, directional_metrics,
    directions, evolve_replica, fit_scale, project_founder, scale_inputs, volatility_groups,
)
from src.market import PROTECTED_HOLDOUT_START
from src.train import write_json


FORMAT = 'forex_m5_multiscale_direction_v1'
MARCH_START = pd.Timestamp('2026-03-01T00:00:00Z')
APRIL_START = pd.Timestamp('2026-04-01T00:00:00Z')
Q1 = ('202601', '202602', '202603')
MARCH_SIGNATURE = (6258, 2859, FROZEN_MARCH_ADMISSIONS_SHA256)
APRIL_SIGNATURE = (1264, 278, 'bf18f03b34dfe8f1b024516bb07bd4ccbacdb937ade87ffc8d6350cd3df444a8')
CONTROL = Path('config/m5-multiscale/gen294.zip')
CONTROL_SHA256 = 'c4a6939dd11d617ab3a1923286b1b83b5980895155d4af10c3e6f28edbb512a9'


def load_control():
    if file_sha256(CONTROL) != CONTROL_SHA256:
        raise ValueError('El fundador debe coincidir con el control de la PR #26')
    with ZipFile(CONTROL) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    ablation = manifest['direction_ablation']
    control = load_founder(CONTROL, None, MARCH_START, MARCH_START, allow_direction_ablation=True)
    if (ablation['frozen_entry_bundle_sha256'] != FROZEN_SEED_SHA256 or ablation['m15']['arm'] != 'control'
            or control.next_generation != FIRST_GENERATION):
        raise ValueError('Linaje de control incompatible con la Gen 315')
    return control


def prepare_window(raw_dir, founder, quant, periods, start, end, signature):
    minutes = load_minutes(raw_dir, periods, end=end)
    data = m5_features(minutes, quant, end=end)
    windows, _ = admission_blocks(data, founder, [('window', start, end)])
    window = windows['window']
    observed = (window.eligible_bars, len(window.rows), window.mask_sha256)
    if observed != signature:
        raise ValueError(f'La ventana {start.date()} difiere de la referencia: {observed}')
    native = native_direction_context(minutes, window.rows.index, end=end)
    features = window.rows.loc[:, list(M5_TACTICAL)].join(native).loc[:, list(FEATURE_NAMES)]
    return window, features.to_numpy(dtype=float)


def genome_hash(genome):
    return sha256(np.asarray(genome, dtype='<f8').tobytes()).hexdigest()


def march_evolution(raw_dir, output):
    seal_path = output/'champions.json'
    if seal_path.exists():
        raise ValueError('Los campeones de esta salida ya están sellados')
    founder, _, quant = load_references()
    control = load_control()
    window, raw_inputs = prepare_window(raw_dir, founder, quant, Q1, MARCH_START, APRIL_START, MARCH_SIGNATURE)
    rows = window.rows
    outcomes, mc = rows.outcome.to_numpy(), rows.mc_side.to_numpy()
    mean, std = fit_scale(raw_inputs)
    inputs = scale_inputs(raw_inputs, mean, std)
    initial = project_founder(control, mean, std)
    groups, names, cutoff = volatility_groups(rows.cond_volatility.to_numpy())
    objective = MultiregimeFitness(outcomes, mc, groups, founder.account)
    tasks = [(arm, seed) for seed in SEEDS for arm in ARM_WIDTHS]
    with ThreadPoolExecutor(max_workers=len(ARM_WIDTHS)) as executor:
        futures = [executor.submit(evolve_replica, inputs, initial, objective, arm, seed) for arm, seed in tasks]
        replicas = [future.result() for future in futures]

    champions = []
    for arm in ARM_WIDTHS:
        replica = max((r for r in replicas if r['arm'] == arm), key=lambda r: champion_key(r['champion']))
        genome = replica['genome']
        side = directions(arm_inputs(inputs, arm), [genome])[0]
        metrics = directional_metrics(side, outcomes, mc, founder.account)
        if not np.isclose(metrics['return_per_admission'], replica['champion']['return_per_admission'],
                          rtol=0., atol=1e-12):
            raise ValueError('El replay del campeón debe reproducir su aptitud de marzo')
        champions.append(dict(arm=arm, seed=replica['seed'], selection=replica['champion'],
                              march=metrics, genome=genome, genome_sha256=genome_hash(genome)))

    blocks = []
    for i, name in enumerate(names):
        times = rows.index[groups == i]
        blocks.append(dict(name=name, admissions=len(times), first=times[0].isoformat(),
                           last=times[-1].isoformat(), mc_return=float(objective.mc_blocks[i])))
    normalization = dict(features=FEATURE_NAMES, mean=mean.tolist(), std=std.tolist(), clip=SCALE_CLIP)
    seal = dict(
        format=FORMAT, entry_checkpoint=founder.provenance, direction_founder=control.provenance,
        founder_genome_sha256=genome_hash(initial), training_start=MARCH_START.isoformat(),
        training_end=APRIL_START.isoformat(), march_signature=MARCH_SIGNATURE,
        training_input_sha256=sha256(inputs.tobytes()).hexdigest(), normalization=normalization,
        garch_p67=cutoff, hidden=HIDDEN, first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION,
        champions=champions, protected_holdout_evaluated=False,
    )
    report = dict(
        first_generation=FIRST_GENERATION, last_generation=LAST_GENERATION, population=POPULATION, seeds=SEEDS,
        evaluations=sum(r['evaluations'] for r in replicas), eligible_bars=window.eligible_bars,
        admissions=len(rows), admissions_sha256=window.mask_sha256, blocks=blocks, garch_p67=cutoff,
        fitness=dict(worst_weight=WORST_WEIGHT, pf_dispersion_weight=PF_DISPERSION_WEIGHT,
                     pf_prior=PF_PRIOR, minimum_stratum=MIN_SUBBLOCK),
        monte_carlo=directional_metrics(mc, outcomes, mc, founder.account),
        missing_fraction=dict(zip(FEATURE_NAMES, np.isnan(raw_inputs).mean(axis=0).tolist(), strict=True)),
        champions=[{key: value for key, value in item.items() if key != 'genome'} for item in champions],
        replicas=[{key: value for key, value in item.items() if key != 'genome'} for item in replicas],
    )
    write_json(output/'march.json', report)
    write_json(seal_path, seal)
    print('MULTISCALE_MARCH ' + json.dumps({
        **{key: value for key, value in report.items() if key != 'replicas'},
        'replicas': [{key: r[key] for key in ('arm', 'seed', 'founder', 'champion', 'evaluations')} for r in replicas],
        'seal_sha256': file_sha256(seal_path),
    }, allow_nan=False), flush=True)


def read_seal(seal_path):
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    normalizer = seal['normalization']
    mean, std = (np.asarray(normalizer[key], dtype=float) for key in ('mean', 'std'))
    champions = seal['champions']
    if (seal['format'] != FORMAT or tuple(seal['march_signature']) != MARCH_SIGNATURE
            or seal['training_start'] != MARCH_START.isoformat() or seal['training_end'] != APRIL_START.isoformat()
            or seal['protected_holdout_evaluated'] is not False or seal['hidden'] != HIDDEN
            or seal['entry_checkpoint']['bundle_sha256'] != FROZEN_SEED_SHA256
            or seal['direction_founder']['bundle_sha256'] != CONTROL_SHA256
            or tuple(normalizer['features']) != FEATURE_NAMES or normalizer['clip'] != SCALE_CLIP
            or mean.shape != (len(FEATURE_NAMES),) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0.).any()
            or tuple(item['arm'] for item in champions) != tuple(ARM_WIDTHS)):
        raise ValueError('Sello de marzo incompatible con el diagnóstico de abril')
    for champion in champions:
        genome = np.asarray(champion['genome'], dtype=float)
        if (genome.shape != (GENOME_WIDTH,) or not np.isfinite(genome).all()
                or genome_hash(genome) != champion['genome_sha256'] or champion['seed'] not in SEEDS
                or not FIRST_GENERATION <= champion['selection']['generation'] <= LAST_GENERATION):
            raise ValueError('Genoma sellado incompatible')
    return seal, mean, std


def april_diagnosis(raw_dir, output):
    report_path = output/'april.json'
    if report_path.exists():
        raise ValueError('El diagnóstico de abril ya existe para esta salida')
    seal_path = output/'champions.json'
    seal, mean, std = read_seal(seal_path)
    seal_sha256 = file_sha256(seal_path)
    founder, _, quant = load_references()
    window, raw_inputs = prepare_window(raw_dir, founder, quant, (*Q1, '202604'),
                                       APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE)
    rows = window.rows
    outcome, mc = rows.outcome.to_numpy(), rows.mc_side.to_numpy()
    inputs = scale_inputs(raw_inputs, mean, std)
    high = rows.cond_volatility.to_numpy() > seal['garch_p67']
    policies = {'monte_carlo': mc}
    for champion in seal['champions']:
        arm = champion['arm']
        policies[arm] = directions(arm_inputs(inputs, arm), [champion['genome']])[0]
    metrics = {
        name: {regime: directional_metrics(side[mask], outcome[mask], mc[mask], founder.account)
               for regime, mask in (('all', np.ones(len(rows), dtype=bool)), ('moderate', ~high), ('high', high))
               if mask.any()}
        for name, side in policies.items()
    }
    report = dict(
        start=APRIL_START.isoformat(), end=PROTECTED_HOLDOUT_START.isoformat(), diagnostic=True,
        eligible_bars=window.eligible_bars, admissions=len(rows), days=int(rows.index.normalize().nunique()),
        admissions_sha256=window.mask_sha256, seal_sha256=seal_sha256, stop_applied=False,
        protected_holdout_evaluated=False, policies=metrics,
    )
    write_json(report_path, report)
    print('MULTISCALE_APRIL ' + json.dumps(report, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('evolve', 'diagnose'))
    parser.add_argument('--raw-dir', type=Path, default=Path('data/histdata_scalping'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/direction-multiscale'))
    args = parser.parse_args()
    if args.phase == 'evolve':
        march_evolution(args.raw_dir, args.output_dir)
    else:
        april_diagnosis(args.raw_dir, args.output_dir)


if __name__ == '__main__':
    main()
