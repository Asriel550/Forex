"""Replay operativo de Gen 333 y extensión direccional aislada en marzo."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_direction_benchmark import load_references
from scripts.run_m5_multiscale import (
    APRIL_SIGNATURE, APRIL_START, MARCH_SIGNATURE, MARCH_START, Q1, archived_march_opens,
    genome_hash, prepare_window, read_seal,
)
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.evolution_lineage import MARKET
from src.evolution_multiscale import (
    FEATURE_NAMES, HIDDEN, MultiregimeFitness, POPULATION, SEEDS, arm_inputs, champion_key,
    directional_metrics, directions, evolve_replica, scale_inputs, volatility_groups,
)
from src.market import PROTECTED_HOLDOUT_START
from src.replay_engine import ContractExits, replay_contracts
from src.trading_profiles import resolve_profile
from src.artifacts import write_json


SEAL_SHA256 = 'e201a87c12131af0441b03855175e43afccc4f79252587efdfadea7e8010612c'
GEN333_SHA256 = 'f3b6bca4f9b60705b354522f4a78651927e458c46948237a7058840d24e32ef1'
ARM = 'm15'
FIRST, LAST = 336, 355
HOURLY_CAP = 2


def sealed_gen333(path):
    if file_sha256(path) != SEAL_SHA256:
        raise ValueError('Se requiere el sello original de la corrida 42507157819')
    seal, mean, std = read_seal(path)
    champion = next(c for c in seal['champions'] if c['arm'] == ARM)
    if champion['selection']['generation'] != 333 or champion['seed'] != 42 or champion['genome_sha256'] != GEN333_SHA256:
        raise ValueError('El fundador debe ser la Gen 333, semilla 42, de la rama M5+M15')
    return seal, champion, mean, std


def replay(raw_dir, output, seal_path, exits):
    seal, champion, mean, std = sealed_gen333(seal_path)
    founder, _, quant = load_references()
    profile = resolve_profile('scalping', 'M5')
    risk = RiskConfig(min_confidence=profile['FOREX_MIN_CONFIDENCE'],
                      min_expected_edge=profile['FOREX_MIN_EXPECTED_EDGE'], mc_z=profile['FOREX_MC_Z'],
                      drawdown_limit=founder.drawdown_limit)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output/'march-inputs.npz'
    settings = dict(n_paths=quant.n_paths, max_volatility=seal['garch_p67'], hourly_cap=HOURLY_CAP)
    windows = (
        ('march', Q1, MARCH_START, APRIL_START, MARCH_SIGNATURE, archived_march_opens()),
        ('april', (*Q1, '202604'), APRIL_START, PROTECTED_HOLDOUT_START, APRIL_SIGNATURE, None),
    )
    results, traces = {}, []
    for name, periods, start, end, signature, opens in windows:
        block, raw = prepare_window(raw_dir, founder, quant, periods, start, end, signature, opens=opens)
        rows = block.rows
        inputs = scale_inputs(raw, mean, std)
        if name == 'march':
            np.savez_compressed(snapshot, inputs=inputs, outcome=rows.outcome.to_numpy(),
                                mc=rows.mc_side.to_numpy(), volatility=rows.cond_volatility.to_numpy(),
                                opens_ns=(rows.index - MARKET.step).as_unit('ns').asi8)
        side = directions(arm_inputs(inputs, ARM), [champion['genome']])[0]
        result, trace = replay_contracts(rows, side, founder.account, risk,
                                         observed_hours=block.observed_hours, exits=exits, **settings)
        result.update(admissions_sha256=block.mask_sha256, start=start.isoformat(), end=end.isoformat())
        results[name] = result
        traces.append(trace.assign(window=name))
    report = dict(
        generation=333, seed=42, genome_sha256=GEN333_SHA256, seal_sha256=SEAL_SHA256,
        gate=dict(confidence=risk.min_confidence, min_expected_edge=risk.min_expected_edge,
                  wilson_z=risk.mc_z, drawdown_stop_fraction=risk.drawdown_limit, **settings),
        stake_usd=founder.account.stake, payout=founder.account.payout,
        early_exits=dict(mode='historical_sell_quotes' if exits is not None else 'unavailable',
                         stop_loss_fraction=exits.stop_loss if exits is not None else None,
                         take_profit_fraction=exits.take_profit if exits is not None else None),
        march_snapshot_sha256=file_sha256(snapshot), windows=results, protected_holdout_evaluated=False,
    )
    pd.concat(traces, ignore_index=True).to_csv(output/'replay-admissions.csv', index=False)
    write_json(output/'replay.json', report)
    print('GEN333_REPLAY ' + json.dumps(report, allow_nan=False), flush=True)


def extend(output, seal_path):
    seal, champion, _, _ = sealed_gen333(seal_path)
    snapshot = output/'march-inputs.npz'
    report = json.loads((output/'replay.json').read_text(encoding='utf-8'))
    if report['seal_sha256'] != SEAL_SHA256 or file_sha256(snapshot) != report['march_snapshot_sha256']:
        raise ValueError('El snapshot debe ser el mismo marzo verificado durante el replay')
    with np.load(snapshot, allow_pickle=False) as data:
        inputs, outcome, mc, volatility, opens = (
            data[name] for name in ('inputs', 'outcome', 'mc', 'volatility', 'opens_ns')
        )
    if (inputs.shape != (MARCH_SIGNATURE[1], len(FEATURE_NAMES)) or not np.isfinite(inputs).all()
            or sha256(opens.tobytes()).hexdigest() != MARCH_SIGNATURE[2]):
        raise ValueError('La extensión requiere la cohorte congelada de marzo')
    founder, _, _ = load_references()
    initial = np.asarray(champion['genome'], dtype=float)
    groups, names, cutoff = volatility_groups(volatility)
    objective = MultiregimeFitness(outcome, mc, groups, founder.account)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = [executor.submit(evolve_replica, inputs, initial, objective, ARM, seed,
                                   first_generation=FIRST, last_generation=LAST) for seed in SEEDS]
        replicas = [future.result() for future in futures]
    baseline_side = directions(arm_inputs(inputs, ARM), [initial])[0]
    baseline = directional_metrics(baseline_side, outcome, mc, founder.account)
    for replica in replicas:
        genome = replica['genome']
        side = directions(arm_inputs(inputs, ARM), [genome])[0]
        metrics = directional_metrics(side, outcome, mc, founder.account)
        if not np.isclose(metrics['return_per_admission'], replica['champion']['return_per_admission'],
                          rtol=0., atol=1e-12):
            raise ValueError('El replay del genoma ganador debe reproducir su aptitud')
        replica.update(genome_sha256=genome_hash(genome), march=metrics,
                       nonnegative_evaluations=sum(g['nonnegative_candidates'] for g in replica['generations']),
                       max_candidate_return=max(g['max_candidate_return'] for g in replica['generations']))
    best = max(replicas, key=lambda r: champion_key(r['champion']))
    matrix_sha256 = sha256(inputs.tobytes()).hexdigest()
    result = dict(
        arm=ARM, hidden=HIDDEN, first_generation=FIRST, last_generation=LAST, population=POPULATION, seeds=SEEDS,
        evaluations=sum(r['evaluations'] for r in replicas), founder_genome_sha256=GEN333_SHA256,
        founder_previous_march=champion['march'], founder_replay_march=baseline,
        march_signature=MARCH_SIGNATURE, march_snapshot_sha256=report['march_snapshot_sha256'],
        training_input_sha256=matrix_sha256, matrix_matches_original=matrix_sha256 == seal['training_input_sha256'],
        blocks=names, garch_p67=cutoff, normalization=seal['normalization'],
        best_seed=best['seed'], best_genome_sha256=best['genome_sha256'], replicas=replicas,
        protected_holdout_evaluated=False,
    )
    write_json(output/'extension.json', result)
    summary = dict(result, replicas=[{k: v for k, v in r.items() if k not in ('genome', 'generations')}
                                    for r in replicas])
    print('GEN333_EXTENSION ' + json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('replay', 'extend'))
    parser.add_argument('--raw-dir', type=Path, default=Path('data/histdata_scalping'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/m5-gen333'))
    parser.add_argument('--seal', type=Path, default=Path('data/m5-gen333-source/champions.json'))
    parser.add_argument('--exit-quotes', type=Path)
    parser.add_argument('--stop-loss', type=float, help='Fracción de stake para cerrar con pérdida')
    parser.add_argument('--take-profit', type=float, help='Fracción de stake para cerrar con ganancia')
    args = parser.parse_args()
    exit_arguments = (args.exit_quotes, args.stop_loss, args.take_profit)
    if any(arg is not None for arg in exit_arguments) and (
        args.phase != 'replay' or any(arg is None for arg in exit_arguments)
    ):
        parser.error('SL/TP requiere replay, cotizaciones y los dos umbrales')
    if args.phase == 'replay':
        exits = (ContractExits.from_csv(args.exit_quotes, stop_loss=args.stop_loss, take_profit=args.take_profit)
                 if args.exit_quotes is not None else None)
        replay(args.raw_dir, args.output_dir, args.seal, exits)
    else:
        extend(args.output_dir, args.seal)


if __name__ == '__main__':
    main()
