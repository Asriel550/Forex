"""Evoluciona dirección con la admisión de la #84 y reproduce marzo al final."""

import argparse
from hashlib import sha256
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.acquire_histdata import acquire
from scripts.build_histdata_dataset import read_native_minutes
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_context import build_observations, m15_direction_bias
from src.evolution_direction import M15_ARMS, FrozenDirectionSimulation, frozen_simulation, with_direction
from src.evolution_engine import EvolutionConfig, M5Simulation, assess_fitness, run_evolution
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.market import PROTECTED_HOLDOUT_START
from src.quant_engine import FEATURES, QuantConfig, enrich_features
from src.resampling import aggregate
from src.train import write_json


FROZEN_SEED_SHA256 = '57a8634b5957e4a89353218d4cf3c8fd689eb470aa028670308e66744dd508da'
PR25_DIRECTION_SHA256 = 'f28cd91e37a927e6f67bed82b668b6bde99a4ef544db1fb7a2cd3ed3180ba491'
FROZEN_MARCH_ADMISSIONS_SHA256 = '4c02cfbc1ff86d6ff2e6422ecb03be2633b8473deada1d23b1780da970e0dfe6'


def load_features(raw_dir, periods, quant, *, end=None):
    sources = json.loads(Path('datasets/eurusd/manifest.json').read_text(encoding='utf-8'))['sources']
    archives = [source for source in sources if source['period'] in periods]
    if {source['period'] for source in archives} != set(periods):
        raise ValueError('Faltan fuentes M1 fijadas en el manifiesto')
    raw_dir.mkdir(parents=True, exist_ok=True)
    for source in archives:
        period = source['period']
        if not (raw_dir / source['zip_file']).is_file():
            acquire(raw_dir, int(period[:4]), int(period[4:]) if len(period) > 4 else None)
    minutes, _ = read_native_minutes(raw_dir, archives)
    if end is not None:
        minutes = minutes.loc[minutes.timestamp < end]
    m5 = aggregate(minutes, 1, 'M5', now=end)
    features = enrich_features(m5, quant, timeframe='M5', expiry_candles=1)
    return features.dropna(subset=FEATURES).reset_index(drop=True)


def report_metrics(metrics, simulation, proposed, payout):
    keys = ('total_trades', 'win_rate_excluding_refunds', 'profit_factor', 'trades_per_observed_hour',
            'max_drawdown_pct', 'net_profit', 'risk_stop', 'signals_after_activation', 'directional')
    last = proposed[metrics['total_trades'] - 1] if metrics['total_trades'] else None
    return {key: metrics[key] for key in keys} | dict(
        last_expiry_utc=simulation.timestamps[simulation.entries[last] + 1].isoformat() if last is not None else None,
        checks=assess_fitness(metrics, payout=payout)['checks'],
    )


def direction_changes(simulation, indices, before, after):
    before, after = np.sign(before[indices]), np.sign(after[indices])
    actual = simulation.change[indices]
    flipped = before != after
    resolved = actual != 0.
    return dict(flips=int(flipped.sum()),
                corrected=int((flipped & resolved & (after == actual)).sum()),
                regressed=int((flipped & resolved & (before == actual)).sum()),
                refunds=int((flipped & ~resolved).sum()))


def founder_observations(prepared, founder):
    """Construye el mismo contexto bruto y normalizado de las entradas congeladas."""
    context = build_observations(prepared)
    normalizer = founder.normalizer
    width = len(MARKET_FEATURES)
    market_normalizer = {key: normalizer[key][:width] for key in ('features', 'mean', 'std')}
    market_normalizer['clip'] = normalizer['clip']
    normalized = ((context - np.asarray(normalizer['mean'][width:]))
                  / np.asarray(normalizer['std'][width:])).clip(-normalizer['clip'], normalizer['clip'])
    return context, normalized, market_normalizer


def march_comparison(data, founder, candidate, *, direction_seed=None, m15_config=None, m15_weight=0.):
    prepared = prepare_market_data(data, market=MARKET)
    prepared = prepared.loc[prepared.timestamp + MARKET.step <= pd.Timestamp('2026-04-01T00:00:00Z')]
    if prepared.timestamp.iloc[-1] + MARKET.step > PROTECTED_HOLDOUT_START:
        raise ValueError('La comparación invade el holdout protegido')
    _, context, market_normalizer = founder_observations(prepared, founder)

    def environment(month):
        return BinaryOptionsEnv(
            prepared.loc[prepared.timestamp.dt.month.eq(month)], market=MARKET, prepared=True,
            account=founder.account, normalizer=market_normalizer,
            risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit),
        )

    feb = environment(2)
    env = environment(3)
    simulation = M5Simulation(env, context.loc[env._times])
    frozen = FrozenDirectionSimulation(simulation, founder.genome)
    admission_hash = sha256(env._times[simulation.entries[frozen.proposed]].asi8.tobytes()).hexdigest()
    if ((len(simulation.entries), len(frozen.proposed)) != (6258, 2859)
            or admission_hash != FROZEN_MARCH_ADMISSIONS_SHA256):
        raise ValueError(f'La máscara de marzo difiere de la auditoría de referencia: '
                         f'{len(simulation.entries)} velas, {len(frozen.proposed)} admisiones, hash {admission_hash}')
    march = prepared.loc[prepared.timestamp.dt.month.eq(3)]
    bias = m15_direction_bias(prepared) if m15_config and m15_config['arm'] != 'control' else None
    trial = frozen_simulation(simulation, founder.genome, march, bias, m15_config)
    winner_name = f"Gen {candidate.provenance['source_generation']} ablada"
    models = {
        'Gen 268': (frozen, founder.genome, 0.),
        'Monte Carlo': (frozen, with_direction(founder.genome, 0.), 0.),
    }
    if direction_seed is not None:
        models[f"Gen {direction_seed.provenance['source_generation']} de referencia"] = (frozen, direction_seed.genome, 0.)
    models[winner_name] = (trial, candidate.genome, m15_weight)
    evaluated = {name: sim.evaluate(genome, m15_weight=weight) for name, (sim, genome, weight) in models.items()}
    if (evaluated['Gen 268']['total_trades'], evaluated['Gen 268']['net_profit'],
            evaluated['Monte Carlo']['total_trades'], evaluated['Monte Carlo']['net_profit']) != (101, -198., 463, -184.):
        raise ValueError('Las políticas de control no reproducen el replay de marzo')

    volatility_cutoff = float(np.quantile(feb._sigma[feb._eligible], 2/3))
    high = env._sigma[simulation.entries] > volatility_cutoff
    metrics = {}
    for name, (sim, genome, weight) in models.items():
        result = report_metrics(evaluated[name], simulation, frozen.proposed, founder.account.payout)
        raw, direction = sim.direction_scores(genome, m15_weight=weight)
        side = np.sign(direction)
        by_volatility = {}
        for regime, mask in (('moderada', ~high), ('alta', high)):
            resolved = mask & (simulation.change != 0.)
            score, actual = raw[resolved], simulation.change[resolved]
            rho = (float(spearmanr(score, actual).statistic)
                   if len(actual) > 1 and np.ptp(score) and np.ptp(actual) else None)
            by_volatility[regime] = dict(
                resolved_bars=len(actual), spearman=rho,
                accuracy=float(np.mean(side[resolved] == actual)) if len(actual) else None,
            )
        result['directional_by_volatility'] = by_volatility
        metrics[name] = result

    winner = metrics[winner_name]
    control = metrics['Monte Carlo']
    result = dict(
        eligible_bars=len(simulation.entries), frozen_admissions=len(frozen.proposed),
        frozen_admission_sha256=admission_hash,
        observed_hours=simulation.observed_hours, volatility_cutoff_feb_p67=volatility_cutoff,
        policies=metrics, winner=winner_name,
        targets=dict(over_450_trades=winner['total_trades'] > 450,
                     pf_at_least_090=(winner['profit_factor'] or 0.) >= .90,
                     matches_or_beats_mc=(winner['total_trades'] >= control['total_trades']
                                          and (winner['profit_factor'] or 0.) >= control['profit_factor'])),
    )
    if m15_config is not None:
        _, base = frozen.direction_scores(founder.genome)
        _, without_m15 = frozen.direction_scores(candidate.genome)
        _, with_m15 = trial.direction_scores(candidate.genome, m15_weight=m15_weight)
        first_101 = frozen.proposed[:101]
        result['m15'] = dict(
            **m15_config, learned_weight=m15_weight,
            first_101_vs_gen268=direction_changes(simulation, first_101, base, with_m15),
            first_101_m15_term_only=direction_changes(simulation, first_101, without_m15, with_m15),
            all_admitted_m15_term_only=direction_changes(simulation, frozen.proposed, without_m15, with_m15),
            same_winner_without_m15=report_metrics(frozen.evaluate(candidate.genome), simulation,
                                                    frozen.proposed, founder.account.payout),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-bundle', type=Path, required=True)
    parser.add_argument('--direction-seed-bundle', type=Path)
    parser.add_argument('--m15-arm', choices=M15_ARMS, action='append',
                        help='Repetir para comparar brazos sobre los mismos datos preparados')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/histdata_scalping'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/direction-ablation'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if sha256(args.seed_bundle.read_bytes()).hexdigest() != FROZEN_SEED_SHA256:
        raise ValueError('La primera ablación requiere el checkpoint exacto de la corrida #84')
    if (args.m15_arm is None) != (args.direction_seed_bundle is None):
        parser.error('--m15-arm y --direction-seed-bundle deben proporcionarse juntos')
    arms = args.m15_arm or [None]
    if len(set(arms)) != len(arms):
        parser.error('Cada brazo M15 se debe especificar una sola vez')
    if args.direction_seed_bundle and sha256(args.direction_seed_bundle.read_bytes()).hexdigest() != PR25_DIRECTION_SHA256:
        raise ValueError('La comparación M15 requiere el checkpoint direccional exacto de la PR #25')
    with ZipFile(args.seed_bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    quant = QuantConfig.from_metadata(manifest['quant_config'])
    train_end, search_end = (pd.Timestamp(manifest['splits'][phase]['end']) for phase in ('train', 'search'))
    selected = []
    with TemporaryDirectory() as temporary:
        input_path = Path(temporary) / 'features-2025.parquet'
        load_features(args.raw_dir, ['2025'], quant).to_parquet(input_path, index=False)
        for arm in arms:
            output = args.output_dir / arm if len(arms) > 1 else args.output_dir
            search = run_evolution(input_path, output, EvolutionConfig(population_size=128, generations=20),
                                   seed_bundle=args.seed_bundle, direction_only=True,
                                   direction_seed_bundle=args.direction_seed_bundle, m15_arm=arm)
            for phase in ('train', 'search', 'confirmation'):
                if search['splits'][phase] != manifest['splits'][phase]:
                    raise ValueError(f'El tramo {phase} difiere de la corrida de referencia')
            first_generation = 294 if arm else 273
            if search['lineage']['first_generation'] != first_generation:
                raise ValueError(f'Se esperaba iniciar la ablación en la generación {first_generation}')
            selected.append((output, search))

    # Todos los campeones y sus checkpoints ya están fijados al cargar las velas de 2026.
    founder = load_founder(args.seed_bundle, None, train_end, search_end)
    direction_seed = (load_founder(args.direction_seed_bundle, None, train_end, search_end,
                                   allow_direction_ablation=True) if args.direction_seed_bundle else None)
    fresh = load_features(args.raw_dir, ['202601', '202602', '202603'], quant)
    for output, search in selected:
        candidate = load_founder(output/'lineage.zip', None, train_end, search_end,
                                 allow_direction_ablation=True)
        result = march_comparison(fresh, founder, candidate, direction_seed=direction_seed,
                                  m15_config=search['direction_ablation'].get('m15'),
                                  m15_weight=search['champion'].get('m15_weight', 0.))
        write_json(output/'march.json', result)
        print('DIRECTION_ABLATION ' + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
