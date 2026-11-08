"""Evoluciona dirección desde la #84 y reproduce marzo como diagnóstico final."""

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
from src.evolution_context import build_observations
from src.evolution_direction import FrozenDirectionSimulation, with_direction
from src.evolution_engine import EvolutionConfig, M5Simulation, assess_fitness, run_evolution
from src.evolution_lineage import MARKET, MARKET_FEATURES, load_founder
from src.market import PROTECTED_HOLDOUT_START
from src.quant_engine import FEATURES, QuantConfig, enrich_features
from src.resampling import aggregate
from src.train import write_json


FROZEN_SEED_SHA256 = '57a8634b5957e4a89353218d4cf3c8fd689eb470aa028670308e66744dd508da'


def load_features(raw_dir, periods, quant):
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
    m5 = aggregate(minutes, 1, 'M5')
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


def march_comparison(data, founder, candidate):
    prepared = prepare_market_data(data, market=MARKET)
    prepared = prepared.loc[prepared.timestamp + MARKET.step <= pd.Timestamp('2026-04-01T00:00:00Z')]
    if prepared.timestamp.iloc[-1] + MARKET.step > PROTECTED_HOLDOUT_START:
        raise ValueError('La comparación invade el holdout protegido')
    context = build_observations(prepared)
    normalizer = founder.normalizer
    width = len(MARKET_FEATURES)
    market_normalizer = {key: normalizer[key][:width] for key in ('features', 'mean', 'std')}
    market_normalizer['clip'] = normalizer['clip']
    context = ((context - np.asarray(normalizer['mean'][width:]))
               / np.asarray(normalizer['std'][width:])).clip(-normalizer['clip'], normalizer['clip'])

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
    models = {'Gen 268': founder.genome, 'Monte Carlo': with_direction(founder.genome, 0.),
              f"Gen {candidate.provenance['source_generation']} ablada": candidate.genome}
    evaluated = {name: frozen.evaluate(genome) for name, genome in models.items()}
    if (len(simulation.entries), len(frozen.proposed)) != (6258, 2859):
        raise ValueError('La máscara de marzo difiere de la auditoría de referencia')
    if (evaluated['Gen 268']['total_trades'], evaluated['Gen 268']['net_profit'],
            evaluated['Monte Carlo']['total_trades'], evaluated['Monte Carlo']['net_profit']) != (101, -198., 463, -184.):
        raise ValueError('Las políticas de control no reproducen el replay de marzo')

    volatility_cutoff = float(np.quantile(feb._sigma[feb._eligible], 2/3))
    high = env._sigma[simulation.entries] > volatility_cutoff
    metrics = {}
    for name, genome in models.items():
        result = report_metrics(evaluated[name], simulation, frozen.proposed, founder.account.payout)
        _, raw = simulation.decision_scores(genome)
        side = np.sign(np.where(raw != 0., raw, frozen.mc_direction))
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

    winner = metrics[next(reversed(metrics))]
    control = metrics['Monte Carlo']
    return dict(
        eligible_bars=len(simulation.entries), frozen_admissions=len(frozen.proposed),
        frozen_admission_sha256=sha256(env._times[simulation.entries[frozen.proposed]].asi8.tobytes()).hexdigest(),
        observed_hours=simulation.observed_hours, volatility_cutoff_feb_p67=volatility_cutoff,
        policies=metrics,
        targets=dict(over_450_trades=winner['total_trades'] > 450,
                     pf_at_least_090=(winner['profit_factor'] or 0.) >= .90,
                     matches_or_beats_mc=(winner['total_trades'] >= control['total_trades']
                                          and (winner['profit_factor'] or 0.) >= control['profit_factor'])),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-bundle', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/histdata_scalping'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/direction-ablation'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    if sha256(args.seed_bundle.read_bytes()).hexdigest() != FROZEN_SEED_SHA256:
        raise ValueError('La primera ablación requiere el checkpoint exacto de la corrida #84')
    with ZipFile(args.seed_bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    quant = QuantConfig.from_metadata(manifest['quant_config'])
    with TemporaryDirectory() as temporary:
        input_path = Path(temporary) / 'features-2025.parquet'
        load_features(args.raw_dir, ['2025'], quant).to_parquet(input_path, index=False)
        search = run_evolution(input_path, args.output_dir, EvolutionConfig(population_size=128, generations=20),
                               seed_bundle=args.seed_bundle, direction_only=True)

    # El campeón y su checkpoint ya están fijados al cargar las velas de 2026.
    train_end, search_end = (pd.Timestamp(manifest['splits'][phase]['end']) for phase in ('train', 'search'))
    for phase in ('train', 'search', 'confirmation'):
        if search['splits'][phase] != manifest['splits'][phase]:
            raise ValueError(f'El tramo {phase} difiere de la corrida de referencia')
    founder = load_founder(args.seed_bundle, None, train_end, search_end)
    candidate = load_founder(args.output_dir/'lineage.zip', None, train_end, search_end,
                             allow_direction_ablation=True)
    if search['lineage']['first_generation'] != 273:
        raise ValueError('Se esperaba iniciar la ablación en la generación 273')
    fresh = load_features(args.raw_dir, ['202601', '202602', '202603'], quant)
    result = march_comparison(fresh, founder, candidate)
    write_json(args.output_dir/'march.json', result)
    print('DIRECTION_ABLATION ' + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
