"""Evolución genética de políticas de entrada para opciones binarias M5."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from .artifacts import file_sha256
from .decision import RiskConfig
from .environment import BinaryOptionsEnv, prepare_market_data
from .evolution_context import CONTEXT_SETTINGS, build_observations
from .evolution_lineage import FORMAT, MARKET, MARKET_FEATURES, genome_payload, load_founder
from .market import PROTECTED_HOLDOUT_START
from .metrics import _atomic_write, calculate_metrics
from .trading_profiles import SCALPING_TARGET_PER_HOUR, resolve_profile
from .train import chronological_market_split, write_json

LOG = logging.getLogger(__name__)
SEARCH_BLOCKS = 4
WORST_BLOCK_WEIGHT = .5
ELITE_FRACTION = .20
MIN_WIN_RATE = .56
WIN_RATE_SHORTFALL_WEIGHT = .5
MIN_PROFIT_FACTOR = 1.20
MAX_DRAWDOWN_PCT = 15.
MIN_TRADES_HOUR, MAX_TRADES_HOUR = SCALPING_TARGET_PER_HOUR['M5']


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 24
    generations: int = 10
    seed: int = 42
    workers: int = 4

    def __post_init__(self):
        for name, lower, upper in (
            ('population_size', 4, 64), ('generations', 1, 20),
            ('seed', 0, 2**32 - 1), ('workers', 1, 8),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'{name} debe ser un entero de {lower} a {upper}')


class M5Simulation:
    """Simulación vectorizada, con la misma caja y ventanas del entorno M5.

    La política puntúa únicamente features de entrada. Los precios futuros
    se guardan por separado para liquidar las entradas seleccionadas.
    """

    def __init__(self, env: BinaryOptionsEnv, context=None):
        if env.market != MARKET:
            raise ValueError('La evolución requiere M5 binario y vencimiento de una vela')
        self.entries = env._eligible
        self.features = env._market[self.entries]
        up, down, _ = env._probabilities[self.entries].T
        self.mc_margin = up - down
        self.observed_hours = sum(end - start for start, end in env.segments) / 12
        if context is not None:
            if not context.index.equals(env._times):
                raise ValueError('El contexto debe coincidir con el índice de velas M5')
            values = context.to_numpy(dtype=np.float32)[self.entries]
            if not np.isfinite(values).all():
                raise ValueError('El contexto requiere observaciones finitas')
            self.features = np.column_stack((self.features, values))
        self.change = np.sign(env._close[self.entries + 1] - env._close[self.entries])
        self.account = env.account
        self.timestamps = env._times + env.bar_step
        self.stop_limit = min(self.account.max_drawdown, env.risk.drawdown_limit)

    def evaluate(self, genome):
        """El genoma decide entrada y dirección. Los retornos futuros solo liquidan."""
        heads = genome.reshape(2, self.features.shape[1] + 1)
        scores = np.einsum('ij,kj->ik', self.features, heads[:, :-1]) + heads[:, -1]
        direction = self.mc_margin + scores[:, 1]
        proposed = np.flatnonzero((scores[:, 0] > 0.) & (direction != 0.))
        side = np.sign(direction[proposed])
        actions = np.where(side > 0., 2, 0)
        change = self.change[proposed]
        won, tied = change == side, change == 0.
        outcomes = np.where(tied, 'tie', np.where(won, 'win', 'loss'))
        profits = self.account.stake * np.where(
            won, self.account.payout, np.where(tied & (self.account.tie_policy == 'refund'), 0., -1.),
        )
        initial = self.account.initial_balance
        balance = np.r_[initial, initial + np.cumsum(profits)]
        drawdown = 1 - balance / np.maximum.accumulate(balance)
        stops = np.flatnonzero((balance < self.account.stake) | (drawdown + 1e-12 >= self.stop_limit))
        if stops.size:
            # La posición cero de balance es previa al primer contrato.
            filled = slice(None, stops[0])
            proposed, actions, profits, outcomes = proposed[filled], actions[filled], profits[filled], outcomes[filled]

        equity = np.zeros(len(self.timestamps))
        equity[self.entries[proposed] + 1] = profits
        equity = initial + np.cumsum(equity)
        trades = pd.DataFrame(dict(action=actions, profit=profits, outcome=outcomes))
        result = calculate_metrics(trades, pd.DataFrame(dict(timestamp=self.timestamps, balance=equity)),
                                   initial_balance=initial, periods_per_year=252 * 24 * 12)
        result.update(return_basis='M5 equity including inactivity',
                      observed_hours=self.observed_hours, decision_opportunities=len(self.entries),
                      trades_per_observed_hour=len(trades) / self.observed_hours,
                      calls=int((actions == 2).sum()), puts=int((actions == 0).sum()),
                      risk_stop=bool(stops.size))
        return result


def assess_fitness(metrics):
    """Puntúa a toda la población. Los checks describen el gate del campeón."""
    win = metrics['win_rate_excluding_refunds']
    pf = metrics['profit_factor']
    no_losses = metrics['profit_factor_status'] == 'infinite_no_losses'
    rate = metrics['trades_per_observed_hour']
    drawdown = metrics['max_drawdown_pct']
    checks = dict(
        win_rate=win is not None and win >= MIN_WIN_RATE,
        profit_factor=no_losses or (pf is not None and pf > MIN_PROFIT_FACTOR),
        frequency=MIN_TRADES_HOUR <= rate <= MAX_TRADES_HOUR,
        drawdown=drawdown < MAX_DRAWDOWN_PCT,
    )
    pf_term = 1. if no_losses else pf / (1 + pf) if pf is not None else 0.
    win_term = win if win is not None else 0.
    activity_weight = min(1., rate / MIN_TRADES_HOUR)
    distance = max(0., MIN_TRADES_HOUR - rate, rate - MAX_TRADES_HOUR)
    win_shortfall = WIN_RATE_SHORTFALL_WEIGHT * max(0., MIN_WIN_RATE - win_term)
    score = activity_weight * (win_term + pf_term) - distance / MIN_TRADES_HOUR - drawdown / 100 - win_shortfall
    return dict(checks=checks, passed=all(checks.values()), score=score)


def assess_search_fitness(metrics, blocks):
    """Exige rentabilidad y drawdown controlado en cada bloque para ser élite."""
    fitness = assess_fitness(metrics)
    scores = [assess_fitness(block)['score'] for block in blocks]
    score = (1 - WORST_BLOCK_WEIGHT) * float(np.mean(scores)) + WORST_BLOCK_WEIGHT * min(scores)
    fitness['checks']['block_consistency'] = all(
        block['net_profit'] > 0 and block['max_drawdown_pct'] < MAX_DRAWDOWN_PCT for block in blocks
    )
    return dict(fitness, score=score, passed=all(fitness['checks'].values()), block_scores=scores)


def selection_rank(fitness):
    return fitness['checks']['block_consistency'], fitness['score']


def seeded_population(rng, founder, count, sigma):
    population = np.repeat(founder[None, :], count, axis=0)
    population[1:] += rng.normal(0., sigma, size=population[1:].shape)
    return population


def breed(rng, population, elite_indices, parent_indices, sigma):
    """Copia las élites y rellena la población por cruce uniforme y mutación."""
    elites = population[elite_indices]
    parents = population[parent_indices]
    count = len(population) - len(elites)
    left = rng.integers(len(parents), size=count)
    right = (left + rng.integers(1, len(parents), size=count)) % len(parents) if len(parents) > 1 else left
    shape = (count, population.shape[1])
    children = np.where(rng.random(shape) < .5, parents[left], parents[right])
    children += rng.normal(0., sigma, size=shape)
    return np.vstack((elites, children))


def save_candidate(path, genome, normalizer, manifest):
    payload = {
        'genome.json': json.dumps(genome_payload(genome), allow_nan=False).encode('utf-8'),
        'normalizer.json': json.dumps(normalizer, allow_nan=False).encode('utf-8'),
    }
    manifest = dict(manifest, sha256={name: sha256(body).hexdigest() for name, body in payload.items()})
    payload['manifest.json'] = json.dumps(manifest, indent=2, allow_nan=False).encode('utf-8')

    def archive(temporary):
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED) as bundle:
            for name, body in payload.items():
                bundle.writestr(name, body)

    _atomic_write(path, archive)
    return file_sha256(path)


def write_summary(path, report):
    lineage = report['lineage']
    lines = [
        '# Evolución Scalping M5 binario', '',
        'Las élites cumplen consistencia en todos los bloques. Solo el campeón pasa por confirmación.',
        f"Genoma fundador de la generación {lineage['source_generation']}. "
        f"Corrida de la generación {lineage['first_generation']} a la {lineage['last_generation']}.",
        'Holdout histórico reservado.', '',
        '| Fase | Generación | Aptitud | Contratos | Acierto sin reembolsos | PF | Contratos/h | DD | Cumple límites |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in [*report['generations'], report['confirmation']]:
        m = row['metrics']
        wr = m['win_rate_excluding_refunds']
        wr = 'N/D' if wr is None else f'{wr:.2%}'
        pf = m['profit_factor']
        pf = f'{pf:.3f}' if pf is not None else (
            '∞' if m['profit_factor_status'] == 'infinite_no_losses' else 'N/D'
        )
        gate = 'Sí' if row['fitness']['passed'] else 'No'
        score = f"{row['fitness']['score']:.4f}" if row['phase'] == 'Búsqueda' else '—'
        lines.append(f"| {row['phase']} | {row['generation']} | {score} | "
                     f"{m['total_trades']} | {wr} | {pf} | "
                     f"{m['trades_per_observed_hour']:.3f} | {m['max_drawdown_pct']:.2f}% | {gate} |")
    status = 'cumplido' if report['qualified'] else 'rechazado'
    lines.extend(['', f"Gate de la corrida **{status}**. Campeón preseleccionado en generación "
                      f"{report['champion']['generation']}.", '',
                  '| Tramo | Entradas evaluables | CALL ejecutados | PUT ejecutados |',
                  '| --- | ---: | ---: | ---: |'])
    for row in (report['champion'], report['confirmation']):
        metrics = row['metrics']
        lines.append(f"| {row['phase']} | {metrics['decision_opportunities']} | "
                     f"{metrics['calls']} | {metrics['puts']} |")
    lines.extend(['', '## Bloques del campeón', '',
                  '| Bloque | Entradas evaluables | Contratos | PnL USD | DD |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for block, metrics in zip(report['search_blocks'], report['champion']['blocks']):
        lines.append(f"| {block['block']} | {metrics['decision_opportunities']} | {metrics['total_trades']} | "
                     f"{metrics['net_profit']:.2f} | {metrics['max_drawdown_pct']:.2f}% |")
    lines.append('')
    _atomic_write(path, lambda p: Path(p).write_text('\n'.join(lines), encoding='utf-8'))


def run_evolution(input_path, output_dir, config=None, *, seed_bundle=None):
    """Guarda cada ganador de generación y comprueba al campeón fuera de búsqueda."""
    config = config or EvolutionConfig()
    input_path, root = Path(input_path), Path(output_dir)
    source_hash = file_sha256(input_path)
    raw = pd.read_parquet(input_path)
    if file_sha256(input_path) != source_hash:
        raise ValueError('El Parquet cambió durante la lectura')
    preset = resolve_profile('scalping', 'M5')
    quant = raw.attrs.get('quant_config', {})
    if (quant.get('drift_mode'), quant.get('drift_scale_floor'), quant.get('drift_scale')) != (
        'momentum', preset['FOREX_DRIFT_SCALE_FLOOR'], preset['FOREX_DRIFT_SCALE']
    ):
        raise ValueError('Regenerar las features de scalping M5 con drift adaptativo 1.4 a 2.0')

    prepared = prepare_market_data(raw, market=MARKET)
    splits = chronological_market_split(prepared, MARKET)
    validation = splits['validation']
    cut = 3 * len(validation) // 4
    search, confirmation = validation.iloc[:cut], validation.iloc[cut + 1:]
    if len(search) < 2 * SEARCH_BLOCKS or len(confirmation) < 2:
        raise ValueError('Datos insuficientes para bloques de búsqueda y confirmación')
    bounds = np.linspace(0, len(search), SEARCH_BLOCKS + 1, dtype=int)
    blocks = [search.iloc[start:end] for start, end in zip(bounds, bounds[1:])]
    context_source = prepared.loc[prepared.timestamp <= validation.timestamp.iloc[-1]]
    context = build_observations(context_source)
    train = splits['train']
    founder = load_founder(seed_bundle, context.loc[train.timestamp],
                           train.timestamp.iloc[-1], search.timestamp.iloc[-1])
    normalizer = founder.normalizer
    width = len(MARKET_FEATURES)
    market_normalizer = {k: normalizer[k][:width] for k in ('features', 'mean', 'std')}
    market_normalizer['clip'] = normalizer['clip']
    values = (context.to_numpy() - np.asarray(normalizer['mean'][width:])) / np.asarray(normalizer['std'][width:])
    context.iloc[:, :] = np.clip(values, -normalizer['clip'], normalizer['clip'])
    risk = RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit)
    env_options = dict(market=MARKET, prepared=True, risk=risk, normalizer=market_normalizer,
                       account=founder.account)

    def simulate(frame):
        return M5Simulation(BinaryOptionsEnv(frame, **env_options), context.loc[frame.timestamp])

    simulation = simulate(search)
    block_sims = [simulate(block) for block in blocks]

    def evaluate(genome):
        metrics = simulation.evaluate(genome)
        by_block = [sim.evaluate(genome) for sim in block_sims]
        return metrics, by_block, assess_search_fitness(metrics, by_block)

    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('La evolución requiere un directorio de salida vacío')

    first_generation = founder.next_generation
    last_generation = first_generation + config.generations
    metadata = dict(
        format=FORMAT, profile='scalping', market=asdict(MARKET),
        policy='positive entry head enables CALL/PUT by signed MC margin plus direction residual',
        input_sha256=source_hash, quant_config=quant, account=asdict(simulation.account),
        risk=dict(drawdown_limit=risk.drawdown_limit), config=asdict(config), holdout_test_evaluated=False,
        protected_holdout_start=PROTECTED_HOLDOUT_START.isoformat(),
        normalizer_fit='inherited market statistics; context fitted on train at migration',
        lineage=dict(founder.provenance, first_generation=first_generation, last_generation=last_generation),
        fitness_limits=dict(min_win_rate=MIN_WIN_RATE, exclusive_min_profit_factor=MIN_PROFIT_FACTOR,
                            trades_per_hour=[MIN_TRADES_HOUR, MAX_TRADES_HOUR],
                            exclusive_max_drawdown_pct=MAX_DRAWDOWN_PCT),
        fitness_selection=dict(method='economic_chronological_blocks_v3', blocks=SEARCH_BLOCKS,
                               worst_block_weight=WORST_BLOCK_WEIGHT,
                               win_rate_shortfall_weight=WIN_RATE_SHORTFALL_WEIGHT),
        context_observations=CONTEXT_SETTINGS,
        search_blocks=[dict(block=i, rows=len(frame), start=frame.timestamp.iloc[0].isoformat(),
                            end=frame.timestamp.iloc[-1].isoformat())
                       for i, frame in enumerate(blocks, start=1)],
        elite_fraction=ELITE_FRACTION, confirmation_purge_bars=1,
        splits={name: dict(rows=len(frame), start=frame.timestamp.iloc[0].isoformat(),
                           end=frame.timestamp.iloc[-1].isoformat())
                for name, frame in (('train', train), ('search', search),
                                    ('confirmation', confirmation))},
    )
    rng = np.random.default_rng([config.seed, first_generation])
    sigma = founder.mutation_sigma
    population = seeded_population(rng, founder.genome, config.population_size, sigma)
    elite_count = math.ceil(ELITE_FRACTION * len(population))
    history, candidate_rows = [], []
    champion = champion_genome = None
    LOG.info('Linaje M5. Fundador gen %d. Generaciones %d a %d',
             founder.provenance['source_generation'], first_generation, last_generation)
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for generation in range(first_generation, last_generation + 1):
            metrics, by_block, fitness = zip(*executor.map(evaluate, population))
            parents = sorted(range(len(population)), key=lambda i: selection_rank(fitness[i]),
                             reverse=True)[:elite_count]
            elites = [i for i in parents if fitness[i]['checks']['block_consistency']]
            best = parents[0]
            name = f'bundle_scalping_gen_{generation}.zip'
            record = dict(phase='Búsqueda', generation=generation, candidate=best, bundle=name,
                          metrics=metrics[best], blocks=by_block[best], fitness=fitness[best],
                          elites=len(elites), parents=len(parents),
                          qualified_candidates=sum(f['passed'] for f in fitness))
            improved = champion is None or selection_rank(fitness[best]) > selection_rank(champion['fitness'])
            if generation > first_generation:
                sigma = max(.025, sigma * .85) if improved else min(.35, sigma * 1.25)
            record['mutation_sigma'] = sigma if generation < last_generation else None
            digest = save_candidate(root/'candidates'/name, population[best], normalizer,
                                    dict(metadata, selection=record))
            record['bundle_sha256'] = digest
            history.append(record)
            if improved:
                champion, champion_genome = record, population[best].copy()
            candidate_rows.extend(dict(
                generation=generation, candidate=i, fitness=f['score'], elite=i in elites,
                parent=i in parents, founder=generation == first_generation and i == 0,
                passed=f['passed'], **{f'pass_{k}': v for k, v in f['checks'].items()},
                **{k: m[k] for k in ('total_trades', 'win_rate_excluding_refunds', 'profit_factor',
                                     'profit_factor_status', 'trades_per_observed_hour', 'max_drawdown_pct',
                                     'net_profit', 'risk_stop', 'calls', 'puts', 'decision_opportunities')},
                **{f'block_{j}_{key}': block[key] for j, block in enumerate(candidate_blocks, start=1)
                   for key in ('net_profit', 'max_drawdown_pct', 'total_trades')},
            ) for i, (m, f, candidate_blocks) in enumerate(zip(metrics, fitness, by_block)))
            LOG.info('Generación %d. Aptitud %.4f, élites %d, padres %d, candidatos dentro de límites %d',
                     generation, fitness[best]['score'], len(elites), len(parents), record['qualified_candidates'])
            if generation < last_generation:
                population = breed(rng, population, elites, parents, sigma)

    # El genoma queda congelado antes de construir el evaluador de confirmación.
    confirm_sim = simulate(confirmation)
    confirmed = confirm_sim.evaluate(champion_genome)
    final = dict(phase='Confirmación', generation=champion['generation'],
                 metrics=confirmed, fitness=assess_fitness(confirmed))
    lineage_state = dict(last_completed_generation=last_generation, next_mutation_sigma=sigma)
    lineage_digest = save_candidate(root/'lineage.zip', champion_genome, normalizer,
                                    dict(metadata, selection=champion, lineage_state=lineage_state))
    report = dict(metadata, generations=history, champion=champion, confirmation=final,
                  lineage_state=lineage_state, lineage_bundle_sha256=lineage_digest,
                  qualified=champion['fitness']['passed'] and final['fitness']['passed'])
    write_json(root/'evolution.json', report)
    _atomic_write(root/'evolution.csv', lambda p: pd.DataFrame(candidate_rows).to_csv(p, index=False))
    write_summary(root/'evolution.md', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = EvolutionConfig()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('reports/evolution'))
    parser.add_argument('--seed-bundle', type=Path)
    parser.add_argument('--population-size', type=int, default=defaults.population_size)
    parser.add_argument('--generations', type=int, default=defaults.generations)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--workers', type=int, default=defaults.workers)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = EvolutionConfig(args.population_size, args.generations, args.seed, args.workers)
    run_evolution(args.input, args.output_dir, config, seed_bundle=args.seed_bundle)


if __name__ == '__main__':
    main()
