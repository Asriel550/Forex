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
from .environment import BinaryOptionsEnv, fit_normalizer, prepare_market_data
from .evolution_context import build_entry_context, filter_setups
from .market import MarketConfig, PROTECTED_HOLDOUT_START
from .metrics import _atomic_write, calculate_metrics
from .trading_profiles import SCALPING_TARGET_PER_HOUR, resolve_profile
from .train import chronological_market_split, write_json

LOG = logging.getLogger(__name__)
MARKET = MarketConfig('M5', 'binary', 1)
SEARCH_BLOCKS = 4
WORST_BLOCK_WEIGHT = .5
COMPLEXITY_WEIGHT = .10
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
        if env.market != MARKET or not env.risk.enabled:
            raise ValueError('La evolución requiere M5 binario, una vela y filtro activo')
        entries = env._eligible[env._setup_mask[env._eligible]]
        up, down, _ = env._probabilities[entries].T
        actions = np.where(up > down, 2, 0)
        self.observed_hours = sum(end - start for start, end in env.segments) / 12
        self.context_audit = None
        if context is not None:
            if not context.index.equals(env._times):
                raise ValueError('El contexto debe coincidir con el índice de velas M5')
            allowed, audit = filter_setups(context, entries, actions, env._eligible)
            entries, actions = entries[allowed], actions[allowed]
            capacity = len(entries) / self.observed_hours
            self.context_audit = dict(audit, observed_hours=self.observed_hours,
                                      capacity_per_hour=capacity,
                                      frequency_minimum_reachable=capacity >= MIN_TRADES_HOUR)
        self.entries = entries
        self.features = env._market[entries]
        self.actions = actions
        change = env._close[self.entries + 1] - env._close[self.entries]
        won = np.where(self.actions == 2, change > 0, change < 0)
        tied = change == 0
        self.outcomes = np.where(tied, 'tie', np.where(won, 'win', 'loss'))
        self.account = env.account
        self.profits = self.account.stake * np.where(
            won, self.account.payout, np.where(tied & (self.account.tie_policy == 'refund'), 0., -1.),
        )
        self.timestamps = env._times + env.bar_step
        self.stop_limit = min(self.account.max_drawdown, env.risk.drawdown_limit)

    def evaluate(self, genome):
        """Acepta la dirección MC cuando el producto escalar más sesgo es positivo."""
        proposed = np.flatnonzero(np.einsum('ij,j->i', self.features, genome[:-1]) + genome[-1] > 0)
        initial = self.account.initial_balance
        balance = np.r_[initial, initial + np.cumsum(self.profits[proposed])]
        drawdown = 1 - balance / np.maximum.accumulate(balance)
        stops = np.flatnonzero((balance < self.account.stake) | (drawdown + 1e-12 >= self.stop_limit))
        if stops.size:
            # La posición cero de balance es previa al primer contrato.
            proposed = proposed[:stops[0]]

        equity = np.zeros(len(self.timestamps))
        equity[self.entries[proposed] + 1] = self.profits[proposed]
        equity = initial + np.cumsum(equity)
        trades = pd.DataFrame(dict(action=self.actions[proposed], profit=self.profits[proposed],
                                   outcome=self.outcomes[proposed]))
        result = calculate_metrics(trades, pd.DataFrame(dict(timestamp=self.timestamps, balance=equity)),
                                   initial_balance=initial, periods_per_year=252 * 24 * 12)
        result.update(return_basis='M5 equity including inactivity',
                      observed_hours=self.observed_hours, eligible_setups=len(self.entries),
                      trades_per_observed_hour=len(trades) / self.observed_hours,
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


def effective_features(genome):
    """Número efectivo de pesos. Es invariante a reescalar el genoma."""
    weights = np.abs(genome[:-1])
    scale = weights.max()
    if scale == 0.:
        return 0.
    normalized = weights / scale
    return float(normalized.sum() ** 2 / np.dot(normalized, normalized))


def assess_search_fitness(metrics, blocks, genome):
    """Exige rentabilidad y drawdown controlado en cada bloque para ser élite."""
    fitness = assess_fitness(metrics)
    scores = [assess_fitness(block)['score'] for block in blocks]
    complexity = effective_features(genome)
    penalty = COMPLEXITY_WEIGHT * complexity / (len(genome) - 1)
    score = (1 - WORST_BLOCK_WEIGHT) * float(np.mean(scores)) + WORST_BLOCK_WEIGHT * min(scores) - penalty
    fitness['checks']['block_consistency'] = all(
        block['net_profit'] > 0 and block['max_drawdown_pct'] < MAX_DRAWDOWN_PCT for block in blocks
    )
    return dict(fitness, score=score, passed=all(fitness['checks'].values()), block_scores=scores,
                effective_features=complexity, complexity_penalty=penalty)


def selection_rank(fitness):
    return fitness['checks']['block_consistency'], fitness['score']


def initial_population(rng, count, genes):
    return rng.normal(0., 1. / math.sqrt(genes), size=(count, genes))


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
        'genome.json': json.dumps(dict(weights=genome[:-1].tolist(), bias=float(genome[-1])),
                                  allow_nan=False).encode('utf-8'),
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
    lines = [
        '# Evolución Scalping M5 binario', '',
        'Las élites cumplen consistencia en todos los bloques. Solo el campeón pasa por confirmación.',
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
                  '## Filtros de entrada', '',
                  '| Tramo | MC | Tras H1 | Tras H4 | Tras sesión | Tras volatilidad | Máximo/h |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for label, audit in report['context_audit'].items():
        stages = ' | '.join(str(audit['stages'][key]) for key in ('MC', 'H1', 'H4', 'session', 'volatility'))
        lines.append(f"| {label} | {stages} | {audit['capacity_per_hour']:.3f} |")
    below_minimum = [label for label, audit in report['context_audit'].items()
                     if not audit['frequency_minimum_reachable']]
    if below_minimum:
        phases = ' y '.join(below_minimum)
        lines.extend(['', f"La capacidad de {phases} queda por debajo del mínimo de "
                          f"{MIN_TRADES_HOUR:g} contratos por hora del gate."])
    lines.extend(['', '## Bloques del campeón', '',
                  '| Bloque | Setups tras filtros | Contratos | PnL USD | DD |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for block, metrics in zip(report['search_blocks'], report['champion']['blocks']):
        setups = block['context_audit']['stages']['volatility']
        lines.append(f"| {block['block']} | {setups} | {metrics['total_trades']} | "
                     f"{metrics['net_profit']:.2f} | {metrics['max_drawdown_pct']:.2f}% |")
    lines.append('')
    _atomic_write(path, lambda p: Path(p).write_text('\n'.join(lines), encoding='utf-8'))


def run_evolution(input_path, output_dir, config=None):
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
    context, context_settings = build_entry_context(context_source, splits['train'].timestamp.iloc[-1])
    normalizer = fit_normalizer(splits['train'])
    risk = RiskConfig(min_confidence=preset['FOREX_MIN_CONFIDENCE'],
                      min_expected_edge=preset['FOREX_MIN_EXPECTED_EDGE'], mc_z=preset['FOREX_MC_Z'])
    env_options = dict(market=MARKET, prepared=True, risk=risk, normalizer=normalizer)
    def simulate(frame):
        return M5Simulation(BinaryOptionsEnv(frame, **env_options), context.loc[frame.timestamp])

    simulation = simulate(search)
    block_sims = [simulate(block) for block in blocks]

    def evaluate(genome):
        metrics = simulation.evaluate(genome)
        by_block = [sim.evaluate(genome) for sim in block_sims]
        return metrics, by_block, assess_search_fitness(metrics, by_block, genome)

    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('La evolución requiere un directorio de salida vacío')

    metadata = dict(
        format='forex_m5_evolution_candidate_v3', profile='scalping', market=asdict(MARKET),
        policy='apply MC, H1/H4, session and realized volatility filters before scoring entries with the genome',
        input_sha256=source_hash, quant_config=quant, account=asdict(simulation.account),
        risk=asdict(risk), config=asdict(config), holdout_test_evaluated=False,
        protected_holdout_start=PROTECTED_HOLDOUT_START.isoformat(), normalizer_fit='train only',
        fitness_limits=dict(min_win_rate=MIN_WIN_RATE, exclusive_min_profit_factor=MIN_PROFIT_FACTOR,
                            trades_per_hour=[MIN_TRADES_HOUR, MAX_TRADES_HOUR],
                            exclusive_max_drawdown_pct=MAX_DRAWDOWN_PCT),
        fitness_selection=dict(method='chronological_blocks_v2', blocks=SEARCH_BLOCKS,
                               worst_block_weight=WORST_BLOCK_WEIGHT, complexity_weight=COMPLEXITY_WEIGHT,
                               win_rate_shortfall_weight=WIN_RATE_SHORTFALL_WEIGHT),
        entry_context=context_settings,
        search_blocks=[dict(block=i, rows=len(frame), start=frame.timestamp.iloc[0].isoformat(),
                            end=frame.timestamp.iloc[-1].isoformat(), context_audit=sim.context_audit)
                       for i, (frame, sim) in enumerate(zip(blocks, block_sims), start=1)],
        elite_fraction=ELITE_FRACTION, confirmation_purge_bars=1,
        splits={name: dict(rows=len(frame), start=frame.timestamp.iloc[0].isoformat(),
                           end=frame.timestamp.iloc[-1].isoformat())
                for name, frame in (('train', splits['train']), ('search', search),
                                    ('confirmation', confirmation))},
    )
    rng = np.random.default_rng(config.seed)
    population = initial_population(rng, config.population_size, simulation.features.shape[1] + 1)
    elite_count = math.ceil(ELITE_FRACTION * len(population))
    history, candidate_rows = [], []
    champion = champion_genome = None
    sigma = .12
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for generation in range(config.generations + 1):
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
            if generation:
                sigma = max(.025, sigma * .85) if improved else min(.35, sigma * 1.25)
            record['mutation_sigma'] = sigma if generation < config.generations else None
            digest = save_candidate(root/'candidates'/name, population[best], normalizer,
                                    dict(metadata, selection=record))
            record['bundle_sha256'] = digest
            history.append(record)
            if improved:
                champion, champion_genome = record, population[best].copy()
            candidate_rows.extend(dict(
                generation=generation, candidate=i, fitness=f['score'], elite=i in elites,
                parent=i in parents, effective_features=f['effective_features'], complexity_penalty=f['complexity_penalty'],
                passed=f['passed'], **{f'pass_{k}': v for k, v in f['checks'].items()},
                **{k: m[k] for k in ('total_trades', 'win_rate_excluding_refunds', 'profit_factor',
                                     'profit_factor_status', 'trades_per_observed_hour', 'max_drawdown_pct',
                                     'net_profit', 'risk_stop')},
                **{f'block_{j}_{key}': block[key] for j, block in enumerate(candidate_blocks, start=1)
                   for key in ('net_profit', 'max_drawdown_pct', 'total_trades')},
            ) for i, (m, f, candidate_blocks) in enumerate(zip(metrics, fitness, by_block)))
            LOG.info('Generación %d. Aptitud %.4f, élites %d, padres %d, candidatos dentro de límites %d',
                     generation, fitness[best]['score'], len(elites), len(parents), record['qualified_candidates'])
            if generation < config.generations:
                population = breed(rng, population, elites, parents, sigma)

    # El genoma queda congelado antes de construir el evaluador de confirmación.
    confirm_sim = simulate(confirmation)
    confirmed = confirm_sim.evaluate(champion_genome)
    final = dict(phase='Confirmación', generation=champion['generation'],
                 metrics=confirmed, fitness=assess_fitness(confirmed))
    report = dict(metadata, generations=history, champion=champion, confirmation=final,
                  context_audit={'Búsqueda': simulation.context_audit, 'Confirmación': confirm_sim.context_audit},
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
    parser.add_argument('--population-size', type=int, default=defaults.population_size)
    parser.add_argument('--generations', type=int, default=defaults.generations)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--workers', type=int, default=defaults.workers)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = EvolutionConfig(args.population_size, args.generations, args.seed, args.workers)
    run_evolution(args.input, args.output_dir, config)


if __name__ == '__main__':
    main()
