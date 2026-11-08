"""Evolución genética de políticas de entrada para opciones binarias M5."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import chain
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
from .evolution_context import CONTEXT_SETTINGS, M15_DIRECTION_CLIP, build_observations, m15_direction_bias
from .evolution_direction import (
    DISAGREEMENT_WEIGHT, M15_ANCHOR_WEIGHT, M15_ARMS, M15_WEIGHT_LIMIT, SPEARMAN_WEIGHT,
    direction_genes, frozen_simulation, with_direction,
)
from .evolution_lineage import FORMAT, MARKET, MARKET_FEATURES, genome_payload, load_founder
from .market import PROTECTED_HOLDOUT_START
from .metrics import _atomic_write, calculate_metrics
from .trading_profiles import resolve_profile
from .train import chronological_market_split, write_json

LOG = logging.getLogger(__name__)
MAX_POPULATION_SIZE = 128
SEARCH_BLOCKS = 4
OVERALL_SCORE_WEIGHT = .25
WORST_FORWARD_WEIGHT = .5
FORWARD_REFERENCE_WEIGHT = .25
FORWARD_DROP_TOLERANCE = .05
ELITE_FRACTION = .20
CROSSOVER_WEIGHT_RANGE = (.25, .75)
WIDE_MUTATION_FRACTION = .1875
WIDE_MUTATION_SIGMA_MULTIPLIER = 3.
WIDE_MUTATION_SIGMA_FLOOR = .35
MIN_WIN_RATE = .56
WIN_RATE_SHORTFALL_WEIGHT = .5
MIN_PROFIT_FACTOR = 1.20
MAX_DRAWDOWN_PCT = 15.
DRAWDOWN_SOFT_START_PCT = 12.
DRAWDOWN_EXP_STEP_PCT = 3.
DRAWDOWN_EXP_WEIGHT = .04
DRAWDOWN_EXP_CAP = .50
MIN_TRADES_HOUR, MAX_TRADES_HOUR = 1.2, 2.5
LOW_FREQUENCY_WEIGHT = 2.
MIN_ACTIVATION_REJECTION, MAX_ACTIVATION_REJECTION = .20, .30
ACTIVATION_REJECTION_WEIGHT = .15
DIRECTIONAL_FITNESS_WEIGHT = 1.


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = MAX_POPULATION_SIZE
    generations: int = 10
    seed: int = 42
    workers: int = 4

    def __post_init__(self):
        for name, lower, upper in (
            ('population_size', 4, MAX_POPULATION_SIZE), ('generations', 1, 20),
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

    def decision_scores(self, genome):
        heads = genome[:-1].reshape(2, self.features.shape[1] + 1)
        scores = np.einsum('ij,kj->ik', self.features, heads[:, :-1]) + heads[:, -1]
        return scores[:, 0], self.mc_margin + scores[:, 1]

    def evaluate(self, genome):
        """El genoma decide entrada y dirección. Los retornos futuros solo liquidan."""
        entry, direction = self.decision_scores(genome)
        activation_threshold = float(genome[-1])
        margin = np.abs(direction)
        entries = (entry > 0.) & (margin > 0.)
        proposed = np.flatnonzero(entries & (margin > activation_threshold))
        return self.simulate_trades(direction, proposed, signals=int(entries.sum()),
                                    activation_threshold=activation_threshold)

    def simulate_trades(self, direction, proposed, *, signals, activation_threshold):
        """Liquida una secuencia de entradas bajo las mismas reglas de caja y stop."""
        accepted = len(proposed)
        rejected = signals - accepted
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
                      activation_threshold=activation_threshold, signals_before_activation=signals,
                      activation_rejections=rejected, activation_rejection_rate=rejected / max(1, signals),
                      signals_after_activation=accepted, signals_unfilled_due_to_stop=accepted - len(trades),
                      calls=int((actions == 2).sum()), puts=int((actions == 0).sum()),
                      risk_stop=bool(stops.size))
        return result


def _effective_win_target(payout):
    return max(MIN_WIN_RATE, MIN_PROFIT_FACTOR / (MIN_PROFIT_FACTOR + payout))


def assess_fitness(metrics, *, payout=.8):
    """Puntúa actividad, acierto y riesgo. El gate también comprueba PF estricto."""
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
    win_term = win if win is not None else 0.
    activity_weight = min(1., rate / MIN_TRADES_HOUR)
    frequency_penalty = (LOW_FREQUENCY_WEIGHT * max(0., MIN_TRADES_HOUR - rate)
                         + max(0., rate - MAX_TRADES_HOUR)) / MIN_TRADES_HOUR
    dd_excess = max(0., drawdown - DRAWDOWN_SOFT_START_PCT) / DRAWDOWN_EXP_STEP_PCT
    drawdown_penalty = drawdown / 100 + min(DRAWDOWN_EXP_CAP, DRAWDOWN_EXP_WEIGHT * math.expm1(dd_excess))
    win_shortfall = WIN_RATE_SHORTFALL_WEIGHT * max(0., _effective_win_target(payout) - win_term)
    score = activity_weight * win_term - frequency_penalty - drawdown_penalty - win_shortfall
    return dict(checks=checks, passed=all(checks.values()), score=score,
                penalties=dict(frequency=frequency_penalty, drawdown=drawdown_penalty, win_rate=win_shortfall))


def assess_search_fitness(metrics, blocks, *, payout=.8):
    """Puntúa al mismo genoma en transiciones cronológicas internas de búsqueda."""
    fitness = assess_fitness(metrics, payout=payout)
    scores = [assess_fitness(block, payout=payout)['score'] for block in blocks]
    forward_scores, forward_penalties = [], []
    for previous, following in zip(scores, scores[1:]):
        penalty = FORWARD_REFERENCE_WEIGHT * max(0., previous - following - FORWARD_DROP_TOLERANCE)
        forward_penalties.append(penalty)
        forward_scores.append(FORWARD_REFERENCE_WEIGHT * previous
                              + (1 - FORWARD_REFERENCE_WEIGHT) * following - penalty)
    forward_score = ((1 - WORST_FORWARD_WEIGHT) * float(np.mean(forward_scores))
                     + WORST_FORWARD_WEIGHT * min(forward_scores))
    win_rates = [block['win_rate_excluding_refunds'] or 0. for block in blocks]
    win_rate_variance = float(np.var(win_rates))
    rejection = metrics.get('activation_rejection_rate', 0.)
    rejection_distance = max(0., MIN_ACTIVATION_REJECTION - rejection, rejection - MAX_ACTIVATION_REJECTION)
    score = (OVERALL_SCORE_WEIGHT * fitness['score'] + (1 - OVERALL_SCORE_WEIGHT) * forward_score
             - ACTIVATION_REJECTION_WEIGHT * rejection_distance)
    fitness['checks']['block_consistency'] = all(
        block['net_profit'] > 0 and block['max_drawdown_pct'] < MAX_DRAWDOWN_PCT for block in blocks
    )
    return dict(fitness, score=score, passed=all(fitness['checks'].values()),
                block_scores=scores, forward_scores=forward_scores, forward_penalties=forward_penalties,
                block_win_rate_variance=win_rate_variance)


def improves_on_parent(candidate, parent):
    """Un salto amplio debe mejorar la aptitud sin ceder terreno por bloque."""
    block_scores = zip(candidate['block_scores'], parent['block_scores'], strict=True)
    return (
        candidate['score'] > parent['score']
        and all(child >= base for child, base in block_scores)
        and all(not passed or candidate['checks'][key] for key, passed in parent['checks'].items())
    )


def _wide_mutants(rng, parent, count, sigma):
    wide_sigma = max(WIDE_MUTATION_SIGMA_FLOOR, WIDE_MUTATION_SIGMA_MULTIPLIER * sigma)
    return parent + rng.normal(0., wide_sigma, size=(count, parent.size))


def seeded_population(rng, founder, count, sigma, *, wide_count=0):
    population = np.repeat(founder[None, :], count, axis=0)
    regular = population[1:count - wide_count]
    regular += rng.normal(0., sigma, size=regular.shape)
    population[count - wide_count:] = _wide_mutants(rng, founder, wide_count, sigma)
    return population


def breed(rng, population, elite_indices, parent_indices, sigma, *, wide_count=0):
    """Copia las élites y mezcla dos genomas por hijo antes de la mutación ordinaria."""
    elites = population[elite_indices]
    parents = population[parent_indices]
    count = len(population) - len(elites) - wide_count
    left = rng.integers(len(parents), size=count)
    right = (left + rng.integers(1, len(parents), size=count)) % len(parents) if len(parents) > 1 else left
    weight = rng.uniform(*CROSSOVER_WEIGHT_RANGE, size=(count, 1))
    primary = parents[left]
    children = primary + weight * (parents[right] - primary)
    children += rng.normal(0., sigma, size=children.shape)
    if wide_count:
        children = np.vstack((children, _wide_mutants(rng, parents[0], wide_count, sigma)))
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
    baseline = report['founder_evaluation']
    lines = [
        '# Evolución Scalping M5 binario', '',
        'Se conserva el mejor genoma y las élites adicionales cumplen consistencia por bloque.',
        'Solo el campeón pasa por confirmación.',
        f"Genoma fundador de la generación {lineage['source_generation']}. "
        f"Corrida de la generación {lineage['first_generation']} a la {lineage['last_generation']}.",
        f"Aptitud reevaluada del fundador {baseline['fitness']['score']:.4f}.",
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
    wide_eligible = sum(row['wide_eligible'] for row in report['generations'])
    wide_total = report['mutation_policy']['wide_per_generation'] * len(report['generations'])
    lines.extend(['', f"Mutación amplia. {wide_eligible} de {wide_total} descendientes superaron a su padre.", '',
                  f"Gate de la corrida **{status}**. Campeón preseleccionado en generación "
                      f"{report['champion']['generation']}.", '',
                  '| Tramo | Entradas evaluables | Umbral | Señales previas | Descartes por umbral | Sin ejecutar por stop | CALL ejecutados | PUT ejecutados |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for row in (report['champion'], report['confirmation']):
        metrics = row['metrics']
        lines.append(f"| {row['phase']} | {metrics['decision_opportunities']} | "
                     f"{metrics['activation_threshold']:.4f} | {metrics['signals_before_activation']} | "
                     f"{metrics['activation_rejections']} ({metrics['activation_rejection_rate']:.2%}) | "
                     f"{metrics['signals_unfilled_due_to_stop']} | "
                     f"{metrics['calls']} | {metrics['puts']} |")
    lines.extend(['', '## Bloques del campeón', '',
                  '| Bloque | Entradas evaluables | Contratos | PnL USD | DD |',
                  '| --- | ---: | ---: | ---: | ---: |'])
    for block, metrics in zip(report['search_blocks'], report['champion']['blocks']):
        lines.append(f"| {block['block']} | {metrics['decision_opportunities']} | {metrics['total_trades']} | "
                     f"{metrics['net_profit']:.2f} | {metrics['max_drawdown_pct']:.2f}% |")
    lines.append('')
    _atomic_write(path, lambda p: Path(p).write_text('\n'.join(lines), encoding='utf-8'))


def _load_direction_seed(bundle, founder, train_end, search_end, m15_arm):
    """Continúa solo desde una ablación que usó las mismas entradas congeladas."""
    with ZipFile(bundle) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    ablation = manifest.get('direction_ablation', {})
    if ablation.get('frozen_entry_bundle_sha256') != founder.provenance['bundle_sha256']:
        raise ValueError('El checkpoint direccional usó otras entradas congeladas')
    prior_m15 = ablation.get('m15')
    if prior_m15 is not None and prior_m15['arm'] != m15_arm:
        raise ValueError('El checkpoint direccional pertenece a otro brazo M15')
    direction_seed = load_founder(bundle, None, train_end, search_end, allow_direction_ablation=True)
    if (direction_seed.normalizer != founder.normalizer or direction_seed.account != founder.account
            or direction_seed.drawdown_limit != founder.drawdown_limit
            or not np.array_equal(direction_seed.genome, with_direction(founder.genome, direction_genes(direction_seed.genome)))):
        raise ValueError('El checkpoint direccional modificó los genes congelados o la simulación')
    m15_weight = float(manifest['selection'].get('m15_weight', 0.))
    if not np.isfinite(m15_weight) or abs(m15_weight) > M15_WEIGHT_LIMIT:
        raise ValueError('Peso M15 inválido en el checkpoint direccional')
    return direction_seed, m15_weight


def run_evolution(input_path, output_dir, config=None, *, seed_bundle=None, direction_only=False,
                  direction_seed_bundle=None, m15_arm=None):
    """Guarda cada ganador de generación y comprueba al campeón fuera de búsqueda."""
    config = config or EvolutionConfig()
    if direction_only and seed_bundle is None:
        raise ValueError('La ablación direccional requiere un checkpoint de entrada explícito')
    if ((direction_seed_bundle is not None or m15_arm is not None) and not direction_only
            or (m15_arm is not None and m15_arm not in M15_ARMS)):
        raise ValueError('El checkpoint direccional y M15 requieren una ablación compatible')
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
    direction_seed, initial_m15_weight = founder, 0.
    if direction_seed_bundle is not None:
        direction_seed, initial_m15_weight = _load_direction_seed(
            direction_seed_bundle, founder, train.timestamp.iloc[-1], search.timestamp.iloc[-1], m15_arm,
        )
    m15_config = m15_bias = None
    if m15_arm is not None:
        m15_config = dict(
            arm=m15_arm, feature='closed_M15_ema8_minus_ema24_over_mean_range',
            feature_clip=M15_DIRECTION_CLIP, learned_weight_limit=M15_WEIGHT_LIMIT,
            high_volatility_cutoff=float(train.cond_volatility.quantile(2/3)),
            anchor_weight=M15_ANCHOR_WEIGHT if m15_arm == 'anchor' else 0.,
        )
        if m15_arm != 'control':
            m15_bias = m15_direction_bias(context_source)
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
        sim = M5Simulation(BinaryOptionsEnv(frame, **env_options), context.loc[frame.timestamp])
        return frozen_simulation(sim, founder.genome, frame, m15_bias, m15_config) if direction_only else sim

    simulation = simulate(search)
    block_sims = [simulate(block) for block in blocks]
    payout = founder.account.payout

    def full_genome(genes):
        if m15_config is not None:
            genes = genes[:-1]
        return with_direction(founder.genome, genes) if direction_only else genes

    def m15_weight(genes):
        return float(genes[-1]) if m15_bias is not None else 0.

    def evaluate_simulation(sim, genome, weight):
        return sim.evaluate(genome, m15_weight=weight) if direction_only else sim.evaluate(genome)

    def evaluate(genes):
        genome = full_genome(genes)
        weight = m15_weight(genes)
        metrics = evaluate_simulation(simulation, genome, weight)
        by_block = [evaluate_simulation(sim, genome, weight) for sim in block_sims]
        fitness = assess_search_fitness(metrics, by_block, payout=payout)
        if direction_only:
            scores = [block['directional']['score'] for block in by_block]
            robustness = .5 * (float(np.mean(scores)) + min(scores))
            fitness['economic_score'] = fitness['score']
            fitness['directional_robustness'] = robustness
            fitness['score'] += DIRECTIONAL_FITNESS_WEIGHT * robustness
        return metrics, by_block, fitness

    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('La evolución requiere un directorio de salida vacío')

    first_generation = direction_seed.next_generation
    last_generation = first_generation + config.generations
    wide_count = max(1, round(WIDE_MUTATION_FRACTION * config.population_size))
    wide_start = config.population_size - wide_count
    metadata = dict(
        format=FORMAT, profile='scalping', market=asdict(MARKET),
        policy='positive entry head and absolute adjusted direction margin above inherited activation threshold',
        input_sha256=source_hash, quant_config=quant, account=asdict(founder.account),
        risk=dict(drawdown_limit=risk.drawdown_limit), config=asdict(config), holdout_test_evaluated=False,
        protected_holdout_start=PROTECTED_HOLDOUT_START.isoformat(),
        normalizer_fit='inherited market statistics; context fitted on train at migration',
        lineage=dict(direction_seed.provenance, first_generation=first_generation, last_generation=last_generation),
        fitness_limits=dict(min_win_rate=MIN_WIN_RATE, exclusive_min_profit_factor=MIN_PROFIT_FACTOR,
                            trades_per_hour=[MIN_TRADES_HOUR, MAX_TRADES_HOUR],
                            exclusive_max_drawdown_pct=MAX_DRAWDOWN_PCT),
        fitness_selection=dict(method='economic_walk_forward_v6', blocks=SEARCH_BLOCKS,
                               overall_score_weight=OVERALL_SCORE_WEIGHT,
                               worst_forward_weight=WORST_FORWARD_WEIGHT,
                               forward_reference_weight=FORWARD_REFERENCE_WEIGHT,
                               forward_drop_tolerance=FORWARD_DROP_TOLERANCE,
                               effective_win_target=_effective_win_target(payout),
                               win_rate_shortfall_weight=WIN_RATE_SHORTFALL_WEIGHT,
                               low_frequency_weight=LOW_FREQUENCY_WEIGHT,
                               drawdown_soft_start_pct=DRAWDOWN_SOFT_START_PCT,
                               drawdown_exp_step_pct=DRAWDOWN_EXP_STEP_PCT,
                               drawdown_exp_weight=DRAWDOWN_EXP_WEIGHT,
                               drawdown_exp_cap=DRAWDOWN_EXP_CAP,
                               activation_rejection_target=[MIN_ACTIVATION_REJECTION, MAX_ACTIVATION_REJECTION],
                               activation_rejection_weight=ACTIVATION_REJECTION_WEIGHT),
        mutation_policy=dict(regular_crossover='weighted_arithmetic', crossover_weight_range=CROSSOVER_WEIGHT_RANGE,
                             wide_fraction=WIDE_MUTATION_FRACTION, wide_per_generation=wide_count,
                             wide_sigma_multiplier=WIDE_MUTATION_SIGMA_MULTIPLIER,
                             wide_sigma_floor=WIDE_MUTATION_SIGMA_FLOOR,
                             wide_parent='incumbent', wide_selection='higher_score_no_block_or_gate_regression'),
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
    genes = direction_seed.genome
    if direction_only:
        genes = direction_genes(genes)
        metadata['policy'] = 'frozen founder admissions; evolve direction residual only'
        metadata['direction_ablation'] = dict(
            frozen_entry_bundle_sha256=founder.provenance['bundle_sha256'],
            frozen_entry_generation=founder.provenance['source_generation'],
            disagreement_weight=DISAGREEMENT_WEIGHT, spearman_weight=SPEARMAN_WEIGHT,
            fitness_weight=DIRECTIONAL_FITNESS_WEIGHT, initial_mc_candidate=1,
            tie_policy='MC direction, then frozen founder direction',
        )
        if m15_config is not None:
            # El control reserva el mismo gen inactivo para igualar el presupuesto y el RNG inicial.
            genes = np.r_[genes, initial_m15_weight]
            metadata['direction_ablation']['m15'] = m15_config
            metadata['direction_ablation']['initial_mc_candidate'] = 1 if m15_arm != 'anchor' else None
    LOG.info('Linaje M5. Fundador gen %d. Generaciones %d a %d',
             direction_seed.provenance['source_generation'], first_generation, last_generation)
    founder_result = evaluate(genes)
    founder_metrics, founder_blocks, founder_fitness = founder_result
    founder_evaluation = dict(generation=direction_seed.provenance['source_generation'],
                              metrics=founder_metrics, blocks=founder_blocks, fitness=founder_fitness)
    LOG.info('Aptitud reevaluada del fundador: %.4f', founder_fitness['score'])
    rng = np.random.default_rng([config.seed, first_generation])
    sigma = direction_seed.mutation_sigma
    population = seeded_population(rng, genes, config.population_size, sigma, wide_count=wide_count)
    if direction_only:
        # Un residuo cero representa a Monte Carlo con las entradas congeladas.
        population[1] = 0.
    elite_count = math.ceil(ELITE_FRACTION * len(population))
    history, candidate_rows = [], []
    champion = champion_genome = None
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for generation in range(first_generation, last_generation + 1):
            # Refleja mutaciones negativas del umbral para mantener una escala no negativa.
            if not direction_only:
                population[:, -1] = np.abs(population[:, -1])
            elif m15_config is not None:
                population[:, -1] = np.clip(population[:, -1], -M15_WEIGHT_LIMIT, M15_WEIGHT_LIMIT)
            if generation == first_generation:
                evaluations = chain((founder_result,), executor.map(evaluate, population[1:]))
            else:
                evaluations = executor.map(evaluate, population)
            metrics, by_block, fitness = zip(*evaluations)
            # La primera posición conserva al padre de los exploradores, incluso si falló el gate.
            eligible = [i < wide_start or improves_on_parent(f, fitness[0]) for i, f in enumerate(fitness)]
            ranks = [(f['score'], m['net_profit'] / m['observed_hours']) for m, f in zip(metrics, fitness)]
            parents = sorted((i for i, admitted in enumerate(eligible) if admitted), key=ranks.__getitem__,
                             reverse=True)[:elite_count]
            best = parents[0]
            elites = [best, *(i for i in parents[1:] if fitness[i]['checks']['block_consistency'])]
            name = f'bundle_scalping_gen_{generation}.zip'
            record = dict(phase='Búsqueda', generation=generation, candidate=best, bundle=name,
                          metrics=metrics[best], blocks=by_block[best], fitness=fitness[best],
                          net_profit_per_observed_hour=ranks[best][1],
                          elites=len(elites), parents=len(parents),
                          wide_eligible=sum(eligible[wide_start:]),
                          wide_parents=sum(i >= wide_start for i in parents),
                          qualified_candidates=sum(f['passed'] for f in fitness))
            if m15_config is not None:
                record['m15_weight'] = m15_weight(population[best])
            improved = champion is None or ranks[best] > (
                champion['fitness']['score'], champion['net_profit_per_observed_hour'],
            )
            if generation > first_generation:
                sigma = max(.025, sigma * .85) if improved else min(.35, sigma * 1.25)
            record['mutation_sigma'] = sigma if generation < last_generation else None
            winner = full_genome(population[best])
            digest = save_candidate(root/'candidates'/name, winner, normalizer,
                                    dict(metadata, selection=record))
            record['bundle_sha256'] = digest
            history.append(record)
            if improved:
                champion, champion_genome = record, winner.copy()
            candidate_rows.extend(dict(
                generation=generation, candidate=i, fitness=f['score'], elite=i in elites,
                wide_mutation=i >= wide_start, selection_eligible=eligible[i],
                block_win_rate_variance=f['block_win_rate_variance'],
                net_profit_per_observed_hour=ranks[i][1],
                parent=i in parents, founder=generation == first_generation and i == 0,
                passed=f['passed'], **{f'pass_{k}': v for k, v in f['checks'].items()},
                **{f'block_{j}_score': score for j, score in enumerate(f['block_scores'], start=1)},
                **{f'forward_to_{j}_penalty': penalty for j, penalty in enumerate(f['forward_penalties'], start=2)},
                **{f'directional_{k}': v for k, v in m.get('directional', {}).items()},
                **{k: m[k] for k in ('total_trades', 'win_rate_excluding_refunds', 'profit_factor',
                                     'profit_factor_status', 'trades_per_observed_hour', 'max_drawdown_pct',
                                     'net_profit', 'risk_stop', 'calls', 'puts', 'decision_opportunities',
                                     'activation_threshold', 'signals_before_activation', 'activation_rejections',
                                     'activation_rejection_rate', 'signals_after_activation',
                                     'signals_unfilled_due_to_stop')},
                **{f'block_{j}_{key}': block[key] for j, block in enumerate(candidate_blocks, start=1)
                   for key in ('net_profit', 'max_drawdown_pct', 'total_trades')},
            ) for i, (m, f, candidate_blocks) in enumerate(zip(metrics, fitness, by_block)))
            LOG.info('Generación %d. Aptitud %.4f, élites %d, padres %d, candidatos dentro de límites %d, '
                     'mutantes amplios elegibles %d/%d',
                     generation, fitness[best]['score'], len(elites), len(parents), record['qualified_candidates'],
                     record['wide_eligible'], wide_count)
            if generation < last_generation:
                population = breed(rng, population, elites, parents, sigma, wide_count=wide_count)

    # El genoma queda congelado antes de construir el evaluador de confirmación.
    confirm_sim = simulate(confirmation)
    confirmed = evaluate_simulation(confirm_sim, champion_genome, champion.get('m15_weight', 0.))
    final = dict(phase='Confirmación', generation=champion['generation'],
                 metrics=confirmed, fitness=assess_fitness(confirmed, payout=payout))
    lineage_state = dict(last_completed_generation=last_generation, next_mutation_sigma=sigma)
    lineage_digest = save_candidate(root/'lineage.zip', champion_genome, normalizer,
                                    dict(metadata, selection=champion, lineage_state=lineage_state))
    report = dict(metadata, founder_evaluation=founder_evaluation, generations=history, champion=champion,
                  confirmation=final,
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
    parser.add_argument('--direction-only', action='store_true')
    parser.add_argument('--direction-seed-bundle', type=Path)
    parser.add_argument('--m15-arm', choices=M15_ARMS)
    parser.add_argument('--population-size', type=int, default=defaults.population_size)
    parser.add_argument('--generations', type=int, default=defaults.generations)
    parser.add_argument('--seed', type=int, default=defaults.seed)
    parser.add_argument('--workers', type=int, default=defaults.workers)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    config = EvolutionConfig(args.population_size, args.generations, args.seed, args.workers)
    run_evolution(args.input, args.output_dir, config, seed_bundle=args.seed_bundle,
                  direction_only=args.direction_only, direction_seed_bundle=args.direction_seed_bundle,
                  m15_arm=args.m15_arm)


if __name__ == '__main__':
    main()
