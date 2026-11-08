"""Contratos, selección, aislamiento temporal y publicación de candidatos."""

import base64
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

from src import trading_profiles
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.environment import AccountConfig, BinaryOptionsEnv, market_feature_names
from src.evolution_context import (
    CONTEXT_FEATURES, build_observations, closed_trend_bias, m15_direction_bias,
    realized_volatility, session_observations,
)
from src.evolution_direction import FrozenDirectionSimulation, direction_genes, frozen_simulation, with_direction
from src.evolution_engine import (
    MARKET, SEARCH_BLOCKS, EvolutionConfig, M5Simulation, assess_fitness, assess_search_fitness,
    breed, improves_on_parent, run_evolution, seeded_population,
)
from src.evolution_lineage import (
    CONTEXT_FORMAT, FORMAT, LEGACY_FORMAT, MARKET_FEATURES, PINNED_SEED, load_founder,
)
from src.market_evaluation import evaluate_market_policy


def candles(close, probabilities=.8):
    frame = pd.DataFrame(dict(
        timestamp=pd.date_range('2025-01-06', periods=len(close), freq='5min', tz='UTC'),
        open=close, close=close, cond_volatility=.0002, dist_p95=.002, dist_p05=-.002,
        mc_prob_up=probabilities, mc_prob_tie=0.,
    ))
    frame['high'], frame['low'] = frame.close + .001, frame.close - .001
    frame['mc_prob_down'] = 1 - frame.mc_prob_up
    frame.attrs.update(timeframe='M5', direction_probability_horizon=1,
                       quant_config=dict(drift_mode='momentum', drift_scale_floor=1.4, drift_scale=2.))
    return frame


@pytest.fixture
def legacy_seed(tmp_path):
    width = len(MARKET_FEATURES)
    payload = {
        'genome.json': json.dumps(dict(weights=[0.] * width, bias=.3)),
        'normalizer.json': json.dumps(dict(features=MARKET_FEATURES, mean=[0.] * width,
                                           std=[1.] * width, clip=10.)),
    }
    split = dict(end='2025-01-06T00:00:00+00:00')
    manifest = dict(format=LEGACY_FORMAT, profile='scalping', market=asdict(MARKET),
                    holdout_test_evaluated=False, splits=dict(train=split, search=split),
                    selection=dict(generation=20), account=asdict(AccountConfig()),
                    risk=dict(drawdown_limit=.20),
                    sha256={name: sha256(body.encode()).hexdigest() for name, body in payload.items()})
    path = tmp_path/'founder.zip'
    with ZipFile(path, 'w') as archive:
        for name, body in payload.items():
            archive.writestr(name, body)
        archive.writestr('manifest.json', json.dumps(manifest))
    return path


@pytest.mark.parametrize('tie_policy', ['refund', 'loss'])
def test_vectorized_execution_matches_m5_env_with_gaps_ties_and_stop(tie_policy):
    frame = candles([1.10, 1.11, 1.12, 1.12, 1.11, 1.11, 1.20, 1.19, 1.18, 1.17, 1.16, 1.15],
                    [.8, .8, .8, .2, .5, .8, .8, .8, .8, .8, .8, .8])
    frame.loc[6:, 'timestamp'] += pd.Timedelta(days=2)
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           account=AccountConfig(initial_balance=100., tie_policy=tie_policy, max_drawdown=.20),
                           risk=RiskConfig(enabled=False))
    sim = M5Simulation(env)
    width = len(market_feature_names(1))
    genes = np.r_[np.zeros(width), 1., np.zeros(width + 2)]

    class AlwaysFollowMonteCarlo:
        def predict(self, obs, deterministic=True):
            return (2 if obs[6] > obs[7] else 0 if obs[6] < obs[7] else 1), None

    actual = sim.evaluate(genes)
    reference = evaluate_market_policy(AlwaysFollowMonteCarlo(), env)
    for key in ('net_profit', 'total_trades', 'ties', 'breakeven_trades', 'winning_trades',
                'losing_trades', 'profit_factor', 'max_drawdown_pct', 'risk_stop',
                'win_rate_excluding_refunds', 'trades_per_observed_hour'):
        assert actual[key] == pytest.approx(reference[key])
    assert actual['ties'] == 1 and actual['risk_stop']
    assert actual['decision_opportunities'] == len(env._eligible)
    assert actual['signals_before_activation'] > actual['total_trades']
    assert actual['activation_rejections'] == 0
    assert actual['signals_after_activation'] == actual['signals_before_activation']
    assert actual['signals_unfilled_due_to_stop'] == actual['signals_after_activation'] - actual['total_trades']
    assert actual['trades_per_observed_hour'] == pytest.approx(actual['total_trades'] / sim.observed_hours)


def test_activation_threshold_rejects_weak_call_and_put_signals_before_stop():
    frame = candles(1.1 + np.arange(12) * .00001, np.tile([.80, .51, .49, .20], 3))
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           risk=RiskConfig(enabled=False))
    sim = M5Simulation(env)
    width = sim.features.shape[1]
    founder = np.r_[np.zeros(width), 1., np.zeros(width + 2)]
    stronger = founder.copy()
    stronger[-1] = .1
    baseline, filtered = sim.evaluate(founder), sim.evaluate(stronger)
    rejected = int((np.abs(sim.mc_margin) <= .1).sum())
    assert rejected > 0
    assert filtered['signals_before_activation'] == baseline['signals_before_activation']
    assert filtered['activation_rejections'] == rejected
    assert filtered['activation_rejection_rate'] == pytest.approx(rejected / len(sim.entries))
    assert filtered['signals_after_activation'] == filtered['total_trades']
    assert filtered['signals_unfilled_due_to_stop'] == 0
    assert 0 < filtered['calls'] < baseline['calls'] and 0 < filtered['puts'] < baseline['puts']


def test_direction_ablation_keeps_admissions_and_scores_all_eligible_bars():
    close = 1.1 + np.tile([0., .0001, .0002, .0001], 5)
    frame = candles(close, np.tile([.8, .51, .50, .20], 5))
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           risk=RiskConfig(enabled=False))
    sim = M5Simulation(env)
    width = sim.features.shape[1]
    founder = np.r_[np.zeros(width), 1., np.zeros(width), .5, .25]
    control = with_direction(founder, 0.)
    frozen = FrozenDirectionSimulation(sim, founder)
    result = frozen.evaluate(control)
    assert result['signals_after_activation'] == sim.evaluate(founder)['signals_after_activation']
    assert result['signals_after_activation'] != sim.evaluate(control)['signals_after_activation']
    assert result['directional']['eligible_bars'] == len(sim.entries)
    assert result['directional']['eligible_bars'] > result['total_trades']
    assert result['directional']['paired_gain_per_eligible_bar'] == 0.
    assert result['directional']['spearman'] == result['directional']['mc_spearman']
    assert result['directional']['score'] == 0.
    opposite = with_direction(founder, np.r_[np.zeros(width), -2.])
    assert frozen.evaluate(opposite)['directional']['disagreements'] > 0
    assert frozen.evaluate(opposite)['signals_after_activation'] == result['signals_after_activation']


def test_m15_anchor_is_high_volatility_only_and_keeps_frozen_admissions():
    frame = candles(1.1 + np.arange(24) * .00001, .6)
    frame['cond_volatility'] = np.tile([.0001, .0003], 12)
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           risk=RiskConfig(enabled=False))
    simulation = M5Simulation(env)
    width = simulation.features.shape[1]
    founder = np.r_[np.zeros(width), 1., np.zeros(width), .5, .25]
    frame = frame.loc[frame.timestamp.isin(env._times)]
    bias = pd.Series(-1., index=frame.timestamp)
    frozen = FrozenDirectionSimulation(simulation, founder)
    feature = frozen_simulation(simulation, founder, frame, bias,
                                dict(anchor_weight=0., high_volatility_cutoff=.0002))
    anchor = frozen_simulation(simulation, founder, frame, bias,
                               dict(anchor_weight=12., high_volatility_cutoff=.0002))
    base, _ = frozen.direction_scores(founder)
    regular, _ = feature.direction_scores(founder, m15_weight=2.)
    strong, _ = anchor.direction_scores(founder, m15_weight=2.)
    high = frame.cond_volatility.to_numpy()[simulation.entries] > .0002
    np.testing.assert_allclose(regular, base - 2.)
    np.testing.assert_allclose(strong, regular - 12. * high)
    np.testing.assert_array_equal(anchor.proposed, frozen.proposed)
    np.testing.assert_array_equal(feature.proposed, frozen.proposed)


def test_gate_enforces_all_strict_cutoffs_without_nullifying_fitness():
    metrics = dict(win_rate_excluding_refunds=.56, profit_factor=1.21, profit_factor_status='finite',
                   trades_per_observed_hour=1.2, max_drawdown_pct=14.99)
    assert assess_fitness(metrics)['passed']
    assert assess_fitness(metrics | dict(trades_per_observed_hour=2.5))['passed']
    for update, failed in (
        (dict(win_rate_excluding_refunds=.5599), 'win_rate'),
        (dict(profit_factor=1.20), 'profit_factor'),
        (dict(trades_per_observed_hour=1.199), 'frequency'),
        (dict(trades_per_observed_hour=2.501), 'frequency'),
        (dict(max_drawdown_pct=15.), 'drawdown'),
    ):
        result = assess_fitness(metrics | update)
        assert not result['passed'] and not result['checks'][failed]
        assert np.isfinite(result['score'])

    scores = [assess_fitness(metrics | dict(win_rate_excluding_refunds=p))['score']
              for p in (.48, .52, .538, .562)]
    assert all(a < b for a, b in zip(scores, scores[1:]))
    below, target, above = [assess_fitness(metrics | dict(win_rate_excluding_refunds=w))['score']
                            for w in (.59, .60, .61)]
    assert target - below > above - target > 0
    assert assess_fitness(metrics | dict(win_rate_excluding_refunds=.59), payout=.90)['penalties']['win_rate'] == 0.
    assert assess_fitness(metrics | dict(profit_factor=.9))['score'] == assess_fitness(metrics)['score']
    empty = assess_fitness(dict(win_rate_excluding_refunds=None, profit_factor=None,
                                profit_factor_status='no_trades', trades_per_observed_hour=0.,
                                max_drawdown_pct=0.))
    assert not empty['passed'] and np.isfinite(empty['score'])


def test_fitness_doubles_low_frequency_penalty_and_caps_exponential_drawdown_penalty():
    metrics = dict(win_rate_excluding_refunds=.60, profit_factor=1.20, profit_factor_status='finite',
                   trades_per_observed_hour=1.0, max_drawdown_pct=12.)
    penalties = assess_fitness(metrics)['penalties']
    assert penalties['frequency'] == pytest.approx(2 * (.2 / 1.2))
    dd = [assess_fitness(metrics | dict(max_drawdown_pct=d))['penalties']['drawdown']
          for d in (12., 13., 15., 16., 30.)]
    assert dd[3] - dd[2] > dd[1] - dd[0] > 0.
    assert dd[-1] == pytest.approx(.30 + .50)


def test_search_fitness_penalizes_unstable_win_rates_and_low_or_high_activity():
    overall = dict(win_rate_excluding_refunds=.56, profit_factor=1.25, profit_factor_status='finite',
                   trades_per_observed_hour=1.8, max_drawdown_pct=10., net_profit=10.)
    blocks = [overall] * SEARCH_BLOCKS
    stable = assess_search_fitness(overall, blocks)
    volatile = assess_search_fitness(overall, [
        overall | dict(win_rate_excluding_refunds=win) for win in (.46, .56, .56, .66)
    ])
    assert stable['block_win_rate_variance'] == 0.
    assert volatile['block_win_rate_variance'] == pytest.approx(.005)
    assert volatile['score'] < stable['score']

    for rate in (0., .6, 3.):
        # La baja actividad del tramo completo importa aunque los bloques operen.
        candidate = assess_search_fitness(overall | dict(trades_per_observed_hour=rate), blocks)
        assert candidate['score'] < stable['score']
    neutral = dict(win_rate_excluding_refunds=None, profit_factor=None, profit_factor_status='no_trades',
                   trades_per_observed_hour=0., max_drawdown_pct=0., net_profit=0.)
    inactive = assess_search_fitness(neutral, [neutral] * SEARCH_BLOCKS)
    assert inactive['block_win_rate_variance'] == 0.
    assert inactive['score'] < stable['score']
    in_range = assess_search_fitness(overall | dict(activation_rejection_rate=.25), blocks)
    assert in_range['score'] > stable['score']
    excessive = assess_search_fitness(overall | dict(activation_rejection_rate=.40), blocks)
    assert in_range['score'] > excessive['score']
    inactive_with_rejections = assess_search_fitness(
        neutral | dict(activation_rejection_rate=.25), [neutral] * SEARCH_BLOCKS,
    )
    assert not inactive_with_rejections['passed']


def test_walk_forward_penalizes_degradation_instead_of_late_recovery():
    strong = dict(win_rate_excluding_refunds=.64, profit_factor=1.42, profit_factor_status='finite',
                  trades_per_observed_hour=1.8, max_drawdown_pct=10., net_profit=10.)
    weak = strong | dict(win_rate_excluding_refunds=.52, profit_factor=.87, net_profit=-10.)
    degraded = assess_search_fitness(strong, [strong, strong, strong, weak])
    recovered = assess_search_fitness(strong, [weak, strong, strong, strong])
    assert degraded['block_win_rate_variance'] == pytest.approx(recovered['block_win_rate_variance'])
    assert degraded['forward_penalties'][:2] == [0., 0.]
    assert degraded['forward_penalties'][-1] > 0.
    assert recovered['forward_penalties'] == [0., 0., 0.]
    assert degraded['score'] < recovered['score']
    small_drop = assess_search_fitness(strong, [strong, strong, strong, strong | dict(win_rate_excluding_refunds=.63)])
    assert small_drop['forward_penalties'] == [0., 0., 0.]


@pytest.mark.parametrize('timeframe,width', [('M15', 3), ('H1', 12), ('H4', 48)])
def test_higher_trend_uses_only_complete_candles_and_expires_at_next_close(timeframe, width):
    frame = candles(1.1 + np.arange(width * 30) * .00001)
    trend = closed_trend_bias(frame, timeframe)
    first = width * 24 - 1
    assert np.isnan(trend[:first]).all() and trend[first] > 0.
    prefix = frame.iloc[:first + 4]
    np.testing.assert_allclose(closed_trend_bias(prefix, timeframe), trend[:len(prefix)])

    with_gap = frame.drop(index=width * 24)
    aligned = pd.Series(closed_trend_bias(with_gap, timeframe), index=with_gap.index)
    assert aligned.loc[width * 25 - 2] == trend[first]
    assert np.isnan(aligned.loc[width * 25 - 1])
    assert aligned.loc[width * 26 - 1] > 0.
    down = candles(1.2 - np.arange(len(frame)) * .00001)
    lateral = candles(np.full(len(frame), 1.1))
    assert closed_trend_bias(down, timeframe)[-1] < 0.
    assert closed_trend_bias(lateral, timeframe)[-1] == 0.
    assert np.isnan(closed_trend_bias(frame.iloc[:6], timeframe)).all()


def test_m15_direction_feature_is_causal_and_in_its_own_range_units():
    frame = candles(1.1 + np.arange(120) * .00001)
    full = m15_direction_bias(frame)
    assert full.iloc[:71].eq(0.).all() and full.iloc[71] > 0.
    assert full.abs().le(3.).all()
    pd.testing.assert_series_equal(full.iloc[:100], m15_direction_bias(frame.iloc[:100]))
    assert list(build_observations(frame).columns) == list(CONTEXT_FEATURES)


def test_policy_can_trade_all_valid_entries_and_learn_context_and_direction():
    frame = candles(1.1 + np.arange(12) * .00001, np.tile([.51, .5, .49], 4))
    frame.timestamp += pd.Timedelta(hours=21, minutes=25)
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           risk=RiskConfig(enabled=False))
    context = build_observations(frame).loc[env._times]
    assert context.ny_rollover.eq(1.).all() and context.m5_volatility_available.eq(0.).all()
    context['h1_bias'] = -1.
    context['h1_available'] = 1.
    assert context.h4_available.eq(0.).all()
    sim = M5Simulation(env, context)
    np.testing.assert_array_equal(sim.entries, env._eligible)
    width = sim.features.shape[1]
    genes = np.zeros((2, width + 1))
    genes[0, len(market_feature_names(1)) + CONTEXT_FEATURES.index('ny_rollover')] = 1.
    genes[1, len(market_feature_names(1)) + CONTEXT_FEATURES.index('h1_bias')] = -1.
    result = sim.evaluate(np.r_[genes.ravel(), 0.])
    assert result['calls'] == result['total_trades'] == result['decision_opportunities'] == len(env._eligible)
    genes[1] *= -1.
    assert sim.evaluate(np.r_[genes.ravel(), 0.])['puts'] == len(env._eligible)
    genes[0] *= -1.
    assert sim.evaluate(np.r_[genes.ravel(), 0.])['total_trades'] == 0
    with pytest.raises(ValueError, match='índice'):
        M5Simulation(env, context.iloc[::-1])


def test_session_observations_include_expiry_and_follow_local_dst():
    timestamps = pd.to_datetime([
        '2025-01-15T21:15Z', '2025-01-15T21:20Z', '2025-01-15T21:25Z',
        '2025-07-15T20:15Z', '2025-07-15T20:20Z', '2025-07-15T20:25Z',
        '2025-07-15T21:50Z', '2025-07-15T21:55Z', '2025-07-15T22:50Z',
        '2025-07-16T00:50Z', '2025-07-16T00:55Z',
    ], utc=True)
    observations = session_observations(timestamps)
    np.testing.assert_array_equal(observations['ny_rollover'], [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(observations['tokyo_early'], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0])


def test_context_is_prefix_causal_and_volatility_resets_on_gap():
    frame = candles(1.1 * np.exp(np.arange(72) * .0001))
    frame.loc[20:, 'timestamp'] += MARKET.step
    observed = realized_volatility(frame)
    assert np.isnan(observed[:12]).all() and np.isnan(observed[20:32]).all()
    assert observed[32] == pytest.approx(np.sqrt(12) * .0001)
    context = build_observations(frame)
    pd.testing.assert_frame_equal(context.iloc[:44], build_observations(frame.iloc[:44]))
    assert len(context) == len(frame) and np.isfinite(context.to_numpy()).all()
    assert context.m5_volatility_available.iloc[20:32].eq(0.).all()
    assert context.m5_realized_volatility.iloc[32] == observed[32]


def test_block_fitness_rejects_one_bad_block():
    overall = dict(win_rate_excluding_refunds=.61, profit_factor=1.25, profit_factor_status='finite',
                   trades_per_observed_hour=1.7, max_drawdown_pct=12., net_profit=10.)
    blocks = [overall] * SEARCH_BLOCKS
    robust = assess_search_fitness(overall, blocks)
    assert robust['passed']
    bad = overall | dict(max_drawdown_pct=15.)
    candidate = assess_search_fitness(overall, [*blocks[:-1], bad])
    assert not candidate['checks']['block_consistency'] and not candidate['passed']
    assert candidate['score'] < robust['score']
    flat = assess_search_fitness(overall, [*blocks[:-1], overall | dict(net_profit=0.)])
    assert not flat['checks']['block_consistency']


def test_arithmetic_crossover_blends_both_parents_before_mutation():
    population = np.arange(40, dtype=float).reshape(8, 5)
    elites = [3, 5]
    crossed = breed(np.random.default_rng(5), population, elites, elites, 0.)
    mutated = breed(np.random.default_rng(5), population, elites, elites, .12)
    np.testing.assert_array_equal(crossed[:2], population[elites])
    np.testing.assert_array_equal(mutated[:2], population[elites])
    first, second = population[elites]
    fractions = (crossed[2:] - first) / (second - first)
    assert ((fractions >= .25) & (fractions <= .75)).all()
    np.testing.assert_allclose(fractions, np.broadcast_to(fractions[:, :1], fractions.shape))
    assert not np.array_equal(mutated[2:], crossed[2:])


def test_wide_mutation_changes_weights_and_threshold_in_reserved_slots():
    founder = np.linspace(-3., 5.6, 71)
    population = np.repeat(founder[None, :], 128, axis=0)
    seeded = seeded_population(np.random.default_rng(5), founder, 128, .35, wide_count=24)
    bred = breed(np.random.default_rng(5), population, [0], [0, 1], .35, wide_count=24)
    for candidates in (seeded, bred):
        assert candidates.shape == (128, 71)
        np.testing.assert_array_equal(candidates[0], founder)
        regular, wide = candidates[1:104] - founder, candidates[104:] - founder
        assert np.std(wide) > 2.5 * np.std(regular)
        assert (wide[:, -1] != 0.).all()


def test_wide_mutant_requires_strict_gain_and_no_regression():
    parent = dict(score=.85, block_scores=[.9, .8, .85, .85], checks=dict(win_rate=True, frequency=True))
    assert improves_on_parent(dict(parent, score=.86), parent)
    assert not improves_on_parent(parent, parent)
    assert not improves_on_parent(dict(parent, score=.89, block_scores=[.98, .79, .89, .9]), parent)
    assert not improves_on_parent(dict(parent, score=.89, checks=dict(win_rate=True, frequency=False)), parent)


@pytest.mark.parametrize(('profitable', 'population_size', 'parent_count', 'wide_count'), [
    (True, 6, 2, 1),
    (False, 128, 26, 24),
])
def test_evolution_preserves_best_and_other_consistent_elites(
    tmp_path, monkeypatch, legacy_seed, profitable, population_size, parent_count, wide_count,
):
    source = tmp_path/'m5.parquet'
    candles(1.1 + np.arange(816) * .00001).to_parquet(source)
    calls = []

    def evaluate(self, genome):
        calls.append(genome.copy())
        return dict(total_trades=100, decision_opportunities=len(self.entries), calls=40, puts=60,
                    win_rate_excluding_refunds=.52 + .03 * float(np.tanh(genome[-2])),
                    profit_factor=1.1 if profitable else .9, profit_factor_status='finite',
                    trades_per_observed_hour=1.5, max_drawdown_pct=10.,
                    net_profit=5. if profitable else -5., risk_stop=False, observed_hours=self.observed_hours,
                    activation_threshold=genome[-1], signals_before_activation=100,
                    activation_rejections=0, activation_rejection_rate=0.,
                    signals_after_activation=100, signals_unfilled_due_to_stop=0)

    monkeypatch.setattr(M5Simulation, 'evaluate', evaluate)
    output = tmp_path/'run'
    report = run_evolution(source, output, EvolutionConfig(population_size=population_size, generations=2, workers=1),
                           seed_bundle=legacy_seed)
    assert not report['qualified']
    scores = pd.read_csv(output/'evolution.csv')
    assert scores.fitness.notna().all()
    assert scores.groupby('generation').parent.sum().eq(parent_count).all()
    assert scores.groupby('generation').elite.sum().eq(parent_count if profitable else 1).all()
    assert scores.groupby('generation').wide_mutation.sum().eq(wide_count).all()
    assert not scores.loc[~scores.selection_eligible, ['parent', 'elite']].any().any()
    for _, group in scores.groupby('generation'):
        explorers = group.loc[group.wide_mutation]
        assert (explorers.selection_eligible == explorers.fitness.gt(group.iloc[0].fitness)).all()
    assert all(g['qualified_candidates'] == 0 for g in report['generations'])
    assert len(calls) == population_size * 3 * (1 + SEARCH_BLOCKS) + 1
    generations = np.asarray(calls[:-1:1 + SEARCH_BLOCKS]).reshape(3, population_size, -1)
    start = report['lineage']['first_generation']
    for generation, (before, after) in enumerate(zip(generations, generations[1:]), start=start):
        parents = scores.loc[(scores.generation == generation) & scores.parent]
        selected = parents.sort_values('fitness', ascending=False, kind='stable').candidate.to_numpy()
        if profitable:
            np.testing.assert_array_equal(after[:parent_count], before[selected])
        else:
            np.testing.assert_array_equal(after[0], before[selected[0]])
            assert not any(np.array_equal(child, parent) for child in after[1:] for parent in before)
    winner = report['champion']
    np.testing.assert_array_equal(calls[-1], generations[winner['generation'] - start, winner['candidate']])
    assert winner['fitness']['score'] == pytest.approx(scores.loc[scores.selection_eligible, 'fitness'].max())


def test_evolution_saves_each_generation_and_selection_ignores_confirmation_and_holdout(tmp_path, legacy_seed):
    frame = candles(1.1 + np.arange(2400) * .00001)
    source = tmp_path/'m5.parquet'
    frame.to_parquet(source)
    output = tmp_path/'original'
    first = run_evolution(source, output, EvolutionConfig(population_size=6, generations=2, workers=1),
                          seed_bundle=legacy_seed)
    assert [g['generation'] for g in first['generations']] == [21, 22, 23]
    assert first['holdout_test_evaluated'] is False
    assert len(first['search_blocks']) == SEARCH_BLOCKS
    for previous, following in zip(first['search_blocks'], first['search_blocks'][1:]):
        assert pd.Timestamp(previous['end']) < pd.Timestamp(following['start'])
    champion = first['champion']
    fold_entries = sum(b['decision_opportunities'] for b in champion['blocks'])
    assert fold_entries == champion['metrics']['decision_opportunities'] - (SEARCH_BLOCKS - 1)
    start = pd.Timestamp(first['splits']['confirmation']['start'])
    assert start > pd.Timestamp(first['splits']['search']['end']) + MARKET.step
    future = frame.timestamp >= start
    frame.loc[future, ['open', 'high', 'low', 'close']] -= np.arange(future.sum())[:, None] * .0001
    source_changed = tmp_path/'m5_changed.parquet'
    frame.to_parquet(source_changed)
    second = run_evolution(source_changed, tmp_path/'changed',
                           EvolutionConfig(population_size=6, generations=2, workers=2), seed_bundle=legacy_seed)
    assert first['champion']['generation'] == second['champion']['generation']
    assert first['champion']['metrics'] == second['champion']['metrics']
    assert first['founder_evaluation'] == second['founder_evaluation']
    assert first['confirmation']['metrics'] != second['confirmation']['metrics']
    assert len(pd.read_csv(output/'evolution.csv')) == 18
    assert (output/'evolution.md').is_file()
    for generation in first['generations']:
        path = output/'candidates'/generation['bundle']
        other = tmp_path/'changed'/'candidates'/generation['bundle']
        assert file_sha256(path) == generation['bundle_sha256']
        with ZipFile(path) as archive, ZipFile(other) as changed:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['format'] == FORMAT
            assert manifest['market']['timeframe'] == 'M5'
            for name, digest in manifest['sha256'].items():
                assert sha256(archive.read(name)).hexdigest() == digest
                assert archive.read(name) == changed.read(name)
    with ZipFile(output/'lineage.zip') as checkpoint, ZipFile(tmp_path/'changed'/'lineage.zip') as changed:
        assert all(checkpoint.read(name) == changed.read(name) for name in ('genome.json', 'normalizer.json'))


def test_archived_gen20_preserves_legacy_genes_normalizer_and_cutoffs():
    pinned = json.loads(PINNED_SEED.read_text())
    bundle = base64.b64decode(pinned['bundle_base64'])
    assert sha256(bundle).hexdigest() == pinned['bundle_sha256']
    with ZipFile(BytesIO(bundle)) as archive:
        manifest, genome, normalizer = [
            json.loads(archive.read(name)) for name in ('manifest.json', 'genome.json', 'normalizer.json')
        ]
    context = build_observations(candles(1.1 + np.arange(48) * .00001))
    train_end, search_end = (pd.Timestamp(manifest['splits'][phase]['end']) for phase in ('train', 'search'))
    founder = load_founder(None, context, train_end, search_end)
    assert founder.next_generation == 21
    assert founder.provenance['run_id'] == 42421933732
    assert founder.provenance['bundle_sha256'] == pinned['bundle_sha256']
    entry, direction = founder.genome[:-1].reshape(2, -1)
    assert founder.genome[-1] == 0.
    width = len(MARKET_FEATURES)
    np.testing.assert_array_equal(entry[:width], genome['weights'])
    np.testing.assert_array_equal(entry[width:-1], 0.)
    np.testing.assert_array_equal(direction, 0.)
    assert entry[-1] == genome['bias']
    for key in ('features', 'mean', 'std'):
        assert founder.normalizer[key][:width] == normalizer[key]
    with pytest.raises(ValueError, match='corte actual de search'):
        load_founder(None, context, train_end, search_end - MARKET.step)


def test_next_run_keeps_search_champion_and_continues_after_last_generation(tmp_path, legacy_seed):
    source = tmp_path/'m5.parquet'
    candles(1.1 + np.arange(816) * .00001).to_parquet(source)
    config = EvolutionConfig(population_size=4, generations=2, workers=1)
    output = tmp_path/'run'
    first = run_evolution(source, output, config, seed_bundle=legacy_seed)
    assert not first['qualified']
    checkpoint = output/'lineage.zip'
    assert file_sha256(checkpoint) == first['lineage_bundle_sha256']
    with ZipFile(checkpoint) as archive, ZipFile(output/'candidates'/first['champion']['bundle']) as best:
        for name in ('genome.json', 'normalizer.json'):
            assert archive.read(name) == best.read(name)
    resumed = tmp_path/'resumed'
    report = run_evolution(source, resumed, config, seed_bundle=checkpoint)
    assert [g['generation'] for g in report['generations']] == [24, 25, 26]
    assert report['lineage']['source_generation'] == first['champion']['generation']
    assert report['lineage']['bundle_sha256'] == first['lineage_bundle_sha256']
    assert report['founder_evaluation']['generation'] == first['champion']['generation']
    assert report['founder_evaluation']['fitness'] == first['champion']['fitness']
    founder_rows = pd.read_csv(resumed/'evolution.csv').query('founder')
    assert len(founder_rows) == 1
    assert founder_rows.fitness.iloc[0] == pytest.approx(first['champion']['fitness']['score'])

    with ZipFile(checkpoint) as archive:
        genome = json.loads(archive.read('genome.json'))
        normalizer = archive.read('normalizer.json')
        manifest = json.loads(archive.read('manifest.json'))
    assert founder_rows.activation_threshold.iloc[0] == pytest.approx(genome.pop('activation_threshold'))
    genome_data = json.dumps(genome).encode()
    manifest['format'] = CONTEXT_FORMAT
    manifest['sha256']['genome.json'] = sha256(genome_data).hexdigest()
    v4 = tmp_path/'previous.zip'
    with ZipFile(v4, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr('genome.json', genome_data)
        archive.writestr('normalizer.json', normalizer)
    train_end, search_end = (pd.Timestamp(manifest['splits'][phase]['end']) for phase in ('train', 'search'))
    migrated = load_founder(v4, None, train_end, search_end)
    assert migrated.next_generation == 24 and migrated.genome[-1] == 0.
    for head, name in zip(migrated.genome[:-1].reshape(2, -1), ('entry', 'direction_residual')):
        np.testing.assert_array_equal(head, [*genome[name]['weights'], genome[name]['bias']])


def test_direction_only_evolution_freezes_entry_genes_and_isolates_checkpoint(tmp_path, legacy_seed):
    source = tmp_path/'m5.parquet'
    candles(1.1 + np.arange(816) * .00001).to_parquet(source)
    output = tmp_path/'direction'
    with pytest.raises(ValueError, match='checkpoint de entrada explícito'):
        run_evolution(source, output, direction_only=True)
    report = run_evolution(source, output, EvolutionConfig(population_size=4, generations=1, workers=1),
                           seed_bundle=legacy_seed, direction_only=True)
    assert report['direction_ablation']['frozen_entry_bundle_sha256'] == file_sha256(legacy_seed)
    admitted = report['founder_evaluation']['metrics']['signals_after_activation']
    candidates = pd.read_csv(output/'evolution.csv')
    assert candidates.signals_after_activation.eq(admitted).all()
    assert candidates.directional_eligible_bars.eq(candidates.decision_opportunities).all()
    assert candidates.loc[(candidates.generation == 21) & (candidates.candidate == 1),
                          'directional_score'].iloc[0] == 0.
    checkpoint = output/'lineage.zip'
    train_end, search_end = (pd.Timestamp(report['splits'][phase]['end']) for phase in ('train', 'search'))
    with pytest.raises(ValueError, match='entradas congeladas'):
        load_founder(checkpoint, None, train_end, search_end)
    winner = load_founder(checkpoint, None, train_end, search_end, allow_direction_ablation=True)
    assert winner.genome[-1] == 0.
    np.testing.assert_array_equal(winner.genome[:len(direction_genes(winner.genome))-1], 0.)
    assert winner.genome[len(direction_genes(winner.genome))-1] == .3


def test_m15_arms_continue_from_same_direction_seed(tmp_path, legacy_seed):
    source = tmp_path/'m5.parquet'
    candles(1.1 + np.arange(816) * .00001).to_parquet(source)
    config = EvolutionConfig(population_size=4, generations=1, workers=1)
    prior = tmp_path/'prior'
    original = run_evolution(source, prior, config, seed_bundle=legacy_seed, direction_only=True)
    for arm in ('control', 'feature', 'anchor'):
        output = tmp_path/arm
        report = run_evolution(source, output, config, seed_bundle=legacy_seed, direction_only=True,
                               direction_seed_bundle=prior/'lineage.zip', m15_arm=arm)
        assert report['lineage']['first_generation'] == 23
        assert report['lineage']['source_generation'] == original['champion']['generation']
        assert report['direction_ablation']['frozen_entry_bundle_sha256'] == file_sha256(legacy_seed)
        assert report['direction_ablation']['m15']['arm'] == arm
        assert report['champion']['metrics']['signals_after_activation'] == original['champion']['metrics']['signals_after_activation']
        with ZipFile(output/'lineage.zip') as archive:
            manifest = json.loads(archive.read('manifest.json'))
        assert manifest['selection']['m15_weight'] == report['champion']['m15_weight']


def test_dispatch_defaults_to_m5_and_separates_artifacts(monkeypatch, capsys):
    for name in ('FOREX_TRADING_PROFILE', 'FOREX_TIMEFRAME', 'FOREX_TRADING_MODE',
                 'FOREX_EXPIRY_CANDLES', 'FOREX_MIN_CONFIDENCE', 'FOREX_DRIFT_SCALE', 'GITHUB_ENV'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('RESEARCH_MODE', 'evolution')
    monkeypatch.setenv('FOREX_ENT_COEF', '.03')
    trading_profiles.main()
    resolved = dict(line.split('=') for line in capsys.readouterr().out.splitlines())
    assert resolved['FOREX_TRADING_PROFILE'] == 'scalping' and resolved['FOREX_TIMEFRAME'] == 'M5'
    assert resolved['FOREX_TRADING_MODE'] == 'binary' and resolved['FOREX_EXPIRY_CANDLES'] == '1'
    monkeypatch.setenv('FOREX_EXPIRY_CANDLES', '2')
    with pytest.raises(ValueError, match='una vela'):
        trading_profiles.main()

    path = Path(__file__).resolve().parents[1]/'.github/workflows/pipeline.yml'
    workflow = yaml.safe_load(path.read_text())
    # PyYAML interpreta la clave on como booleana.
    inputs = workflow.get('on', workflow.get(True))['workflow_dispatch']['inputs']
    assert inputs['population_size']['default'] == EvolutionConfig().population_size == 128
    assert EvolutionConfig(population_size=64).population_size == 64
    with pytest.raises(ValueError, match='population_size debe ser un entero de 4 a 128'):
        EvolutionConfig(population_size=129)
    jobs = workflow['jobs']
    assert 'evolution' in jobs['research-pipeline']['if'] and 'evolution' in jobs['unit-tests']['if']
    assert 'evolution' not in jobs['telegram-alert']['if']
    steps = jobs['research-pipeline']['steps']
    evolution = next(s for s in steps if s.get('name') == 'Evolve isolated M5 population')
    assert evolution['if'] == "inputs.mode == 'evolution'"
    parent = next(s for s in steps if s.get('id') == 'evolution-parent')
    assert parent['if'] == "inputs.mode == 'evolution'"
    assert 'a.workflow_run?.head_branch === branch' in parent['with']['script']
    assert 'Date.parse(b.created_at) - Date.parse(a.created_at)' in parent['with']['script']
    assert 'run.conclusion' not in parent['with']['script']
    download = next(s for s in steps if s.get('name') == 'Download previous M5 research lineage')
    assert download['with']['artifact-ids'] == '${{ steps.evolution-parent.outputs.artifact_id }}'
    lineage = next(s for s in steps if s.get('name') == 'Save M5 research lineage')
    gate = next(s for s in steps if s.get('name') == 'Enforce evolutionary validation gate')
    assert lineage['with']['path'] == 'reports/evolution/lineage.zip'
    assert steps.index(evolution) < steps.index(lineage) < steps.index(gate)
    upload = next(s for s in steps if s.get('name') == 'Upload generational research candidates')
    assert upload['if'] == "always() && inputs.mode == 'evolution'"
    assert upload['with']['path'] == 'reports/evolution/candidates/bundle_scalping_gen_*.zip'
    publish = next(s for s in steps if s.get('name', '').startswith('Publish isolated v4'))
    assert publish['if'] == "success() && inputs.mode == 'train'"
