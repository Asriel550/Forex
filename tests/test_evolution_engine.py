"""Contratos, selección, aislamiento temporal y publicación de candidatos."""

from hashlib import sha256
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
from src.evolution_context import build_entry_context, closed_trend_direction, realized_volatility, session_allowed
from src.evolution_engine import (
    MARKET, SEARCH_BLOCKS, EvolutionConfig, M5Simulation, assess_fitness, assess_search_fitness,
    breed, effective_features, run_evolution,
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


@pytest.mark.parametrize('tie_policy', ['refund', 'loss'])
def test_vectorized_execution_matches_m5_env_with_gaps_ties_and_stop(tie_policy):
    frame = candles([1.10, 1.11, 1.12, 1.12, 1.11, 1.11, 1.20, 1.19, 1.18, 1.17, 1.16, 1.15],
                    [.8, .8, .8, .2, .5, .8, .8, .8, .8, .8, .8, .8])
    frame.loc[6:, 'timestamp'] += pd.Timedelta(days=2)
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1,
                           account=AccountConfig(initial_balance=100., tie_policy=tie_policy),
                           risk=RiskConfig(min_confidence=.57, mc_z=2.24, min_expected_edge=.01))
    sim = M5Simulation(env)
    genes = np.r_[np.zeros(len(market_feature_names(1))), 1.]

    class AlwaysFollowMonteCarlo:
        def predict(self, obs, deterministic=True):
            return 2 if obs[6] > obs[7] else 0, None

    actual = sim.evaluate(genes)
    reference = evaluate_market_policy(AlwaysFollowMonteCarlo(), env)
    for key in ('net_profit', 'total_trades', 'ties', 'breakeven_trades', 'winning_trades',
                'losing_trades', 'profit_factor', 'max_drawdown_pct', 'risk_stop',
                'win_rate_excluding_refunds', 'trades_per_observed_hour'):
        assert actual[key] == pytest.approx(reference[key])
    assert actual['ties'] == 1 and actual['risk_stop']
    assert actual['eligible_setups'] == reference['stochastic_proposals']['after_economic_filter']['count']


def test_gate_enforces_all_strict_cutoffs_without_nullifying_fitness():
    metrics = dict(win_rate_excluding_refunds=.56, profit_factor=1.21, profit_factor_status='finite',
                   trades_per_observed_hour=1.5, max_drawdown_pct=14.99)
    assert assess_fitness(metrics)['passed']
    assert assess_fitness(metrics | dict(trades_per_observed_hour=3.))['passed']
    for update, failed in (
        (dict(win_rate_excluding_refunds=.5599), 'win_rate'),
        (dict(profit_factor=1.20), 'profit_factor'),
        (dict(trades_per_observed_hour=3.001), 'frequency'),
        (dict(max_drawdown_pct=15.), 'drawdown'),
    ):
        result = assess_fitness(metrics | update)
        assert not result['passed'] and not result['checks'][failed]
        assert np.isfinite(result['score'])

    scores = [assess_fitness(metrics | dict(win_rate_excluding_refunds=p))['score']
              for p in (.48, .52, .538, .562)]
    assert all(a < b for a, b in zip(scores, scores[1:]))
    near_cutoff = [assess_fitness(metrics | dict(win_rate_excluding_refunds=.56 + offset))['score']
                   for offset in (-1e-6, 1e-6)]
    assert 0 < near_cutoff[1] - near_cutoff[0] < 3e-6
    empty = assess_fitness(dict(win_rate_excluding_refunds=None, profit_factor=None,
                                profit_factor_status='no_trades', trades_per_observed_hour=0.,
                                max_drawdown_pct=0.))
    assert not empty['passed'] and np.isfinite(empty['score'])


@pytest.mark.parametrize('timeframe,width', [('H1', 12), ('H4', 48)])
def test_higher_trend_uses_only_complete_candles_and_expires_at_next_close(timeframe, width):
    frame = candles(1.1 + np.arange(width * 30) * .00001)
    trend = closed_trend_direction(frame, timeframe)
    first = width * 24 - 1
    assert np.isnan(trend[:first]).all() and trend[first] == 1.
    prefix = frame.iloc[:first + 4]
    np.testing.assert_allclose(closed_trend_direction(prefix, timeframe), trend[:len(prefix)])

    with_gap = frame.drop(index=width * 24 + 2)
    aligned = pd.Series(closed_trend_direction(with_gap, timeframe), index=with_gap.index)
    assert aligned.loc[width * 25 - 2] == 1.
    assert np.isnan(aligned.loc[width * 25 - 1])
    assert aligned.loc[width * 26 - 1] == 1.
    down = candles(1.2 - np.arange(len(frame)) * .00001)
    lateral = candles(np.full(len(frame), 1.1))
    assert closed_trend_direction(down, timeframe)[-1] == -1.
    assert closed_trend_direction(lateral, timeframe)[-1] == 0.
    assert np.isnan(closed_trend_direction(frame.iloc[:6], timeframe)).all()


def test_setups_must_pass_both_trends_session_and_volatility_before_scoring():
    frame = candles(1.1 + np.arange(12) * .00001, [.8, .8, .2, .8, .8, .8, .8, .8, .8, .8, .5, .8])
    env = BinaryOptionsEnv(frame, market=MARKET, lookback=1, indicator_period=1)
    context = pd.DataFrame(dict(
        H1=[1., -1., -1., 0., np.nan, 1., 1., 1., 1., 1., 1.],
        H4=[1., -1., 1., 1., 1., -1., np.nan, 1., 1., 1., 1.],
        session_allowed=True, volatility_allowed=True,
    ), index=env._times)
    context.iloc[7, context.columns.get_loc('session_allowed')] = False
    context.iloc[8, context.columns.get_loc('volatility_allowed')] = False
    sim = M5Simulation(env, context)
    np.testing.assert_array_equal(sim.entries, [0, 1])
    np.testing.assert_array_equal(sim.actions, [2, 0])
    assert sim.context_audit['stages'] == dict(MC=9, H1=6, H4=4, session=3, volatility=2)
    assert sim.context_audit['rejected'] == {
        'H1': dict(opposite=1, lateral=1, unavailable=1),
        'H4': dict(opposite=1, lateral=0, unavailable=1), 'session': 1, 'volatility': 1,
    }
    genes = np.r_[np.zeros(len(market_feature_names(1))), 1.]
    assert sim.evaluate(genes)['total_trades'] == 2
    with pytest.raises(ValueError, match='índice'):
        M5Simulation(env, context.iloc[::-1])


def test_session_blackouts_check_expiry_and_follow_local_dst():
    timestamps = pd.to_datetime([
        '2025-01-15T21:15Z', '2025-01-15T21:20Z', '2025-01-15T21:25Z',
        '2025-07-15T20:15Z', '2025-07-15T20:20Z', '2025-07-15T20:25Z',
        '2025-07-15T21:50Z', '2025-07-15T21:55Z', '2025-07-15T22:50Z',
        '2025-07-16T00:50Z', '2025-07-16T00:55Z',
    ], utc=True)
    np.testing.assert_array_equal(session_allowed(timestamps),
                                  [True, False, False, True, False, False, False, True, False, False, True])


def test_realized_volatility_resets_on_gap_and_floor_uses_only_train():
    frame = candles(1.1 * np.exp(np.arange(72) * .0001))
    frame.loc[20:, 'timestamp'] += MARKET.step
    observed = realized_volatility(frame)
    assert np.isnan(observed[:12]).all() and np.isnan(observed[20:32]).all()
    assert observed[32] == pytest.approx(np.sqrt(12) * .0001)
    cutoff = frame.timestamp.iloc[43]
    _, config = build_entry_context(frame, cutoff)
    changed = frame.copy()
    changed.loc[44:, 'close'] *= np.exp(np.arange(len(changed) - 44) * .001)
    _, other = build_entry_context(changed, cutoff)
    assert config['realized_volatility'] == other['realized_volatility']


def test_block_fitness_rejects_one_bad_block_and_penalizes_scale_invariant_complexity():
    overall = dict(win_rate_excluding_refunds=.61, profit_factor=1.25, profit_factor_status='finite',
                   trades_per_observed_hour=1.7, max_drawdown_pct=12., net_profit=10.)
    dense = np.array([1., -1., 1., 0.])
    sparse = np.array([1., 0., 0., 5.])
    assert effective_features(dense) == pytest.approx(3.)
    assert effective_features(dense) == effective_features(dense * .01)
    assert effective_features(np.array([0., 0., 0., 1.])) == 0.
    blocks = [overall] * SEARCH_BLOCKS
    robust = assess_search_fitness(overall, blocks, sparse)
    assert robust['passed']
    assert robust['score'] > assess_search_fitness(overall, blocks, dense)['score']
    bad = overall | dict(max_drawdown_pct=15.)
    candidate = assess_search_fitness(overall, [*blocks[:-1], bad], sparse)
    assert not candidate['checks']['block_consistency'] and not candidate['passed']
    assert candidate['score'] < robust['score']
    flat = assess_search_fitness(overall, [*blocks[:-1], overall | dict(net_profit=0.)], sparse)
    assert not flat['checks']['block_consistency']


def test_breeding_retains_elites_and_uses_only_their_genes_before_mutation():
    population = np.arange(40, dtype=float).reshape(8, 5)
    elites = [3, 5]
    crossed = breed(np.random.default_rng(5), population, elites, elites, 0.)
    mutated = breed(np.random.default_rng(5), population, elites, elites, .12)
    np.testing.assert_array_equal(crossed[:2], population[elites])
    np.testing.assert_array_equal(mutated[:2], population[elites])
    assert np.logical_or(crossed[2:] == population[3], crossed[2:] == population[5]).all()
    assert not np.array_equal(mutated[2:], crossed[2:])


@pytest.mark.parametrize('profitable', [True, False])
def test_evolution_clones_only_consistent_elites_and_preserves_parents(tmp_path, monkeypatch, profitable):
    source = tmp_path/'m5.parquet'
    candles(1.1 + np.arange(816) * .00001).to_parquet(source)
    calls = []

    def evaluate(self, genome):
        calls.append(genome.copy())
        return dict(total_trades=100, win_rate_excluding_refunds=.52 + .03 * float(np.tanh(genome[-1])),
                    profit_factor=1.1 if profitable else .9, profit_factor_status='finite',
                    trades_per_observed_hour=1.5, max_drawdown_pct=10.,
                    net_profit=5. if profitable else -5., risk_stop=False)

    monkeypatch.setattr(M5Simulation, 'evaluate', evaluate)
    output = tmp_path/'run'
    report = run_evolution(source, output, EvolutionConfig(population_size=6, generations=2, workers=1))
    assert not report['qualified']
    scores = pd.read_csv(output/'evolution.csv')
    assert scores.fitness.notna().all()
    assert scores.groupby('generation').parent.sum().eq(2).all()
    assert scores.groupby('generation').elite.sum().eq(2 if profitable else 0).all()
    assert all(g['qualified_candidates'] == 0 for g in report['generations'])
    assert len(calls) == 6 * 3 * (1 + SEARCH_BLOCKS) + 1
    generations = np.asarray(calls[:-1:1 + SEARCH_BLOCKS]).reshape(3, 6, -1)
    for generation, (before, after) in enumerate(zip(generations, generations[1:])):
        parents = scores.loc[(scores.generation == generation) & scores.parent]
        selected = parents.sort_values('fitness', ascending=False, kind='stable').candidate.to_numpy()
        if profitable:
            np.testing.assert_array_equal(after[:2], before[selected])
        else:
            assert not any(np.array_equal(child, parent) for child in after for parent in before)
    winner = report['champion']
    np.testing.assert_array_equal(calls[-1], generations[winner['generation'], winner['candidate']])
    assert winner['fitness']['score'] == pytest.approx(scores.fitness.max())


def test_evolution_saves_each_generation_and_selection_ignores_confirmation_and_holdout(tmp_path):
    frame = candles(1.1 + np.arange(2400) * .00001)
    source = tmp_path/'m5.parquet'
    frame.to_parquet(source)
    output = tmp_path/'original'
    first = run_evolution(source, output, EvolutionConfig(population_size=6, generations=2, workers=1))
    assert len(first['generations']) == 3
    assert first['normalizer_fit'] == 'train only' and first['holdout_test_evaluated'] is False
    assert len(first['search_blocks']) == SEARCH_BLOCKS
    for previous, following in zip(first['search_blocks'], first['search_blocks'][1:]):
        assert pd.Timestamp(previous['end']) < pd.Timestamp(following['start'])
    fold_setups = sum(b['context_audit']['stages']['MC'] for b in first['search_blocks'])
    assert fold_setups == first['context_audit']['Búsqueda']['stages']['MC'] - (SEARCH_BLOCKS - 1)
    start = pd.Timestamp(first['splits']['confirmation']['start'])
    assert start > pd.Timestamp(first['splits']['search']['end']) + MARKET.step
    future = frame.timestamp >= start
    frame.loc[future, ['open', 'high', 'low', 'close']] -= np.arange(future.sum())[:, None] * .0001
    source_changed = tmp_path/'m5_changed.parquet'
    frame.to_parquet(source_changed)
    second = run_evolution(source_changed, tmp_path/'changed',
                           EvolutionConfig(population_size=6, generations=2, workers=2))
    assert first['champion']['generation'] == second['champion']['generation']
    assert first['champion']['metrics'] == second['champion']['metrics']
    assert first['confirmation']['metrics'] != second['confirmation']['metrics']
    assert first['entry_context']['realized_volatility'] == second['entry_context']['realized_volatility']
    assert len(pd.read_csv(output/'evolution.csv')) == 18
    assert (output/'evolution.md').is_file()
    for generation in first['generations']:
        path = output/'candidates'/generation['bundle']
        other = tmp_path/'changed'/'candidates'/generation['bundle']
        assert file_sha256(path) == generation['bundle_sha256']
        with ZipFile(path) as archive, ZipFile(other) as changed:
            manifest = json.loads(archive.read('manifest.json'))
            assert manifest['market']['timeframe'] == 'M5'
            for name, digest in manifest['sha256'].items():
                assert sha256(archive.read(name)).hexdigest() == digest
                assert archive.read(name) == changed.read(name)


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
    jobs = workflow['jobs']
    assert 'evolution' in jobs['research-pipeline']['if'] and 'evolution' in jobs['unit-tests']['if']
    assert 'evolution' not in jobs['telegram-alert']['if']
    steps = jobs['research-pipeline']['steps']
    evolution = next(s for s in steps if s.get('name') == 'Evolve isolated M5 population')
    assert evolution['if'] == "inputs.mode == 'evolution'"
    upload = next(s for s in steps if s.get('name') == 'Upload generational research candidates')
    assert upload['if'] == "always() && inputs.mode == 'evolution'"
    assert upload['with']['path'] == 'reports/evolution/candidates/bundle_scalping_gen_*.zip'
    publish = next(s for s in steps if s.get('name', '').startswith('Publish isolated v4'))
    assert publish['if'] == "success() && inputs.mode == 'train'"
