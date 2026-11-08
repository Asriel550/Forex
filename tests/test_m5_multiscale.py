import numpy as np
import pandas as pd
import pytest

from scripts.run_m5_multiscale import archived_march_opens, load_control
from src.environment import AccountConfig
from src.evolution_context import H1_STRUCTURAL, M15_TACTICAL, native_direction_context
from src.evolution_multiscale import (
    ARM_WIDTHS, FEATURE_NAMES, GENOME_WIDTH, HIDDEN, MultiregimeFitness, arm_inputs,
    directional_metrics, directions, evolve_replica, fit_scale, project_founder, scale_inputs, volatility_groups,
)


def test_native_indicators_use_closed_complete_minutes_and_expire_on_a_gap():
    start = pd.Timestamp('2026-02-25T00:00:00Z')
    minutes = pd.date_range(start, periods=33 * 60, freq='min')
    close = 1.1 + np.arange(len(minutes)) * .000001
    raw = pd.DataFrame(dict(timestamp=minutes, open=close, high=close+.0001,
                            low=close-.0001, close=close, volume=0.))
    cut = start + pd.Timedelta(hours=31)
    decisions = pd.date_range(cut - pd.Timedelta(minutes=15), cut, freq='5min')
    full = native_direction_context(raw, decisions, end=minutes[-1] + pd.Timedelta(minutes=1))
    prefix = native_direction_context(raw.loc[raw.timestamp < cut], decisions, end=cut)
    pd.testing.assert_frame_equal(full, prefix)
    assert (prefix.iloc[-1] > 0.).all()
    missing = raw.loc[raw.timestamp != cut - pd.Timedelta(minutes=5)]
    with_gap = native_direction_context(missing, decisions, end=cut)
    assert with_gap.loc[cut, list(M15_TACTICAL)].isna().all()
    assert with_gap.loc[cut, list(H1_STRUCTURAL)].isna().all()
    pd.testing.assert_series_equal(with_gap.iloc[0], prefix.iloc[0])


def test_projected_founder_and_inactive_features_share_the_topology():
    assert len(archived_march_opens()) == 2859
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(64, len(FEATURE_NAMES)))
    mean, std = fit_scale(raw)
    inputs = scale_inputs(raw, mean, std)
    founder = project_founder(load_control(), mean, std)
    assert founder.shape == (GENOME_WIDTH,)
    baseline = directions(arm_inputs(inputs, 'm5'), [founder])
    for arm in ARM_WIDTHS:
        np.testing.assert_array_equal(directions(arm_inputs(inputs, arm), [founder]), baseline)

    genome = np.zeros(GENOME_WIDTH)
    genome[(len(FEATURE_NAMES) - 1) * HIDDEN] = 1.
    genome[-HIDDEN - 1] = 1.
    assert np.all(directions(arm_inputs(inputs, 'm15'), [genome]) == 1)
    np.testing.assert_array_equal(directions(arm_inputs(inputs, 'm15_h1'), [genome])[0],
                                  np.where(inputs[:, -1] >= 0., 1, -1))


def test_fitness_prefers_even_payoff_and_uses_the_fixed_denominator():
    outcomes = np.tile(np.r_[np.ones(10), 0], 2)
    groups = np.repeat([0, 1], 11)
    balanced = np.tile(np.r_[np.ones(6), -np.ones(5)], 2)
    uneven = np.r_[np.ones(8), -np.ones(3), np.ones(4), -np.ones(7)]
    mc = np.tile(np.r_[np.ones(5), -np.ones(6)], 2)
    account = AccountConfig()
    fitness = MultiregimeFitness(outcomes, mc, groups, account).evaluate(np.array([balanced, uneven]))
    stats = directional_metrics(balanced, outcomes, mc, account)
    assert stats['refunds'] == 2
    np.testing.assert_allclose(fitness['return_per_admission'], [stats['return_per_admission']] * 2)
    assert fitness['delta_vs_mc'][0] == pytest.approx(stats['delta_vs_mc'])
    assert fitness['score'][0] > fitness['score'][1]
    assert fitness['pf_log_std'][0] < fitness['pf_log_std'][1]
    assert fitness['qualifies'].tolist() == [True, False]
    train_groups, names, _ = volatility_groups(np.tile(np.arange(400), 4))
    assert len(names) == 8
    assert np.bincount(train_groups).min() >= 64


def test_evolution_continues_numbering_and_rechecks_the_supplied_founder():
    rng = np.random.default_rng(42)
    inputs = rng.normal(size=(24, len(FEATURE_NAMES)))
    founder = rng.normal(size=GENOME_WIDTH)
    outcomes = np.tile([-1, 0, 1], 8)
    mc = np.ones(24, dtype=int)
    groups = np.repeat([0, 1], 12)
    objective = MultiregimeFitness(outcomes, mc, groups, AccountConfig())
    expected = objective.evaluate(directions(arm_inputs(inputs, 'm15'), [founder]))
    result = evolve_replica(inputs, founder, objective, 'm15', 43, first_generation=336, last_generation=337)
    assert [g['generation'] for g in result['generations']] == [336, 337]
    assert result['evaluations'] == 256
    assert result['founder']['score'] == pytest.approx(expected['score'][0])
