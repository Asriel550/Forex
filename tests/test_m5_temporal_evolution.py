"""PR #33: objective robusto Q1/Q2, convicción causal y loss tanh."""
import numpy as np
import pandas as pd
import pytest

from scripts.run_m5_adaptive_gate import simulate_admission
from scripts.run_m5_temporal_evolution import causal_reasons, frozen_gen375
from src.evolution_engine import assess_temporal_multiregime_fitness
from src.model_training import directional_tanh_loss
from scripts.run_m5_multiscale import genome_hash
from scripts.run_m5_adaptive_gate import GEN354_SHA256


def quarter(pnl, dd=5., rate=1.3, pf=1.5, admissions=200, stopped=False):
    return dict(pnl_usd=pnl, max_drawdown_pct=dd,
                contracts_per_observed_hour=rate, profit_factor=pf,
                profit_factor_status='finite', admissions=admissions,
                observed_hours=100., account_stop=dict(reason='drawdown') if stopped else None)


def test_balanced_returns_outscore_one_lucky_quarter():
    good = [quarter(20), quarter(18)]
    fragile = [quarter(100, dd=20, pf=5), quarter(-40, dd=20, pf=.5)]
    stable = assess_temporal_multiregime_fitness(good, stake=10., direction_loss=.4)
    lucky = assess_temporal_multiregime_fitness(fragile, stake=10., direction_loss=.4)
    assert stable['qualifies'] is True
    assert lucky['qualifies'] is False
    assert stable['score'] > lucky['score']
    assert stable['quarter_return_variance'] < lucky['quarter_return_variance']


def test_drawdown_stops_loss_and_frequency_reduce_score():
    blocks = [quarter(20), quarter(20)]
    baseline = assess_temporal_multiregime_fitness(blocks, stake=10., direction_loss=.5)
    risk = assess_temporal_multiregime_fitness([
        quarter(20, dd=18, stopped=True),
        quarter(20, dd=18, stopped=True),
    ], stake=10., direction_loss=.5)
    no_density = assess_temporal_multiregime_fitness([
        quarter(20, rate=.1), quarter(20, rate=.1)
    ], stake=10., direction_loss=.5)
    high_loss = assess_temporal_multiregime_fitness(
        blocks, stake=10., direction_loss=1.5
    )
    assert baseline['score'] > risk['score']
    assert baseline['score'] > no_density['score']
    assert baseline['score'] > high_loss['score']
    with pytest.raises(ValueError):
        assess_temporal_multiregime_fitness(blocks[:1], stake=10., direction_loss=.4)


def test_high_volatility_false_positives_cost_more_without_gate_volatility_filter():
    score = np.array([2., 2., -2., -2.])
    labels = np.array([-1, -1, -1, -1])
    high = np.array([True, False, True, False])
    observed = directional_tanh_loss(score, labels, high)
    control = directional_tanh_loss(score, labels, np.zeros(4, dtype=bool))
    assert observed['total'] > control['total']
    assert observed['high_vol_wrong_count'] == 1
    assert observed['settled_count'] == 4
    zeros = directional_tanh_loss([3., -3., 1.], [0, 0, 1], [True, True, True])
    assert zeros['settled_count'] == 1
    with pytest.raises(ValueError):
        directional_tanh_loss([1., 2.], [0, 0], [True, False])


@pytest.mark.parametrize('percentile', (.85, .80, .75))
def test_fast_restrictive_decisions_equal_exact_causal_gate(percentile):
    entries = pd.date_range('2026-01-01T00:00Z', periods=660, freq='5min')
    rows = pd.DataFrame({'outcome': np.tile([1, -1], 330)}, index=entries)
    scores = np.linspace(.05, 4.0, len(rows))
    expected = simulate_admission(rows, scores, (), percentile=percentile, elastic=True)
    actual = causal_reasons(rows, scores, percentile)
    np.testing.assert_array_equal(actual, expected.reason.to_numpy())
    assert np.all(actual[:500] == 'conviction_warmup')
    assert np.any(actual[500:] == 'accepted')


def test_frozen_founder_rejects_incompatible_archive():
    evolution = dict(parent_sha256=GEN354_SHA256, protected_holdout_evaluated=False,
                     winners={'p80': 43}, replicas=[
                         dict(seed=43, percentile=.80, genome=[2., 3.],
                              genome_sha256=genome_hash([2., 3.]),
                              champion={'generation': 375})
                     ])
    out = frozen_gen375(GEN354_SHA256, evolution)
    np.testing.assert_array_equal(out, [2., 3.])
    evolution['replicas'][0]['genome'][0] = 1.
    with pytest.raises(ValueError, match='sellado'):
        frozen_gen375(GEN354_SHA256, evolution)
