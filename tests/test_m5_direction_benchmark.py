from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.run_m5_direction_benchmark import (
    MC_COLUMNS, MODEL_FEATURES, admission_blocks, fit_models, load_references, summarize,
)
from src.environment import AccountConfig
from src.evolution_lineage import MARKET, OBSERVATION_FEATURES
from src.market import PROTECTED_HOLDOUT_START


def test_frozen_admissions_use_one_candle_and_respect_gaps_and_holdout():
    founder, _, _ = load_references()
    width = len(OBSERVATION_FEATURES)
    founder = replace(founder, genome=np.r_[np.zeros(width), 1., np.zeros(width), .5, .25])
    end, step = PROTECTED_HOLDOUT_START, MARKET.step
    times = pd.date_range(end=end - step, periods=72, freq=step)
    close = 1.1 + np.tile([0., .0002, .0002, .0001], 18)
    raw = pd.DataFrame(dict(timestamp=times, open=close, close=close, high=close+.001, low=close-.001,
                            cond_volatility=.0002, dist_p05=-.002, dist_p95=.002,
                            mc_prob_up=.6, mc_prob_down=.39, mc_prob_tie=.01))
    raw.attrs['timeframe'] = 'M5'
    raw = raw.drop(index=34)
    split = times[52]
    windows = [('early', times[0], split), ('late', split, end)]
    blocks, _ = admission_blocks(raw, founder, windows)
    early, late = (blocks[name].rows for name in ('early', 'late'))
    assert times[34] not in early.index
    assert early.index[-1] + step <= split < late.index[0]
    assert late.index[-1] + step == end
    prices = raw.set_index('timestamp').close
    for rows in (early, late):
        expected = np.sign(prices.loc[rows.index].to_numpy() - prices.loc[rows.index-step].to_numpy())
        np.testing.assert_array_equal(rows.outcome, expected)
        assert list(rows.columns[:len(MODEL_FEATURES)]) == list(MODEL_FEATURES)
    fixed = {'early': early.index[::2] - step, 'late': late.index[1::2] - step}
    subset, _ = admission_blocks(raw, founder, windows, fixed_opens=fixed)
    pd.testing.assert_frame_equal(subset['early'].rows, early.iloc[::2])
    pd.testing.assert_frame_equal(subset['late'].rows, late.iloc[1::2])
    with pytest.raises(ValueError, match='aperturas elegibles'):
        admission_blocks(raw, founder, windows, fixed_opens=fixed | {'early': times[[33]]})
    with pytest.raises(ValueError, match='anteriores al holdout'):
        admission_blocks(raw.assign(timestamp=raw.timestamp + step), founder, windows)


def test_calibrators_require_disjoint_windows_and_predict_three_classes():
    rng = np.random.default_rng(0)
    times = pd.date_range('2025-11-25T21:00:00Z', periods=72, freq='5min')
    rows = pd.DataFrame(rng.normal(size=(len(times), len(MODEL_FEATURES))), index=times, columns=MODEL_FEATURES)
    rows.loc[:, list(MC_COLUMNS)] = rng.dirichlet([2., 1., 2.], size=len(times))
    rows['outcome'] = np.tile([-1, 0, 1], 24)
    fit, calibration = rows.iloc[:36], rows.iloc[36:]
    models = fit_models(fit, calibration)
    for model in models:
        np.testing.assert_array_equal(model.classes_, [-1, 0, 1])
    with pytest.raises(ValueError, match='cronológicos y disjuntos'):
        fit_models(fit, rows.iloc[35:])


def test_paired_payoff_keeps_refunds_and_bootstraps_whole_days():
    starts = pd.date_range('2026-01-01', periods=12, freq='D', tz='UTC')
    times = starts.repeat(3) + pd.to_timedelta(np.tile([0, 5, 10], 12), unit='min')
    rows = pd.DataFrame(dict(outcome=np.tile([1, -1, 0], 12), mc_side=np.tile([-1, -1, 1], 12),
                             gen306_side=np.tile([-1, -1, 1], 12)), index=times)
    probs = np.tile([[.1, .05, .85], [.85, .05, .1], [.1, .8, .1]], (12, 1))
    forecasts = dict(mc=probs, mc_calibrated=probs, logistic_calibrated=probs)
    metrics = summarize(rows, forecasts, AccountConfig())
    assert metrics['admissions'] == 36
    assert metrics['days'] == 12
    logistic = metrics['policies']['logistic_calibrated']
    assert (logistic['wins'], logistic['losses'], logistic['refunds']) == (24, 0, 12)
    assert logistic['delta_vs_mc'] == pytest.approx(.6)
    assert logistic['delta_vs_mc_ci95'] == pytest.approx([.6, .6])
    short = summarize(rows.iloc[:3], {name: p[:3] for name, p in forecasts.items()}, AccountConfig())
    assert short['policies']['logistic_calibrated']['delta_vs_mc_ci95'] is None
