import hashlib
import json
import zipfile

import pandas as pd
import pytest

from scripts.prepare_scalping_data import prepare
from src import data_loader
from src.trading_profiles import resolve_profile


def test_profile_defaults_and_timeframe_selection():
    intraday = resolve_profile()
    assert (intraday['FOREX_TIMEFRAME'], intraday['FOREX_EXPIRY_CANDLES']) == ('M15', 1)
    assert intraday['FOREX_MIN_CONFIDENCE'] == .60
    assert intraday['FOREX_ENT_COEF'] == .02
    assert intraday['FOREX_NEUTRAL_OPPORTUNITY_PENALTY'] == .05

    scalping = resolve_profile(timeframe='M1', expiry_candles=3, confidence='.53')
    assert scalping['FOREX_TRADING_PROFILE'] == 'scalping'
    assert scalping['FOREX_EXPIRY_CANDLES'] == 3
    assert scalping['FOREX_MIN_CONFIDENCE'] == .53
    assert scalping['FOREX_REFIT_EVERY'] == 480
    assert intraday['FOREX_REFIT_EVERY'] == 96
    assert scalping['FOREX_ENT_COEF'] > intraday['FOREX_ENT_COEF']
    assert scalping['FOREX_NEUTRAL_OPPORTUNITY_PENALTY'] > intraday['FOREX_NEUTRAL_OPPORTUNITY_PENALTY']
    assert resolve_profile('daily')['FOREX_TIMEFRAME'] == 'H1'
    assert resolve_profile('swing')['FOREX_TREND_WINDOW'] > intraday['FOREX_TREND_WINDOW']
    assert resolve_profile('swing', trading_mode='forex_spot')['FOREX_EXPIRY_CANDLES'] == 1


@pytest.mark.parametrize('overrides', [
    {'timeframe': 'M15'}, {'expiry_candles': 4}, {'expiry_candles': 1.5},
    {'confidence': '.6'},
])
def test_inconsistent_scalping_contract_fails_before_ingestion(overrides):
    with pytest.raises(ValueError):
        resolve_profile('scalping', **overrides)


@pytest.mark.parametrize('timeframe,expected_rows', [('M1', 15), ('M5', 3)])
def test_pinned_histdata_archive_ingested_at_requested_resolution(tmp_path, timeframe, expected_rows):
    name = 'HISTDATA_COM_ASCII_EURUSD_M1_2025.zip'
    archive = tmp_path / name
    timestamps = pd.date_range('2025-01-06 00:00', periods=15, freq='min')
    rows = [
        f'{t:%Y%m%d %H%M%S};1.10;1.12;1.09;{1.10 + i/10000:.5f};1'
        for i, t in enumerate(timestamps)
    ]
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('DAT_ASCII_EURUSD_M1_2025.csv', '\n'.join(rows))
    manifest = tmp_path / 'sources.json'
    manifest.write_text(json.dumps({'sources': [{
        'period': '2025', 'zip_file': name,
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'source_url': 'https://www.histdata.com/',
    }]}))
    out = prepare(tmp_path, tmp_path / timeframe, timeframe, manifest)
    bars = data_loader.load_data(
        inputs=[out], input_timeframe=timeframe, timeframe=timeframe,
        closures=data_loader.read_closures(out.parent/'observed_gaps.json'),
        output=tmp_path/'market.parquet', now='2025-01-07T00:00Z',
    )
    assert len(bars) == expected_rows
    assert bars.timestamp.iloc[0] == pd.Timestamp('2025-01-06T05:00Z')
    assert bars.volume.iloc[0] == 15 // expected_rows
    assert bars.close.iloc[0] == pytest.approx(1.10 + (15//expected_rows - 1)/10000)
    assert bars.attrs['source_provenance']['source_timeframe'] == 'M1'

    archive.write_bytes(b'invalid archive')
    with pytest.raises(ValueError, match='SHA256'):
        prepare(tmp_path, tmp_path/timeframe, timeframe, manifest)


def test_direct_native_m5_bars_are_preserved_and_m15_is_rejected(tmp_path):
    raw = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-06T00:00Z', periods=4, freq='5min'),
        'open': [1.1, 1.2, 1.1, 1.0], 'high': [1.2, 1.3, 1.2, 1.1],
        'low': [1., 1.1, 1., .9], 'close': [1.2, 1.1, 1.0, 1.1], 'volume': [7., 8., 9., 4.],
    })
    source = tmp_path/'EURUSD_M5_UTC.csv'
    raw.to_csv(source, index=False)
    bars = data_loader.load_data(inputs=[source], timeframe='M5', output=tmp_path/'m5.parquet')
    pd.testing.assert_frame_equal(raw, bars, check_freq=False)
    coarse = tmp_path/'EURUSD_M15_UTC.csv'
    raw.iloc[::3].to_csv(coarse, index=False)
    with pytest.raises(data_loader.DataError, match='incompatible'):
        data_loader.load_data(inputs=[coarse], input_timeframe='M5', timeframe='M5',
                              output=tmp_path/'wrong.parquet')


def test_only_an_isolated_high_frequency_garch_refit_can_retain_parameters(monkeypatch):
    import numpy as np
    from src import quant_engine as quant

    config = quant.QuantConfig(window=110, min_observations=100, refit_every=5)
    prices = 1.10 + np.linspace(0, .001, 112)
    attempts = []

    def fit(sample, iterations):
        attempts.append(len(sample))
        if len(attempts) == 2:
            raise quant.GarchConvergenceError('test convergence failure')
        return .1, .1, .8, .5

    monkeypatch.setattr(quant, '_fit_garch', fit)
    with pytest.raises(quant.GarchConvergenceError):
        quant.causal_garch_volatility(prices, config)
    attempts.clear()
    volatility, events = quant.causal_garch_volatility(prices, config, retain_on_nonconvergence=True)
    assert events[1] == {'row': 105, 'retained_from_row': 100}
    assert events[2]['row'] == 110
    assert np.isfinite(volatility[100:]).all()

    def fails_twice(sample, iterations):
        if len(sample) > config.min_observations:
            raise quant.GarchConvergenceError('repeated convergence failure')
        return .1, .1, .8, .5

    monkeypatch.setattr(quant, '_fit_garch', fails_twice)
    with pytest.raises(quant.GarchConvergenceError):
        quant.causal_garch_volatility(prices, config, retain_on_nonconvergence=True)


def test_scalping_microtrend_releases_direction_without_dropping_break_even_gate():
    from dataclasses import replace
    import numpy as np
    from src.decision import RiskConfig, resolve_action
    from src.quant_engine import MonteCarloGBM, QuantConfig, causal_drift

    preset = resolve_profile('scalping', 'M1', confidence='.52')
    assert resolve_profile()['FOREX_DRIFT_SCALE'] == 1.5
    assert resolve_profile()['FOREX_MIN_EXPECTED_EDGE'] == .02
    assert resolve_profile('scalping', 'M1')['FOREX_DRIFT_SCALE'] == 1.8
    risk = RiskConfig(min_confidence=preset['FOREX_MIN_CONFIDENCE'],
                      min_expected_edge=preset['FOREX_MIN_EXPECTED_EDGE'])
    below_break_even = resolve_action(2, .55, .45, 0., payout=.8, tie_policy='refund', risk=risk)
    assert below_break_even['filter_reason'] == 'expected_edge'

    close = 1.1 * np.exp(np.arange(20) * .00008)
    sigma = np.full(len(close), .0002)
    old = QuantConfig()
    new = replace(old, drift_window=preset['FOREX_TREND_WINDOW'],
                  drift_scale=preset['FOREX_DRIFT_SCALE'], drift_noise_z=preset['FOREX_DRIFT_NOISE_Z'],
                  drift_cap_sigma=preset['FOREX_DRIFT_CAP_SIGMA'])
    old_mu = causal_drift(close, sigma, old)['mc_drift'][-1]
    new_mu = causal_drift(close, sigma, new)['mc_drift'][-1]
    mc = MonteCarloGBM()
    old_probability = mc.direction_probabilities(sigma[-1], old_mu)
    new_probability = mc.direction_probabilities(sigma[-1], new_mu)
    assert resolve_action(2, *old_probability, payout=.8, tie_policy='refund', risk=risk)['blocked']
    assert resolve_action(2, *new_probability, payout=.8, tie_policy='refund', risk=risk)['action'] == 2


def test_m5_quality_profile_raises_the_directional_and_economic_gates():
    from src.decision import RiskConfig, resolve_action

    profile = resolve_profile('scalping', 'M5', confidence='.57')
    assert resolve_profile('scalping', 'M5')['FOREX_MIN_CONFIDENCE'] == .58
    assert profile['FOREX_DRIFT_SCALE_FLOOR'] == 1.4
    assert profile['FOREX_MC_Z'] == 2.24
    risk = RiskConfig(min_confidence=profile['FOREX_MIN_CONFIDENCE'],
                      min_expected_edge=profile['FOREX_MIN_EXPECTED_EDGE'], mc_z=profile['FOREX_MC_Z'])
    for probability, expected in ((.5705, 'expected_edge'), (.585, 'accepted')):
        decision = resolve_action(2, probability, 1 - probability, 0.,
                                  payout=.8, tie_policy='refund', risk=risk)
        assert decision['filter_reason'] == expected
