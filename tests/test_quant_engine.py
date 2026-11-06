"""Corrección, causalidad, reproducibilidad y presupuestos amplios de rendimiento.

Los límites temporales detectan regresiones graves; no certifican latencias
universales en milisegundos. Las pruebas no descargan datos ni usan GPU.
"""
from time import perf_counter

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from src import quant_engine as qe


def market(n=700):
    rng = np.random.default_rng(731)
    variance = 4e-8
    returns = np.zeros(n)
    for t in range(1, n):
        variance = 1e-9 + 0.08 * returns[t-1]**2 + 0.90 * variance
        returns[t] = np.sqrt(variance) * rng.standard_normal()
    close = 1.1 * np.exp(np.cumsum(returns))
    return pd.DataFrame(dict(timestamp=pd.date_range('2020-01-06', periods=n,
                                                    freq='15min', tz='UTC'),
                             open=close, high=close*1.0001, low=close*0.9999,
                             close=close, volume=0.0))


@pytest.fixture
def config():
    return qe.QuantConfig(window=384, min_observations=128, refit_every=96,drift_mode="constant")


def test_full_matrix_shape_determinism_and_positive_prices():
    first = qe.MonteCarloGBM(seed=17)
    second = qe.MonteCarloGBM(seed=17)
    a, b = first.paths(1.1, 0.0005), second.paths(1.1, 0.0005)
    assert a.shape == (10_000, 32)
    assert a.dtype == np.float64
    assert np.isfinite(a).all() and (a > 0).all()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, qe.MonteCarloGBM(seed=18).paths(1.1, 0.0005))


def test_empirical_percentiles_equal_full_matrix_linear_interpolation():
    mc = qe.MonteCarloGBM(n_paths=10_000, horizon=32, seed=4)
    spots = np.array([1.1, 1.2, 1.3, 1.4])
    sigma = np.array([0., 0.0001, 0.0005, 0.01])
    drift = np.array([0., 0.00001, -0.0001, 0.0003])
    p05, p95 = mc.terminal_quantiles(spots, sigma, drift)
    for i in range(len(spots)):
        expected = np.quantile(mc.paths(spots[i], sigma[i], drift[i])[:, -1], [0.05, 0.95])
        np.testing.assert_allclose([p05[i], p95[i]], expected, rtol=1e-14)
    assert (p05 <= p95).all()
    assert p05[0] == p95[0] == spots[0]


def test_gbm_terminal_distribution_and_global_rng_untouched():
    np.random.seed(91)
    before = np.random.get_state()
    mc = qe.MonteCarloGBM(seed=42)
    terminal = np.log(mc.paths(1.1, 0.001)[:, -1] / 1.1)
    assert abs(terminal.mean() - (-0.5 * .001**2 * 32)) < .0002
    assert np.isclose(terminal.var(), .001**2 * 32, rtol=.06)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_garch_matches_arch_one_step_forecast(config):
    frame = market(129)
    sigma, fits = qe.causal_garch_volatility(frame.close, config)
    returns = np.diff(np.log(frame.close.to_numpy())) * qe.SCALE
    rms = np.sqrt(np.mean(returns**2))
    model = arch_model(returns/rms, mean='Zero', vol='GARCH', p=1, q=1, rescale=False)
    reference = model.fit(disp='off', update_freq=0,
                          options={'maxiter':500, 'ftol':1e-8})
    expected = np.sqrt(reference.forecast(horizon=1).variance.iloc[-1, 0]) * rms / qe.SCALE
    np.testing.assert_allclose(sigma[-1], expected, rtol=1e-12)
    assert np.isnan(sigma[:128]).all() and len(fits) == 1


def test_recursion_between_refits_uses_current_return(config):
    frame = market(132)
    sigma, fits = qe.causal_garch_volatility(frame.close, config)
    fit = fits[0]
    r129 = np.log(frame.close.iloc[129] / frame.close.iloc[128]) * qe.SCALE
    variance = fit['omega'] + fit['alpha'] * r129**2 + fit['beta'] * (sigma[128]*qe.SCALE)**2
    np.testing.assert_allclose(sigma[129], np.sqrt(variance)/qe.SCALE, rtol=1e-10)


def test_garch_and_features_are_causal_and_deterministic(config):
    frame = market()
    full = qe.enrich_features(frame, config)
    prefix = qe.enrich_features(frame.iloc[:400], config)
    modified = frame.copy()
    modified.loc[400:, 'close'] *= np.exp(np.linspace(0.001, 0.004, len(frame)-400))
    changed = qe.enrich_features(modified, config)
    cols = qe.FEATURES + ['mc_p05', 'mc_p95']
    pd.testing.assert_frame_equal(full.loc[:399, cols], prefix[cols], check_exact=True)
    pd.testing.assert_frame_equal(full.loc[:399, cols], changed.loc[:399, cols], check_exact=True)
    repeat = qe.enrich_features(frame, config)
    pd.testing.assert_frame_equal(full, repeat, check_exact=True)
    assert len(full.attrs['garch_fits']) == 1 + (len(frame)-1-128)//96


def test_output_features_units_warmup_and_preserved_input(config):
    frame = market()
    original = frame.copy(deep=True)
    out = qe.enrich_features(frame, config)
    pd.testing.assert_frame_equal(frame, original)
    assert out.index.equals(frame.index) and len(out) == len(frame)
    assert out[qe.FEATURES].iloc[:128].isna().all().all()
    valid = out.iloc[128:]
    assert np.isfinite(valid[qe.FEATURES]).all().all()
    assert (valid.cond_volatility > 0).all()
    assert valid.cond_volatility.median() < 0.01  # decimal por barra, no puntos básicos
    assert (valid.dist_p95 > 0).all() and (valid.dist_p05 < 0).all()
    np.testing.assert_allclose(valid.dist_p95, (valid.mc_p95-valid.close)/valid.close)
    np.testing.assert_allclose(valid.dist_p05, (valid.mc_p05-valid.close)/valid.close)


@pytest.mark.parametrize('bad', ['zero', 'nan', 'inf', 'short', 'constant', 'reverse', 'duplicate'])
def test_invalid_input_fails_explicitly(config, bad):
    frame = market()
    if bad == 'zero': frame.loc[1, 'close'] = 0
    if bad == 'nan': frame.loc[1, 'close'] = np.nan
    if bad == 'inf': frame.loc[1, 'close'] = np.inf
    if bad == 'short': frame = frame.iloc[:100]
    if bad == 'constant': frame['close'] = 1.1
    if bad == 'reverse': frame = frame.iloc[::-1]
    if bad == 'duplicate': frame.loc[1, 'timestamp'] = frame.timestamp.iloc[0]
    with pytest.raises((qe.QuantError, ValueError)):
        qe.enrich_features(frame, config)


@pytest.mark.parametrize('kwargs', [dict(window=99), dict(refit_every=0), dict(n_paths=0),
                                    dict(seed=-1), dict(horizon=0), dict(drift_per_bar=np.nan)])
def test_invalid_configuration(kwargs):
    with pytest.raises(qe.QuantError): qe.QuantConfig(**kwargs)


def test_nonconvergence_is_not_silently_replaced(config, monkeypatch):
    from arch.utility.exceptions import ConvergenceWarning
    import warnings
    class Failed:
        def fit(self, **kwargs): warnings.warn('failed', ConvergenceWarning)
    monkeypatch.setattr(qe, 'arch_model', lambda *args, **kwargs: Failed())
    with pytest.raises(qe.QuantError, match='convergió'):
        qe.enrich_features(market(), config)


def test_vectorized_monte_carlo_performance_and_no_per_bar_tensor(record_property, monkeypatch):
    start = perf_counter()
    mc = qe.MonteCarloGBM()
    paths = mc.paths(1.1, .0003)
    elapsed_matrix = perf_counter() - start
    assert paths.shape == (10_000, 32)
    monkeypatch.setattr(mc, 'paths', lambda *a, **k: pytest.fail('No materializar por vela'))
    start = perf_counter()
    low, high = mc.terminal_quantiles(np.full(200_000, 1.1), np.full(200_000, .0003))
    elapsed_batch = perf_counter() - start
    assert len(low) == len(high) == 200_000
    record_property('matrix_seconds', elapsed_matrix)
    record_property('200k_origins_seconds', elapsed_batch)
    assert elapsed_matrix < 5.0 and elapsed_batch < 5.0


def test_garch_performance_budget_and_refit_count(config, record_property):
    start = perf_counter()
    sigma, fits = qe.causal_garch_volatility(market(1024).close, config)
    elapsed = perf_counter() - start
    record_property('1024_bars_seconds', elapsed)
    assert len(fits) == 1 + (1024-1-128)//96
    assert np.isfinite(sigma[128:]).all()
    assert elapsed < 30.0


def test_garch_retries_deterministic_starting_values(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    failed = SimpleNamespace(convergence_flag=4)
    success = SimpleNamespace(convergence_flag=0,
                              params={'omega':0.05, 'alpha[1]':0.1, 'beta[1]':0.85},
                              conditional_volatility=np.array([1.0]))
    model = SimpleNamespace(fit=Mock(side_effect=[failed, success]))
    monkeypatch.setattr(qe, 'arch_model', lambda *a, **k: model)
    sample = np.random.default_rng(42).normal(size=200)
    omega, alpha, beta, h = qe._fit_garch(sample, 500)
    assert model.fit.call_count == 2
    assert model.fit.call_args.kwargs['starting_values'] is not None
    assert omega > 0 and h > 0 and alpha + beta < 1

