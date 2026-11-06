"""Drift causal sin prometer aciertos: matrices, signos, huecos y compatibilidad."""
from dataclasses import asdict,replace
from time import perf_counter
import numpy as np
import pandas as pd
import pytest
from src.quant_engine import (QuantConfig,MonteCarloGBM,causal_drift,enrich_features,
    project_monte_carlo,DRIFT_AUDIT_FEATURES,PROBABILITY_FEATURES,QuantError)
from src.decision import ensure_probabilities,resolve_action,RiskConfig
from src.telegram_bot import extend_quant_history


def prices(mean,n=40):
    noise=np.tile([-.00001,.00001],n//2+1)[:n]
    return 1.1*np.exp(np.cumsum(mean+noise))


def frame(n=700):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(9).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq='15min',tz='UTC'),
        open=close,high=close+.0002,low=close-.0002,close=close,volume=0.))


@pytest.mark.parametrize('mean,lower,upper',[(.00018,.54,.60),(.00024,.62,.68),(.00029,.70,.76)])
def test_clear_momentum_can_yield_55_to_75_percent_under_model(mean,lower,upper):
    cfg=QuantConfig();sigma=np.full(40,.0002);mc=MonteCarloGBM()
    pos=causal_drift(prices(mean),sigma,cfg)
    neg=causal_drift(prices(-mean),sigma,cfg)
    up,_,_=mc.direction_probabilities(sigma[-1],pos['mc_drift'][-1])
    _,down,_=mc.direction_probabilities(sigma[-1],neg['mc_drift'][-1])
    assert lower<up<upper and lower<down<upper
    if mean==.00029:
        assert resolve_action(2,float(up),float(1-up),0.,payout=.8,tie_policy='refund',risk=RiskConfig())['action']==2


def test_arithmetic_mu_conversion_and_bands_share_same_drift_and_paths():
    cfg=QuantConfig();sigma=np.full(40,.0002);mc=MonteCarloGBM()
    for mean in (.00029,-.00029):
        result=causal_drift(prices(mean),sigma,cfg)
        mu=result['mc_drift'][-1];log_mu=result['mc_log_drift'][-1]
        assert mu==pytest.approx(log_mu+.5*sigma[-1]**2)
        projection=project_monte_carlo(mc,1.1,sigma[-1],mu)
        matrix=mc.paths(1.1,sigma[-1],mu)
        assert matrix.shape==(10000,32)
        np.testing.assert_allclose([projection['mc_p05'],projection['mc_p95']],np.quantile(matrix[:,-1],[.05,.95]),rtol=1e-14)
        assert projection['mc_prob_up']==pytest.approx((matrix[:,0]>1.1).mean())
        assert projection['mc_prob_down']==pytest.approx((matrix[:,0]<1.1).mean())
        if mean>0:
            assert projection['dist_p05']>0 and projection['dist_p95']>0
            assert projection['mc_band_bias']==1 and projection['mc_band_position']<0
        else:
            assert projection['dist_p05']<0 and projection['dist_p95']<0
            assert projection['mc_band_bias']==-1 and projection['mc_band_position']>1


def test_noise_alternation_and_single_outlier_do_not_inject_direction():
    cfg=QuantConfig();sigma=np.full(40,.0002)
    alternating=1.1*np.exp(np.cumsum(np.tile([.0002,-.0002],20)))
    out=causal_drift(alternating,sigma,cfg)
    assert np.abs(out['mc_log_drift']).max()<1e-14
    jump=np.full(40,1.1);jump[20:]*=np.exp(.02)
    out=causal_drift(jump,sigma,cfg)
    assert np.abs(out['mc_log_drift']).max()<1e-14


def test_drift_is_shrunk_and_bounded_without_forcing_minimum_confidence():
    cfg=QuantConfig();sigma=np.full(40,.0002)
    a=causal_drift(prices(.002),sigma,cfg)
    assert np.max(np.abs(a['mc_log_drift']))<=cfg.drift_cap_sigma*.0002+1e-15
    none=causal_drift(prices(.002),sigma,replace(cfg,drift_shrinkage=0.))
    assert (none['mc_log_drift']==0).all()
    weak=causal_drift(prices(.00003),sigma,cfg)
    assert (weak['mc_log_drift']==0).all()


def test_causal_prefix_and_future_mutations_are_bit_identical():
    close=prices(.00024,100);sigma=np.full(100,.0002);cfg=QuantConfig()
    full=causal_drift(close,sigma,cfg)
    prefix=causal_drift(close[:70],sigma[:70],cfg)
    modified=close.copy();modified[70:]*=1.2
    future=causal_drift(modified,sigma,cfg)
    for name in DRIFT_AUDIT_FEATURES:
        np.testing.assert_array_equal(full[name][:70],prefix[name])
        np.testing.assert_array_equal(full[name][:70],future[name][:70])


def test_gap_resets_momentum_and_requires_eight_contiguous_returns():
    cfg=QuantConfig();close=prices(.00029,50);sigma=np.full(50,.0002)
    times=pd.Series(pd.date_range('2024-01-08',periods=50,freq='15min',tz='UTC'))
    times.iloc[20:]+=pd.Timedelta(days=2)
    close[20:]*=1.03
    out=causal_drift(close,sigma,cfg,times)
    assert (out['mc_log_drift'][20:28]==0).all()
    assert out['mc_log_drift'][28]>0


def test_batch_vs_incremental_with_refits_and_time_gap_match():
    cfg=QuantConfig(window=384,min_observations=128,refit_every=96)
    raw=frame(900);raw.loc[705:,'timestamp']+=pd.Timedelta(days=2)
    base=enrich_features(raw.iloc[:700],cfg)
    full=enrich_features(raw,cfg)
    extended=extend_quant_history(base,raw.iloc[699:])
    names=DRIFT_AUDIT_FEATURES+PROBABILITY_FEATURES+['cond_volatility','mc_p05','mc_p95','dist_p05','dist_p95','mc_band_position','mc_band_bias']
    np.testing.assert_allclose(extended[names],full[names],rtol=1e-10,atol=1e-13,equal_nan=True)
    assert (extended.mc_log_drift.iloc[705:713]==0).all()


def test_reconstruction_requires_archived_dynamic_mu_and_matches_existing_probabilities():
    out=enrich_features(frame(),QuantConfig(window=384,min_observations=128))
    reconstructed=ensure_probabilities(out.drop(columns=PROBABILITY_FEATURES))
    pd.testing.assert_frame_equal(reconstructed[PROBABILITY_FEATURES],out[PROBABILITY_FEATURES])
    with pytest.raises(ValueError,match='drift causal archivado'):
        ensure_probabilities(out.drop(columns=PROBABILITY_FEATURES+['mc_drift']))


def test_legacy_metadata_remains_constant_and_is_not_reinterpreted_as_momentum():
    old=dict(window=384,min_observations=128,refit_every=96,n_paths=10000,horizon=32,seed=42,drift_per_bar=0.,max_iterations=500)
    cfg=QuantConfig.from_metadata(old)
    assert cfg.drift_mode=='constant'
    mu=causal_drift(prices(.001),np.full(40,.0002),cfg)['mc_drift']
    assert (mu==0.).all()
    raw=frame(900);base=enrich_features(raw.iloc[:700],cfg)
    base.attrs['quant_config']=old
    extended=extend_quant_history(base,raw.iloc[699:])
    full=enrich_features(raw,cfg)
    np.testing.assert_allclose(extended[PROBABILITY_FEATURES],full[PROBABILITY_FEATURES],rtol=0,atol=1e-14,equal_nan=True)


def test_drift_warmup_and_zero_volatility_are_explicit():
    cfg=QuantConfig();close=prices(.0003);sigma=np.full(40,.0002);sigma[:10]=np.nan
    out=causal_drift(close,sigma,cfg)
    assert np.isnan(out['mc_drift'][:10]).all()
    zero=causal_drift(close,np.zeros(40),cfg)
    assert np.isfinite(zero['mc_drift']).all() and (zero['mc_drift']==0).all()
    p=project_monte_carlo(MonteCarloGBM(),1.1,0.,0.)
    assert p['mc_prob_tie']==1 and p['mc_band_position']==.5


@pytest.mark.parametrize('kwargs',[dict(drift_window=3),dict(drift_window=True),dict(drift_shrinkage=1.1),dict(drift_noise_z=-1),dict(drift_cap_sigma=0),dict(drift_clip_sigma=np.nan),dict(drift_mode='future'),dict(drift_per_bar=.001)])
def test_invalid_momentum_configuration(kwargs):
    with pytest.raises(QuantError):QuantConfig(**kwargs)


def test_vectorized_drift_200k_origins_budget():
    n=200000;close=prices(.00002,n);sigma=np.full(n,.0002)
    start=perf_counter();out=causal_drift(close,sigma,QuantConfig())
    assert len(out['mc_drift'])==n and perf_counter()-start<5.


def test_truncated_baseline_cannot_silently_zero_missing_momentum_history():
    from src.telegram_bot import AlertError
    cfg=QuantConfig(window=384,min_observations=128,drift_window=600)
    raw=frame(800)
    base=enrich_features(raw.iloc[:700],cfg).dropna(subset=['cond_volatility']).reset_index(drop=True)
    with pytest.raises(AlertError,match='momentum causal'):
        extend_quant_history(base,raw.iloc[699:])
