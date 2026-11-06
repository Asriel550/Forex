"""Sensibilidad sin holdout, paridad de caja con Gymnasium y payoff real."""
from dataclasses import asdict,replace
import json
import numpy as np
import pandas as pd
import pytest
from src.quant_engine import QuantConfig,causal_drift,enrich_features,reproject_drift_scale,PROBABILITY_FEATURES
from src.decision import RiskConfig,resolve_action,mc_probability_lower
from src.environment import AccountConfig,prepare_market_data,market_feature_names
from src.backtest import simulate_holdout
from src.sensitivity import evaluate_mc_baseline,run_sensitivity
from src.metrics import decision_metrics,CONFIDENCE_THRESHOLDS


def raw(n=700):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(33).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq='15min',tz='UTC'),
        open=close,high=close+.0002,low=close-.0002,close=close,volume=0.))


def test_defaults_are_amplified_and_legacy_metadata_preserves_one():
    assert QuantConfig().drift_scale==1.5 and RiskConfig().min_confidence==.60
    old=asdict(QuantConfig(drift_scale=1.));old.pop('drift_scale')
    assert QuantConfig.from_metadata(old).drift_scale==1.
    close=1.1*np.exp(np.arange(40)*.00024);sigma=np.full(40,.0002)
    base=causal_drift(close,sigma,QuantConfig(drift_scale=1.))['mc_log_drift']
    amplified=causal_drift(close,sigma,QuantConfig())['mc_log_drift']
    np.testing.assert_allclose(amplified,base*1.5)
    assert amplified.max()<=QuantConfig().drift_cap_sigma*sigma[-1]


@pytest.mark.parametrize('scale',[0,-1,3.1,np.nan,np.inf])
def test_scaling_is_bounded(scale):
    with pytest.raises(ValueError):QuantConfig(drift_scale=scale)


def test_reproject_scale_matches_full_causal_recomputation():
    source=raw();cfg=QuantConfig(window=384,min_observations=128,drift_scale=1.)
    a=enrich_features(source,cfg)
    b=reproject_drift_scale(a,1.5)
    expected=enrich_features(source,replace(cfg,drift_scale=1.5))
    cols=PROBABILITY_FEATURES+['mc_drift','mc_log_drift','mc_p05','mc_p95','dist_p05','dist_p95']
    pd.testing.assert_frame_equal(b[cols],expected[cols],check_exact=True)
    assert a.attrs['quant_config']['drift_scale']==1.


@pytest.mark.parametrize('threshold',[.58,.60,.62,.65])
def test_operating_thresholds_preserve_payout_edge(threshold):
    risk=RiskConfig(min_confidence=threshold)
    p=threshold+.001
    assert resolve_action(2,p,1-p,0.,payout=.8,tie_policy='refund',risk=risk)['action']==2
    # El mismo porcentaje no cubre necesariamente un payout menor.
    decision=resolve_action(2,.59,.41,0.,payout=.5,tie_policy='refund',risk=RiskConfig(min_confidence=.58))
    assert decision['action']==1 and decision['filter_reason']=='expected_edge'
    assert mc_probability_lower(p)<p


@pytest.mark.parametrize('tie_policy',['refund','loss'])
@pytest.mark.parametrize('payout',[.8,.73])
def test_vector_baseline_matches_actual_environment_cash_and_stop(tie_policy,payout):
    data=raw(300);data['cond_volatility']=.0002;data['dist_p05']=-.002;data['dist_p95']=.002
    prepared=prepare_market_data(data,1,1).iloc[:180].copy().reset_index(drop=True)
    prepared.loc[50:,'timestamp']+=pd.Timedelta(days=2)
    prepared.loc[10,'close']=prepared.loc[9,'close']
    probability=np.resize([.61,.7,.9,.4,.2,.51],len(prepared))
    prepared['mc_prob_up']=probability;prepared['mc_prob_down']=1-probability;prepared['mc_prob_tie']=0.
    risk=RiskConfig();account=AccountConfig(initial_balance=100,stake=10,payout=payout,tie_policy=tie_policy)
    names=market_feature_names(1);up=names.index('mc_prob_up');down=names.index('mc_prob_down')
    class Direction:
        def predict(self,obs,deterministic=True):return (2 if obs[up]>obs[down] else 0 if obs[down]>obs[up] else 1),None
    trades,equity,signals,meta=simulate_holdout(Direction(),prepared,account=account,lookback=1,normalizer=None,risk=risk)
    result=evaluate_mc_baseline(prepared,risk=risk,account=account,n_paths=10000)
    assert result['total_trades']==len(trades)
    assert result['net_profit']==pytest.approx(trades.profit.sum())
    assert result['final_balance']==pytest.approx(equity.balance.iloc[-1])
    assert result['risk_stop']==meta['risk_stop']
    assert result['put']==int(trades.action.eq(0).sum()) and result['call']==int(trades.action.eq(2).sum())


def test_sensitivity_never_uses_holdout_outcomes_and_exports_three_formats(tmp_path):
    cfg=QuantConfig(window=384,min_observations=128)
    frame=enrich_features(raw(),cfg).dropna(subset=['cond_volatility']).reset_index(drop=True)
    path=tmp_path/'features.parquet';frame.to_parquet(path,index=False)
    first=run_sensitivity(path,report_dir=tmp_path/'a',scales=(1.,1.5),thresholds=(.58,.65))
    modified=frame.copy();i=int(len(frame)*.90)
    modified.loc[i:,['open','high','low','close']]*=1.2
    modified.to_parquet(path,index=False)
    second=run_sensitivity(path,report_dir=tmp_path/'b',scales=(1.,1.5),thresholds=(.58,.65))
    assert first['results']==second['results']
    assert {r['phase'] for r in first['results']}=={'train','validation'}
    assert first['automatic_selection'] is False
    for ext in ('json','csv','md'):assert (tmp_path/'a'/f'sensitivity.{ext}').exists()
    json.dumps(first,allow_nan=False)
    with pytest.raises(ValueError):run_sensitivity(path,report_dir=tmp_path/'bad',thresholds=(.55,))


def test_reports_cover_new_thresholds_and_correct_break_even():
    signals=pd.DataFrame(dict(action=[2,0,2],requested_action=[2,0,2],mc_prob_up=[.59,.39,.66],
        mc_prob_down=[.41,.61,.34],mc_confidence=[.59,.61,.66],profit=[8.,-10.,8.],blocked=False))
    result=decision_metrics(signals,payout=.8)
    assert tuple(row['threshold'] for row in result['confidence_segments'])==CONFIDENCE_THRESHOLDS
    first=result['confidence_segments'][0]
    assert first['trades']==3 and first['profit_factor']==1.6
    assert first['empirical_win_rate_excluding_refunds']==2/3
    assert first['break_even_win_rate']==pytest.approx(1/1.8)
    assert first['empirical_edge_per_resolved_stake']==pytest.approx(.2)
