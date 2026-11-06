"""Filtro causal, probabilidades a una vela, reward separado de caja y auditoría."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from src.decision import RiskConfig, resolve_action, ensure_probabilities
from src.quant_engine import MonteCarloGBM, QuantConfig, enrich_features, PROBABILITY_FEATURES
from src.environment import BinaryOptionsEnv, AccountConfig, fit_normalizer, prepare_market_data
from src.backtest import simulate_holdout
from src.metrics import decision_metrics,calculate_metrics,export_reports
from src.train import chronological_split,train


def raw(n=300):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(312).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq='15min',tz='UTC'),
        open=close,high=close+.0002,low=close-.0002,close=close,volume=0.,
        cond_volatility=.0002,dist_p95=.002,dist_p05=-.002))


def env(prices,up=.9,risk=None,account=None):
    frame=raw(max(30,len(prices)+5))
    prepared=prepare_market_data(frame,lookback=1,indicator_period=1).iloc[:len(prices)].copy()
    prepared['close']=prices
    prepared['mc_prob_up']=up;prepared['mc_prob_down']=1-up;prepared['mc_prob_tie']=0.
    return BinaryOptionsEnv(prepared,prepared=True,lookback=1,risk=risk,account=account)


@pytest.mark.parametrize('sigma,drift',[(.0002,0.),(.0003,.0001),(.01,-.001),(0.,0.),(0.,.001),(0.,-.001)])
def test_mc_probability_matches_same_10000_paths_at_one_bar(sigma,drift):
    mc=MonteCarloGBM();paths=mc.paths(1.1,sigma,drift)
    up,down,tie=mc.direction_probabilities(sigma,drift)
    np.testing.assert_allclose([up,down,tie],[(paths[:,0]>1.1).mean(),(paths[:,0]<1.1).mean(),(paths[:,0]==1.1).mean()],atol=1e-14)
    assert up+down+tie==pytest.approx(1.)


def test_mc_zero_drift_has_no_70_percent_directional_edge():
    mc=MonteCarloGBM();up,down,tie=mc.direction_probabilities(np.array([1e-5,.0002,.001]))
    assert np.max(np.maximum(up,down))<.55
    assert not np.any(tie>.001)


def test_causal_probabilities_ignore_future_close_and_use_expiry_not_32_bars():
    frame=raw(350);config=QuantConfig(window=256,min_observations=128,refit_every=96)
    full=enrich_features(frame,config);prefix=enrich_features(frame.iloc[:250],config)
    changed=frame.copy();changed.loc[250:,'close']*=1.01
    other=enrich_features(changed,config)
    pd.testing.assert_frame_equal(full[PROBABILITY_FEATURES].iloc[:250],prefix[PROBABILITY_FEATURES],check_exact=True)
    pd.testing.assert_frame_equal(full[PROBABILITY_FEATURES].iloc[:250],other[PROBABILITY_FEATURES].iloc[:250],check_exact=True)
    assert full.attrs['direction_probability_horizon']==1


def test_strict_threshold_direction_and_wilson_guard():
    kwargs=dict(payout=.8,tie_policy='refund',risk=RiskConfig(min_confidence=.70))
    assert resolve_action(2,.70,.30,0.,**kwargs)['action']==1
    assert resolve_action(2,.71,.29,0.,**kwargs)['action']==2
    assert resolve_action(0,.71,.29,0.,**kwargs)['action']==1
    assert resolve_action(0,.15,.85,0.,**kwargs)['action']==0
    assert resolve_action(2,.71,.29,0.,n_paths=10,**kwargs)['filter_reason']=='expected_edge'
    assert resolve_action(1,.9,.1,0.,**kwargs)['action']==1


def test_neutral_is_free_and_blocked_action_never_changes_cash_even_if_future_wins():
    risk=RiskConfig()
    blocked=env([1.1,1.2],up=.51,risk=risk);blocked.reset()
    _,reward,_,_,info=blocked.step(2)
    assert reward==-risk.blocked_penalty and info['action']==1 and info['requested_action']==2
    assert info['profit']==0 and info['balance']==1000 and info['trades']==0 and info['blocked']
    neutral=env([1.1,1.2],up=.51,account=AccountConfig(inactivity_penalty=1.));neutral.reset()
    assert neutral.step(1)[1]==0.


def test_reward_shaping_preserves_actual_payout_and_penalizes_overactivity():
    risk=RiskConfig(target_activity=.01,activity_decay=.5,overtrade_penalty=2.)
    e=env([1.1,1.2,1.3],risk=risk);e.reset()
    _,first,_,_,a=e.step(2);_,second,_,_,b=e.step(2)
    assert a['profit']==b['profit']==8. and b['balance']==1016.
    assert second<first and b['trade_activity']>a['trade_activity']
    assert first!=.8


def test_drawdown_has_stronger_reward_cost_and_terminates_at_risk_limit():
    e=env([1.1,1.0,1.1],risk=RiskConfig(drawdown_limit=.05),account=AccountConfig(initial_balance=100,stake=10))
    e.reset();_,reward,terminated,truncated,info=e.step(2)
    assert terminated and not truncated and info['end_reason']=='max_drawdown'
    assert reward<-5 and info['profit']==-10 and info['balance']==90


def test_next_close_cannot_change_current_gate_or_observation():
    a=env([1.1,1.2],up=.9);b=env([1.1,1.0],up=.9)
    np.testing.assert_array_equal(a.reset()[0],b.reset()[0])
    ia,ib=a.step(2)[4],b.step(2)[4]
    assert ia['mc_confidence']==ib['mc_confidence'] and ia['action']==ib['action']==2
    assert ia['profit']==8 and ib['profit']==-10


def test_audit_separates_executed_requested_and_neutral_confidence(tmp_path):
    signals=pd.DataFrame(dict(action=[2,0,1,1],requested_action=[2,0,1,0],
        mc_prob_up=[.9,.2,.51,.9],mc_prob_down=[.1,.8,.49,.1],mc_confidence=[.9,.8,.51,.1],
        profit=[8.,-10.,0.,0.],blocked=[False,False,False,True]))
    result=decision_metrics(signals,thresholds=(.70,.80,.95))
    assert result['decisions']['executed']['NEUTRO']['percentage']==50
    assert result['decisions']['requested']['PUT']['count']==2
    assert result['decisions']['executed']['NEUTRO']['mean_mc_confidence']==pytest.approx(.705)
    assert result['confidence_segments'][0]['profit_factor']==.8
    assert result['confidence_segments'][1]['trades']==1 # .80 exacto queda fuera
    assert result['confidence_segments'][2]['profit_factor'] is None
    assert result['confidence_segments'][2]['profit_factor_status']=='no_trades'
    json.dumps(result,allow_nan=False)


def test_backtest_executes_filtered_action_and_keeps_rejected_requests():
    prepared=prepare_market_data(raw(),1,1).iloc[:4].copy()
    prepared['mc_prob_up']=[.51,.9,.1,.5];prepared['mc_prob_down']=1-prepared.mc_prob_up;prepared['mc_prob_tie']=0.
    class Policy:
        def __init__(self):self.actions=iter([2,2,0])
        def predict(self,obs,deterministic=True):return next(self.actions),None
    trades,equity,signals,meta=simulate_holdout(Policy(),prepared,account=AccountConfig(),lookback=1,normalizer=fit_normalizer(prepared,1))
    assert signals.action.tolist()==[1,2,0] and signals.requested_action.tolist()==[2,2,0]
    assert len(trades)==2 and meta['signal_distribution']==dict(put=1,neutral=1,call=1)
    assert trades.mc_confidence.tolist()==[.9,.9]


def test_train_freezes_risk_contract_and_never_uses_holdout_for_normalization(tmp_path):
    path=tmp_path/'source.parquet';frame=raw();frame.to_parquet(path,index=False)
    m=train(path,output_dir=tmp_path/'models',total_timesteps=16,n_steps=16,batch_size=8,n_epochs=1,eval_freq=16,lookback=1,indicator_period=1,verbose=0)
    assert m['manifest_version']==3 and m['risk']['min_confidence']==.60
    assert m['holdout_test_evaluated'] is False and m['reward_version']=='risk_adjusted_v1'
    normal=json.loads((tmp_path/'models/normalizer.json').read_text())
    expected=fit_normalizer(chronological_split(prepare_market_data(frame,1,1))['train'],1)
    assert normal==expected
    from src.backtest import run_backtest
    result=run_backtest(path,model_path=tmp_path/'models/best_ppo_agent.zip',report_dir=tmp_path/'reports',plots=False)
    assert result['total_trades']==0 and result['profit_factor'] is None
    assert result['decision_analysis']['decisions']['executed']['NEUTRO']['percentage']==100
    assert (tmp_path/'reports/summary.md').exists()


@pytest.mark.parametrize('kwargs',[dict(min_confidence=.4),dict(drawdown_limit=0),dict(activity_decay=1),dict(trade_penalty=-1),dict(mc_z=np.nan)])
def test_invalid_risk_configuration(kwargs):
    with pytest.raises(ValueError):RiskConfig(**kwargs)


def test_32_bar_directional_probability_cannot_silently_gate_one_bar_expiry():
    frame=ensure_probabilities(raw())
    frame.attrs['direction_probability_horizon']=32
    with pytest.raises(ValueError,match='UNA vela'):ensure_probabilities(frame)


def test_partial_or_invalid_probability_columns_are_rejected():
    frame=raw();frame['mc_prob_up']=.9
    with pytest.raises(ValueError):ensure_probabilities(frame)
    frame=ensure_probabilities(raw());frame.loc[10,'mc_prob_up']=.99
    with pytest.raises(ValueError):ensure_probabilities(frame)


def test_risk_hyperparameters_are_restored_from_manifest_and_version_two_is_rejected(tmp_path):
    path=tmp_path/'source.parquet';raw().to_parquet(path,index=False)
    m=train(path,output_dir=tmp_path/'models',total_timesteps=16,n_steps=16,batch_size=8,n_epochs=1,eval_freq=16,lookback=1,indicator_period=1,risk=RiskConfig(min_confidence=.80,trade_penalty=.2),verbose=0)
    from src.backtest import run_backtest,load_holdout,BacktestError
    result=run_backtest(path,model_path=tmp_path/'models/best_ppo_agent.zip',report_dir=tmp_path/'reports',plots=False)
    assert result['risk_config']['min_confidence']==.80 and result['risk_config']['trade_penalty']==.2
    m['manifest_version']=2;mf=tmp_path/'old.json';mf.write_text(json.dumps(m))
    with pytest.raises(BacktestError):
        load_holdout(path,tmp_path/'models/best_ppo_agent.zip',tmp_path/'models/normalizer.json',mf)
