"""Métricas conocidas, aislamiento del holdout y un checkpoint PPO real mínimo."""
import json
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from src import backtest as bt
from src.decision import RiskConfig
from src.environment import AccountConfig,fit_normalizer,prepare_market_data
from src.metrics import calculate_metrics,export_reports
from src.train import train


def raw_frame(n=300):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(71).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2020-01-06',periods=n,freq='15min',tz='UTC'),
                             open=close,high=close+.0002,low=close-.0002,close=close,
                             cond_volatility=.0002,dist_p95=.002,dist_p05=-.002))


def ready_prices(prices,times=None):
    ready=prepare_market_data(raw_frame(),2,2).iloc[:len(prices)].copy();ready['close']=prices
    if times is not None:ready['timestamp']=pd.to_datetime(times,utc=True)
    return ready


class Policy:
    def __init__(self,actions):self.actions=iter(actions);self.observations=[]
    def predict(self,obs,deterministic=True):
        assert deterministic
        self.observations.append(obs.copy());return np.array(next(self.actions)),None


def simulate(frame,actions,account=None):
    policy=Policy(actions)
    result=bt.simulate_holdout(policy,frame,account=account or AccountConfig(),lookback=2,normalizer=fit_normalizer(frame,2),risk=RiskConfig(enabled=False))
    return policy,result


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp('backtest');path=root/'features.parquet';raw_frame().to_parquet(path,index=False)
    manifest=train(path,output_dir=root/'models',total_timesteps=16,n_steps=16,batch_size=8,
                   n_epochs=1,eval_freq=16,lookback=2,indicator_period=2,episode_length=32,verbose=0)
    return path,root/'models',manifest


def test_scripted_liquidations_and_real_win_rate():
    _,(trades,equity,signals,meta)=simulate(ready_prices([1.1,1.11,1.10,1.10,1.12]),[2,0,2,1])
    assert trades.profit.tolist()==[8.,8.,0.]
    assert equity.balance.tolist()==[1000.,1008.,1016.,1016.,1016.]
    assert meta['signal_distribution']==dict(put=1,neutral=1,call=2)
    assert (pd.to_datetime(trades.expiry_time)-pd.to_datetime(trades.entry_time)).eq(pd.Timedelta(minutes=15)).all()
    metrics=calculate_metrics(trades,equity,initial_balance=1000)
    assert metrics['win_rate']==2/3 and metrics['total_trades']==3 and metrics['ties']==1


def test_neutral_policy_and_strict_json(tmp_path):
    _,(trades,equity,signals,_)=simulate(ready_prices([1.1,1.2,1.1]),[1,1],AccountConfig(inactivity_penalty=.01))
    metrics=calculate_metrics(trades,equity,initial_balance=1000)
    assert metrics['net_profit']==0 and metrics['win_rate'] is None and metrics['sharpe_ratio'] is None
    export_reports(tmp_path,metrics,trades,equity,signals)
    assert list(pd.read_csv(tmp_path/'trades.csv'))==bt.TRADE_COLUMNS
    text=(tmp_path/'metrics.json').read_text();assert 'NaN' not in text and 'Infinity' not in text


def test_cash_carries_across_weekend():
    frame=ready_prices([1.1,1.2,1.3,1.4],['2024-01-05T21:30Z','2024-01-05T21:45Z','2024-01-07T22:00Z','2024-01-07T22:15Z'])
    _,(trades,equity,_,meta)=simulate(frame,[2,2])
    assert equity.balance.tolist()==[1000.,1008.,1008.,1016.] and meta['skipped_gap_transitions']==1
    assert trades.balance_before.tolist()==[1000.,1008.]


def test_risk_stop_preserves_entire_holdout_curve():
    policy,(trades,equity,_,meta)=simulate(ready_prices([1.1,1.2,1.3,1.4,1.5]),[0,0],AccountConfig(initial_balance=20,stake=10))
    assert len(policy.observations)==2 and meta['risk_stop']
    assert equity.balance.tolist()==[20.,10.,0.,0.,0.]
    metrics=calculate_metrics(trades,equity,initial_balance=20)
    assert metrics['max_drawdown_pct']==100 and metrics['return_observations']==4


def test_financial_metrics_match_known_values():
    trades=pd.DataFrame(dict(action=[2,0,2],profit=[8.,-10.,8.]))
    equity=pd.DataFrame(dict(timestamp=pd.date_range('2020-01-01',periods=5,freq='15min',tz='UTC'),balance=[100.,108.,98.,106.,106.]))
    result=calculate_metrics(trades,equity,initial_balance=100)
    returns=np.array([.08,-10/108,8/98,0.])
    assert result['net_profit']==6 and result['profit_factor']==1.6 and result['max_drawdown_usd']==10
    assert result['max_drawdown_pct']==pytest.approx(100*10/108)
    assert result['sharpe_ratio']==pytest.approx(returns.mean()/returns.std(ddof=1)*np.sqrt(252*96))


def test_profit_factor_without_losses_is_explicit_null():
    trades=pd.DataFrame(dict(action=[2],profit=[8.]))
    equity=pd.DataFrame(dict(timestamp=pd.date_range('2020-01-01',periods=2,freq='15min',tz='UTC'),balance=[100.,108.]))
    result=calculate_metrics(trades,equity,initial_balance=100)
    assert result['profit_factor'] is None and result['profit_factor_status']=='infinite_no_losses'
    json.dumps(result,allow_nan=False)


def test_invalid_pnl_and_neutral_trade_rejected():
    eq=pd.DataFrame(dict(timestamp=pd.date_range('2020-01-01',periods=2,freq='15min',tz='UTC'),balance=[100.,108.]))
    with pytest.raises(ValueError):calculate_metrics(pd.DataFrame(dict(action=[2],profit=[9.])),eq,initial_balance=100)
    with pytest.raises(ValueError):calculate_metrics(pd.DataFrame(dict(action=[1],profit=[8.])),eq,initial_balance=100)


def test_holdout_reconstructs_only_reserved_dates(bundle):
    path,models,manifest=bundle
    frame,_,_=bt.load_holdout(path,models/'best_ppo_agent.zip',models/'normalizer.json',models/'training_manifest.json')
    assert len(frame)==manifest['splits']['test']['rows']
    assert frame.timestamp.iloc[0].isoformat()==manifest['splits']['test']['start']
    assert frame.timestamp.min()>pd.Timestamp(manifest['splits']['validation']['end'])


@pytest.mark.parametrize('kind',['input','model','normalizer'])
def test_artifact_changes_are_rejected(bundle,tmp_path,kind):
    path,models,_=bundle;paths=dict(input=path,model=models/'best_ppo_agent.zip',normalizer=models/'normalizer.json')
    wrong=tmp_path/'wrong';wrong.write_bytes(paths[kind].read_bytes()+b'changed');paths[kind]=wrong
    with pytest.raises(bt.BacktestError,match='SHA256'):
        bt.load_holdout(paths['input'],paths['model'],paths['normalizer'],models/'training_manifest.json')


def test_legacy_or_changed_manifest_is_rejected(bundle,tmp_path):
    path,models,manifest=bundle
    for flag in ('legacy','dates','incomplete','used','not_dict'):
        value=json.loads(json.dumps(manifest))
        if flag=='legacy':value['manifest_version']=1
        if flag=='dates':value['splits']['test']['start']=value['splits']['train']['start']
        if flag=='incomplete':value.pop('features')
        if flag=='used':value['holdout_test_evaluated']=True
        if flag=='not_dict':value=[]
        file=tmp_path/'manifest.json';file.write_text(json.dumps(value))
        with pytest.raises(bt.BacktestError):bt.load_holdout(path,models/'best_ppo_agent.zip',models/'normalizer.json',file)


def test_policy_only_receives_holdout(bundle,tmp_path,monkeypatch):
    path,models,manifest=bundle;n=manifest['splits']['test']['rows'];policy=Policy([1]*(n-1))
    monkeypatch.setattr(bt,'load_agent',lambda path:policy)
    result=bt.run_backtest(path,model_path=models/'best_ppo_agent.zip',report_dir=tmp_path,plots=False)
    assert len(policy.observations)==n-1 and result['net_profit']==0
    signals=pd.read_csv(tmp_path/'signals.csv')
    assert pd.to_datetime(signals.entry_time,utc=True).min()==pd.Timestamp(manifest['splits']['test']['start'])+pd.Timedelta(minutes=15)


def test_changed_future_cannot_change_earlier_observations():
    a=ready_prices([1.1,1.2,1.1,1.2,1.3]);b=a.copy();b.loc[4,'close']=1.0
    pa,pb=Policy([2]*4),Policy([2]*4);kwargs=dict(account=AccountConfig(),lookback=2,normalizer=fit_normalizer(a.iloc[:2],2),risk=RiskConfig(enabled=False))
    ra=bt.simulate_holdout(pa,a,**kwargs);rb=bt.simulate_holdout(pb,b,**kwargs)
    for left,right in zip(pa.observations,pb.observations):np.testing.assert_array_equal(left,right)
    assert ra[0].profit.iloc[-1]==8 and rb[0].profit.iloc[-1]==-10
    pd.testing.assert_frame_equal(ra[0].iloc[:-1],rb[0].iloc[:-1])


def test_actual_checkpoint_is_repeatable_and_exports_png(bundle,tmp_path):
    path,models,_=bundle
    a=bt.run_backtest(path,model_path=models/'best_ppo_agent.zip',report_dir=tmp_path/'a')
    b=bt.run_backtest(path,model_path=models/'best_ppo_agent.zip',report_dir=tmp_path/'b',plots=False)
    assert a==b
    for name in ('metrics.json','trades.csv','equity.csv','signals.csv'):assert (tmp_path/'a'/name).exists()
    for name in ('signals.png','equity_drawdown.png'):assert (tmp_path/'a'/name).read_bytes().startswith(b'\x89PNG')

