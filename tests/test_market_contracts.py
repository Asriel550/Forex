"""Contratos multi-timeframe, expiración causal y PnL spot independientes."""
from dataclasses import asdict,replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from stable_baselines3.common.env_checker import check_env
from src.market import MarketConfig
from src.resampling import aggregate
from src import data_loader as dl
from src.quant_engine import QuantConfig,enrich_features,MonteCarloGBM
from src.environment import BinaryOptionsEnv,prepare_market_data,AccountConfig
from src.forex_spot import ForexSpotEnv,SpotConfig
from src.decision import RiskConfig
from src.opportunities import OpportunityTraining
from src.train import train


def raw(n=180,timeframe='M15'):
    m=MarketConfig(timeframe)
    close=1.1*np.exp(np.cumsum(np.random.default_rng(99).normal(0,.0002,n)))
    f=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq=m.step,tz='UTC'),
        open=close,high=close+.0004,low=close-.0004,close=close,volume=1.,
        cond_volatility=.0002,dist_p95=.002,dist_p05=-.002,
        mc_prob_up=.9,mc_prob_down=.1,mc_prob_tie=0.))
    f.attrs.update(timeframe=timeframe,direction_probability_horizon=1)
    return f


def prepared(n=8,timeframe='M15',expiry=1):
    f=raw(n+2,timeframe);f.attrs['direction_probability_horizon']=expiry
    return prepare_market_data(f,1,1,market=MarketConfig(timeframe,expiry_candles=expiry))


@pytest.mark.parametrize('timeframe',['M1','M5','M15','H1','H4'])
def test_quant_causal_for_each_frequency_and_expiry(timeframe):
    f=raw(timeframe=timeframe).drop(columns=['cond_volatility','dist_p95','dist_p05','mc_prob_up','mc_prob_down','mc_prob_tie'])
    cfg=QuantConfig(window=128,min_observations=100,refit_every=20)
    a=enrich_features(f,cfg,timeframe=timeframe,expiry_candles=2)
    b=enrich_features(f.iloc[:140],cfg,timeframe=timeframe,expiry_candles=2)
    assert a.attrs['timeframe']==timeframe and a.attrs['direction_probability_horizon']==2
    np.testing.assert_allclose(a.mc_prob_up.iloc[:140],b.mc_prob_up,equal_nan=True)
    np.testing.assert_allclose(a.mc_log_drift.iloc[:140],b.mc_log_drift,equal_nan=True)


@pytest.mark.parametrize('expiry',[1,2,8,32])
def test_mc_probability_uses_matching_column_of_actual_10000_paths(expiry):
    mc=MonteCarloGBM();paths=mc.paths(1.1,.001,.0001)
    up,down,tie=mc.direction_probabilities(.001,.0001,expiry)
    assert up==pytest.approx(np.mean(paths[:,expiry-1]>1.1))
    assert down==pytest.approx(np.mean(paths[:,expiry-1]<1.1))
    assert float(up+down+tie)==pytest.approx(1.)


@pytest.mark.parametrize('timeframe',['M1','M5','M15','H1','H4'])
def test_binary_nonoverlap_and_exact_expiration(timeframe):
    f=prepared(timeframe=timeframe,expiry=2);f['close']=np.linspace(1.1,1.2,len(f))
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig(timeframe,expiry_candles=2),risk=RiskConfig(enabled=False))
    e.reset();obs,reward,t,u,info=e.step(2)
    assert reward==.8 and info['index']==2
    assert pd.Timestamp(info['expiry_time'])-pd.Timestamp(info['decision_time'])==pd.Timedelta(0) # info decision_time describes new state
    assert pd.Timestamp(info['expiry_time'])-(f.timestamp.iloc[0]+e.bar_step)==2*e.bar_step
    _,_,_,_,info=e.step(1);assert info['index']==3 # waiting does not open a contract
    check_env(e,warn=True)


def test_no_contract_crosses_a_gap_or_end_of_episode():
    f=prepared(expiry=2);f.loc[2:,'timestamp']+=pd.Timedelta(days=2)
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig(expiry_candles=2))
    assert e.segments[0][0]==2
    with pytest.raises(ValueError):e.reset(options={'start_index':0})
    _,_=e.reset(options={'start_index':int(e._eligible[-1])})
    _,_,_,truncated,_=e.step(2);assert truncated
    with pytest.raises(RuntimeError):e.step(2)
    with pytest.raises(ValueError):BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())


def test_effective_neutral_penalty_is_causal_and_never_changes_cash():
    up,down=prepared(expiry=2),prepared(expiry=2)
    up['close']=[1.1]*len(up);down['close']=[1.1]*len(down)
    up.loc[2,'close']=1.2;down.loc[2,'close']=1.0
    risk=replace(RiskConfig(),neutral_opportunity_penalty=.05)
    for frame in [up,down]:
        e=BinaryOptionsEnv(frame,prepared=True,lookback=1,market=MarketConfig(expiry_candles=2),risk=risk)
        e.reset();_,reward,_,_,info=e.step(1)
        assert reward==-.05 and info['balance']==1000 and info['profit']==0
    f=prepared();f.mc_prob_up=.5;f.mc_prob_down=.5
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig(),risk=risk)
    e.reset();assert e.step(1)[1]==0


def test_event_training_skips_only_free_wait_and_retains_eligible_neutral():
    f=prepared(n=20);f['mc_prob_up']=.5;f['mc_prob_down']=.5
    f.loc[[0,8],'mc_prob_up']=.9;f.loc[[0,8],'mc_prob_down']=.1
    base=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig(),risk=replace(RiskConfig(),neutral_opportunity_penalty=.05))
    wrapper=OpportunityTraining(base)
    wrapper.reset(options={'start_index':0})
    _,reward,_,_,info=wrapper.step(1)
    assert reward==-.05 and info['index']==8 and info['skipped_no_setup']==7
    assert info['requested_action']==1 and info['missed_opportunity']
    # Verify equivalence with seven explicit free waits; no policy action forced.
    comparison=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig(),risk=base.risk)
    comparison.reset();comparison.step(1)
    for _ in range(7):comparison.step(1)
    np.testing.assert_array_equal(base._observation(),comparison._observation())
    empty=f.copy();empty.mc_prob_up=.5;empty.mc_prob_down=.5
    with pytest.raises(ValueError,match='setups'):OpportunityTraining(BinaryOptionsEnv(empty,prepared=True,lookback=1,market=MarketConfig()))


def test_resampling_complete_groups_utc_and_no_upsampling(tmp_path):
    f=raw(16).iloc[:16][dl.COLUMNS]
    h=aggregate(f,15,'H1',now='2024-01-09T00:00Z')
    assert len(h)==4 and h.volume.iloc[0]==4
    assert h.open.iloc[0]==f.open.iloc[0] and h.close.iloc[0]==f.close.iloc[3]
    partial=aggregate(f.drop(index=1),15,'H1',now='2024-01-09T00:00Z')
    assert len(partial)==3 and partial.attrs['dropped_partial_bars']==1
    with pytest.raises(dl.DataError,match='M1'):aggregate(f,15,'M5')
    minute=f.copy();minute.timestamp=pd.date_range('2024-01-08',periods=len(f),freq='min',tz='UTC')
    m5=aggregate(minute,1,'M5',now='2024-01-09T00:00Z');assert len(m5)==3
    p=tmp_path/'DAT_ASCII_EURUSD_M1_202401.csv';minute.to_csv(p,index=False)
    output=tmp_path/'m5.parquet'
    loaded=dl.load_data(inputs=[p],output=output,timeframe='M5',timezone='UTC',now='2024-01-09T00:00Z')
    assert loaded.attrs['timeframe']=='M5' and len(loaded)==3
    again=dl.load_data(inputs=[p],output=output,timeframe='M5',timezone='UTC',now='2024-01-09T00:00Z')
    pd.testing.assert_frame_equal(loaded,again)


def spot_env(close,risk=None,account=None):
    f=prepared(n=len(close)-1);f=f.iloc[:len(close)].copy();f['close']=close
    return ForexSpotEnv(f,prepared=True,lookback=1,market=MarketConfig(trading_mode='forex_spot'),
        risk=risk or RiskConfig(enabled=False),account=account)


def test_spot_pnl_spread_position_holding_and_flat_close():
    e=spot_env([1.1,1.101,1.102,1.103]);e.reset()
    _,_,_,_,a=e.step(2);assert a['balance']==pytest.approx(1008.5) and a['position']==1
    _,_,_,_,b=e.step(2);assert b['balance']==pytest.approx(1018.5) # no second spread
    _,_,_,_,c=e.step(1);assert c['balance']==pytest.approx(1018.5) and c['position']==0
    assert c['trades']==1 and c['closed_trades'][0]['profit']==pytest.approx(18.5)


def test_spot_short_and_margin_call_are_not_binary_payout():
    e=spot_env([1.1,1.099,1.098]);e.reset();_,_,_,_,a=e.step(0)
    assert a['balance']==pytest.approx(1008.5)
    _,_,_,_,b=e.step(1);assert b['closed_trades'][0]['profit']==pytest.approx(8.5)
    e=spot_env([1.1,1.0,.99]);e.reset();_,_,terminated,_,info=e.step(2)
    assert terminated and info['margin_call'] and info['position']==0
    with pytest.raises(ValueError):MarketConfig(trading_mode='forex_spot',expiry_candles=2)


@pytest.mark.parametrize('mode,expiry',[('binary',1),('binary',2),('forex_spot',1)])
def test_real_ppo_v4_training_freezes_market_contract_and_holdout(tmp_path,mode,expiry):
    market=MarketConfig('H1',mode,expiry);f=raw(300,'H1');f.attrs['direction_probability_horizon']=expiry
    path=tmp_path/'data.parquet';f.to_parquet(path,index=False)
    m=train(path,output_dir=tmp_path/'models',total_timesteps=32,n_steps=16,batch_size=8,
        n_epochs=1,eval_freq=16,lookback=1,indicator_period=1,episode_length=16,market=market,verbose=0)
    assert m['manifest_version']==4 and m['market_config']==asdict(market)
    assert m['holdout_test_evaluated'] is False and m['artifact_status'].startswith('research')
    metrics=json.loads((tmp_path/'models/validation/validation_metrics.json').read_text())
    assert metrics['evaluation'].startswith('chronological full phase')
    if mode=='binary': assert m['risk']['neutral_opportunity_penalty']==.05 and m['opportunity_training']
    else: assert m['spot_config']['lots']==.1 and not m['opportunity_training']


def test_workflow_inputs_are_connected_and_bundle_isolated():
    source=(Path(__file__).resolve().parents[1]/'.github/workflows/pipeline.yml').read_text()
    w=yaml.safe_load(source);inputs=w.get('on',w.get(True))['workflow_dispatch']['inputs']
    assert inputs['timeframe']['options']==['auto','M1','M5','M15','H1','H4']
    assert inputs['trading_mode']['options']==['binary','forex_spot']
    assert inputs['expiry_candles']['type']=='number' and inputs['expiry_candles']['default']==0
    assert 'src.backtest' not in source
    assert 'name: forex-research-bundle-v4-' in source
    assert '--trading-mode "$FOREX_TRADING_MODE"' in source and '--expiry-candles "$FOREX_EXPIRY_CANDLES"' in source


@pytest.mark.parametrize('expiry',[0,1.5,True,33])
def test_bad_expiry_is_rejected(expiry):
    with pytest.raises(ValueError):MarketConfig(expiry_candles=expiry)


def test_spot_reversal_journal_reconciles_equity_and_rejects_insufficient_margin():
    e=spot_env([1.1,1.101,1.100]);e.reset();_,_,_,_,a=e.step(2)
    _,_,_,_,b=e.step(0)
    assert len(b['closed_trades'])==2 and b['trades']==2
    assert sum(t['profit'] for t in b['closed_trades'])==pytest.approx(b['balance']-1000)
    assert b['balance']==pytest.approx(1017.) # 2 x 10 pips, 2 x 1.5 USD entry spread
    e=spot_env([1.1,1.101,1.102],account=AccountConfig(initial_balance=50))
    e.reset();_,_,_,_,info=e.step(2)
    assert info['blocked'] and info['balance']==50 and info['position']==0


def test_open_spot_is_liquidated_before_a_market_gap():
    f=prepared(n=4);f['close']=[1.1,1.101,1.102,1.103,1.104][:len(f)]
    f.loc[2:,'timestamp']+=pd.Timedelta(days=2)
    e=ForexSpotEnv(f,prepared=True,lookback=1,risk=RiskConfig(enabled=False))
    e.reset();_,_,_,truncated,info=e.step(2)
    assert truncated and info['position']==0 and len(info['closed_trades'])==1
    assert info['balance']==pytest.approx(1008.5)


def test_v4_blocked_attempt_without_setup_is_free_but_opposite_setup_costs():
    f=prepared();f.mc_prob_up=.5;f.mc_prob_down=.5
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())
    e.reset();assert e.step(2)[1]==0
    f.mc_prob_up=.9;f.mc_prob_down=.1
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())
    e.reset();assert e.step(0)[1]==-.05
    e.reset();assert e.step(1)[1]==-.05


def test_v4_selection_uses_risk_reward_once_and_spot_can_report_gap_debt():
    from src.market_evaluation import evaluate_market_policy
    class Long:
        def predict(self,obs,deterministic=True):return np.array(2),None
    f=prepared(n=3);f['close']=[1.1,1.099,1.101,1.102][:len(f)]
    env=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())
    result=evaluate_market_policy(Long(),env)
    assert result['max_drawdown']>0 and not result['risk_stop']
    assert result['selection_score']==pytest.approx(result['total_reward']/result['eligible_setups_on_exposure'])
    debt=spot_env([1.1,1.0,.99])
    result=evaluate_market_policy(Long(),debt)
    assert result['risk_stop'] and result['final_balance']<0
    assert result['net_profit']==pytest.approx(-1001.5)
    assert result['selection_score']<0


def test_resampling_cannot_move_historical_holdout_into_training():
    from src.train import chronological_market_split
    from src.market import PROTECTED_HOLDOUT_START
    f=raw(1000,'H4');f['timestamp']=pd.date_range('2026-01-01',periods=len(f),freq='4h',tz='UTC')
    market=MarketConfig('H4')
    ready=prepare_market_data(f,1,1,market=market)
    splits=chronological_market_split(ready,market)
    for phase in ['train','validation']:
        assert (splits[phase].timestamp+market.step<=PROTECTED_HOLDOUT_START).all()
    assert splits['test'].timestamp.min()>=PROTECTED_HOLDOUT_START
    # Prices inside the protected snapshot cannot alter membership or normalization.
    changed=ready.copy();changed.loc[changed.timestamp>=PROTECTED_HOLDOUT_START,'close']=999
    other=chronological_market_split(changed,market)
    pd.testing.assert_frame_equal(splits['train'],other['train'])
    pd.testing.assert_frame_equal(splits['validation'],other['validation'])


def test_frequency_report_counts_valid_entry_grid_and_separates_policy(tmp_path):
    from src.market_evaluation import evaluate_market_policy

    class Neutral:
        def predict(self, obs, deterministic=True):
            return np.array(1), None

    frame = prepared(timeframe='M5').iloc[:6].copy()
    frame['mc_prob_up'] = [.50, .52, .55, .57, .60, .99]
    frame['mc_prob_down'] = 1 - frame.mc_prob_up
    env = BinaryOptionsEnv(
        frame, prepared=True, lookback=1, market=MarketConfig('M5'),
        risk=RiskConfig(min_confidence=.54, min_expected_edge=0.),
    )
    result = evaluate_market_policy(Neutral(), env, tmp_path, trading_profile='scalping')
    proposals = result['stochastic_proposals']
    assert proposals['observed_hours'] == pytest.approx(5/12)
    assert proposals['before_filter']['count'] == 4
    assert proposals['after_confidence']['count'] == 3
    assert proposals['after_economic_filter']['count'] == 2
    assert proposals['after_economic_filter']['per_hour'] == pytest.approx(4.8)
    assert result['trades_per_observed_hour'] == 0
    assert json.loads((tmp_path/'validation_metrics.json').read_text())['stochastic_proposals'] == proposals
    assert 'Superan confianza, Wilson y payout' in (tmp_path/'validation.md').read_text()
