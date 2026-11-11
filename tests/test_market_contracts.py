"""Contratos multi-timeframe, expiración causal y PnL spot independientes."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from src.market import MarketConfig
from src.resampling import aggregate
from src import data_loader as dl
from src.quant_engine import QuantConfig,enrich_features,MonteCarloGBM
from src.environment import AlignmentReward,BinaryOptionsEnv,prepare_market_data
from src.decision import RiskConfig


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


@pytest.mark.parametrize('setup,winning_close', [(0, 1.), (2, 1.2)])
def test_m5_shaping_penalizes_conflict_causally_and_only_rewards_aligned_wins(setup, winning_close):
    frame = prepared(timeframe='M5')
    frame['mc_prob_up'] = .8 if setup == 2 else .2
    frame['mc_prob_down'] = 1 - frame.mc_prob_up
    frame.loc[0, 'close'] = 1.1
    args = dict(prepared=True, lookback=1, market=MarketConfig('M5'),
                risk=RiskConfig(min_confidence=.57, mc_z=2.24, min_expected_edge=.01))
    losing_close = 2.2 - winning_close
    for close, outcome, pnl in ((winning_close, 'win', 8.), (losing_close, 'loss', -10.), (1.1, 'tie', 0.)):
        frame.loc[1, 'close'] = close
        env = BinaryOptionsEnv(frame, alignment_reward=AlignmentReward(), **args)
        env.reset()
        _, reward, _, _, info = env.step(2 - setup)
        assert reward == -1.25 and info['mc_setup_action'] == setup
        assert info['action'] == 1 and info['blocked'] and info['index'] == 1
        assert info['profit'] == 0. and info['balance'] == 1000. and info['trades'] == 0
        env.reset()
        _, reward, _, _, info = env.step(setup)
        assert info['outcome'] == outcome and info['profit'] == pnl
        assert info['balance'] == 1000. + pnl
        if outcome == 'win':
            assert reward == pytest.approx(1.75 - env.risk.trade_penalty)
        else:
            base = BinaryOptionsEnv(frame, **args)
            base.reset()
            assert reward == base.step(setup)[1]


def test_m5_alignment_uses_economic_gate_and_cannot_enable_on_m15():
    alignment = AlignmentReward()
    frame = prepared(timeframe='M5')
    frame['mc_prob_up'], frame['mc_prob_down'] = .5705, .4295
    env = BinaryOptionsEnv(
        frame, prepared=True, lookback=1, market=MarketConfig('M5'), alignment_reward=alignment,
        risk=RiskConfig(min_confidence=.57, min_expected_edge=.01, mc_z=2.24),
    )
    for action in (0, 1, 2):
        env.reset()
        _, reward, _, _, info = env.step(action)
        assert reward == 0. and info['mc_setup_action'] == 1
    with pytest.raises(ValueError, match='requiere M5'):
        BinaryOptionsEnv(prepared(), prepared=True, lookback=1,
                         market=MarketConfig(), alignment_reward=alignment)


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


@pytest.mark.parametrize('expiry',[0,1.5,True,33])
def test_bad_expiry_is_rejected(expiry):
    with pytest.raises(ValueError):MarketConfig(expiry_candles=expiry)


def test_v4_blocked_attempt_without_setup_is_free_but_opposite_setup_costs():
    f=prepared();f.mc_prob_up=.5;f.mc_prob_down=.5
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())
    e.reset();assert e.step(2)[1]==0
    f.mc_prob_up=.9;f.mc_prob_down=.1
    e=BinaryOptionsEnv(f,prepared=True,lookback=1,market=MarketConfig())
    e.reset();assert e.step(0)[1]==-.05
    e.reset();assert e.step(1)[1]==-.05


def test_resampling_cannot_move_historical_holdout_into_training():
    from src.splits import chronological_market_split
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
        risk=RiskConfig(min_confidence=.54, min_expected_edge=0.), alignment_reward=AlignmentReward(),
    )
    result = evaluate_market_policy(Neutral(), env, tmp_path, trading_profile='scalping')
    proposals = result['stochastic_proposals']
    assert proposals['observed_hours'] == pytest.approx(5/12)
    assert proposals['before_filter']['count'] == 4
    assert proposals['after_confidence']['count'] == 3
    assert proposals['after_economic_filter']['count'] == 2
    assert proposals['after_economic_filter']['per_hour'] == pytest.approx(4.8)
    assert result['trades_per_observed_hour'] == 0
    assert result['mc_alignment'] == dict(evaluated_setups=2, aligned_requests=0,
        opposite_requests=0, neutral_requests=2, opposite_rate=0.)
    assert result['validation_quality']['checks']['eligible_setup_conversion']['value'] == 0.
    assert json.loads((tmp_path/'validation_metrics.json').read_text())['stochastic_proposals'] == proposals
    assert 'Superan confianza, Wilson y payout' in (tmp_path/'validation.md').read_text()


def test_m5_quality_checks_actual_execution_and_strict_profit_factor():
    from src.market_evaluation import scalping_m5_quality

    metrics = dict(
        win_rate_excluding_refunds=.56, profit_factor=1.2, profit_factor_status='finite',
        trades_per_observed_hour=1.5, trades_per_hour_upper_bound=3.,
        total_trades=30, eligible_setups_before_policy=100,
    )
    result = scalping_m5_quality(metrics, (1.5, 3.))
    assert result['checks']['win_rate_excluding_refunds']['passed']
    assert not result['checks']['profit_factor']['passed']
    assert not result['passed']
    metrics['profit_factor'] = 1.21
    assert scalping_m5_quality(metrics, (1.5, 3.))['passed']
    for change in (dict(trades_per_observed_hour=.2), dict(trades_per_observed_hour=3.01),
                   dict(eligible_setups_before_policy=101), dict(eligible_setups_before_policy=0)):
        assert not scalping_m5_quality(metrics | change, (1.5, 3.))['passed']


def test_adaptive_drift_suppresses_weak_momentum_and_preserves_strong_momentum():
    from src.quant_engine import momentum_log_drift

    cfg = QuantConfig(drift_scale=2.2, drift_noise_z=.5, drift_cap_sigma=.9)
    adaptive = replace(cfg, drift_scale_floor=1.4)
    mean = np.array([.00008, -.00008])
    se = np.array([.00008, .00002])
    sigma = np.array([.0002, .0002])
    fixed = momentum_log_drift(mean, se, sigma, cfg)
    changed = momentum_log_drift(mean, se, sigma, adaptive)
    assert 0 < changed[0] < fixed[0]
    assert changed[1] == fixed[1]
