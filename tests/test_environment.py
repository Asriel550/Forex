"""Liquidación determinista, límites temporales y observaciones sin futuro."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from stable_baselines3.common.env_checker import check_env

from src.environment import (AccountConfig, BinaryOptionsEnv, market_feature_names,
                             prepare_market_data)


def prepared_frame(closes, times=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    frame = pd.DataFrame(dict(timestamp=times if times is not None else
                               pd.date_range('2024-01-02', periods=n, freq='15min', tz='UTC'),
                               open=closes, high=closes+.0001, low=closes-.0001,
                               close=closes, cond_volatility=.0002,
                               dist_p95=.002, dist_p05=-.002))
    frame['rsi_scaled'], frame['atr_fraction'], frame['spread_scaled'] = .5, .001, 0.
    for lag in range(2):
        frame[f'log_return_lag_{lag}'] = 0.
    return frame


def env_for(closes, **kwargs):
    return BinaryOptionsEnv(prepared_frame(closes), prepared=True, lookback=2, **kwargs)


@pytest.mark.parametrize('action,settlement,expected', [
    (2,1.11,.8), (2,1.09,-1.), (0,1.09,.8), (0,1.11,-1.),
    (1,1.11,0.), (1,1.09,0.), (2,1.10,0.), (0,1.10,0.),
])
def test_rewards_and_cash(action, settlement, expected):
    env = env_for([1.10, settlement])
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    _, reward, terminated, truncated, info = env.step(action)
    assert reward == expected
    assert info['balance'] == pytest.approx(1000+10*expected)
    assert info['profit'] == pytest.approx(10*expected)
    assert not terminated and truncated
    assert info['index'] == 1 and info['open_positions'] == 0
    assert pd.Timestamp(info['expiry_time'])-pd.Timestamp('2024-01-02T00:15Z') == pd.Timedelta(minutes=15)


def test_tie_can_explicitly_be_a_loss():
    env = env_for([1.1,1.1], account=AccountConfig(tie_policy='loss'))
    env.reset()
    _, reward, _, _, info = env.step(2)
    assert reward == -1 and info['balance'] == 990
    assert info['ties'] == 1


def test_inactivity_penalty_does_not_fabricate_cash_loss():
    env = env_for([1.1,1.2], account=AccountConfig(inactivity_penalty=.001))
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert reward == -.001 and info['profit'] == 0 and info['balance'] == 1000
    assert info['trades'] == 0


def test_one_bar_expiration_and_no_overlap():
    env = env_for([1.10,1.11,1.09,1.10])
    env.reset()
    rewards = []
    for action in (2,0,0):
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert info['index'] == info['entry_index']+1
        assert info['open_positions'] == 0
    assert rewards == [.8,.8,-1.]
    assert info['balance'] == 1006
    assert info['trades'] == 3 and info['wins'] == 2 and info['losses'] == 1
    assert truncated and not terminated
    with pytest.raises(RuntimeError): env.step(2)


def test_cannot_step_before_reset_or_use_invalid_actions():
    env = env_for([1.1,1.2,1.3])
    with pytest.raises(RuntimeError): env.step(2)
    env.reset()
    for action in [-1,3,1.5,'CALL']:
        with pytest.raises(ValueError): env.step(action)


def test_insufficient_cash_terminates_without_negative_balance():
    env = env_for([1.1,1.2,1.3], account=AccountConfig(initial_balance=10, stake=10))
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -1 and terminated and not truncated
    assert info['balance'] == 0 and info['end_reason'] == 'insufficient_balance'


def test_drawdown_limit_terminates():
    env = env_for([1.1,1.2,1.3], account=AccountConfig(initial_balance=100, stake=10, max_drawdown=.10))
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info['end_reason'] == 'max_drawdown'


def test_episode_length_truncates_after_exact_number_of_transitions():
    env = env_for(np.linspace(1.1,1.2,10), episode_length=2)
    env.reset()
    assert env.step(1)[3] is False
    _, _, terminated, truncated, info = env.step(1)
    assert truncated and not terminated and info['index'] == 2


def test_gap_never_becomes_a_15_minute_contract():
    times = pd.to_datetime(['2024-01-05T21:15Z','2024-01-05T21:30Z','2024-01-05T21:45Z',
                            '2024-01-07T22:00Z','2024-01-07T22:15Z'])
    env = BinaryOptionsEnv(prepared_frame([1.1,1.2,1.3,1.0,1.1],times), prepared=True, lookback=2)
    assert env.segments == ((0,2),(3,4))
    env.reset()
    env.step(2)
    _, _, _, truncated, info = env.step(2)
    assert truncated and info['settlement_price'] == 1.3
    with pytest.raises(ValueError): env.reset(options={'start_index':2})
    env.reset(options={'start_index':3, 'balance':1016, 'peak_balance':1016})
    assert env.step(2)[4]['balance'] == 1024


def test_random_resets_are_seeded_and_valid():
    env = env_for(np.linspace(1.1,1.2,100), random_start=True, episode_length=10)
    first, info = env.reset(seed=19)
    second, other = env.reset(seed=19)
    np.testing.assert_array_equal(first, second)
    assert info['index'] == other['index']
    env.step(1)
    assert env.reset(seed=19)[1]['balance'] == 1000


def test_current_observation_ignores_future_prices_and_unknown_label_columns():
    closes = 1.1+np.sin(np.arange(90)/5)*.0005
    original = prepared_frame(closes)
    future = original.copy()
    future.loc[50:, ['open','high','low','close']] *= 1.2
    future['future_label'] = np.arange(len(future))*1e8
    first = BinaryOptionsEnv(original, lookback=2, indicator_period=2)
    second = BinaryOptionsEnv(future, lookback=2, indicator_period=2)
    a, _ = first.reset()
    b, _ = second.reset()
    np.testing.assert_array_equal(a,b)
    for _ in range(30):
        a, ra, _, _, ia = first.step(2)
        b, rb, _, _, ib = second.step(2)
        np.testing.assert_array_equal(a,b)
        assert ra == rb and ia['balance'] == ib['balance']


def test_technical_features_prefix_is_identical_without_future_rows():
    raw = prepared_frame(1.1+np.sin(np.arange(100)/5)*.0005)
    full = prepare_market_data(raw, lookback=2, indicator_period=2)
    prefix = prepare_market_data(raw.iloc[:50], lookback=2, indicator_period=2)
    names = list(market_feature_names(2))
    pd.testing.assert_frame_equal(full.iloc[:len(prefix)][names], prefix[names])


def test_environment_does_not_modify_the_input():
    frame = prepared_frame([1.1,1.2,1.3])
    original = frame.copy(deep=True)
    env = BinaryOptionsEnv(frame, prepared=True, lookback=2)
    env.reset()
    env.step(2)
    pd.testing.assert_frame_equal(frame,original)


def test_prepared_microsecond_timestamps_are_handled():
    frame = prepared_frame([1.1,1.2,1.3])
    frame['timestamp'] = frame.timestamp.astype('datetime64[us, UTC]')
    env = BinaryOptionsEnv(frame, prepared=True, lookback=2)
    assert env.segments == ((0,2),)


@pytest.mark.parametrize('kwargs', [dict(stake=0), dict(payout=1.2),dict(initial_balance=5),
                                    dict(tie_policy='win'),dict(inactivity_penalty=-1),dict(payout=np.nan)])
def test_invalid_account_configuration(kwargs):
    with pytest.raises(ValueError): AccountConfig(**kwargs)


def test_missing_features_and_nonconsecutive_data_fail():
    with pytest.raises(ValueError): BinaryOptionsEnv(pd.DataFrame({'close':[1.1,1.2]}))
    frame = prepared_frame([1.1,1.2,1.3])
    frame.loc[1,'cond_volatility'] = np.nan
    with pytest.raises(ValueError): BinaryOptionsEnv(frame, prepared=True, lookback=2)
    frame = prepared_frame([1.1,1.2], pd.to_datetime(['2024-01-01T00:00Z','2024-01-02T00:00Z']))
    with pytest.raises(ValueError): BinaryOptionsEnv(frame, prepared=True, lookback=2)


def test_gymnasium_and_stable_baselines_contract():
    env = env_for(np.linspace(1.1,1.2,100))
    check_env(env, warn=True, skip_render_check=True)



def test_next_close_is_unobservable_until_after_action():
    up = env_for([1.1,1.2])
    down = env_for([1.1,1.0])
    np.testing.assert_array_equal(up.reset()[0],down.reset()[0])
    assert up.step(2)[1] == .8
    assert down.step(2)[1] == -1.
