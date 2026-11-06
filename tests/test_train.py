"""Splits, normalización train-only y un entrenamiento PPO real mínimo."""
import json

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO
from src.decision import RiskConfig

from src.environment import (AccountConfig, BinaryOptionsEnv, fit_normalizer,
                             market_feature_names, prepare_market_data)
from src.train import chronological_split, evaluate_policy, train


def raw_data(n=300):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(19).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2020-01-06',periods=n,freq='15min',tz='UTC'),
                             open=close, high=close+.0002, low=close-.0002, close=close,
                             cond_volatility=.0002,dist_p95=.002,dist_p05=-.002))


def test_chronological_split_and_train_only_normalization():
    ready=prepare_market_data(raw_data(),lookback=2,indicator_period=2)
    split=chronological_split(ready)
    assert split['train'].timestamp.max() < split['validation'].timestamp.min()
    assert split['validation'].timestamp.max() < split['test'].timestamp.min()
    normalizer=fit_normalizer(split['train'],2)
    expected=split['train'][list(market_feature_names(2))].mean().to_numpy()
    np.testing.assert_allclose(normalizer['mean'],expected)
    changed=ready.copy()
    changed.loc[len(split['train']):,'dist_p95']=10000
    other=fit_normalizer(chronological_split(changed)['train'],2)
    assert normalizer == other


@pytest.mark.parametrize('kwargs',[dict(train_fraction=.9,validation_fraction=.2),dict(purge_bars=-1)])
def test_invalid_splits(kwargs):
    with pytest.raises(ValueError): chronological_split(raw_data(),**kwargs)


def test_evaluation_carries_cash_over_weekends_without_new_funding():
    raw=raw_data()
    ready=prepare_market_data(raw,lookback=2,indicator_period=2).iloc[:4].copy()
    ready['close']=[1.1,1.2,1.3,1.4]
    ready['timestamp']=pd.to_datetime(['2024-01-05T21:30Z','2024-01-05T21:45Z',
                                      '2024-01-07T22:00Z','2024-01-07T22:15Z'])
    env=BinaryOptionsEnv(ready,prepared=True,lookback=2,risk=RiskConfig(enabled=False))
    class AlwaysCall:
        def predict(self,obs,deterministic=True): return np.array(2),None
    result=evaluate_policy(AlwaysCall(),env)
    assert result['net_profit'] == 16 and result['final_balance'] == 1016
    assert result['steps'] == result['trades'] == result['wins'] == 2


def test_short_real_ppo_training_saves_loadable_best_and_metadata(tmp_path):
    path=tmp_path/'synthetic.parquet'
    raw_data().to_parquet(path,index=False)
    output=tmp_path/'models'
    manifest=train(path,output_dir=output,total_timesteps=32,n_steps=16,batch_size=8,
                   n_epochs=1,eval_freq=16,lookback=2,indicator_period=2,
                   episode_length=32,verbose=0)
    assert (output/'best_ppo_agent.zip').is_file()
    assert (output/'final_ppo_agent.zip').is_file()
    normalizer=json.loads((output/'normalizer.json').read_text())
    history=json.loads((output/'validation_history.json').read_text())
    best=json.loads((output/'best_validation.json').read_text())
    assert best['net_profit'] == max(item['net_profit'] for item in history)
    assert history[0]['timesteps'] == 0 and history[-1]['timesteps'] >= 32
    assert manifest['holdout_test_evaluated'] is False
    assert manifest['normalizer_fit'] == 'train only'
    ready=prepare_market_data(raw_data(),lookback=2,indicator_period=2)
    validation=chronological_split(ready)['validation']
    env=BinaryOptionsEnv(validation,prepared=True,lookback=2,normalizer=normalizer)
    model=PPO.load(output/'best_ppo_agent.zip',device='cpu')
    obs,_=env.reset()
    action,_=model.predict(obs,deterministic=True)
    assert env.action_space.contains(action)
    assert model.observation_space == env.observation_space

