"""Entropía configurable, incentivos causales, compatibilidad y auditoría PPO."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.agent import build_agent,load_agent
from src.decision import RiskConfig
from src.environment import AccountConfig,BinaryOptionsEnv,prepare_market_data
from src.exploration import policy_snapshot
from src.train import train


def raw(n=300):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(194).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq='15min',tz='UTC'),
        open=close,high=close+.0002,low=close-.0002,close=close,volume=0.,
        cond_volatility=.0002,dist_p05=-.002,dist_p95=.002))


def env(up=.9,risk=None,future=1.12):
    p=prepare_market_data(raw(),1,1).iloc[:2].copy()
    p['close']=[1.1,future];p['mc_prob_up']=up;p['mc_prob_down']=1-up;p['mc_prob_tie']=0.
    return BinaryOptionsEnv(p,prepared=True,lookback=1,risk=risk)


def test_entropy_coefficient_is_real_model_setting_and_survives_checkpoint(tmp_path):
    e=env();model=build_agent(e,n_steps=16,batch_size=8,n_epochs=1,ent_coef=.03,verbose=0)
    assert model.ent_coef==.03
    path=tmp_path/'model.zip';model.save(path)
    assert load_agent(path).ent_coef==.03
    obs,_=e.reset()
    before=torch.get_rng_state().clone();snapshot=policy_snapshot(model,np.array([obs,obs]))
    assert torch.equal(before,torch.get_rng_state())
    assert 0<=snapshot['mean_entropy_nats']<=np.log(3)+1e-5
    assert sum(snapshot['mean_action_probabilities'].values())==pytest.approx(1.,abs=1e-6)


@pytest.mark.parametrize('value',[-.01,float('nan'),float('inf'),.101])
def test_invalid_entropy_coefficients_are_rejected(value):
    with pytest.raises(ValueError,match='ent_coef'):build_agent(env(),n_steps=16,batch_size=8,ent_coef=value,verbose=0)


def test_reduced_blocked_cost_and_free_wait_without_eligible_setup():
    risk=RiskConfig();assert risk.blocked_penalty==.05 and risk.trade_penalty==.025
    e=env(up=.51);e.reset();_,reward,_,_,info=e.step(2)
    assert reward==-.05 and info['profit']==0 and info['balance']==1000
    assert info['action']==1 and info['requested_action']==2 and info['blocked']
    e=env(up=.51);e.reset();_,reward,_,_,info=e.step(1)
    assert reward==0 and not info['missed_opportunity'] and info['balance']==1000


def test_neutral_cost_is_small_optional_and_uses_no_future_outcome():
    up,down=env(future=1.2),env(future=1.0)
    np.testing.assert_array_equal(up.reset()[0],down.reset()[0])
    for e in (up,down):
        _,reward,_,_,info=e.step(1)
        assert reward==-.005 and info['missed_opportunity'] and info['profit']==0
        assert info['trades']==0 and info['balance']==1000
    config=RiskConfig(neutral_opportunity_penalty=0.)
    e=env(risk=config);e.reset();assert e.step(1)[1]==0


def test_legacy_metadata_does_not_acquire_new_neutral_penalty():
    old=asdict(RiskConfig());old.pop('neutral_opportunity_penalty')
    old.update(trade_penalty=.05,blocked_penalty=.25,min_confidence=.7)
    cfg=RiskConfig.from_metadata(old)
    assert cfg.neutral_opportunity_penalty==0 and cfg.trade_penalty==.05 and cfg.blocked_penalty==.25
    e=env(risk=cfg);e.reset();assert e.step(1)[1]==0


def test_opportunity_penalty_requires_payout_edge_not_only_confidence():
    p=prepare_market_data(raw(),1,1).iloc[:2].copy()
    p['mc_prob_up']=.61;p['mc_prob_down']=.39;p['mc_prob_tie']=0.
    e=BinaryOptionsEnv(p,prepared=True,lookback=1,account=AccountConfig(payout=.5),risk=RiskConfig(min_confidence=.58))
    e.reset();_,reward,_,_,info=e.step(1)
    assert reward==0 and not info['missed_opportunity']


def test_manifest_and_rollout_audit_match_actual_steps_and_actions(tmp_path):
    p=tmp_path/'features.parquet';raw().to_parquet(p,index=False)
    risk=RiskConfig(min_confidence=.58,blocked_penalty=.02,neutral_opportunity_penalty=.01)
    m=train(p,output_dir=tmp_path/'models',total_timesteps=32,n_steps=16,batch_size=8,n_epochs=1,eval_freq=16,
            lookback=1,indicator_period=1,ent_coef=.03,risk=risk,verbose=0)
    assert m['ent_coef']==.03 and m['reward_version']=='risk_adjusted_v2'
    assert m['risk']==asdict(risk) and m['holdout_test_evaluated'] is False
    summary=json.loads((tmp_path/'models/exploration_summary.json').read_text())
    history=json.loads((tmp_path/'models/exploration_history.json').read_text())
    assert sum(summary['requested'].values())==m['actual_timesteps']==32
    assert sum(summary['executed'].values())==32 and len(history)==2
    assert sum(r['blocked_attempts'] for r in history)==summary['blocked_attempts']
    assert summary['ent_coef']==.03 and summary['evaluation_deterministic'] is True
    for row in history:assert sum(row['requested'].values())==sum(row['executed'].values())==16
    assert load_agent(tmp_path/'models/best_ppo_agent.zip').ent_coef==.03
    from src.backtest import run_backtest
    result=run_backtest(p,model_path=tmp_path/'models/best_ppo_agent.zip',report_dir=tmp_path/'reports',plots=False)
    assert result['risk_config']==asdict(risk) and 'missed_opportunities' in result['decision_analysis']


def test_workflow_preserves_threshold_scale_and_zero_opportunity_option():
    import yaml
    source=(Path(__file__).resolve().parents[1]/'.github/workflows/pipeline.yml').read_text()
    data=yaml.safe_load(source);trigger=data.get('on',data.get(True))
    inputs=trigger['workflow_dispatch']['inputs']
    assert '0.58' in inputs['confidence_threshold']['options'] and '1.5' in inputs['drift_scale']['options']
    assert inputs['ent_coef']['options']==['auto','0.01','0.02','0.03']
    assert '0' in inputs['neutral_opportunity_penalty']['options']
    assert '--ent-coef "$FOREX_ENT_COEF"' in source
    assert '--neutral-opportunity-penalty "$FOREX_NEUTRAL_OPPORTUNITY_PENALTY"' in source
