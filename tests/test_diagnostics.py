"""Diagnóstico causal, denominadores por exposición y barrido sin holdout."""
import json
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src import diagnostics as d
from src.decision import RiskConfig, resolve_action
from src.environment import AccountConfig, prepare_market_data


def raw(n=300):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(81).normal(0,.0001,n)))
    frame=pd.DataFrame(dict(timestamp=pd.date_range('2024-01-08',periods=n,freq='15min',tz='UTC'),
        open=close,high=close+.0002,low=close-.0002,close=close,volume=0.,
        cond_volatility=.0002,dist_p95=.002,dist_p05=-.002,
        mc_prob_up=.8,mc_prob_down=.2,mc_prob_tie=0.))
    frame.attrs['quant_config']=asdict(d.QuantConfig())
    return frame


class Fixed:
    def __init__(self,action):self.action=action
    def predict(self,obs,deterministic):
        assert deterministic
        return np.array(self.action),None


def test_mc_gate_matches_live_resolver_and_does_not_read_future_close():
    f=raw(6);f['mc_prob_up']=[.9,.5,.59,.601,.1,.5];f['mc_prob_down']=1-f.mc_prob_up
    a=AccountConfig();r=RiskConfig()
    proposal=d.mc_proposals(f,risk=r,account=a)
    changed=f.copy();changed['close']=np.arange(6)+100
    pd.testing.assert_frame_equal(proposal,d.mc_proposals(changed,risk=r,account=a))
    for row in proposal.itertuples():
        source=f.iloc[row.entry_index]
        expected=resolve_action(row.mc_raw_action,source.mc_prob_up,source.mc_prob_down,0.,payout=a.payout,tie_policy=a.tie_policy,risk=r)
        assert row.mc_action==expected['action']


def test_neutral_ppo_vs_mc_and_rejected_opposite_requests_are_separate():
    f=prepare_market_data(raw(10),1,1);r=RiskConfig();a=AccountConfig()
    summary,trace=d.diagnose_policy(Fixed(1),f,normalizer=None,risk=r,account=a,lookback=1)
    n=len(f)-1
    assert summary['mc_opportunities_full_phase']==n
    assert summary['mc_opportunities_skipped_neutral']==n and summary['blocked_requests']==0
    assert summary['raw_ppo_distribution']['NEUTRO']['count']==n
    summary,_=d.diagnose_policy(Fixed(0),f,normalizer=None,risk=r,account=a,lookback=1)
    assert summary['blocked_requests']==n and summary['mc_opportunities_opposed']==n
    assert summary['mc_opportunities_skipped_neutral']==0 and summary['total_trades']==0


def test_risk_stop_and_gaps_have_separate_denominators():
    f=prepare_market_data(raw(10),1,1)
    f['close']=np.linspace(1.2,1.1,len(f))
    # Gap occurs after the first contract (do not settle across it).
    f.loc[2:,'timestamp']+=pd.Timedelta(days=2)
    r=replace(RiskConfig(),drawdown_limit=.01)
    result,trace=d.diagnose_policy(Fixed(2),f,normalizer=None,risk=r,account=AccountConfig(),lookback=1)
    assert result['risk_stop'] and len(trace)==1 and result['gap_transitions']==1
    assert result['mc_opportunities_full_phase']>result['mc_opportunities_on_ppo_exposure']==1
    assert result['skipped_after_stop']==result['consecutive_transitions']-1
    assert sum(x['count'] for x in result['cross_table'])==1


def test_real_multi_seed_smoke_freezes_strategy_and_evaluates_no_holdout(tmp_path,monkeypatch):
    path=tmp_path/'features.parquet';f=raw();f.to_parquet(path,index=False)
    expected=d.chronological_split(prepare_market_data(f,1,1))
    max_time=expected['validation'].timestamp.max()
    original=d.diagnose_policy;seen=[]
    def guarded(model,frame,**kwargs):
        assert frame.timestamp.max()<=max_time
        seen.append((frame.timestamp.min(),frame.timestamp.max()))
        return original(model,frame,**kwargs)
    monkeypatch.setattr(d,'diagnose_policy',guarded)
    original_baseline=d.evaluate_mc_baseline
    def baseline_guard(frame,**kwargs):
        assert frame.timestamp.max()<=max_time
        return original_baseline(frame,**kwargs)
    monkeypatch.setattr(d,'evaluate_mc_baseline',baseline_guard)
    report=d.run_experiments(path,report_dir=tmp_path/'report',seeds=(42,43),total_timesteps=32,
        train_kwargs=dict(lookback=1,indicator_period=1,n_steps=16,batch_size=8,n_epochs=1,eval_freq=16,verbose=0))
    assert len(seen)==4 and report['holdout_evaluated'] is False
    assert report['automatic_promotion'] is False and report['automatic_seed_selection'] is False
    assert report['risk']['min_confidence']==.6 and report['ent_coef']==.02
    for seed in (42,43):
        m=json.loads((tmp_path/f'report/seed_{seed}/training_manifest.json').read_text())
        assert m['seed']==seed and m['holdout_test_evaluated'] is False
    archived=d.diagnose_frozen_bundle(path,tmp_path/'report/seed_42',report_dir=tmp_path/'frozen')
    assert len(archived['results'])==2 and archived['holdout_evaluated'] is False
    assert archived['model_sha256']
    manifest_path=tmp_path/'report/seed_42/training_manifest.json'
    m=json.loads(manifest_path.read_text());m['input_sha256']='0'*64;manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Hash'):d.diagnose_frozen_bundle(path,manifest_path.parent,report_dir=tmp_path/'bad')
    assert report['dispersion']['validation']['total_trades']['finite_count']==2
    assert (tmp_path/'report/diagnostics.md').exists() and (tmp_path/'report/diagnostics.csv').exists()


@pytest.mark.parametrize('seeds',[[],[1,1],[-1],[True],list(range(11))])
def test_bad_seed_grid_fails_before_reading_data(seeds,tmp_path):
    with pytest.raises(ValueError):d.run_experiments(tmp_path/'missing',seeds=seeds)


def test_workflow_freezes_holdout_and_isolates_diagnostics_and_telegram_test():
    source=(Path(__file__).resolve().parents[1]/'.github/workflows/pipeline.yml').read_text()
    workflow=yaml.safe_load(source);trigger=workflow.get('on',workflow.get(True))
    assert {'diagnostics','telegram_test'}.issubset(trigger['workflow_dispatch']['inputs']['mode']['options'])
    job=workflow['jobs']['research-pipeline']
    assert 'workflow_dispatch' in job['if'] and 'default_branch' in job['if']
    assert 'src.backtest' not in source
    publish=next(s for s in job['steps'] if s.get('with',{}).get('name')=='forex-alert-bundle')
    assert publish['if']=="inputs.mode == 'train'"
    diagnostic=next(s for s in job['steps'] if '--seeds' in s.get('run',''))
    assert diagnostic['if']=="inputs.mode == 'diagnostics'"
    assert '--seeds 42 43 44' in diagnostic['run']
    assert any('--model-dir .diagnostic-bundle/models' in s.get('run','') for s in job['steps'])
    tg=workflow['jobs']['telegram-alert']['steps']
    assert all(s.get('if')=="inputs.mode != 'telegram_test'" for s in tg if s.get('id')=='bundle' or s.get('uses','').startswith('actions/download-artifact'))
    test=next(s for s in tg if '--test-delivery' in s.get('run',''))
    assert test['env']['TELEGRAM_TEST_ID']=='${{ inputs.telegram_test_id }}'
    assert '$TELEGRAM_TEST_ID' in test['run']
