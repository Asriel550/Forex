"""Evaluación exclusiva del holdout congelado al entrenar, sin actualizar PPO.

Verifica hashes de datos/modelo/normalizador y límites originales del split.
El saldo persiste entre segmentos. Tras un stop, el resto del holdout queda
sin operar; los huecos no se convierten en contratos de quince minutos.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from filelock import FileLock
import numpy as np
import pandas as pd
from .agent import load_agent
from .artifacts import file_sha256
from .environment import AccountConfig,BinaryOptionsEnv,market_feature_names,prepare_market_data
from .metrics import calculate_metrics,export_reports,plot_reports
from .train import chronological_split

ROOT=Path(__file__).resolve().parents[1]
LOG=logging.getLogger(__name__)
TRADE_COLUMNS=['trade_id','action','direction','entry_index','expiry_index','entry_time','expiry_time',
               'entry_price','settlement_price','stake','payout','outcome','profit','balance_before','balance_after']
SIGNAL_COLUMNS=['entry_index','expiry_index','entry_time','expiry_time','action','outcome','reward','profit','balance_after']


class BacktestError(ValueError):
    """Datos/artefactos no pertenecen al entrenamiento registrado."""


def _verify_hash(path,expected,label):
    if not isinstance(expected,str) or len(expected)!=64 or file_sha256(path)!=expected:
        raise BacktestError(f'{label}: SHA256 distinto del manifiesto')


def load_holdout(input_path,model_path,normalizer_path,manifest_path):
    manifest=json.loads(Path(manifest_path).read_text())
    required={'input_sha256','best_model_sha256','normalizer_sha256','train_fraction',
              'validation_fraction','purge_bars','lookback','indicator_period','features','account','splits'}
    if not isinstance(manifest,dict) or not required.issubset(manifest):
        raise BacktestError('Manifiesto incompleto: volver a entrenar')
    if (manifest.get('manifest_version')!=2 or manifest.get('actual_timesteps',0)<=0
            or manifest.get('normalizer_fit')!='train only' or manifest.get('holdout_test_evaluated') is not False
            or manifest.get('selection')!='validation net_profit'):
        raise BacktestError('Requiere entrenamiento v2 completado y test reservado')
    if not np.isclose(manifest['train_fraction']+manifest['validation_fraction'],.90):
        raise BacktestError('Se exige el 10% final como holdout, antes de la purga')
    for path,key,label in [(input_path,'input_sha256','Parquet'),(model_path,'best_model_sha256','Modelo'),
                           (normalizer_path,'normalizer_sha256','Normalizador')]:
        _verify_hash(path,manifest[key],label)
    raw=pd.read_parquet(input_path);_verify_hash(input_path,manifest['input_sha256'],'Parquet tras lectura')
    ready=prepare_market_data(raw,manifest['lookback'],manifest['indicator_period'])
    splits=chronological_split(ready,train_fraction=manifest['train_fraction'],
                               validation_fraction=manifest['validation_fraction'],purge_bars=manifest['purge_bars'])
    for name,frame in splits.items():
        record=manifest['splits'][name]
        if (len(frame)!=record['rows'] or frame.timestamp.iloc[0].isoformat()!=record['start']
                or frame.timestamp.iloc[-1].isoformat()!=record['end']):
            raise BacktestError(f'Límites de {name} distintos del entrenamiento')
    normalizer=json.loads(Path(normalizer_path).read_text())
    _verify_hash(normalizer_path,manifest['normalizer_sha256'],'Normalizador tras lectura')
    if list(market_feature_names(manifest['lookback']))!=manifest['features'] or manifest['features']!=normalizer['features']:
        raise BacktestError('Orden de features incompatible')
    return splits['test'],normalizer,manifest


def simulate_holdout(model,holdout,*,account,lookback,normalizer):
    """No aprende pesos; el único DataFrame que recibe es el reservado."""
    env=BinaryOptionsEnv(holdout,prepared=True,lookback=lookback,account=account,normalizer=normalizer)
    if hasattr(model,'observation_space') and model.observation_space!=env.observation_space:
        raise BacktestError('Observación incompatible')
    if hasattr(model,'action_space') and model.action_space!=env.action_space:
        raise BacktestError('Se requiere Discrete(3)')
    balances=np.full(len(holdout),np.nan);evaluated=np.zeros(len(holdout),dtype=bool)
    balance=peak=account.initial_balance
    filled=0;trades=[];signals=[];risk_stop=False;stop_reason=None
    try:
        for start,_ in env.segments:
            balances[filled:start+1]=balance;filled=start+1
            observation,state=env.reset(options=dict(start_index=start,balance=balance,peak_balance=peak))
            while True:
                entry_index=state['index']
                entry_time=(pd.Timestamp(holdout.timestamp.iloc[entry_index])+pd.Timedelta(minutes=15)).isoformat()
                action,_=model.predict(observation,deterministic=True);value=np.asarray(action)
                if value.size!=1 or not env.action_space.contains(value.item()):
                    raise BacktestError('Acción inválida de la política')
                action=int(value.item());before=balance
                observation,reward,terminated,truncated,info=env.step(action);state=info
                balance,peak=info['balance'],info['peak_balance']
                balances[info['index']]=balance;evaluated[info['index']]=True;filled=info['index']+1
                signals.append(dict(entry_index=entry_index,expiry_index=info['index'],entry_time=entry_time,
                                    expiry_time=info['expiry_time'],action=action,outcome=info['outcome'],
                                    reward=reward,profit=info['profit'],balance_after=balance))
                if action!=1:
                    trades.append(dict(trade_id=len(trades)+1,action=action,direction='PUT' if action==0 else 'CALL',
                                       entry_index=entry_index,expiry_index=info['index'],entry_time=entry_time,
                                       expiry_time=info['expiry_time'],entry_price=info['entry_price'],
                                       settlement_price=info['settlement_price'],stake=account.stake,payout=account.payout,
                                       outcome=info['outcome'],profit=info['profit'],balance_before=before,balance_after=balance))
                if terminated or truncated:
                    risk_stop=terminated;stop_reason=info['end_reason'] if terminated else None;break
            if risk_stop:break
    finally:env.close()
    balances[filled:]=balance
    equity=pd.DataFrame(dict(timestamp=pd.to_datetime(holdout.timestamp,utc=True)+pd.Timedelta(minutes=15),
                              balance=balances,settlement_evaluated=evaluated))
    meta=dict(holdout_rows=len(holdout),eligible_transitions=sum(b-a for a,b in env.segments),
              evaluated_decisions=len(signals),skipped_gap_transitions=len(holdout)-1-sum(b-a for a,b in env.segments),
              risk_stop=bool(risk_stop),stop_reason=stop_reason,
              signal_distribution=dict(put=sum(s['action']==0 for s in signals),neutral=sum(s['action']==1 for s in signals),
                                       call=sum(s['action']==2 for s in signals)))
    return pd.DataFrame(trades,columns=TRADE_COLUMNS),equity,pd.DataFrame(signals,columns=SIGNAL_COLUMNS),meta


def run_backtest(input_path=ROOT/'data/eurusd_m15_features.parquet',*,model_path=ROOT/'models/best_ppo_agent.zip',
                 normalizer_path=None,manifest_path=None,report_dir=ROOT/'reports',plots=True):
    model_path,report_dir=Path(model_path),Path(report_dir)
    normalizer_path=Path(normalizer_path) if normalizer_path else model_path.parent/'normalizer.json'
    manifest_path=Path(manifest_path) if manifest_path else model_path.parent/'training_manifest.json'
    holdout,normalizer,manifest=load_holdout(input_path,model_path,normalizer_path,manifest_path)
    model=load_agent(model_path);_verify_hash(model_path,manifest['best_model_sha256'],'Modelo tras carga')
    trades,equity,signals,meta=simulate_holdout(model,holdout,account=AccountConfig(**manifest['account']),
                                               lookback=manifest['lookback'],normalizer=normalizer)
    result=calculate_metrics(trades,equity,initial_balance=manifest['account']['initial_balance'])
    result.update(evaluation='frozen_holdout_only',holdout=manifest['splits']['test'],
                  artifact_hashes={k:manifest[k] for k in ['input_sha256','best_model_sha256','normalizer_sha256']},**meta)
    report_dir.mkdir(parents=True,exist_ok=True)
    with FileLock(str(report_dir/'.backtest.lock'),timeout=120):
        export_reports(report_dir,result,trades,equity,signals)
        if plots:plot_reports(report_dir,equity,signals)
    LOG.info('Holdout: %d contratos; PnL %.2f',len(trades),result['net_profit'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/eurusd_m15_features.parquet')
    parser.add_argument('--model',type=Path,default=ROOT/'models/best_ppo_agent.zip')
    parser.add_argument('--normalizer',type=Path);parser.add_argument('--manifest',type=Path)
    parser.add_argument('--report-dir',type=Path,default=ROOT/'reports');parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    run_backtest(args.input,model_path=args.model,normalizer_path=args.normalizer,
                 manifest_path=args.manifest,report_dir=args.report_dir,plots=not args.no_plots)


if __name__=='__main__':main()
