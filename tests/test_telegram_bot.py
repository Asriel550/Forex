"""Inferencia incremental, frescura, secretos y outbox sin envíos reales."""
import json
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
import requests
from src import telegram_bot as tb
from src.environment import BinaryOptionsEnv,AccountConfig,fit_normalizer,prepare_market_data
from src.quant_engine import QuantConfig,enrich_features
from src.train import train


def history(n=850):
    close=1.1*np.exp(np.cumsum(np.random.default_rng(77).normal(0,.0002,n)))
    return pd.DataFrame(dict(timestamp=pd.date_range('2020-01-06',periods=n,freq='15min',tz='UTC'),
                             open=close,high=close+.0001,low=close-.0001,close=close,volume=0.))


def test_incremental_garch_matches_full_engine_across_refits():
    raw=history(900);config=QuantConfig(window=384,min_observations=128,refit_every=96)
    base=enrich_features(raw.iloc[:700],config)
    incremental=tb.extend_quant_history(base,raw.iloc[699:])
    full=enrich_features(raw,config)
    names=['cond_volatility','dist_p95','dist_p05','mc_p05','mc_p95']
    np.testing.assert_allclose(incremental[names],full[names],rtol=1e-12,atol=1e-14,equal_nan=True)
    assert incremental.attrs['source_rows']==900


def test_latest_observation_matches_environment_reset_without_future_inference():
    frame=enrich_features(history(),QuantConfig(window=384,min_observations=128))
    ready=prepare_market_data(frame,lookback=2,indicator_period=2)
    normalizer=fit_normalizer(ready.iloc[:200],2)
    manifest={'lookback':2,'account':dict(initial_balance=1000.,stake=10.,payout=.8,
               tie_policy='refund',inactivity_penalty=0.,max_drawdown=1.)}
    prefix=ready.iloc[:300]
    obs=tb.latest_observation(prefix,normalizer,manifest)
    env=BinaryOptionsEnv(ready,prepared=True,lookback=2,normalizer=normalizer)
    expected,_=env.reset(options={'start_index':299})
    np.testing.assert_array_equal(obs,expected)


@pytest.mark.parametrize('clock',['2026-11-06T10:04Z','2026-11-06T10:15Z','2026-11-06T09:59Z'])
def test_stale_or_incomplete_candle_is_rejected(clock):
    with pytest.raises(tb.SignalSkipped):tb.assert_fresh('2026-11-06T09:45Z',clock)


def test_message_format():
    row=pd.Series(dict(timestamp=pd.Timestamp('2026-11-06T09:45Z'),cond_volatility=.0018,mc_p05=1.082,mc_p95=1.091))
    message=tb.format_signal(2,row)
    assert message.startswith('🔔 SEÑAL EUR/USD M15\nAcción: CALL (Comprar)')
    assert 'P05: 1.0820 | P95: 1.0910' in message


def api_response(status,payload,headers=None):
    r=Mock();r.status_code=status;r.json.return_value=payload;r.headers=headers or {};return r


def test_telegram_retries_explicit_429_only(monkeypatch):
    client=Mock();client.post.side_effect=[api_response(429,{'ok':False,'parameters':{'retry_after':2}}),
                                         api_response(200,{'ok':True,'result':{'message_id':7}})]
    waits=[];monkeypatch.setattr(tb,'wait_seconds',waits.append)
    assert tb.send_telegram('123:secret','-1001','hello',session=client)==7
    assert waits==[2.] and client.post.call_count==2


def test_ambiguous_timeout_is_not_retried_or_leaked():
    client=Mock();client.post.side_effect=requests.ReadTimeout('https://api.telegram.org/botSECRET/sendMessage')
    with pytest.raises(tb.AlertError) as error:tb.send_telegram('SECRET','1','hi',session=client)
    assert 'SECRET' not in str(error.value) and client.post.call_count==1


def test_credentials_error_is_redacted():
    client=Mock();client.post.return_value=api_response(401,{'ok':False,'description':'SECRET'})
    with pytest.raises(tb.TelegramRejected) as error:tb.send_telegram('SECRET','1','hi',session=client)
    assert 'SECRET' not in str(error.value)


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root=tmp_path_factory.mktemp('telegram-bundle')
    raw=history();base=enrich_features(raw.iloc[:800],QuantConfig(window=384,min_observations=128))
    path=root/'features.parquet';base.to_parquet(path,index=False)
    train(path,output_dir=root/'models',total_timesteps=16,n_steps=16,batch_size=8,n_epochs=1,
          lookback=2,indicator_period=2,eval_freq=16,verbose=0)
    return root,path,raw


def test_dry_run_and_outbox_deduplication(bundle,tmp_path,monkeypatch):
    root,path,raw=bundle
    clock=raw.timestamp.iloc[-1]+pd.Timedelta(minutes=16)
    monkeypatch.setattr(tb,'load_data',lambda **kwargs:raw.iloc[799:].copy())
    monkeypatch.setattr(tb,'current_utc',lambda:clock)
    kwargs=dict(model_dir=root/'models',history_path=path,state_path=tmp_path/'state.json',now=clock)
    preview=tb.run_signal(**kwargs,dry_run=True)
    assert preview['status']=='dry_run' and not (tmp_path/'state.json').exists()
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','TEST_SECRET');monkeypatch.setenv('TELEGRAM_CHAT_ID','-1')
    sender=Mock(return_value=42);monkeypatch.setattr(tb,'send_telegram',sender)
    assert tb.run_signal(**kwargs)['status']=='sent'
    with pytest.raises(tb.SignalSkipped):tb.run_signal(**kwargs)
    assert sender.call_count==1
    saved=json.loads((tmp_path/'state.json').read_text())
    assert saved['status']=='sent' and 'TEST_SECRET' not in json.dumps(saved)


def test_uncertain_delivery_remains_pending_and_does_not_repeat(bundle,tmp_path,monkeypatch):
    root,path,raw=bundle;clock=raw.timestamp.iloc[-1]+pd.Timedelta(minutes=16)
    monkeypatch.setattr(tb,'load_data',lambda **kwargs:raw.iloc[799:].copy())
    monkeypatch.setattr(tb,'current_utc',lambda:clock)
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','TEST_SECRET');monkeypatch.setenv('TELEGRAM_CHAT_ID','-1')
    sender=Mock(side_effect=tb.AlertError('Entrega incierta'));monkeypatch.setattr(tb,'send_telegram',sender)
    kwargs=dict(model_dir=root/'models',history_path=path,state_path=tmp_path/'state.json',now=clock)
    with pytest.raises(tb.AlertError):tb.run_signal(**kwargs)
    assert json.loads((tmp_path/'state.json').read_text())['status']=='pending'
    with pytest.raises(tb.SignalSkipped):tb.run_signal(**kwargs)
    assert sender.call_count==1



def test_truncated_warmup_cannot_silently_shorten_refit_window():
    raw=history(900)
    config=QuantConfig(window=1000,min_observations=128,refit_every=96)
    baseline=enrich_features(raw.iloc[:700],config).dropna(subset=['cond_volatility']).reset_index(drop=True)
    # source_rows conserva el origen real; la ventana requerida en 704 tiene
    # más retornos de los que quedaron después de quitar el warmup.
    with pytest.raises(tb.AlertError,match='Contexto histórico insuficiente'):
        tb.extend_quant_history(baseline,raw.iloc[699:])



def test_static_history_cannot_emit_stale_alert(bundle,tmp_path,monkeypatch):
    root,path,raw=bundle
    clock=raw.timestamp.iloc[-1]+pd.Timedelta(minutes=16)
    monkeypatch.setattr(tb,'load_data',lambda **kwargs:raw.iloc[:800].copy())
    sender=Mock();monkeypatch.setattr(tb,'send_telegram',sender)
    model=Mock();monkeypatch.setattr(tb,'load_agent',model)
    with pytest.raises(tb.SignalSkipped):
        tb.run_signal(model_dir=root/'models',history_path=path,now=clock,
                      state_path=tmp_path/'state.json',input_dir=tmp_path/'files')
    sender.assert_not_called();model.assert_not_called()
    assert not (tmp_path/'state.json').exists()
