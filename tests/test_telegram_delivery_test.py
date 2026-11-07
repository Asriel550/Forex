"""Prueba técnica a través del sender/outbox real, HTTP simulado sin secretos."""
import json
from unittest.mock import Mock
import pytest
import requests
from src import telegram_bot as tb


def response(status=200,body=None):
    r=Mock();r.status_code=status;r.headers={}
    r.json.return_value=body or {'ok':True,'result':{'message_id':321}}
    return r


@pytest.fixture
def secrets(monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN','TEST_TOKEN_ONLY')
    monkeypatch.setenv('TELEGRAM_CHAT_ID','TEST_CHAT_ONLY')


def test_send_and_duplicate_use_shared_outbox_without_market_or_model(tmp_path,monkeypatch,secrets):
    monkeypatch.setattr(tb,'load_bundle',Mock(side_effect=AssertionError('No model')))
    monkeypatch.setattr(tb,'load_data',Mock(side_effect=AssertionError('No market')))
    client=Mock();client.post.return_value=response()
    result=tb.run_delivery_test('mobile-v1',state_dir=tmp_path,session=client)
    assert result['status']=='sent' and result['message_id']==321
    message=client.post.call_args.kwargs['json']['text']
    assert 'PRUEBA' in message and 'no contiene una señal' in message
    assert tb.run_delivery_test('mobile-v1',state_dir=tmp_path,session=client)['status']=='already_sent'
    assert client.post.call_count==1
    # Other IDs do not overwrite the first ID's idempotency record.
    tb.run_delivery_test('mobile-v2',state_dir=tmp_path,session=client)
    tb.run_delivery_test('mobile-v1',state_dir=tmp_path,session=client)
    assert client.post.call_count==2
    assert all('TEST_TOKEN_ONLY' not in p.read_text() and 'TEST_CHAT_ONLY' not in p.read_text() for p in tmp_path.glob('*.json'))


@pytest.mark.parametrize('failure',['timeout','5xx','malformed'])
def test_uncertain_delivery_stays_pending_and_never_reposts(tmp_path,secrets,failure):
    client=Mock()
    if failure=='timeout':client.post.side_effect=requests.Timeout('secret-in-exception')
    elif failure=='5xx':client.post.return_value=response(503,{'ok':False})
    else:client.post.return_value=response(200,{'nonsense':True})
    with pytest.raises(tb.AlertError):tb.run_delivery_test('same',state_dir=tmp_path,session=client)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['status']=='pending'
    with pytest.raises(tb.AlertError,match='incierta'):tb.run_delivery_test('same',state_dir=tmp_path,session=client)
    assert client.post.call_count==1


def test_explicit_rejection_can_be_retried_and_429_uses_existing_backoff(tmp_path,monkeypatch,secrets):
    client=Mock();client.post.return_value=response(401,{'ok':False})
    with pytest.raises(tb.TelegramRejected):tb.run_delivery_test('auth',state_dir=tmp_path,session=client)
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['status']=='rejected'
    waits=[];monkeypatch.setattr(tb,'wait_seconds',waits.append)
    client.post.side_effect=[response(429,{'ok':False,'parameters':{'retry_after':2}}),response()]
    assert tb.run_delivery_test('auth',state_dir=tmp_path,session=client)['status']=='sent'
    assert waits==[2.]


@pytest.mark.parametrize('test_id',['','x'*65,'$(echo secret)','../outbox','a b'])
def test_test_id_is_validated_before_http(test_id,tmp_path,secrets):
    client=Mock()
    with pytest.raises(tb.AlertError):tb.run_delivery_test(test_id,state_dir=tmp_path,session=client)
    client.post.assert_not_called()


def test_missing_secret_is_not_an_http_request(tmp_path,monkeypatch):
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN',raising=False)
    client=Mock()
    with pytest.raises(tb.TelegramRejected):tb.run_delivery_test('auth',state_dir=tmp_path,session=client)
    client.post.assert_not_called()
