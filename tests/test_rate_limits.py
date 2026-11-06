"""Reintentos HTTP sin red ni esperas reales."""
import json
from unittest.mock import Mock
import pandas as pd
import pytest
import requests
from src import data_loader as dl


def response(status,headers=None):
    r=requests.Response();r.status_code=status;r.url='https://example.test';r._content_consumed=True
    r.headers.update(headers or {})
    return r


def test_exponential_backoff_and_retry_after(monkeypatch):
    sleeps=[];responses=iter([response(429,{'Retry-After':'5'}),response(503),response(200)])
    monkeypatch.setattr(requests.Session,'request',lambda *a,**k:next(responses))
    monkeypatch.setattr(dl,'wait_seconds',sleeps.append)
    policy=dl.DownloadPolicy(request_pause=0,jitter=0)
    with dl.dukascopy_session(policy) as session:
        assert session.get('https://example.test').status_code==200
    assert [s for s in sleeps if s] == [5.,4.]


def test_retry_after_http_date():
    assert dl.retry_after_seconds('Fri, 06 Nov 2026 10:00:30 GMT',pd.Timestamp('2026-11-06T10:00Z'))==30
    assert dl.retry_after_seconds('invalid')==0
    assert dl.retry_after_seconds('nan')==0


def test_large_retry_after_aborts_instead_of_retrying_early(monkeypatch):
    call=Mock(return_value=response(429,{'Retry-After':'9999'}))
    monkeypatch.setattr(requests.Session,'request',call)
    monkeypatch.setattr(dl,'wait_seconds',lambda n:None)
    with dl.dukascopy_session() as session,pytest.raises(dl.DataError,match='presupuesto'):
        session.get('https://example.test')
    assert call.call_count==1


def test_permanent_403_is_not_retried(monkeypatch):
    call=Mock(return_value=response(403));monkeypatch.setattr(requests.Session,'request',call)
    monkeypatch.setattr(dl,'wait_seconds',lambda n:None)
    with dl.dukascopy_session() as session,pytest.raises(dl.DataError):session.get('https://example.test')
    assert call.call_count==1


def test_max_attempts_is_one_budget(monkeypatch):
    call=Mock(return_value=response(429));monkeypatch.setattr(requests.Session,'request',call)
    monkeypatch.setattr(dl,'wait_seconds',lambda n:None)
    with dl.dukascopy_session(dl.DownloadPolicy(max_attempts=3,jitter=0)) as session,pytest.raises(dl.DataError):
        session.get('https://example.test')
    assert call.call_count==3


def test_small_pages_and_cursor_progress(monkeypatch):
    start=pd.Timestamp('2020-01-06T00:00Z');end=start+pd.Timedelta(days=2)
    calls=[]
    def get(url,params,headers):
        calls.append(params)
        cursor=int(params['last_update'])
        rows=[[cursor+i*900000,1.1,1.2,1.0,1.1,0.] for i in range(int(params['limit']))]
        result=Mock();result.text='forex_history('+json.dumps(rows)+');';return result
    session=Mock();session.get.side_effect=get
    session.__enter__=Mock(return_value=session);session.__exit__=Mock(return_value=False)
    monkeypatch.setattr(dl,'dukascopy_session',lambda:session)
    result=dl.fetch_dukascopy_chunk(start,end)
    assert len(result)==192 and len(calls)==2
    assert all(int(call['limit'])<=96 for call in calls)
    assert int(calls[1]['last_update'])>int(calls[0]['last_update'])
