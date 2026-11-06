"""Pruebas deterministas: nunca necesitan Internet ni MetaTrader."""
import sqlite3
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import data_loader as dl

NOW = pd.Timestamp('2026-11-03T12:00Z')


def candles(start='2026-11-02T10:00Z', periods=8):
    return pd.DataFrame(dict(timestamp=pd.date_range(start, periods=periods, freq='15min'),
                             open=1.10, high=1.12, low=1.09, close=1.11, volume=0.0))


def test_parquet_cache_no_reread_and_subset(tmp_path, monkeypatch):
    source, output = tmp_path/'raw.parquet', tmp_path/'cache.parquet'
    frame = candles()
    frame.to_parquet(source)
    result = dl.load_data(source='local', local=source, output=output, now=NOW)
    pd.testing.assert_frame_equal(frame, result)
    monkeypatch.setattr(dl, 'read_local', Mock(side_effect=AssertionError('Relectura')))
    result = dl.load_data(source='local', local=source, output=output, now=NOW,
                          start=frame.timestamp[2], end=frame.timestamp[5])
    assert len(result) == 3
    assert str(result.timestamp.dt.tz) == 'UTC'


def test_sqlite_epoch_filter_and_quoted_table(tmp_path):
    path = tmp_path/'mt5.sqlite'
    raw = candles().rename(columns={'timestamp':'time', 'volume':'tick_volume'})
    raw['time'] = raw.time.astype('int64') // 10**9
    raw['symbol'] = 'EURUSD'
    with sqlite3.connect(path) as con:
        raw.to_sql('rates"export', con, index=False)
    result = dl.load_data(source='local', local=path, table='rates"export',
                          symbol_column='symbol', epoch_unit='s',
                          output=tmp_path/'cache.parquet', now=NOW)
    assert len(result) == 8
    with pytest.raises(dl.DataError, match='Tabla inexistente'):
        dl.read_local(path, 'rates; DROP TABLE rates', None, 'EURUSD')


@pytest.mark.parametrize('defect', ['nan', 'negative', 'ohlc', 'duplicate', 'grid', 'infinity'])
def test_invalid_data_rejected(defect):
    raw = candles()
    if defect == 'nan': raw.loc[0, 'close'] = np.nan
    if defect == 'negative': raw.loc[0, 'volume'] = -1
    if defect == 'ohlc': raw.loc[0, 'high'] = 1.0
    if defect == 'duplicate':
        other = raw.iloc[[0]].copy()
        other['close'] = 1.105
        raw = pd.concat([raw, other])
    if defect == 'grid': raw.loc[0, 'timestamp'] += pd.Timedelta(minutes=1)
    if defect == 'infinity': raw.loc[0, 'volume'] = np.inf
    with pytest.raises(dl.DataError): dl.normalize(raw, now=NOW)


def test_exact_duplicates_incomplete_bar_timezone():
    raw = candles(start='2026-11-03T11:30Z', periods=3)
    result = dl.normalize(pd.concat([raw, raw.iloc[[0]]]), now=NOW)
    assert len(result) == 2
    raw = candles()
    raw['timestamp'] = raw.timestamp.dt.tz_convert('Europe/Helsinki').dt.tz_localize(None)
    result = dl.normalize(raw, timezone='Europe/Helsinki', now=NOW)
    assert result.timestamp.iloc[0] == candles().timestamp.iloc[0]


def test_gap_and_boundary_coverage():
    frame = candles()
    for broken in [frame.drop(3), frame.iloc[1:], frame.iloc[:-1]]:
        with pytest.raises(dl.DataError, match='Faltan'):
            dl.validate_window(broken, frame.timestamp.iloc[0], frame.timestamp.iloc[-1]+dl.STEP)
    closure = [dict(start=str(frame.timestamp[3]), end=str(frame.timestamp[4]), reason='Pausa comprobada')]
    assert len(dl.validate_window(frame.drop(3), frame.timestamp[0], frame.timestamp[7]+dl.STEP, closure)) == 7


def test_weekend_and_dst():
    assert len(dl.expected_grid('2026-11-06T22:00Z', '2026-11-08T22:00Z')) == 0
    # Domingo anterior al cambio DST de Nueva York abre a las 21 UTC.
    assert len(dl.expected_grid('2026-11-01T21:00Z', '2026-11-01T22:00Z')) == 0
    assert len(dl.expected_grid('2026-10-25T21:00Z', '2026-10-25T22:00Z')) == 4


def test_remote_cache_incremental_and_failure_preserves_file(tmp_path, monkeypatch):
    raw = candles(start='2026-11-03T10:00Z')
    output = tmp_path/'remote.parquet'
    remote = Mock(return_value=raw.iloc[:4])
    monkeypatch.setattr(dl, 'download_dukascopy', remote)
    opts = dict(start=raw.timestamp[0], output=output, publication_delay=0)
    dl.load_data(**opts, now=pd.Timestamp('2026-11-03T11:00Z'))
    dl.load_data(**opts, now=pd.Timestamp('2026-11-03T11:00Z'))
    assert remote.call_count == 1
    remote.return_value = raw
    assert len(dl.load_data(**opts, now=NOW)) == 8
    before = output.read_bytes()
    remote.return_value = raw.iloc[:-1]
    with pytest.raises(dl.DataError):
        dl.load_data(**opts, now=NOW+dl.STEP)
    assert output.read_bytes() == before


def test_local_source_changed_invalidates_cache(tmp_path):
    source, output = tmp_path/'raw.parquet', tmp_path/'cache.parquet'
    candles().iloc[:4].to_parquet(source)
    assert len(dl.load_data(source='local', local=source, output=output, now=NOW)) == 4
    candles().to_parquet(source)
    assert len(dl.load_data(source='local', local=source, output=output, now=NOW)) == 8


def test_remote_has_no_60_day_restriction(tmp_path, monkeypatch):
    raw = candles(start='2018-01-02T10:00Z')
    remote = Mock(return_value=raw)
    monkeypatch.setattr(dl, 'download_dukascopy', remote)
    result = dl.load_data(start=raw.timestamp.iloc[0], end=raw.timestamp.iloc[-1]+dl.STEP,
                          output=tmp_path/'x.parquet', now=NOW)
    assert len(result) == 8
    assert remote.call_args.args[0].year == 2018


def test_numeric_timestamp_requires_unit():
    raw = candles()
    raw['timestamp'] = raw.timestamp.astype('int64')//10**9
    with pytest.raises(dl.DataError, match='epoch_unit'):
        dl.normalize(raw, now=NOW)


def test_monthly_checkpoints_resume_without_network(tmp_path, monkeypatch):
    frame = candles()
    fetch = Mock(return_value=frame)
    monkeypatch.setattr(dl, 'fetch_dukascopy_chunk', fetch)
    kwargs = dict(cache_dir=tmp_path/'chunks', now=NOW)
    first = dl.download_dukascopy(frame.timestamp.iloc[0], frame.timestamp.iloc[-1]+dl.STEP, **kwargs)
    fetch.side_effect = AssertionError('No red al reanudar')
    second = dl.download_dukascopy(frame.timestamp.iloc[0], frame.timestamp.iloc[-1]+dl.STEP, **kwargs)
    pd.testing.assert_frame_equal(first, second)
    assert fetch.call_count == 1


def test_five_year_default_request(tmp_path, monkeypatch):
    remote = Mock(side_effect=RuntimeError('captured'))
    monkeypatch.setattr(dl, 'download_dukascopy', remote)
    with pytest.raises(RuntimeError, match='captured'):
        dl.load_data(output=tmp_path/'x.parquet', now=NOW)
    lo, hi = remote.call_args.args
    assert lo == hi - pd.DateOffset(years=5)


def test_native_dukascopy_api_contract(monkeypatch):
    import json
    from types import SimpleNamespace
    raw = candles().iloc[0]
    row = [int(raw.timestamp.timestamp()*1000)] + raw[dl.COLUMNS[1:]].tolist()
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.get.side_effect = [SimpleNamespace(text='forex_history('+json.dumps([row])+');'),
                               SimpleNamespace(text='forex_history([]);')]
    monkeypatch.setattr(dl, 'dukascopy_session', lambda: session)
    result = dl.fetch_dukascopy_chunk(raw.timestamp, raw.timestamp+2*dl.STEP)
    assert len(result) == 1
    assert session.get.call_args.kwargs['params']['interval'] == '15MIN'
    assert session.get.call_args.kwargs['params']['last_update'] == str(int((raw.timestamp+dl.STEP).timestamp()*1000))
    assert str(result.timestamp.dt.tz) == 'UTC'


def test_dukascopy_rejects_malformed_or_nonadvancing_pages(monkeypatch):
    from types import SimpleNamespace
    import json
    start = pd.Timestamp('2018-01-03T10:00Z')
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(dl, 'dukascopy_session', lambda: session)
    for text in ['<html>unavailable</html>', 'forex_history({"error":"failed"});',
                 'forex_history('+json.dumps([[int((start-dl.STEP).timestamp()*1000),1,1,1,1,0]])+');']:
        session.get.return_value = SimpleNamespace(text=text)
        with pytest.raises(dl.DataError):
            dl.fetch_dukascopy_chunk(start, start+dl.STEP)


def test_live_2018_download_when_explicitly_enabled(tmp_path):
    """Opt-in: FOREX_LIVE_TESTS=1 python -m pytest -q -k live_2018."""
    import os
    if os.environ.get('FOREX_LIVE_TESTS') != '1':
        pytest.skip('Requiere habilitar consulta HTTP real explícitamente')
    frame = dl.load_data(start='2018-01-03T10:00Z', end='2018-01-03T12:00Z',
                          output=tmp_path/'eurusd_2018.parquet')
    assert len(frame) == 8
    assert frame.timestamp.iloc[0] == pd.Timestamp('2018-01-03T10:00Z')
