"""Lectura estática HistData, validación M15, agregación M1 y caché sin red."""
import os
import stat
import zipfile
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
import requests
from src import data_loader as dl

NOW=pd.Timestamp('2026-11-06T12:00Z')

def candles(n=4,start='2024-01-02T05:00Z',freq='15min'):
    return pd.DataFrame(dict(timestamp=pd.date_range(start,periods=n,freq=freq),open=1.10,high=1.12,low=1.09,close=1.11,volume=0.))

def csv_file(path,frame):
    frame.to_csv(path,index=False);return path

def ascii_text(frame):
    frame=frame.copy();frame['timestamp']=frame.timestamp.dt.tz_convert(dl.HISTDATA_TZ).dt.strftime('%Y%m%d %H%M%S')
    return frame.to_csv(index=False,header=False,sep=';')

def load(path,folder,**kw):
    return dl.load_data(inputs=path,output=folder/'cache.parquet',now=NOW,**kw)


def test_csv_utc_contract_without_network(tmp_path,monkeypatch):
    original=candles();path=csv_file(tmp_path/'history.csv',original)
    monkeypatch.setattr(requests.Session,'request',Mock(side_effect=AssertionError('No HTTP')))
    result=load(path,tmp_path)
    pd.testing.assert_frame_equal(result,original)
    assert list(result)==dl.COLUMNS and str(result.timestamp.dt.tz)=='UTC'

@pytest.mark.parametrize('stamp',['20240102 120000','20240603 120000'])
def test_histdata_uses_fixed_est_in_winter_and_summer(tmp_path,stamp):
    path=tmp_path/'history.csv';path.write_text(f'{stamp};1.1;1.2;1.0;1.15;0\n')
    assert load(path,tmp_path).timestamp.iloc[0].hour==17


def test_zip_multiple_parts_ignores_text(tmp_path):
    frame=candles();path=tmp_path/'data.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('a.csv',ascii_text(frame.iloc[:2]));z.writestr('b.csv',ascii_text(frame.iloc[2:]));z.writestr('notes.txt','metadata')
    pd.testing.assert_frame_equal(load(path,tmp_path),frame)


def test_bom_header_and_naive_utc_override(tmp_path):
    frame=candles();frame['timestamp']=frame.timestamp.dt.tz_localize(None)
    frame.columns=[c.upper() for c in frame.columns]
    path=tmp_path/'history.CSV';frame.to_csv(path,index=False,encoding='utf-8-sig')
    result=load(path,tmp_path,timezone='UTC')
    assert result.timestamp.iloc[0]==pd.Timestamp('2024-01-02T05:00Z')


def test_separate_date_and_time(tmp_path):
    path=tmp_path/'history.csv';path.write_text('Date,Time,Open,High,Low,Close,Volume\n2024.01.02,00:00,1.1,1.2,1.0,1.15,0\n')
    assert load(path,tmp_path).timestamp.iloc[0]==pd.Timestamp('2024-01-02T05:00Z')


def test_native_m1_auto_aggregates(tmp_path):
    frame=candles(n=30,freq='1min');frame['volume']=1.
    frame.loc[0,'open']=1.105;frame.loc[14,'close']=1.115;frame.loc[3,'high']=1.13;frame.loc[4,'low']=1.08
    path=tmp_path/'DAT_ASCII_EURUSD_M1_202401.csv';path.write_text(ascii_text(frame))
    result=load(path,tmp_path)
    assert len(result)==2 and result.volume.tolist()==[15.,15.]
    assert result[['open','high','low','close']].iloc[0].tolist()==[1.105,1.13,1.08,1.115]


def test_m1_spanning_files_and_duplicate_does_not_double_count(tmp_path):
    frame=candles(n=15,freq='1min');frame['volume']=1.
    paths=[csv_file(tmp_path/'a.csv',frame.iloc[:10]),csv_file(tmp_path/'b.csv',frame.iloc[9:])]
    result=load(paths,tmp_path,input_timeframe='M1')
    assert len(result)==1 and result.volume.iloc[0]==15


def test_partial_m1_is_not_filled(tmp_path):
    path=csv_file(tmp_path/'m1.csv',candles(n=15,freq='1min').drop(7))
    with pytest.raises(dl.DataError,match='15 minutos'):load(path,tmp_path,input_timeframe='M1')


def test_renamed_m1_needs_explicit_timeframe(tmp_path):
    path=csv_file(tmp_path/'unknown.csv',candles(n=15,freq='1min'))
    with pytest.raises(dl.DataError,match='alineados'):load(path,tmp_path)
    assert len(load(path,tmp_path,input_timeframe='M1'))==1


def test_named_resolution_instrument_and_mixed_timeframes(tmp_path):
    m1=tmp_path/'DAT_ASCII_EURUSD_M1_202401.csv';m1.write_text(ascii_text(candles(n=15,freq='1min')))
    with pytest.raises(dl.DataError,match='incompatible'):load(m1,tmp_path,input_timeframe='M15')
    gbp=tmp_path/'DAT_ASCII_GBPUSD_M1_202401.csv';gbp.write_text(m1.read_text())
    with pytest.raises(dl.DataError,match='Instrumento'):load(gbp,tmp_path)
    m15=csv_file(tmp_path/'m15.csv',candles())
    with pytest.raises(dl.DataError,match='mezclar'):load([m1,m15],tmp_path)

@pytest.mark.parametrize('bad',['nan','inf','negative','ohlc','grid','duplicate'])
def test_invalid_data_rejected(tmp_path,bad):
    frame=candles()
    if bad=='nan':frame.loc[1,'close']=np.nan
    if bad=='inf':frame.loc[1,'volume']=np.inf
    if bad=='negative':frame.loc[1,'volume']=-1
    if bad=='ohlc':frame.loc[1,'high']=1.0
    if bad=='grid':frame.loc[1,'timestamp']+=pd.Timedelta(minutes=1)
    if bad=='duplicate':
        extra=frame.iloc[[0]].copy();extra['close']=1.105;frame=pd.concat([frame,extra])
    with pytest.raises(dl.DataError):load(csv_file(tmp_path/'bad.csv',frame),tmp_path)


def test_identical_duplicates_and_unsorted_input(tmp_path):
    frame=candles();raw=pd.concat([frame.iloc[::-1],frame.iloc[[0]]])
    pd.testing.assert_frame_equal(load(csv_file(tmp_path/'history.csv',raw),tmp_path),frame)


def test_internal_gap_and_documented_closure(tmp_path):
    frame=candles();path=csv_file(tmp_path/'gap.csv',frame.drop(1))
    with pytest.raises(dl.DataError,match='Faltan'):load(path,tmp_path)
    closure=[dict(start=frame.timestamp.iloc[1].isoformat(),end=frame.timestamp.iloc[2].isoformat(),reason='Pausa comprobada')]
    assert len(load(path,tmp_path,closures=closure))==3


def test_weekend_and_market_calendar_dst():
    frame=candles(n=2);frame['timestamp']=pd.to_datetime(['2024-01-05T21:45Z','2024-01-07T22:00Z'])
    assert len(dl.validate_window(frame,frame.timestamp.iloc[0],frame.timestamp.iloc[-1]+dl.STEP))==2
    assert len(dl.expected_grid('2024-06-02T21:00Z','2024-06-02T22:00Z'))==4


def test_cache_avoids_csv_reparse(tmp_path,monkeypatch):
    frame=candles();path=csv_file(tmp_path/'m15.csv',frame);load(path,tmp_path)
    monkeypatch.setattr(dl,'_read_csv',Mock(side_effect=AssertionError('No CSV')))
    assert len(load(path,tmp_path,start=frame.timestamp.iloc[1],end=frame.timestamp.iloc[3]))==2


def test_sha_detects_source_change_with_same_mtime(tmp_path):
    path=csv_file(tmp_path/'history.csv',candles());load(path,tmp_path);old=path.stat()
    csv_file(path,candles(n=5));os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
    assert len(load(path,tmp_path))==5


def test_failed_update_keeps_previous_cache(tmp_path):
    path=csv_file(tmp_path/'history.csv',candles());load(path,tmp_path)
    before=(tmp_path/'cache.parquet').read_bytes();csv_file(path,candles().drop(1))
    with pytest.raises(dl.DataError):load(path,tmp_path)
    assert (tmp_path/'cache.parquet').read_bytes()==before


def test_cache_expires_when_a_previously_open_bar_closes(tmp_path):
    path=csv_file(tmp_path/'current.csv',candles(n=3,start='2026-11-06T10:00Z'))
    options=dict(inputs=path,output=tmp_path/'cache.parquet')
    assert len(dl.load_data(**options,now=pd.Timestamp('2026-11-06T10:30Z')))==2
    assert len(dl.load_data(**options,now=pd.Timestamp('2026-11-06T10:45Z')))==3

@pytest.mark.parametrize('bounds',[dict(start='2024-01-02T04:45Z'),dict(end='2024-01-02T06:15Z')])
def test_query_requires_boundary_coverage(tmp_path,bounds):
    path=csv_file(tmp_path/'history.csv',candles())
    with pytest.raises(dl.DataError,match='Faltan'):load(path,tmp_path,**bounds)

@pytest.mark.parametrize('name',['../evil.csv','/evil.csv','C:/evil.csv'])
def test_unsafe_zip_path_never_extracted(tmp_path,name):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr(name,ascii_text(candles()))
    with pytest.raises(dl.DataError,match='insegura'):load(path,tmp_path)
    assert not (tmp_path.parent/'evil.csv').exists()


def test_zip_symlink_rejected(tmp_path):
    path=tmp_path/'bad.zip';entry=zipfile.ZipInfo('prices.csv');entry.create_system=3;entry.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(path,'w') as z:z.writestr(entry,'/etc/passwd')
    with pytest.raises(dl.DataError,match='insegura'):load(path,tmp_path)


def test_zip_limits_and_corrupt_archive(tmp_path):
    path=tmp_path/'bad.zip';path.write_bytes(b'not zip')
    with pytest.raises(dl.DataError):load(path,tmp_path)
    with zipfile.ZipFile(path,'w') as z:z.writestr('a.csv',ascii_text(candles()));z.writestr('b.txt','x')
    with pytest.raises(dl.DataError,match='límite'):load(path,tmp_path,max_bytes=10)
    with pytest.raises(dl.DataError,match='miembros'):load(path,tmp_path,max_members=1)
    with zipfile.ZipFile(path,'w') as z:z.writestr('readme.txt','no data')
    with pytest.raises(dl.DataError,match='sin CSV'):load(path,tmp_path)


def test_ticks_urls_remote_modes_and_missing_sources_rejected(tmp_path):
    tick=tmp_path/'ticks.csv';tick.write_text('20240102 000000123,1.1,1.2,0\n')
    with pytest.raises(dl.DataError):load(tick,tmp_path)
    with pytest.raises(dl.DataError,match='URLs'):load('https://example.test/a.csv',tmp_path)
    with pytest.raises(dl.DataError,match='remota'):load('unused',tmp_path,source='dukascopy')
    with pytest.raises(dl.DataError,match='inexistente'):load(tmp_path/'absent.csv',tmp_path)
    folder=tmp_path/'empty';folder.mkdir()
    with pytest.raises(dl.DataError,match='No hay'):load(folder,tmp_path)


def test_parquet_epoch_directory_and_no_source_overwrite(tmp_path):
    path=tmp_path/'raw.parquet';candles().to_parquet(path,index=False)
    assert len(load(path,tmp_path,source='local'))==4
    raw=candles();raw['timestamp']=raw.timestamp.astype('int64')//10**9
    epoch=csv_file(tmp_path/'epoch.csv',raw)
    assert len(load(epoch,tmp_path,epoch_unit='s'))==4
    with pytest.raises(dl.DataError):load(epoch,tmp_path,force=True)
    folder=tmp_path/'parts';folder.mkdir();csv_file(folder/'a.csv',candles().iloc[:2]);csv_file(folder/'b.csv',candles().iloc[2:])
    assert len(dl.load_data(input_dir=folder,output=tmp_path/'dir.parquet',now=NOW))==4
    with pytest.raises(dl.DataError,match='sobrescribir'):dl.load_data(inputs=path,output=path,now=NOW)


def test_static_csv_integrates_with_garch_and_prepare_market_data(tmp_path):
    from src.quant_engine import QuantConfig, enrich_features, FEATURES
    from src.environment import prepare_market_data
    frame=candles(350,start='2024-01-02T00:00Z')
    close=1.1*np.exp(np.cumsum(np.random.default_rng(23).normal(0,.0002,len(frame))))
    frame['open']=close;frame['close']=close;frame['high']=close+.0001;frame['low']=close-.0001
    path=csv_file(tmp_path/'history.csv',frame)
    loaded=load(path,tmp_path)
    enriched=enrich_features(loaded,QuantConfig(window=256,min_observations=128))
    prepared=prepare_market_data(enriched)
    assert len(prepared)>100 and str(prepared.timestamp.dt.tz)=='UTC'
    assert np.isfinite(prepared[FEATURES].to_numpy()).all()
    np.testing.assert_allclose(enriched.close,frame.close)
