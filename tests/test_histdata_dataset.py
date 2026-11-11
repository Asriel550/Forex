"""Auditoría del snapshot real y de su transformación reproducible sin red."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scripts.build_histdata_dataset import build
from src import data_loader as dl

ROOT=Path(__file__).resolve().parents[1]


def test_calendar_exclusions_match_naive_mask_with_overlaps_and_outside_bounds():
    start,end='2025-01-06T00:00Z','2025-01-07T00:00Z'
    grid=dl.expected_grid(start,end)
    closures=[dict(start='2025-01-05T23:30Z',end='2025-01-06T00:30Z',reason='test'),
              dict(start='2025-01-06T00:15Z',end='2025-01-06T01:00Z',reason='overlap'),
              dict(start='2025-01-07T00:00Z',end='2025-01-07T00:15Z',reason='outside')]
    mask=np.ones(len(grid),dtype=bool)
    for item in closures:mask &= ~((grid>=pd.Timestamp(item['start']))&(grid<pd.Timestamp(item['end'])))
    pd.testing.assert_index_equal(dl.expected_grid(start,end,closures),grid[mask])


def test_build_excludes_conflicting_minute_and_whole_incomplete_m15(tmp_path,monkeypatch):
    import scripts.build_histdata_dataset as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    raw_dir=tmp_path/'raw';raw_dir.mkdir();dest=tmp_path/'datasets/eurusd'
    ts=pd.date_range('2025-01-06 00:00',periods=60,freq='min')
    frame=pd.DataFrame(dict(timestamp=ts.strftime('%Y%m%d %H%M%S'),open=1.1,high=1.2,low=1.,close=1.11,volume=0))
    conflict=frame.iloc[[20]].copy();conflict['close']=1.12
    # Primer grupo duplicado exacto; segundo con conflicto; tercero incompleto.
    raw=pd.concat([frame.drop(index=35),frame.iloc[[0]],conflict],ignore_index=True)
    p=raw_dir/'official.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('DAT_ASCII_EURUSD_M1_2025.csv',raw.to_csv(sep=';',index=False,header=False))
    source=tmp_path/'sources.json';source.write_text(json.dumps(dict(acquired_at='2026-11-06T00:00:00Z',archives=[dict(period='2025',zip_file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())])))
    report=build(raw_dir,dest,source)
    assert report['rows']==2 and report['incomplete_m15_groups']==2
    assert report['source_audit'][0]['exact_duplicate_rows_removed']==1
    assert len(report['source_audit'][0]['conflicting_m1_timestamps_excluded'])==1
    cleaned=dl.load_data(input_dir=dest,output=tmp_path/'out.parquet',closures=dl.read_closures(tmp_path/'config/closures.json'))
    assert cleaned.timestamp.dt.minute.tolist()==[0,45]
    assert (cleaned.close==1.11).all()
