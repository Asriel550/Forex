"""Build audited, complete M15 bars from previously acquired HistData ZIPs.

No prices are imputed. Only groups with all 15 distinct M1 timestamps survive.
Observed source gaps are documented as exclusions, never as market closures.
"""
from pathlib import Path
import argparse,sys,json,hashlib,zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data_loader import normalize, expected_grid, STEP, COLUMNS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read_native_minutes(raw_dir, archives):
    """Verify pinned archives and remove only duplicate or conflicting M1 observations."""
    frames=[];audit=[]
    for entry in archives:
        path=raw_dir/entry['zip_file']
        if sha(path)!=entry['sha256']:raise ValueError('Source hash mismatch')
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.lower().endswith('.csv')]
            if len(names)!=1:raise ValueError('Expected one official CSV')
            with z.open(names[0]) as source:
                raw=pd.read_csv(source,sep=';',header=None,names=COLUMNS,dtype=str)
        unique=raw.drop_duplicates()
        conflicts=unique.loc[unique.timestamp.duplicated(keep=False),'timestamp'].unique().tolist()
        clean=unique.loc[~unique.timestamp.isin(conflicts)]
        frame=normalize(clean,timezone='EST',input_timeframe='M1')
        audit.append(dict(period=entry['period'],source_rows=len(raw),valid_unique_m1=len(frame),
                          exact_duplicate_rows_removed=len(raw)-len(unique),
                          conflicting_m1_timestamps_excluded=conflicts))
        print(entry['period'],len(frame),'valid M1; conflicts',len(conflicts),flush=True)
        frames.append(frame)
    merged=pd.concat(frames,ignore_index=True).drop_duplicates().sort_values('timestamp')
    if merged.timestamp.duplicated().any():raise ValueError('Conflicting source timestamps')
    return merged.reset_index(drop=True), audit


def build(raw_dir,destination,manifest_path):
    destination.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads(manifest_path.read_text())
    merged,audit=read_native_minutes(raw_dir,source_manifest['archives'])
    grouped=merged.groupby(merged.timestamp.dt.floor('15min'),sort=True)
    counts=grouped.size()
    if (counts>15).any():raise ValueError('Invalid M1 group size')
    full=grouped.agg(open=('open','first'),high=('high','max'),low=('low','min'),
                     close=('close','last'),volume=('volume','sum')).rename_axis('timestamp').reset_index()
    full=full.loc[full.timestamp.isin(counts[counts==15].index)].reset_index(drop=True)
    full=normalize(full)
    # Exempt only specifically observed gaps in the provider's source files.
    expected=expected_grid(full.timestamp.min(),full.timestamp.max()+STEP)
    missing=expected.difference(pd.DatetimeIndex(full.timestamp))
    exemptions=[]
    for stamp in missing:
        count=int(counts.get(stamp,0))
        if exemptions and pd.Timestamp(exemptions[-1]['end'])==stamp and exemptions[-1]['source_year']==stamp.year:
            item=exemptions[-1];item['end']=(stamp+STEP).isoformat()
            item['missing_m15_bars']+=1;item['observed_m1_min']=min(item['observed_m1_min'],count)
            item['observed_m1_max']=max(item['observed_m1_max'],count)
        else:
            exemptions.append(dict(start=stamp.isoformat(),end=(stamp+STEP).isoformat(),
                reason='HistData gap/conflict: excluded; see datasets/eurusd/manifest.json',
                kind='observed_source_gap',source_year=stamp.year,missing_m15_bars=1,
                observed_m1_min=count,observed_m1_max=count))
    assets=[]
    for year,frame in full.groupby(full.timestamp.dt.year):
        member=f'EURUSD_M15_UTC_{year}.csv'
        blob=frame.to_csv(index=False,float_format='%.5f',date_format='%Y-%m-%dT%H:%M:%SZ').encode()
        path=destination/f'EURUSD_M15_UTC_{year}.zip'
        info=zipfile.ZipInfo(member,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:z.writestr(info,blob)
        assets.append(dict(path=str(path.relative_to(ROOT)),rows=len(frame),
                           sha256=sha(path),csv_sha256=hashlib.sha256(blob).hexdigest(),bytes=path.stat().st_size,
                           first_open_utc=frame.timestamp.iloc[0].isoformat(),last_open_utc=frame.timestamp.iloc[-1].isoformat()))
    config=ROOT/'config';config.mkdir(exist_ok=True)
    exclusions=config/'closures.json'
    exclusions.write_text(json.dumps(exemptions,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(provider='HistData.com',instrument='EURUSD',source_timeframe='M1',output_timeframe='M15',
       source_timezone='EST fixed UTC-05:00 without DST',output_timezone='UTC',
       acquired_at=source_manifest['acquired_at'],sources=source_manifest['archives'],source_audit=audit,
       aggregation='open=first high=max low=min close=last volume=sum, exactly 15 unique M1 observations',
       policy='Drop all conflicting M1 timestamps, then incomplete M15 groups. No interpolation, no forward-fill, no invented volume/prices.',
       exclusions_note='Observed source omissions, NOT claims that the Forex market was closed. Environment terminates at every time gap.',
       rows=len(full),unique_m1_rows=len(merged),incomplete_m15_groups=int((counts<15).sum()),
       missing_open_session_m15=len(missing),exclusion_intervals=len(exemptions),
       exclusions_path='config/closures.json',exclusions_sha256=sha(exclusions),assets=assets)
    (destination/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('rows','unique_m1_rows','incomplete_m15_groups','missing_open_session_m15','exclusion_intervals')},indent=2),flush=True)
    print(json.dumps(assets,indent=2),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=ROOT/'data/histdata_raw')
    parser.add_argument('--source-manifest',type=Path,default=ROOT/'data/histdata_sources.json')
    parser.add_argument('--output',type=Path,default=ROOT/'datasets/eurusd')
    args=parser.parse_args()
    build(args.raw_dir,args.output,args.source_manifest)
