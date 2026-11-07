"""OHLCV UTC estático, grupos completos; nunca desagrega M15 a M5."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from filelock import FileLock
from .market import MarketConfig


def aggregate(frame, source_minutes, timeframe, now=None):
    from .data_loader import DataError, COLUMNS, utc, _deduplicate
    market = MarketConfig(timeframe)
    target = int(market.step.total_seconds()/60)
    if source_minutes > target or target % source_minutes:
        raise DataError(f'{timeframe} requiere datos más finos: M15 no permite reconstruir M5. Usar archivos estáticos M1.')
    frame = _deduplicate(frame).sort_values('timestamp')
    if (frame.timestamp!=frame.timestamp.dt.floor(f"{source_minutes}min")).any():
        raise DataError("Fuente no alineada a su resolución declarada")
    clock = utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    groups = frame.groupby(frame.timestamp.dt.floor(market.frequency), sort=True)
    counts = groups.size()
    complete = counts == target//source_minutes
    # El mismo número de timestamps únicos alineados garantiza grupo completo.
    ready = complete & (counts.index+market.step <= clock)
    result = groups.agg(open=('open','first'),high=('high','max'),low=('low','min'),
                        close=('close','last'),volume=('volume','sum'))
    result = result.loc[ready].rename_axis('timestamp').reset_index()[COLUMNS]
    if result.empty: raise DataError('No hay grupos completos y cerrados para la temporalidad solicitada')
    result.attrs.update(timeframe=timeframe, source_minutes=source_minutes,
                        dropped_partial_bars=int((~complete).sum()), timestamp_convention='bar open UTC')
    return result


def load_resampled(*, timeframe, inputs=None, input_dir=None, output=None,
                   timezone='EST', input_timeframe='auto', timestamp_format=None,
                   epoch_unit=None, encoding='utf-8-sig', closures=(), force=False,
                   now=None, max_bytes=512*1024*1024, max_members=2000, start=None, end=None):
    from . import data_loader as dl
    market = MarketConfig(timeframe)
    output = Path(output or dl.ROOT/f'data/eurusd_{timeframe.lower()}.parquet')
    output.parent.mkdir(parents=True,exist_ok=True)
    paths = dl._files(inputs, input_dir or dl.DEFAULT_INPUT_DIR)
    fingerprints = [dict(path=str(p),sha256=dl._sha256(p)) for p in paths]
    clock = dl.utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    signature = json.dumps(dict(version=1,files=fingerprints,timeframe=timeframe,timezone=timezone,
        input_timeframe=input_timeframe,closures=closures,timestamp_format=timestamp_format,
        epoch_unit=epoch_unit,encoding=encoding,closed_cutoff=clock.floor(market.frequency).isoformat()),sort_keys=True)
    def select(frame):
        selected = frame
        if start is not None: selected=selected.loc[selected.timestamp >= dl.utc(start)]
        if end is not None: selected=selected.loc[selected.timestamp+market.step <= dl.utc(end)]
        if selected.empty: raise dl.DataError('Intervalo sin velas completas')
        return selected.reset_index(drop=True)
    with FileLock(str(output)+'.lock',timeout=120):
        if output.exists() and not force:
            import pyarrow.parquet as pq
            if (pq.read_metadata(output).metadata or {}).get(b'forex_static_v1') == signature.encode():
                return select(pd.read_parquet(output))
        frames=[];resolutions=set()
        for raw,name in dl._frames(paths,encoding=encoding,max_bytes=max_bytes,max_members=max_members):
            resolution=dl._resolution(name,input_timeframe);resolutions.add(resolution)
            if len(resolutions)>1: raise dl.DataError('No mezclar resoluciones en una ingesta')
            frames.append(dl.normalize(raw,timezone,epoch_unit,clock,input_timeframe=resolution,timestamp_format=timestamp_format))
        source=1 if resolutions=={'M1'} else 15
        combined=dl._deduplicate(pd.concat(frames,ignore_index=True)).sort_values('timestamp').reset_index(drop=True)
        missing=dl.expected_grid(combined.timestamp.iloc[0],combined.timestamp.iloc[-1]+pd.Timedelta(minutes=source),closures,step=pd.Timedelta(minutes=source)).difference(pd.DatetimeIndex(combined.timestamp))
        if len(missing): raise dl.DataError(f'Fuente tiene {len(missing)} huecos no justificados; declarar exclusiones observadas')
        result=aggregate(combined,source,timeframe,clock)
        if fingerprints != [dict(path=str(p),sha256=dl._sha256(p)) for p in paths]:
            raise dl.DataError('Fuente cambió durante ingesta')
        dl.write_cache(result,output,signature)
        return select(result)
