"""Ingesta EUR/USD estática: CSV, ZIP con CSV y Parquet, sin red.

Salida exacta: timestamp (apertura UTC), open, high, low, close, volume.
HistData Generic ASCII: YYYYMMDD HHMMSS;O;H;L;C;V, sin cabecera y EST fijo
UTC-05:00, SIN horario de verano. También acepta CSV con cabecera estándar.
M15 se valida; M1 se agrega a M15 únicamente con 15 minutos completos.
No descarga archivos, no llama APIs y no interpola precios/volumen.
"""
from __future__ import annotations

import argparse
import csv
from datetime import timezone as fixed_timezone, timedelta
import hashlib
import io
import json
import logging
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import os
import zipfile

from filelock import FileLock
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT/'datasets/eurusd'
DEFAULT_OUTPUT = ROOT/'data/eurusd_m15.parquet'
COLUMNS = ['timestamp','open','high','low','close','volume']
STEP = pd.Timedelta(minutes=15)
HISTDATA_TZ = fixed_timezone(timedelta(hours=-5), name='EST')
LOG = logging.getLogger(__name__)


class DataError(ValueError):
    """Archivo inválido, datos incompletos o configuración ambigua."""


def utc(value):
    """Límites de consulta: sin zona significa UTC, no la zona del CSV."""
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise DataError('Timestamp vacío')
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def read_closures(path=None):
    """Intervalos [start,end) UTC, documentados por el operador/proveedor."""
    items = json.loads(Path(path).read_text(encoding='utf-8')) if path else []
    if not isinstance(items,list):
        raise DataError('Cierres debe ser una lista JSON')
    for item in items:
        if not isinstance(item,dict) or not item.get('reason') or utc(item['start'])>=utc(item['end']):
            raise DataError('Cada cierre requiere start < end y reason')
    return items


def expected_grid(start,end,closures=(),step=STEP):
    """Convención Forex domingo 17–viernes 17 Nueva York, con DST.

    Este calendario es independiente de EST fijo utilizado en los archivos.
    Festivos y exclusiones del proveedor requieren intervalos documentados.
    Una exclusión por falta de datos no implica que el mercado estuviera cerrado.
    """
    grid = pd.date_range(utc(start).ceil(step),utc(end),freq=step,inclusive='left')
    ny = grid.tz_convert('America/New_York')
    day,hour = ny.dayofweek,ny.hour
    active = (day<4)|((day==4)&(hour<17))|((day==6)&(hour>=17))
    if closures:
        # O(n + k log n), incluso con miles de exclusiones del feed.
        starts = pd.DatetimeIndex([utc(item['start']) for item in closures])
        ends = pd.DatetimeIndex([utc(item['end']) for item in closures])
        changes = np.zeros(len(grid)+1,dtype=np.int64)
        np.add.at(changes,grid.searchsorted(starts,side='left'),1)
        np.add.at(changes,grid.searchsorted(ends,side='left'),-1)
        active &= np.cumsum(changes[:-1])==0
    return grid[active]


def _timestamps(values,timezone,epoch_unit,timestamp_format):
    if pd.api.types.is_numeric_dtype(values):
        if epoch_unit not in ('s','ms','us','ns'):
            raise DataError('Timestamps numéricos requieren --epoch-unit')
        return pd.to_datetime(values,unit=epoch_unit,utc=True,errors='raise')
    # Conserva timestamps datetime y fechas ISO con offset, sin reinterpretarlos.
    if isinstance(values.dtype,pd.DatetimeTZDtype):
        return pd.to_datetime(values,utc=True,errors='raise')
    if pd.api.types.is_datetime64_dtype(values):
        parsed = values
    else:
        text = values.astype('string').str.strip()
        if text.isna().any():
            raise DataError('Timestamp nulo')
        if epoch_unit:
            return pd.to_datetime(pd.to_numeric(text,errors='raise'),unit=epoch_unit,utc=True,errors='raise')
        aware = text.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True,case=False)
        if aware.any():
            if not aware.all():
                raise DataError('No mezclar timestamps con zona y sin zona')
            return pd.to_datetime(text,format=timestamp_format or 'ISO8601',utc=True,errors='raise')
        fmt = timestamp_format
        if fmt is None:
            if text.str.fullmatch(r'\d{8} \d{6}').all():
                fmt = '%Y%m%d %H%M%S'
            elif text.str.fullmatch(r'\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}').all():
                fmt = '%Y.%m.%d %H:%M'
            else:
                fmt = 'ISO8601'
        parsed = pd.to_datetime(text,format=fmt,errors='raise')
    zone = HISTDATA_TZ if timezone.upper() in ('EST','HISTDATA') else timezone
    return parsed.dt.tz_localize(zone,ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')


def normalize(raw,timezone='UTC',epoch_unit=None,now=None,*,input_timeframe='M15',timestamp_format=None):
    """Limpieza sin relleno. Duplicados exactos se eliminan; conflictos se rechazan.

    Esta función conserva el default UTC para sus consumidores existentes.
    load_data/CLI usan EST por defecto para las fuentes HistData sin zona.
    """
    if input_timeframe not in ('M1','M15'):
        raise DataError('Resolución de normalización debe ser M1 o M15')
    df = raw.copy()
    df.columns = [str(c).strip().strip('<>').lower().replace(' ','') for c in df.columns]
    if not df.columns.is_unique:
        raise DataError('Columnas duplicadas')
    if 'date' in df and 'time' in df and 'timestamp' not in df:
        df['timestamp'] = df['date'].astype('string').str.strip()+' '+df['time'].astype('string').str.strip()
    if 'timestamp' not in df:
        candidates = [c for c in ('datetime','date','time') if c in df]
        if len(candidates)==1:
            df = df.rename(columns={candidates[0]:'timestamp'})
        elif isinstance(df.index,pd.DatetimeIndex):
            df.insert(0,'timestamp',df.index)
    if 'volume' not in df and 'tick_volume' in df:
        df = df.rename(columns={'tick_volume':'volume'})
    if not set(COLUMNS).issubset(df):
        raise DataError(f'Se requieren {COLUMNS}; ticks Bid/Ask no son velas OHLCV')
    df = df[COLUMNS].copy()
    try:
        df['timestamp'] = _timestamps(df.timestamp,timezone,epoch_unit,timestamp_format)
        df[COLUMNS[1:]] = df[COLUMNS[1:]].apply(pd.to_numeric,errors='raise').astype('float64')
    except (ValueError,TypeError,OverflowError) as exc:
        raise DataError(f'Timestamp o valor OHLCV inválido: {exc}') from exc
    if df.isna().any().any() or not np.isfinite(df[COLUMNS[1:]].to_numpy()).all():
        raise DataError('Nulos o infinitos en OHLCV/timestamp')
    prices = df[['open','high','low','close']]
    if ((prices<=0).any().any() or (df.volume<0).any()
            or (df.high<prices.max(axis=1)).any() or (df.low>prices.min(axis=1)).any()):
        raise DataError('OHLC incoherente o volumen negativo')
    frequency = '1min' if input_timeframe=='M1' else '15min'
    if (df.timestamp!=df.timestamp.dt.floor(frequency)).any():
        raise DataError(f'Timestamps no alineados a {input_timeframe}; si el archivo es M1, indicarlo explícitamente')
    df = _deduplicate(df)
    clock = utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    pending = df.loc[df.timestamp.dt.floor('15min')+STEP>clock,'timestamp']
    refresh_at = (pending.min().floor('15min')+STEP).isoformat() if len(pending) else None
    period = pd.Timedelta(minutes=1 if input_timeframe=='M1' else 15)
    df = df.loc[df.timestamp+period<=clock]
    if df.empty:
        raise DataError('No hay velas cerradas')
    result = df.sort_values('timestamp').reset_index(drop=True)
    result.attrs['refresh_at'] = refresh_at
    return result


def _deduplicate(df):
    df = df.drop_duplicates()
    if df.timestamp.duplicated().any():
        raise DataError('Duplicados conflictivos para el mismo timestamp')
    return df


def validate_window(df,start,end,closures=()):
    result = df.loc[(df.timestamp>=utc(start))&(df.timestamp<utc(end))].copy()
    missing = expected_grid(start,end,closures).difference(pd.DatetimeIndex(result.timestamp))
    if len(missing):
        raise DataError(f'Faltan {len(missing)} velas en sesión abierta; primeras: {list(missing[:5])}')
    if result.empty:
        raise DataError('Intervalo sin velas')
    return result.reset_index(drop=True)


def resample_m1(df,now=None):
    """Agrega únicamente grupos de 15 minutos distintos, sin inventar cotizaciones.

    Se fusionan los archivos ANTES de agrupar, para admitir límites de archivo
    dentro de una vela y deduplicar solapamientos sin duplicar el volumen.
    """
    df = _deduplicate(df).sort_values('timestamp')
    clock = utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    bins = df.timestamp.dt.floor('15min')
    df = df.loc[bins+STEP<=clock].copy()
    grouped = df.groupby(df.timestamp.dt.floor('15min'),sort=True)
    counts = grouped.size()
    bad = counts[counts!=15]
    if len(bad):
        raise DataError(f'M1 incompleto: cada vela M15 requiere 15 minutos; primer grupo: {bad.index[0]} ({bad.iloc[0]}/15)')
    result = grouped.agg(open=('open','first'),high=('high','max'),low=('low','min'),
                         close=('close','last'),volume=('volume','sum')).rename_axis('timestamp').reset_index()
    if result.empty:
        raise DataError('Sin velas M15 completas')
    return result[COLUMNS]


def _resolution(name,requested):
    # Solo el nombre oficial declara instrumento/resolución; nunca se adivina por precios.
    match = re.search(r'DAT_(?:ASCII|MT)_([A-Z]{6})_(M1|M15|T)(?:_|\.)',Path(name).name.upper())
    named = None
    if match:
        if match[1]!='EURUSD':
            raise DataError(f'Instrumento distinto de EURUSD: {name}')
        named = match[2]
        if named=='T':
            raise DataError('HistData Tick no es un archivo de velas OHLCV')
    if requested!='auto' and named and requested!=named:
        raise DataError(f'{name} declara {named}, incompatible con --input-timeframe {requested}')
    return named if requested=='auto' and named else 'M15' if requested=='auto' else requested


def _read_csv(stream,name,encoding):
    text = io.TextIOWrapper(stream,encoding=encoding,newline='')
    try:
        first = ''
        for _ in range(100):
            first = text.readline()
            if not first or first.strip():
                break
        if not first.strip():
            raise DataError(f'CSV vacío: {name}')
        delimiters = [d for d in (';',',','\t') if first.count(d)>=5]
        if len(delimiters)!=1:
            raise DataError(f'Separador no reconocido o archivo de ticks: {name}')
        sep = delimiters[0]
        fields = next(csv.reader([first],delimiter=sep))
        names = [v.strip().strip('<>').lower().replace(' ','') for v in fields]
        header = bool(set(names)&{'timestamp','datetime','date','time','open'})
        if not header and len(fields) not in (6,7):
            raise DataError(f'CSV OHLCV requiere 6 campos (o fecha/hora separadas): {name}')
        text.seek(0)
        assigned = None if header else COLUMNS if len(fields)==6 else ['date','time',*COLUMNS[1:]]
        return pd.read_csv(text,sep=sep,header=0 if header else None,names=assigned,
                           dtype=str,on_bad_lines='error',skip_blank_lines=True)
    finally:
        text.close()


def _files(inputs,input_dir):
    requested = [inputs] if isinstance(inputs,(str,Path)) else list(inputs or [input_dir])
    found = []
    for item in requested:
        if str(item).lower().startswith(('http://','https://','ftp://','sftp://')):
            raise DataError('Solo archivos estáticos locales; no se admiten URLs ni APIs')
        path = Path(item).expanduser().resolve()
        if path.is_dir():
            found.extend(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.csv','.zip','.parquet'))
        elif path.is_file():
            found.append(path)
        else:
            raise DataError(f'Origen inexistente: {path}. Subir CSV/ZIP a datasets/eurusd/')
    found = sorted(set(found))
    if not found:
        raise DataError('No hay CSV/ZIP/Parquet en el origen; se requieren archivos reales, no una descarga automática')
    if any(p.suffix.lower() not in ('.csv','.zip','.parquet') for p in found):
        raise DataError('Extensión no admitida')
    return found


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def _frames(paths,*,encoding,max_bytes,max_members):
    """Lee ZIP sin extraer entradas; limita miembros/tamaño descomprimido."""
    total = 0
    for path in paths:
        if path.suffix.lower()=='.zip':
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos)>max_members:
                    raise DataError('ZIP supera el límite de miembros')
                csv_files = []
                for info in infos:
                    name = info.filename.replace('\\','/')
                    parts = PurePosixPath(name)
                    if (parts.is_absolute() or '..' in parts.parts or re.match(r'^[A-Za-z]:',name)
                            or stat.S_ISLNK(info.external_attr>>16)):
                        raise DataError('Entrada ZIP insegura')
                    if not info.is_dir() and parts.suffix.lower()=='.csv':
                        if info.flag_bits&1:
                            raise DataError('ZIP cifrado no admitido')
                        total += info.file_size
                        if total>max_bytes or info.file_size>max_bytes:
                            raise DataError('ZIP supera el límite de datos descomprimidos')
                        if info.file_size/max(1,info.compress_size)>1000:
                            raise DataError('Relación de compresión ZIP excesiva')
                        csv_files.append(info)
                if not csv_files:
                    raise DataError(f'ZIP sin CSV: {path.name}')
                for info in sorted(csv_files,key=lambda i:i.filename):
                    with archive.open(info) as member:
                        yield _read_csv(member,info.filename,encoding),info.filename
        else:
            total += path.stat().st_size
            if total>max_bytes:
                raise DataError('Se supera el límite de tamaño de entrada')
            if path.suffix.lower()=='.parquet':
                yield pd.read_parquet(path),path.name
            else:
                with path.open('rb') as file:
                    yield _read_csv(file,path.name,encoding),path.name


def write_cache(df,path,signature,refresh_at=None):
    """Parquet Zstandard con metadatos y sustitución atómica."""
    path = Path(path)
    table = pa.Table.from_pandas(df,preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b'forex_static_v1'] = signature.encode()
    if refresh_at:
        metadata[b'forex_static_refresh_at'] = refresh_at.encode()
    table = table.replace_schema_metadata(metadata)
    fd,temp = tempfile.mkstemp(dir=path.parent,suffix='.parquet.tmp');os.close(fd)
    try:
        pq.write_table(table,temp,compression='zstd');os.replace(temp,path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_data(*,inputs=None,input_dir=DEFAULT_INPUT_DIR,output=DEFAULT_OUTPUT,start=None,end=None,
              timezone='EST',input_timeframe='auto',timestamp_format=None,epoch_unit=None,
              encoding='utf-8-sig',closures=(),force=False,now=None,max_bytes=512*1024*1024,
              max_members=2000,source='static',local=None,timeframe=None):
    """Carga archivos, devuelve M15 UTC y publica data/eurusd_m15.parquet.

    inputs acepta uno o varios archivos/directorios; por defecto datasets/eurusd.
    auto reconoce M1 SOLO por nombres oficiales de HistData; lo demás se trata
    como M15. Para M1 renombrado se exige input_timeframe='M1'. No mezcla M1/M15.
    source='local' y local=path se conservan como aliases de importación local.
    start/end son UTC [start,end); omitirlos carga el histórico cerrado disponible,
    sin exigir que llegue hasta hoy. Para alertas, la frescura se valida aparte.
    """
    if source not in ('static','local','files','histdata'):
        raise DataError('Se eliminó la ingesta remota: usar archivos locales CSV/ZIP')
    if input_timeframe not in ('auto','M1','M15') or max_bytes<1 or max_members<1:
        raise DataError('Resolución o límites de entrada inválidos')
    if local is not None:
        if inputs is not None:
            raise DataError('Usar inputs o local, no ambos')
        inputs = local
    if timeframe is not None:
        from .resampling import load_resampled
        return load_resampled(timeframe=timeframe,inputs=inputs,input_dir=input_dir,
            output=ROOT/f"data/eurusd_{timeframe.lower()}.parquet" if Path(output)==DEFAULT_OUTPUT else output,
            start=start,end=end,timezone=timezone,input_timeframe=input_timeframe,
            timestamp_format=timestamp_format,epoch_unit=epoch_unit,encoding=encoding,
            closures=closures,force=force,now=now,max_bytes=max_bytes,max_members=max_members)
    paths = _files(inputs,input_dir)
    output = Path(output).resolve()
    if output in paths:
        raise DataError('La salida no puede sobrescribir un archivo de origen')
    fingerprints = [dict(path=str(p),sha256=_sha256(p)) for p in paths]
    signature = hashlib.sha256(json.dumps(dict(files=fingerprints,timezone=timezone,timeframe=input_timeframe,
        timestamp_format=timestamp_format,epoch_unit=epoch_unit,encoding=encoding,closures=list(closures),
        max_bytes=max_bytes,max_members=max_members),sort_keys=True).encode()).hexdigest()
    clock = utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    output.parent.mkdir(parents=True,exist_ok=True)
    def select(frame):
        lower = utc(start).ceil('15min') if start is not None else frame.timestamp.iloc[0]
        upper = min(utc(end).floor('15min'),clock.floor('15min')) if end is not None else frame.timestamp.iloc[-1]+STEP
        if lower>=upper:
            raise DataError('start debe ser anterior a end efectivo')
        return validate_window(frame,lower,upper,closures)
    with FileLock(str(output)+'.lock',timeout=120):
        if output.exists() and not force:
            try:
                meta = pq.read_metadata(output).metadata or {}
                refresh = meta.get(b'forex_static_refresh_at')
                if meta.get(b'forex_static_v1')==signature.encode() and (not refresh or clock<utc(refresh.decode())):
                    cached = normalize(pd.read_parquet(output),now=clock)
                    result = select(cached)
                    LOG.info('Caché válida: %d velas',len(result))
                    return result
            except (DataError,OSError,ValueError,pa.ArrowException):
                LOG.info('Caché insuficiente o inválida; releer archivos estáticos')
        frames,resolutions = [],set()
        try:
            for raw,name in _frames(paths,encoding=encoding,max_bytes=max_bytes,max_members=max_members):
                resolution = _resolution(name,input_timeframe)
                resolutions.add(resolution)
                if len(resolutions)>1:
                    raise DataError('No mezclar archivos M1 y M15 en una misma ingesta')
                frames.append(normalize(raw,timezone,epoch_unit,clock,input_timeframe=resolution,
                                         timestamp_format=timestamp_format))
        except (zipfile.BadZipFile,UnicodeError,pd.errors.ParserError) as exc:
            raise DataError(f'Archivo CSV/ZIP corrupto o no compatible: {exc}') from exc
        refreshes = [utc(frame.attrs['refresh_at']) for frame in frames if frame.attrs.get('refresh_at')]
        combined = _deduplicate(pd.concat(frames,ignore_index=True)).sort_values('timestamp').reset_index(drop=True)
        if resolutions=={'M1'}:
            combined = resample_m1(combined,clock)
        combined = normalize(combined,now=clock)
        validate_window(combined,combined.timestamp.iloc[0],combined.timestamp.iloc[-1]+STEP,closures)
        result = select(combined)
        if [dict(path=str(p),sha256=_sha256(p)) for p in paths]!=fingerprints:
            raise DataError('Los archivos de origen cambiaron durante la lectura')
        write_cache(combined,output,signature,min(refreshes).isoformat() if refreshes else None)
        LOG.info('Guardado %s: %d velas M15; consulta: %d',output,len(combined),len(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',type=Path,help='Archivos o directorios; nunca URLs')
    parser.add_argument('--input-dir',type=Path,default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--timezone',default='EST',help='Zona de timestamps sin offset; HistData usa EST fijo')
    parser.add_argument('--input-timeframe',choices=['auto','M1','M15'],default='auto')
    parser.add_argument('--timeframe',choices=['M5','M15','H1','H4'])
    parser.add_argument('--timestamp-format')
    parser.add_argument('--epoch-unit',choices=['s','ms','us','ns'])
    parser.add_argument('--encoding',default='utf-8-sig')
    parser.add_argument('--start');parser.add_argument('--end')
    parser.add_argument('--closures',type=Path,default=ROOT/'config/closures.json' if (ROOT/'config/closures.json').exists() else None)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--max-bytes',type=int,default=512*1024*1024)
    parser.add_argument('--max-members',type=int,default=2000)
    args = vars(parser.parse_args());args['closures']=read_closures(args['closures'])
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    try:
        result = load_data(**args)
    except Exception:
        LOG.exception('Ingesta estática fallida; no se sustituye la salida por datos inválidos')
        raise SystemExit(1)
    LOG.info('Rango UTC: %s a %s',result.timestamp.iloc[0],result.timestamp.iloc[-1])


if __name__=='__main__':
    main()
