"""Ingesta EUR/USD M15 sin terminal MetaTrader ni dependencia de Windows.

Contrato: timestamp es APERTURA de vela, UTC; intervalo [start, end).
No interpola OHLC, no inventa volumen y no rellena huecos. Volumen Forex
puede ser cero o tick_volume: no equivale al volumen global negociado.

Dukascopy: descarga M15 multianual por HTTP (chart/json3),
sin Node, terminal, cuenta ni credenciales. Fragmentación
diaria, checkpoints Parquet y sesiones HTTP con timeout/reintentos.
Local es únicamente una opción de importación genérica; nunca es necesario.

Calendario aproximado: domingo 17:00 a viernes 17:00 America/New_York,
con DST. Festivos y pausas del proveedor requieren --closures explícito.
Cada cierre es {"start": "...Z", "end": "...Z", "reason": "..."}.
Los intervalos son semiabiertos y justifican ausencias, no borran velas.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import random
import math
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from filelock import FileLock
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)
STEP = pd.Timedelta(minutes=15)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/eurusd_m15.parquet"


class DataError(ValueError):
    """Datos inválidos, incompletos o incompatibles con el contrato M15."""


def utc(value) -> pd.Timestamp:
    """Normaliza límites de consulta; una fecha sin zona significa UTC."""
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise DataError("Fecha vacía o inválida")
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def read_closures(path: Path | None) -> list[dict]:
    """Lee únicamente cierres documentados por el usuario/proveedor."""
    items = json.loads(path.read_text(encoding="utf-8")) if path else []
    for item in items:
        if not item.get("reason") or utc(item["start"]) >= utc(item["end"]):
            raise DataError("Cada cierre necesita start < end y reason")
    return items


def expected_grid(start, end, closures=()) -> pd.DatetimeIndex:
    """Aperturas M15 esperadas; admite DST y cierres declarados."""
    grid = pd.date_range(utc(start).ceil("15min"), utc(end),
                         freq="15min", inclusive="left")
    local = grid.tz_convert("America/New_York")
    day, hour = local.dayofweek, local.hour
    active = ((day < 4) | ((day == 4) & (hour < 17))
              | ((day == 6) & (hour >= 17)))
    for item in closures:
        active &= ~((grid >= utc(item["start"])) & (grid < utc(item["end"])))
    return grid[active]


def normalize(raw: pd.DataFrame, timezone="UTC", epoch_unit=None,
              now=None) -> pd.DataFrame:
    """Valida OHLCV, timestamps, duplicados y elimina velas sin cerrar.

    Solo se eliminan duplicados completamente idénticos. Duplicados con
    precios diferentes y nulos se rechazan: requieren revisar el origen.
    No se redondean timestamps fuera de la cuadrícula M15.
    """
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        raise DataError("Se esperaba un único instrumento, no columnas MultiIndex")
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"datetime": "timestamp", "date": "timestamp", "time": "timestamp"}
    if "timestamp" not in df:
        candidates = [c for c in aliases if c in df]
        if len(candidates) == 1:
            df = df.rename(columns={candidates[0]: "timestamp"})
        elif isinstance(df.index, pd.DatetimeIndex):
            df.insert(0, "timestamp", df.index)
    if "volume" not in df and "tick_volume" in df:
        df = df.rename(columns={"tick_volume": "volume"})
    if not df.columns.is_unique or not set(COLUMNS).issubset(df.columns):
        raise DataError(f"Columnas requeridas: {COLUMNS}; recibidas: {list(df.columns)}")
    df = df[COLUMNS].copy()
    values = df["timestamp"]
    if pd.api.types.is_numeric_dtype(values):
        if epoch_unit not in ("s", "ms", "us", "ns"):
            raise DataError("Timestamp numérico: especificar epoch_unit=s/ms/us/ns")
        ts = pd.to_datetime(values, unit=epoch_unit, utc=True, errors="raise")
    else:
        ts = pd.to_datetime(values, errors="raise")
        if ts.dt.tz is None:
            ts = ts.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts
    df[COLUMNS[1:]] = df[COLUMNS[1:]].apply(pd.to_numeric, errors="raise").astype("float64")
    if df.isna().any().any() or not np.isfinite(df[COLUMNS[1:]].to_numpy()).all():
        raise DataError("Nulos o infinitos en OHLCV/timestamp")
    if (df["timestamp"] != df["timestamp"].dt.floor("15min")).any():
        raise DataError("Timestamp fuera de M15; exportar/resamplear velas antes de cargar")
    if ((df[["open", "high", "low", "close"]] <= 0).any().any()
            or (df["volume"] < 0).any()
            or (df["high"] < df[["open", "low", "close"]].max(axis=1)).any()
            or (df["low"] > df[["open", "high", "close"]].min(axis=1)).any()):
        raise DataError("Precios OHLC incoherentes o volumen negativo")
    df = df.drop_duplicates()
    if df["timestamp"].duplicated().any():
        raise DataError("Hay velas conflictivas para el mismo timestamp")
    cutoff = utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    df = df.loc[df["timestamp"] + STEP <= cutoff]
    if df.empty:
        raise DataError("No hay velas M15 cerradas")
    return df.sort_values("timestamp").reset_index(drop=True)


def validate_window(df, start, end, closures=()) -> pd.DataFrame:
    """Comprueba huecos internos Y cobertura de ambos extremos solicitados."""
    selected = df.loc[(df.timestamp >= start) & (df.timestamp < end)].copy()
    missing = expected_grid(start, end, closures).difference(
        pd.DatetimeIndex(selected.timestamp))
    if len(missing):
        raise DataError(f"Faltan {len(missing)} velas en sesión abierta; "
                        f"primeras: {list(missing[:5])}. Revisar origen o cierres documentados.")
    if selected.empty:
        raise DataError("El intervalo solicitado no contiene velas")
    return selected.reset_index(drop=True)


def read_local(path: Path, table: str, symbol_column: str | None,
               symbol: str) -> pd.DataFrame:
    """SQLite en modo solo lectura; tabla/columna verificadas y símbolo parametrizado.

    El archivo/tabla debe contener exclusivamente EURUSD M15, o utilizar
    symbol_column para seleccionar EURUSD. No se infiere temporalidad ni
    instrumento por los precios. SQLite admite time epoch + tick_volume.
    """
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.loc[df[symbol_column] == symbol] if symbol_column else df
    if path.suffix.lower() not in (".sqlite", ".sqlite3", ".db"):
        raise DataError("Origen local debe ser .parquet, .sqlite, .sqlite3 o .db")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if table not in tables:
            raise DataError(f"Tabla inexistente: {table}")
        quote = lambda name: '"' + name.replace('"', '""') + '"'
        query = f"SELECT * FROM {quote(table)}"
        params = ()
        if symbol_column:
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({quote(table)})")}
            if symbol_column not in cols:
                raise DataError(f"Columna de símbolo inexistente: {symbol_column}")
            query += f" WHERE {quote(symbol_column)} = ?"
            params = (symbol,)
        return pd.read_sql_query(query, con, params=params)


@dataclass(frozen=True)
class DownloadPolicy:
    """Un único presupuesto de reintentos, sin multiplicarlos entre capas."""
    max_attempts: int = 6
    base_backoff: float = 2.0
    max_backoff: float = 120.0
    max_wait_budget: float = 300.0
    request_pause: float = 1.0
    jitter: float = 0.5

    def __post_init__(self):
        if self.max_attempts < 1 or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts debe ser entero positivo")
        values = (self.base_backoff,self.max_backoff,self.max_wait_budget,self.request_pause,self.jitter)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Esperas deben ser finitas y no negativas")


def retry_after_seconds(value, now=None):
    """Retry-After puede ser segundos o una fecha HTTP; no se ignora al servidor."""
    if value is None:
        return 0.0
    try:
        seconds = float(value)
        return max(0., seconds) if math.isfinite(seconds) else 0.
    except (ValueError, TypeError):
        try:
            deadline = utc(parsedate_to_datetime(value))
            clock = utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
            return max(0., (deadline-clock).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 0.


def wait_seconds(seconds):
    """Espera larga fraccionada para facilitar interrupciones del proceso."""
    while seconds > 0:
        part = min(seconds, 60.)
        time.sleep(part)
        seconds -= part


def dukascopy_session(policy=None):
    """GET secuencial con pacing, backoff + jitter, Retry-After y presupuesto.

    No cambia IPs, no evita controles de acceso y no reintenta 401/403.
    Si Retry-After supera el presupuesto, aborta sin solicitar prematuramente.
    """
    import requests
    from requests.adapters import HTTPAdapter
    policy = policy or DownloadPolicy()

    class PacedSession(requests.Session):
        def request(self, method, url, **kwargs):
            kwargs.setdefault("timeout", (10, 45))
            waited = 0.
            for attempt in range(policy.max_attempts):
                wait_seconds(policy.request_pause)
                retry_after = 0.
                try:
                    response = super().request(method, url, **kwargs)
                    if response.status_code not in (429,500,502,503,504):
                        response.raise_for_status()
                        return response
                    status = response.status_code
                    retry_after = retry_after_seconds(response.headers.get("Retry-After"))
                    response.close()
                except requests.RequestException as exc:
                    if isinstance(exc, requests.HTTPError):
                        raise DataError("Dukascopy rechazó la petición; no se reintenta un 4xx permanente") from None
                    status = "network"
                if attempt + 1 == policy.max_attempts:
                    raise DataError(f"Dukascopy no disponible ({status}); agotados {policy.max_attempts} intentos")
                backoff = min(policy.max_backoff, policy.base_backoff * 2**attempt)
                delay = max(retry_after, backoff + random.uniform(0., policy.jitter))
                if waited + delay > policy.max_wait_budget:
                    raise DataError("Dukascopy exige una espera superior al presupuesto; reanudar más tarde")
                LOG.warning("Dukascopy %s: reintento %d/%d en %.1fs",status,attempt+2,policy.max_attempts,delay)
                wait_seconds(delay)
                waited += delay
            raise AssertionError("unreachable")

    session = PacedSession()
    # Se desactivan los reintentos del adaptador: la política anterior es única.
    for scheme in ("https://", "http://"):
        session.mount(scheme, HTTPAdapter(max_retries=0))
    return session


def fetch_dukascopy_chunk(start, end, side="bid") -> pd.DataFrame:
    """Adaptador HTTP del histórico Dukascopy chart/json3 (velas M15 nativas).

    Usa el mismo protocolo público de dukascopy-python/feed v1, sin importar
    una versión del wrapper. Las páginas tienen como máximo 96 registros (un día M15).
    Se valida JSONP y estructura; cursor avanza por timestamp, no por longitud.
    No hay límite artificial de 60 días. Solo EUR/USD, precios bid O ask.
    """
    if side not in ("bid", "ask"):
        raise DataError("side debe ser bid o ask")
    cursor, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    callback = "forex_history"
    records = []
    with dukascopy_session() as session:
        while cursor < end_ms:
            response = session.get(
                "https://freeserv.dukascopy.com/2.0/index.php",
                params=dict(path="chart/json3", instrument="EUR/USD", interval="15MIN",
                            offer_side="B" if side == "bid" else "A",
                            time_direction="N", last_update=str(cursor),
                            limit=str(min(96, max(1, (end_ms-cursor+899_999)//900_000))),
                            splits="true", stocks="true", jsonp=callback),
                headers={"User-Agent": "ForexResearch/1.0",
                         "Referer": "https://freeserv.dukascopy.com/2.0/"},
            )
            text = response.text.strip()
            prefix, suffix = callback + "(", ");"
            if not (text.startswith(prefix) and text.endswith(suffix)):
                raise DataError("Respuesta Dukascopy no es el JSONP esperado")
            page = json.loads(text[len(prefix):-len(suffix)])
            if not isinstance(page, list):
                raise DataError("Dukascopy devolvió un error o estructura inesperada")
            if not page:
                break  # validate_window decidirá si falta cobertura.
            if any(not isinstance(row, list) or len(row) != 6 for row in page):
                raise DataError("Esquema Dukascopy inesperado: se requieren 6 campos")
            stamps = np.asarray([row[0] for row in page], dtype=np.float64)
            if (not np.isfinite(stamps).all() or (stamps != np.floor(stamps)).any()
                    or (np.diff(stamps) <= 0).any() or stamps[0] < cursor):
                raise DataError("Página Dukascopy desordenada o cursor sin avance")
            records.extend(row for row in page if cursor <= row[0] < end_ms)
            next_cursor = int(stamps[-1]) + int(STEP.total_seconds() * 1000)
            if next_cursor <= cursor:
                raise DataError("Dukascopy no avanzó el cursor")
            cursor = next_cursor
    frame = pd.DataFrame(records, columns=COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame


def download_dukascopy(start, end, *, cache_dir, side="bid", closures=(),
                       force=False, now=None) -> pd.DataFrame:
    """Descarga/reanuda fragmentos diarios; nunca ignora fallos de cobertura.

    Cada checkpoint se publica atómicamente después de validarlo. Un runner
    nuevo puede reanudar si GitHub Actions restaura data/chunks. Los fragmentos
    históricos se reutilizan; el bloque parcial tiene una clave de fin distinta
    cuando avanza la consulta. --force vuelve a consultar también los chunks.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    cursor = start
    while cursor < end:
        boundary = cursor.normalize() + pd.Timedelta(days=1)
        stop = min(boundary, end)
        if not len(expected_grid(cursor, stop, closures)):
            cursor = stop
            continue
        key = hashlib.sha256(json.dumps(dict(feed=1, side=side, closures=list(closures)),
                                         sort_keys=True).encode()).hexdigest()
        name = f"eurusd_{side}_{cursor.strftime('%Y%m%dT%H%M')}_{stop.strftime('%Y%m%dT%H%M')}.parquet"
        path = cache_dir / name
        chunk = None
        if path.exists() and not force:
            try:
                if (pq.read_metadata(path).metadata or {}).get(b"forex_source_v1") == key.encode():
                    chunk = validate_window(normalize(pd.read_parquet(path), now=now),
                                            cursor, stop, closures)
            except Exception as exc:
                LOG.warning("Checkpoint inválido %s: %s", path.name, exc)
        if chunk is None:
            LOG.info("Dukascopy %s [%s, %s)", side, cursor, stop)
            # HTTP tiene su propio presupuesto. No se reinicia al agotar 429.
            chunk = normalize(fetch_dukascopy_chunk(cursor, stop, side), now=now)
            chunk = validate_window(chunk, cursor, stop, closures)
            write_cache(chunk, path, key)
        pieces.append(chunk)
        cursor = stop
    if not pieces:
        raise DataError("Intervalo sin sesiones de mercado")
    return pd.concat(pieces, ignore_index=True).drop_duplicates("timestamp").sort_values("timestamp")


def write_cache(df, path: Path, signature: str):
    """Parquet Zstandard y metadatos en una única escritura atómica."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"forex_source_v1"] = signature.encode()
    table = table.replace_schema_metadata(metadata)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        pq.write_table(table, name, compression="zstd")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def load_data(*, source="dukascopy", local=None, output=DEFAULT_OUTPUT,
              start=None, end=None, timezone="UTC", epoch_unit=None,
              table="rates", symbol_column=None, symbol="EURUSD",
              closures=(), force=False, publication_delay=30, now=None,
              years=5, side="bid", chunk_dir=None) -> pd.DataFrame:
    """Devuelve OHLCV UTC validado y mantiene el archivo procesado.

    Caché: origen, configuración y mtime/tamaño local deben coincidir;
    además se valida cobertura del rango pedido, no solo edad del archivo.
    Dukascopy descarga fragmentos diarios reanudables y conserva el histórico.
    No mezcla feeds locales y remotos. Un fallo remoto no usa datos caducos.
    El bloqueo evita dos escritores simultáneos en el mismo equipo.
    """
    if source not in ("dukascopy", "local") or publication_delay < 0 or years < 1 or side not in ("bid", "ask"):
        raise DataError("source inválido o publication_delay negativo")
    output = Path(output).resolve()
    local = Path(local).resolve() if local else None
    if source == "local" and (local is None or not local.is_file()):
        raise DataError("source=local requiere un archivo local existente")
    if local == output:
        raise DataError("El origen local y la caché deben ser archivos diferentes")
    clock = utc(now if now is not None else pd.Timestamp.now(tz="UTC"))
    last_end = (clock - pd.Timedelta(minutes=publication_delay if source == "dukascopy" else 0)).floor("15min")
    hi = min(utc(end).floor("15min"), last_end) if end is not None else last_end
    lo = utc(start).ceil("15min") if start is not None else None
    if source == "dukascopy" and lo is None:
        lo = (hi - pd.DateOffset(years=years)).ceil("15min")
    if lo is not None and lo >= hi:
        raise DataError("start debe ser anterior a end efectivo")
    provenance = dict(version=3, source=source, feed_version=1, side=side, timezone=timezone,
                      epoch_unit=epoch_unit, table=table, symbol_column=symbol_column,
                      symbol=symbol, closures=list(closures))
    if local:
        stat = local.stat()
        provenance.update(path=str(local), mtime_ns=stat.st_mtime_ns, size=stat.st_size)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=120):
        cached = None
        if output.exists() and not force:
            try:
                metadata = pq.read_metadata(output).metadata or {}
                if metadata.get(b"forex_source_v1") == signature.encode():
                    cached = normalize(pd.read_parquet(output), now=clock)
            except Exception as exc:
                LOG.warning("Caché no legible: %s", exc)
        if cached is not None:
            lower = lo if lo is not None else cached.timestamp.iloc[0]
            upper = hi if source == "dukascopy" or end is not None else cached.timestamp.iloc[-1] + STEP
            try:
                result = validate_window(cached, lower, upper, closures)
                LOG.info("Caché válida: %d velas", len(result))
                return result
            except DataError:
                LOG.info("Caché no cubre la consulta; actualizando origen")
        if source == "local":
            raw = read_local(local, table, symbol_column, symbol)
            fresh = normalize(raw, timezone, epoch_unit, clock)
            lower = lo if lo is not None else fresh.timestamp.iloc[0]
            upper = hi if end is not None else fresh.timestamp.iloc[-1] + STEP
        else:
            fresh = download_dukascopy(
                lo, hi, cache_dir=chunk_dir or output.parent / "chunks", side=side,
                closures=closures, force=force, now=clock,
            )
            if cached is not None:
                fresh = pd.concat([cached, fresh]).drop_duplicates("timestamp", keep="last")
                fresh = fresh.sort_values("timestamp").reset_index(drop=True)
            lower, upper = lo, hi
        result = validate_window(fresh, lower, upper, closures)
        # También valida el archivo completo antes de persistirlo.
        validate_window(fresh, fresh.timestamp.iloc[0], fresh.timestamp.iloc[-1] + STEP, closures)
        write_cache(fresh, output, signature)
        LOG.info("Guardado %s (%d velas); consulta: %d", output, len(fresh), len(result))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["dukascopy", "local"], default="dukascopy")
    parser.add_argument("--years", type=int, default=5, help="Años si no se indica --start")
    parser.add_argument("--side", choices=["bid", "ask"], default="bid")
    parser.add_argument("--chunk-dir", type=Path, help="Checkpoints diarios reanudables")
    parser.add_argument("--local", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", help="Inclusivo; sin zona se interpreta UTC")
    parser.add_argument("--end", help="Exclusivo; limitado a velas cerradas/publicadas")
    parser.add_argument("--timezone", default="UTC", help="Zona de timestamps naive locales")
    parser.add_argument("--epoch-unit", choices=["s", "ms", "us", "ns"])
    parser.add_argument("--table", default="rates")
    parser.add_argument("--symbol-column")
    parser.add_argument("--symbol", default="EURUSD", help="Filtro del archivo local")
    parser.add_argument("--closures", type=Path, help="JSON con cierres justificados")
    parser.add_argument("--publication-delay", type=int, default=30, help="Minutos; solo descarga remota")
    parser.add_argument("--force", action="store_true", help="Ignorar caché y releer el origen")
    args = vars(parser.parse_args())
    args["closures"] = read_closures(args["closures"])
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        df = load_data(**args)
    except Exception:
        LOG.exception("Ingesta fallida; no se publica una caché nueva")
        raise SystemExit(1)
    LOG.info("Rango UTC: %s a %s", df.timestamp.iloc[0], df.timestamp.iloc[-1])


if __name__ == "__main__":
    main()
