"""Stage a frozen, hash-pinned HistData M1 archive as M1 or complete M5 OHLCV."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.acquire_histdata import acquire
from scripts.build_histdata_dataset import read_native_minutes, sha
from src.data_loader import expected_grid, write_cache
from src.market import MarketConfig
from src.resampling import aggregate

SOURCE_YEAR = 2025
SOURCE_MANIFEST = ROOT / 'datasets/eurusd/manifest.json'


def prepare(raw_dir, output_dir, timeframe, source_manifest=SOURCE_MANIFEST):
    if timeframe not in ('M1', 'M5'):
        raise ValueError('Preparación de scalping requiere M1 o M5')
    manifest = json.loads(Path(source_manifest).read_text(encoding='utf-8'))
    source = next(row for row in manifest['sources'] if row['period'] == str(SOURCE_YEAR))
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / source['zip_file']
    if not archive.exists():
        acquire(raw_dir, SOURCE_YEAR)
    if sha(archive) != source['sha256']:
        raise ValueError('El ZIP HistData no coincide con el SHA256 del snapshot 2025 registrado')

    minutes, audit = read_native_minutes(raw_dir, [source])
    bars = aggregate(minutes, 1, timeframe)
    step = MarketConfig(timeframe).step
    missing = expected_grid(
        bars.timestamp.iloc[0], bars.timestamp.iloc[-1] + step, step=step,
    ).difference(pd.DatetimeIndex(bars.timestamp))
    exclusions = []
    for stamp in missing:
        end = (stamp + step).isoformat()
        if exclusions and exclusions[-1]['end'] == stamp.isoformat():
            exclusions[-1]['end'] = end
        else:
            exclusions.append(dict(
                start=stamp.isoformat(), end=end, kind='observed_source_gap',
                reason='Hueco observado en el snapshot HistData verificado',
            ))

    provenance = dict(
        provider='HistData.com', period=str(SOURCE_YEAR), source_url=source['source_url'],
        source_zip_sha256=source['sha256'], source_timeframe='M1', output_timeframe=timeframe,
        transformation='native M1 OHLCV' if timeframe == 'M1' else 'complete groups of 5 native M1 candles',
        source_audit=audit, rows=len(bars), missing_open_session_bars=len(missing),
        dropped_partial_bars=bars.attrs['dropped_partial_bars'],
    )
    bars.attrs['source_provenance'] = provenance
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f'EURUSD_{timeframe}_UTC_{SOURCE_YEAR}.parquet'
    write_cache(bars, output, source['sha256'])
    (output_dir / 'observed_gaps.json').write_text(json.dumps(exclusions) + '\n', encoding='utf-8')
    (output_dir / 'source_audit.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'{timeframe}: {len(bars)} velas cerradas y {len(missing)} ausencias documentadas')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeframe', choices=['M1', 'M5'], required=True)
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/histdata_scalping')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.timeframe)


if __name__ == '__main__':
    main()
