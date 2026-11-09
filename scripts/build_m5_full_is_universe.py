"""PR36: migrar universo M5 ya sellado a señales M5+M15 exclusivamente.

Los datos M1 se emplean SOLO como fuente de agregación OHLCV de M5 y M15.
No se calculan señales por minuto, GARCH nuevo, ni etiquetas desde el futuro.
El universo v1 era una prueba descartada con features M1; v2 elimina esos
campos e incorpora OHLC M5 + la última M15 completamente cerrada.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_adaptive_gate import RAW
from scripts.run_m5_direction_ablation import load_minutes
from scripts.run_m5_semester_replay import (
    END, PERIODS, SOURCE_MANIFEST, START, assert_isolated, verify_source_periods
)
from scripts.verify_m5_durable_snapshot import verify as verify_golden, SNAPSHOT_SHA256
from src.artifacts import file_sha256
from src.evolution_lineage import MARKET
from src.evolution_multiscale import FEATURE_NAMES
from src.reproducibility import market_frame_fingerprint, array_sha256
from src.resampling import aggregate
from src.upstream_filters import SIGNAL_NAMES, timeframe_features
from src.train import write_json


FORMAT = "pr36_full_m5_m15_is_universe_v2"
PARENT_FORMAT = "pr36_full_m5_is_universe_v1"
PARENT_SHA256 = "d8de896967ec419b89e92be61b0f6cad45165ddff297b8353fbf936db53a0d8d"
PARENT_MANIFEST_SHA256 = "ad0fd0e33f636c826f6eb63a1bb57b84f8837dff3c3c9e616bccda12ea0839b2"
ARRAYS = ("row_times_ns", "outcome", "volatility", "entry", "margin", "timeframe",
          "direction_inputs", "legacy_mask")
OLD_ARRAYS = tuple("micro" if key == "timeframe" else key for key in ARRAYS)
OUTPUT = Path("reports/m5-upstream-is")


def assert_universe(golden, archive):
    timestamps = np.asarray(archive["row_times_ns"], dtype="<i8")
    count = len(timestamps)
    if (not 30000 <= count <= 60000
            or archive["direction_inputs"].shape != (count, len(FEATURE_NAMES))
            or archive["timeframe"].shape != (count, len(SIGNAL_NAMES))
            or any(archive[key].shape != (count,) for key in
                   ("legacy_mask","outcome","entry","margin","volatility"))):
        raise ValueError("Universo M5+M15 incompleto")
    stamps = pd.to_datetime(timestamps, utc=True, unit="ns")
    if (not stamps.is_unique or not stamps.is_monotonic_increasing
            or stamps.min() < START or stamps.max() >= END
            or not np.isin(archive["outcome"], (-1, 0, 1)).all()
            or any(not np.isfinite(archive[key]).all() for key in
                   ("direction_inputs","timeframe","entry","margin","volatility"))):
        raise ValueError("El universo vulnera causalidad o intervalo IS")
    mask = archive["legacy_mask"]
    if (mask.dtype != np.dtype("?") or not mask.any()
            or not np.array_equal(timestamps[mask], golden["row_times_ns"])
            or not np.array_equal(archive["outcome"][mask], golden["outcome"])
            or not np.array_equal(archive["direction_inputs"][mask], golden["inputs"])
            or not np.array_equal(archive["volatility"][mask], golden["cond_volatility"])):
        raise ValueError("Las propuestas previas no reproducen EXACTAMENTE el snapshot Release")


def _read_golden(path, manifest):
    verify_golden(path, manifest)
    with np.load(path, allow_pickle=False) as handle:
        return {key:handle[key].copy() for key in handle.files}


def prepare(*, parent_path, parent_manifest, golden_path, golden_manifest,
            output_path=OUTPUT / "universe.npz", manifest_path=OUTPUT / "manifest.json"):
    """Migración determinista v1->v2: nunca se recalcula GARCH."""
    assert_isolated(PERIODS)
    golden = _read_golden(golden_path, golden_manifest)
    parent_meta = json.loads(Path(parent_manifest).read_text(encoding="utf-8"))
    if (file_sha256(parent_path) != PARENT_SHA256
            or file_sha256(parent_manifest) != PARENT_MANIFEST_SHA256
            or parent_meta.get("schema") != PARENT_FORMAT
            or parent_meta.get("golden_is_sha256") != SNAPSHOT_SHA256
            or parent_meta.get("oos_accessed") is not False):
        raise ValueError("La Release upstream v1 fuente no es la original sellada")
    with np.load(parent_path, allow_pickle=False) as handle:
        if set(handle.files) != set(OLD_ARRAYS):
            raise ValueError("El universo anterior contiene arrays desconocidos")
        old = {key:handle[key].copy() for key in OLD_ARRAYS}

    # La única lectura M1 es para agregar OHLCV a M5/M15 completos.
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    verify_source_periods(sources)
    minutes = load_minutes(RAW, PERIODS, end=END)
    m5 = aggregate(minutes, 1, "M5", now=END)
    m15 = aggregate(minutes, 1, "M15", now=END)
    gold_info = json.loads(Path(golden_manifest).read_text(encoding="utf-8"))
    m5_fingerprint = market_frame_fingerprint(
        m5, columns=("open","high","low","close","volume")
    )
    if m5_fingerprint != gold_info["fingerprints"]["m5"]:
        raise ValueError("Las velas OHLCV M5 no son idénticas a las fuentes originales")

    features = timeframe_features(m5, m15)
    opens = pd.to_datetime(old["row_times_ns"] - MARKET.step.value, utc=True, unit="ns")
    aligned = features.reindex(opens)
    if aligned.isna().any().any() or not aligned.index.equals(opens):
        raise ValueError("La lista congelada incluye una M5 que no existe en M5+M15")
    archive = {key:old[key] for key in OLD_ARRAYS if key != "micro"}
    archive["timeframe"] = aligned.to_numpy(dtype="<f8")
    assert_universe(golden, archive)
    target, meta_path = Path(output_path), Path(manifest_path)
    if target.exists() or meta_path.exists():
        raise ValueError("No se sobrescribe el snapshot M5+M15 sellado")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        np.savez_compressed(f, **archive)
    metadata = dict(
        schema=FORMAT, golden_is_sha256=SNAPSHOT_SHA256,
        golden_manifest_sha256=file_sha256(golden_manifest),
        parent_universe_sha256=PARENT_SHA256,
        parent_manifest_sha256=PARENT_MANIFEST_SHA256,
        source_manifest_sha256=file_sha256(SOURCE_MANIFEST),
        is_months=list(PERIODS), oos_accessed=False,
        start_utc=START.isoformat(), end_exclusive_utc=END.isoformat(),
        universe_sha256=file_sha256(target),
        rows=len(archive["outcome"]), original_proposals=int(archive["legacy_mask"].sum()),
        feature_names=list(FEATURE_NAMES), timeframe_signal_names=list(SIGNAL_NAMES),
        months_observed_hours=parent_meta["months_observed_hours"],
        m5_fingerprint=m5_fingerprint,
        arrays_sha256={key:array_sha256(value,dtype=(
            "<i8" if key == "row_times_ns" else "<i1" if key == "outcome"
            else "?" if key == "legacy_mask" else "<f8"
        )) for key,value in archive.items()},
        signal_timeframes=["M5","M15"],
        m1_role="raw ingestion and candle aggregation only; no M1 head inputs",
        status="IS research only: no July-September 2026 data accessed",
    )
    write_json(meta_path, metadata)
    print("FROZEN_M5_M15_IS " + json.dumps({
        "sha256":metadata["universe_sha256"],"rows":metadata["rows"],
        "old_proposals":metadata["original_proposals"],
        "signal_timeframes":metadata["signal_timeframes"],"oos_accessed":False,
    }), flush=True)


def load_universe(path, manifest, golden_path, golden_manifest):
    golden = _read_golden(golden_path, golden_manifest)
    info = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (info.get("schema") != FORMAT or info.get("golden_is_sha256") != SNAPSHOT_SHA256
            or info.get("golden_manifest_sha256") != file_sha256(golden_manifest)
            or info.get("parent_universe_sha256") != PARENT_SHA256
            or info.get("parent_manifest_sha256") != PARENT_MANIFEST_SHA256
            or info.get("source_manifest_sha256") != file_sha256(SOURCE_MANIFEST)
            or info.get("is_months") != list(PERIODS)
            or info.get("signal_timeframes") != ["M5","M15"]
            or info.get("oos_accessed") is not False
            or info.get("universe_sha256") != file_sha256(path)
            or info.get("feature_names") != list(FEATURE_NAMES)
            or info.get("timeframe_signal_names") != list(SIGNAL_NAMES)):
        raise ValueError("Universo M5+M15 no sellado o fuera del IS")
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != set(ARRAYS):
            raise ValueError("El universo contiene señales adicionales a M5+M15")
        archive = {key:f[key].copy() for key in ARRAYS}
    for name,value in archive.items():
        dtype = ("<i8" if name=="row_times_ns" else "<i1" if name=="outcome"
                 else "?" if name=="legacy_mask" else "<f8")
        if array_sha256(value,dtype=dtype) != info["arrays_sha256"][name]:
            raise ValueError("Array binario alterado: " + name)
    assert_universe(golden, archive)
    return archive, info


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("golden","golden_manifest","universe","manifest","parent_universe","parent_manifest"):
        parser.add_argument("--"+name.replace("_","-"),type=Path)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    path=args.universe or OUTPUT/"universe.npz"
    manifest=args.manifest or OUTPUT/"manifest.json"
    if args.golden is None or args.golden_manifest is None:
        parser.error("--golden y --golden-manifest obligatorios")
    if args.verify_only:
        _,meta=load_universe(path,manifest,args.golden,args.golden_manifest)
        print("M5_M15_IS_VERIFIED " + json.dumps({
            "sha256":meta["universe_sha256"],"rows":meta["rows"],
            "signal_timeframes":meta["signal_timeframes"],"oos_accessed":False,
        }),flush=True)
    else:
        if args.parent_universe is None or args.parent_manifest is None:
            parser.error("--parent-universe y --parent-manifest obligatorios para migrar v1")
        prepare(parent_path=args.parent_universe,parent_manifest=args.parent_manifest,
                golden_path=args.golden,golden_manifest=args.golden_manifest,
                output_path=path,manifest_path=manifest)
