"""Construcción IS causal de TODAS las M5 elegibles para PR36.

El snapshot duradero PR35 solo contiene 9.968 propuestas del upstream viejo.
Una entrada nueva requiere features y outcome de las otras velas; no se permite
extrapolarlos desde ese subconjunto. Este constructor compara las 9.968 filas
históricas, M1/M5/GARCH, scores y etiquetas ANTES de sellar las demás.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_adaptive_gate import RAW
from scripts.run_m5_conviction_evolution import setup, OUTPUT as GEN375_OUTPUT
from scripts.run_m5_direction_ablation import load_minutes, m5_features, founder_observations
from scripts.run_m5_semester_replay import (END, MONTHS, PERIODS, Q2_START, START, SOURCE_MANIFEST,
                                            assert_isolated, verify_source_periods)
from scripts.verify_m5_durable_snapshot import verify as verify_golden, SNAPSHOT_SHA256
from src.artifacts import file_sha256
from src.decision import RiskConfig
from src.environment import BinaryOptionsEnv, prepare_market_data
from src.evolution_context import native_direction_context
from src.evolution_engine import M5Simulation
from src.evolution_lineage import MARKET, MARKET_FEATURES
from src.evolution_multiscale import M5_TACTICAL, FEATURE_NAMES, arm_inputs, scale_inputs
from scripts.run_m5_gen333 import ARM
from src.reproducibility import market_frame_fingerprint, array_sha256
from src.upstream_filters import PROXY_NAMES, microstructure_proxies, legacy_upstream
from src.train import write_json


FORMAT = "pr36_full_m5_is_universe_v1"
ARRAYS = ("row_times_ns", "outcome", "volatility", "entry", "margin", "micro",
          "direction_inputs", "legacy_mask")
HOURS = "observed_hours"
OUTPUT = Path("reports/m5-upstream-is")


def assert_universe(golden, archive):
    timestamps = archive["row_times_ns"]
    if (len(timestamps) < 30000 or len(timestamps) > 60000
            or archive["direction_inputs"].shape != (len(timestamps), len(FEATURE_NAMES))
            or archive["micro"].shape != (len(timestamps), len(PROXY_NAMES))
            or archive["legacy_mask"].shape != (len(timestamps),)
            or archive["outcome"].shape != (len(timestamps),)
            or archive["entry"].shape != (len(timestamps),)
            or archive["margin"].shape != (len(timestamps),)
            or archive["volatility"].shape != (len(timestamps),)):
        raise ValueError("Universo M5 incompleto")
    stamps = pd.to_datetime(timestamps, utc=True, unit="ns")
    if (not stamps.is_unique or not stamps.is_monotonic_increasing
            or stamps.min() < START or stamps.max() >= END
            or not np.isfinite(archive["direction_inputs"]).all()
            or not np.isfinite(archive["micro"]).all()
            or not np.isfinite(archive["entry"]).all()
            or not np.isfinite(archive["margin"]).all()
            or not np.isfinite(archive["volatility"]).all()
            or not np.isin(archive["outcome"], (-1, 0, 1)).all()):
        raise ValueError("El universo vulnera causalidad o intervalo IS")
    mask = archive["legacy_mask"]
    if (mask.dtype != np.dtype("?") or not mask.any()
            or not np.array_equal(timestamps[mask], golden["row_times_ns"])
            or not np.array_equal(archive["outcome"][mask], golden["outcome"])
            or not np.array_equal(archive["direction_inputs"][mask], golden["inputs"])
            or not np.array_equal(archive["volatility"][mask], golden["cond_volatility"])):
        raise ValueError("Las propuestas previas no reproducen EXACTAMENTE el snapshot Release")


def prepare(*, golden_path, golden_manifest, output_path=OUTPUT / "universe.npz",
            manifest_path=OUTPUT / "manifest.json"):
    assert_isolated(PERIODS)
    reference = verify_golden(golden_path, golden_manifest)
    golden_meta = json.loads(Path(golden_manifest).read_text(encoding="utf-8"))
    with np.load(golden_path, allow_pickle=False) as f:
        golden = {name: f[name].copy() for name in f.files}
    sources = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    verify_source_periods(sources)
    _, founder, quant, _, _, _, _, mean, std = setup()

    minutes = load_minutes(RAW, PERIODS, end=END)
    data = m5_features(minutes, quant, end=END)
    actual_quant = market_frame_fingerprint(
        data, columns=("cond_volatility","mc_prob_up","mc_prob_down","mc_prob_tie")
    )
    if actual_quant != golden_meta["fingerprints"]["garch_and_mc"]:
        raise ValueError("GARCH no reproduce SHA256 completo del snapshot canónico")
    prepared = prepare_market_data(data, market=MARKET)
    context, normalized, market_normalizer = founder_observations(prepared, founder)
    micro = microstructure_proxies(minutes, prepared)
    rows, original_masks, sigmas, entries, margins, outcomes, proxies, hours = [], [], [], [], [], [], [], {}
    for period, start, end in MONTHS:
        frame = prepared.loc[
            (prepared.timestamp >= start) & (prepared.timestamp + MARKET.step <= end)
        ]
        env = BinaryOptionsEnv(
            frame, market=MARKET, prepared=True, account=founder.account,
            normalizer=market_normalizer,
            risk=RiskConfig(enabled=False, drawdown_limit=founder.drawdown_limit),
        )
        sim = M5Simulation(env, normalized.loc[env._times])
        old_entry, old_direction = sim.decision_scores(founder.genome)
        margin = np.abs(old_direction)
        mask = legacy_upstream(old_entry, margin, float(founder.genome[-1]))
        opened = env._times[sim.entries]
        candidate_frame = frame.iloc[sim.entries]
        timestamps = opened + MARKET.step
        raw = pd.concat((
            candidate_frame.set_index("timestamp").loc[:, list(MARKET_FEATURES)],
            context.loc[opened],
        ), axis=1)
        raw.index = timestamps
        rows.append(raw)
        original_masks.append(mask)
        entries.append(old_entry)
        margins.append(margin)
        outcomes.append(sim.change.astype(np.int8))
        sigmas.append(candidate_frame.cond_volatility.to_numpy(float))
        proxies.append(micro.loc[opened, list(PROXY_NAMES)].to_numpy(dtype=float))
        hours[period] = float(sim.observed_hours)

    raw = pd.concat(rows)
    native = native_direction_context(minutes, raw.index, end=END)
    model_inputs = arm_inputs(scale_inputs(
        raw.loc[:, list(M5_TACTICAL)].join(native).loc[:, list(FEATURE_NAMES)].to_numpy(float),
        mean, std
    ), ARM)
    archive = dict(
        row_times_ns=raw.index.as_unit("ns").asi8.astype("<i8"),
        outcome=np.concatenate(outcomes).astype("i1"),
        volatility=np.concatenate(sigmas).astype("<f8"),
        entry=np.concatenate(entries).astype("<f8"),
        margin=np.concatenate(margins).astype("<f8"),
        micro=np.vstack(proxies).astype("<f8"),
        direction_inputs=np.asarray(model_inputs, dtype="<f8"),
        legacy_mask=np.concatenate(original_masks).astype("?"),
    )
    assert_universe(golden, archive)
    for period in PERIODS:
        t = pd.to_datetime(archive["row_times_ns"], utc=True, unit="ns")
        month = t.strftime("%Y%m") == period
        expected = golden_meta["fingerprints"]["month_admissions"][period]
        opens = t[month & archive["legacy_mask"]] - MARKET.step
        expected_sha = sha256(opens.as_unit("ns").asi8.tobytes()).hexdigest()
        if len(opens) != expected["proposals"] or expected_sha != expected["opens_sha256"]:
            raise ValueError("La máscara histórica mensual no coincide: " + period)
    path, manifest = Path(output_path), Path(manifest_path)
    if path.exists() or manifest.exists():
        raise ValueError("El universo IS es sellado: no sobrescribir binarios existentes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **archive)
    metadata = dict(
        schema=FORMAT,
        golden_is_sha256=reference["snapshot_sha256"],
        golden_manifest_sha256=file_sha256(golden_manifest),
        source_manifest_sha256=file_sha256(SOURCE_MANIFEST),
        is_months=list(PERIODS), oos_accessed=False,
        start_utc=START.isoformat(), end_exclusive_utc=END.isoformat(),
        universe_sha256=file_sha256(path),
        rows=len(archive["outcome"]), original_proposals=int(archive["legacy_mask"].sum()),
        feature_names=list(FEATURE_NAMES), proxy_names=list(PROXY_NAMES),
        months_observed_hours=hours,
        arrays_sha256={name:array_sha256(value,dtype=("<i8" if name=="row_times_ns"
                          else "<i1" if name=="outcome" else "?" if name=="legacy_mask"
                          else "<f8")) for name,value in archive.items()},
        status="IS research only: no July-September 2026 data accessed",
    )
    write_json(manifest, metadata)
    print("FROZEN_FULL_IS " + json.dumps({
        "sha256":metadata["universe_sha256"],"rows":metadata["rows"],
        "old_proposals":metadata["original_proposals"],"oos_accessed":False,
    }), flush=True)


def load_universe(path, manifest, golden_path, golden_manifest):
    verify_golden(golden_path, golden_manifest)
    info = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (info.get("schema") != FORMAT or info.get("golden_is_sha256") != SNAPSHOT_SHA256
            or info.get("golden_manifest_sha256") != file_sha256(golden_manifest)
            or info.get("source_manifest_sha256") != file_sha256(SOURCE_MANIFEST)
            or info.get("is_months") != list(PERIODS) or info.get("oos_accessed") is not False
            or info.get("universe_sha256") != file_sha256(path)
            or info.get("feature_names") != list(FEATURE_NAMES)
            or info.get("proxy_names") != list(PROXY_NAMES)):
        raise ValueError("Universo upstream no sellado o fuera del IS")
    with np.load(path, allow_pickle=False) as f:
        if set(f.files) != set(ARRAYS):
            raise ValueError("Array inesperado en el universo upstream")
        archive = {key:f[key].copy() for key in ARRAYS}
    for name,value in archive.items():
        dtype = "<i8" if name=="row_times_ns" else "<i1" if name=="outcome" else "?" if name=="legacy_mask" else "<f8"
        if array_sha256(value, dtype=dtype) != info["arrays_sha256"][name]:
            raise ValueError("Array individual del universo upstream alterado")
    with np.load(golden_path, allow_pickle=False) as f:
        golden = {key:f[key].copy() for key in f.files}
    assert_universe(golden, archive)
    return archive, info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--golden", type=Path, required=True)
    p.add_argument("--golden-manifest", type=Path, required=True)
    p.add_argument("--universe", type=Path, default=OUTPUT / "universe.npz")
    p.add_argument("--manifest", type=Path, default=OUTPUT / "manifest.json")
    p.add_argument("--verify-only", action="store_true",
                   help="Verificar Release congelada; NUNCA vuelve a ajustar GARCH")
    a = p.parse_args()
    if a.verify_only:
        archive, info = load_universe(a.universe, a.manifest, a.golden, a.golden_manifest)
        print("FROZEN_FULL_IS_VERIFIED " + json.dumps({
            "sha256":info["universe_sha256"],
            "rows":len(archive["outcome"]),
            "original_proposals":int(archive["legacy_mask"].sum()),
            "oos_accessed":False,
        }), flush=True)
    else:
        prepare(golden_path=a.golden,golden_manifest=a.golden_manifest,
                output_path=a.universe,manifest_path=a.manifest)
