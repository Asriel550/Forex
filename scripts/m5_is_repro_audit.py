"""Sellado y auditoría de dos reconstrucciones independientes de IS (Q1/Q2).

El experimento NO contiene archivos ni fechas de Q3. Cada proceso verifica los
ZIP M1 por SHA256 y mide M5, GARCH/MC, admisiones y matriz neuronal antes de
guardar un snapshot binario; la evolución solo consume el snapshot verificado.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_m5_adaptive_gate import GEN354_SHA256
from scripts.run_m5_conviction_evolution import OUTPUT as EVOLUTION_OUTPUT, setup
from scripts.run_m5_multiscale import MARCH_SIGNATURE, genome_hash
from scripts.run_m5_semester_replay import (
    END, OOS_END, PERIODS, Q2_START, START, SOURCE_MANIFEST, assert_isolated,
)
from src.artifacts import file_sha256
from src.evolution_lineage import MARKET
from src.reproducibility import array_sha256


SCHEMA = "pr34_m5_is_frozen_v1"
EXPECTED = {"row_times_ns", "outcome", "cond_volatility", "inputs",
            "high_volatility", "volatility_cutoff"}


def save_verified_snapshot(sample, path, report_path):
    """No sobrescribe un sello ni acepta datos que hayan cruzado 2026-07-01."""
    assert_isolated(PERIODS)
    path, report_path = Path(path), Path(report_path)
    if path.exists() or report_path.exists():
        raise ValueError("Los snapshots son inmutables; elegir nuevas rutas")
    rows = sample["rows"]
    if (rows.empty or rows.index.min() < START or rows.index.max() >= END
            or not rows.index.is_unique or not rows.index.is_monotonic_increasing
            or sample["hashes"]["march_mask_sha256"] != MARCH_SIGNATURE[2]):
        raise ValueError("IS no ha reproducido exactamente el histórico de marzo")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "row_times_ns": rows.index.as_unit("ns").asi8.astype("<i8"),
        "outcome": rows.outcome.to_numpy(dtype="<i1"),
        "cond_volatility": rows.cond_volatility.to_numpy(dtype="<f8"),
        "inputs": np.asarray(sample["inputs"], dtype="<f8"),
        "high_volatility": np.asarray(sample["high_vol"], dtype="?"),
        "volatility_cutoff": np.array([sample["vol_cut"]], dtype="<f8"),
    }
    with path.open("xb") as file:
        np.savez_compressed(file, **payload)
    report = dict(
        schema=SCHEMA, period_start=START.isoformat(), period_end_exclusive=END.isoformat(),
        oos_start=END.isoformat(), oos_end_exclusive=OOS_END.isoformat(),
        oos_accessed=False, sources=PERIODS,
        snapshot_sha256=file_sha256(path),
        hashes=sample["hashes"], fingerprints=sample["fingerprints"],
        quarter_observed_hours={q["name"]: q["observed_hours"] for q in sample["quarters"]},
        quarter_capacity_contracts_per_observed_hour={
            q['name']: q['capacity_contracts_per_observed_hour'] for q in sample['quarters']
        },
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def load_verified_snapshot(path, report_path):
    """Rechaza bytes, feeds, dimensiones y límites distintos del sello."""
    assert_isolated(PERIODS)
    if report_path is None:
        raise ValueError("Es obligatorio proporcionar el manifiesto SHA256 del snapshot")
    path, report_path = Path(path), Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema") != SCHEMA or report.get("snapshot_sha256") != file_sha256(path)
            or report.get("oos_accessed") is not False
            or tuple(report.get("sources", [])) != PERIODS
            or report.get("period_start") != START.isoformat()
            or report.get("period_end_exclusive") != END.isoformat()
            or report.get("oos_start") != END.isoformat()
            or report.get("oos_end_exclusive") != OOS_END.isoformat()
            or report["hashes"]["march_mask_sha256"] != MARCH_SIGNATURE[2]
            or report["hashes"]["source_manifest_sha256"] != file_sha256(SOURCE_MANIFEST)
            or report["hashes"]["gen375_evolution_sha256"] != file_sha256(
                EVOLUTION_OUTPUT / "evolution.json")):
        raise ValueError("Snapshot no sellado o incompatible con IS y fuentes")

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED:
            raise ValueError("Componentes binarios del snapshot incompatibles")
        arrays = {name: archive[name].copy() for name in EXPECTED}
    row_times = arrays["row_times_ns"]
    n = len(row_times)
    index = pd.to_datetime(row_times, utc=True, unit="ns")
    if (n < 501 or arrays["inputs"].ndim != 2 or arrays["inputs"].shape[0] != n
            or arrays["outcome"].shape != (n,) or arrays["cond_volatility"].shape != (n,)
            or arrays["high_volatility"].shape != (n,)
            or arrays["volatility_cutoff"].shape != (1,)
            or not index.is_unique or not index.is_monotonic_increasing
            or index[0] < START or index[-1] >= END
            or not np.isin(arrays["outcome"], [-1, 0, 1]).all()
            or not np.isfinite(arrays["inputs"]).all()
            or not np.isfinite(arrays["cond_volatility"]).all()
            or not np.array_equal(arrays["high_volatility"],
                                  arrays["cond_volatility"] > arrays["volatility_cutoff"][0])):
        raise ValueError("Contenido IS inválido o cruzando OOS")
    signatures = report["fingerprints"]
    if (signatures["model_inputs_sha256"] != array_sha256(arrays["inputs"])
            or signatures["outcomes_sha256"] != array_sha256(arrays["outcome"], dtype="<i1")
            or signatures["model_row_times_sha256"] != array_sha256(row_times, dtype="<i8")):
        raise ValueError("Datos neuronales difieren del registro de auditoría")
    march = index[(index >= pd.Timestamp("2026-03-01T00:00Z")) & (index < Q2_START)]
    from hashlib import sha256
    march_sig = sha256((march - MARKET.step).as_unit("ns").asi8.tobytes()).hexdigest()
    if len(march) != MARCH_SIGNATURE[1] or march_sig != MARCH_SIGNATURE[2]:
        raise ValueError("Snapshot adulteró las 2.859 admisiones originales")

    reference, founder, _, risk, _, _, _, _, _ = setup()
    evolution = json.loads((EVOLUTION_OUTPUT / "evolution.json").read_text(encoding="utf-8"))
    from scripts.run_m5_temporal_evolution import frozen_gen375
    parent = frozen_gen375(GEN354_SHA256, evolution)
    if genome_hash(parent) != report["hashes"]["founding_genome_sha256"]:
        raise ValueError("El snapshot no usa el fundador congelado Gen 375/43")
    rows = pd.DataFrame({
        "outcome": arrays["outcome"].astype(np.int8),
        "cond_volatility": arrays["cond_volatility"],
    }, index=index)
    masks = (index < Q2_START, index >= Q2_START)
    quarters = tuple(dict(
        name=name, rows=rows.loc[mask], mask=np.asarray(mask),
        observed_hours=float(report["quarter_observed_hours"][name]),
        capacity_contracts_per_observed_hour=float(
            report['quarter_capacity_contracts_per_observed_hour'][name]),
    ) for name, mask in zip(("Q1", "Q2"), masks, strict=True))
    if not all(q["observed_hours"] > 0 and not q["rows"].empty for q in quarters):
        raise ValueError("Horas de trimestre inválidas")
    return dict(
        rows=rows, inputs=arrays["inputs"], quarters=quarters,
        high_vol=arrays["high_volatility"], vol_cut=float(arrays["volatility_cutoff"][0]),
        parent=parent, account=founder.account, risk=risk,
        cap=reference["gate"]["hourly_cap"],
        hashes=report["hashes"], fingerprints=report["fingerprints"],
    )


def compare_independent_builds(one_path, two_path):
    a = json.loads(Path(one_path).read_text(encoding="utf-8"))
    b = json.loads(Path(two_path).read_text(encoding="utf-8"))
    for key in ("schema", "sources", "period_start", "period_end_exclusive",
                "hashes", "quarter_observed_hours", "fingerprints", "snapshot_sha256",
                "quarter_capacity_contracts_per_observed_hour"):
        if a[key] != b[key]:
            if key == "fingerprints":
                fields = sorted(set(a[key]) | set(b[key]))
                differences = [name for name in fields if a[key].get(name) != b[key].get(name)]
                raise ValueError("Divergencia entre reconstrucciones en: " + ", ".join(differences))
            raise ValueError(f"Divergencia entre reconstrucciones en: {key}")
    return a["snapshot_sha256"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    args = cli.parse_args()
    if args.action == "build":
        from scripts.run_m5_temporal_evolution import prepare_is
        audit_path = args.manifest.with_suffix(".stage-fingerprints.json")
        sample = prepare_is(audit_path=audit_path)
        result = save_verified_snapshot(sample, args.snapshot, args.manifest)
        # Validar que lo que consumirá el optimizador reproduce el sello.
        load_verified_snapshot(args.snapshot, args.manifest)
        print("IS_SNAPSHOT " + json.dumps({
            "sha256": result["snapshot_sha256"],
            "march_mask_sha256": result["hashes"]["march_mask_sha256"],
            "fingerprints": result["fingerprints"],
            "oos_accessed": False,
        }, allow_nan=False), flush=True)
    else:
        sha = compare_independent_builds(args.first, args.second)
        print("IS_REPRODUCIBLE " + sha, flush=True)


if __name__ == "__main__":
    main()
