"""Contrato de integridad del dataset IS M5 persistido en una GitHub Release privada.

La Release (no GitHub Actions artifacts) mantiene los bytes sin expiración
programada. El CI rechaza cambios, eliminaciones o sustituciones de activos
por SHA256. Ni el publicador ni el consumidor consulta Q3 2026.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

RELEASE_TAG = "data-m5-is-2026h1-garch-v1"
SNAPSHOT_NAME = "m5-is-2026h1-v1.npz"
MANIFEST_NAME = "m5-is-2026h1-v1.json"
SNAPSHOT_SHA256 = "7feb467de7ca66810ee336b3e29c0ba809f7587ecac1b335f05ac92007803504"
MANIFEST_SHA256 = "d115db5f4799aa3b6920d5d7539778c8b611452844211a98fc3e8ccc7f689f04"
GARCH_MC_SHA256 = "7e796d512ad12d5a546e8b6dacf94e72dd2c23ff7639efc65737687cbe037ef6"
MARCH_OPENS_SHA256 = "4c02cfbc1ff86d6ff2e6422ecb03be2633b8473deada1d23b1780da970e0dfe6"
PERIODS = tuple(f"2026{i:02d}" for i in range(1, 7))
EXPECTED_MEMBERS = {"row_times_ns.npy", "outcome.npy", "cond_volatility.npy",
                    "inputs.npy", "high_volatility.npy", "volatility_cutoff.npy"}


def sha_file(path):
    digest = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(snapshot, manifest):
    """Sello exacto de bytes, esquema, rango IS y fechas OOS excluidas."""
    snapshot, manifest = Path(snapshot), Path(manifest)
    if sha_file(snapshot) != SNAPSHOT_SHA256:
        raise ValueError("Snapshot GARCH distinto del SHA256 histórico canónico")
    if sha_file(manifest) != MANIFEST_SHA256:
        raise ValueError("Manifiesto distinto del SHA256 histórico canónico")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    if (meta.get("schema") != "pr34_m5_is_frozen_v1"
            or meta.get("snapshot_sha256") != SNAPSHOT_SHA256
            or meta.get("fingerprints", {}).get("garch_and_mc", {}).get("values_sha256")
            != GARCH_MC_SHA256
            or meta.get("hashes", {}).get("march_mask_sha256") != MARCH_OPENS_SHA256
            or tuple(meta.get("sources", [])) != PERIODS
            or meta.get("period_start") != "2026-01-01T00:00:00+00:00"
            or meta.get("period_end_exclusive") != "2026-07-01T00:00:00+00:00"
            or meta.get("oos_start") != "2026-07-01T00:00:00+00:00"
            or meta.get("oos_end_exclusive") != "2026-10-01T00:00:00+00:00"
            or meta.get("oos_accessed") is not False
            or meta.get("fingerprints", {}).get("march_audit", {}).get("matched") is not True
            or meta.get("fingerprints", {}).get("month_admissions", {}).get("202603", {}).get("proposals")
            != 2859):
        raise ValueError("El snapshot no respeta IS y máscara histórica de marzo")
    with ZipFile(snapshot) as data:
        if set(data.namelist()) != EXPECTED_MEMBERS or data.testzip() is not None:
            raise ValueError("Los componentes del snapshot binario están dañados")
    return {"tag": RELEASE_TAG, "snapshot_sha256": SNAPSHOT_SHA256,
            "manifest_sha256": MANIFEST_SHA256, "march_proposals": 2859,
            "periods": PERIODS, "oos_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.snapshot, args.manifest), sort_keys=True))
