"""Huellas de archivos de un bundle; no sustituyen una firma digital."""
import json
import os
from pathlib import Path
import tempfile
import hashlib


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    """Metadatos atómicos, sin permitir NaN/Infinity silenciosos."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
