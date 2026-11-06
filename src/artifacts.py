"""Huellas de archivos de un bundle; no sustituyen una firma digital."""
from pathlib import Path
import hashlib

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
