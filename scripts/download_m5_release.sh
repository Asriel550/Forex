#!/usr/bin/env bash
# El CI consume la GitHub Release PRIVADA con un tag fijo, nunca artefactos temporales.
set -euo pipefail
dest="${1:-reports/m5-is-canonical}"
: "${GH_TOKEN:?GH_TOKEN de GitHub Actions o token read:contents requerido}"
: "${GITHUB_REPOSITORY:=Asriel550/Forex}"
mkdir -p "$dest"
# gh release create path#label conserva el nombre de origen como nombre del
# asset; los nombres descriptivos son etiquetas, no rutas descargables.
gh release download data-m5-is-2026h1-garch-v1 \
  --repo "$GITHUB_REPOSITORY" \
  --pattern 'a.npz' --pattern 'a.json' \
  --dir "$dest" --clobber
python -m scripts.verify_m5_durable_snapshot --snapshot "$dest/a.npz" --manifest "$dest/a.json"
