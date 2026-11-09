#!/usr/bin/env bash
# El CI consume la GitHub Release PRIVADA con un tag fijo, nunca artefactos temporales.
set -euo pipefail
dest="${1:-reports/m5-is-canonical}"
: "${GH_TOKEN:?GH_TOKEN de GitHub Actions o token read:contents requerido}"
: "${GITHUB_REPOSITORY:=Asriel550/Forex}"
mkdir -p "$dest"
gh release download data-m5-is-2026h1-garch-v1 \
  --repo "$GITHUB_REPOSITORY" \
  --pattern 'm5-is-2026h1-v1.*' \
  --dir "$dest" --clobber
mv -f "$dest/m5-is-2026h1-v1.npz" "$dest/a.npz"
mv -f "$dest/m5-is-2026h1-v1.json" "$dest/a.json"
python -m scripts.verify_m5_durable_snapshot --snapshot "$dest/a.npz" --manifest "$dest/a.json"
