#!/usr/bin/env bash
set -euo pipefail

if [[ "${ALLOW_LEGACY_R269:-0}" != "1" ]]; then
  echo "R269 is a superseded fixed-MTP3 short-context benchmark profile." >&2
  echo "Use: bash scripts/quickstart.sh" >&2
  echo "For historical reproduction only: ALLOW_LEGACY_R269=1 bash scripts/quickstart_r269.sh" >&2
  exit 3
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU}"

bash "$ROOT_DIR/scripts/prepare_model.sh"
bash "$ROOT_DIR/scripts/serve_r269_mtp3.sh"
