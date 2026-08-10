#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="metax-tech/Qwen3.6-35B-A3B-W8A8"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU}"

if ! command -v modelscope >/dev/null 2>&1; then
  echo "ERROR: modelscope CLI not found." >&2
  echo "Install it first: python3 -m pip install -U modelscope" >&2
  exit 2
fi

mkdir -p "$MODEL_DIR"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Downloading $MODEL_ID to $MODEL_DIR"
  modelscope download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
else
  echo "Model directory already contains config.json; download step skipped."
fi

python3 "$ROOT_DIR/scripts/apply_dcu_config.py" --model-dir "$MODEL_DIR"
echo
printf 'Prepared model: %s\n' "$MODEL_DIR"
printf 'Next: MODEL_DIR=%q GPU_ID=0 PORT=8000 bash scripts/serve_r269_mtp3.sh\n' "$MODEL_DIR"
