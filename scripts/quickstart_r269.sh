#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU}"

bash "$ROOT_DIR/scripts/prepare_model.sh"
bash "$ROOT_DIR/scripts/serve_r269_mtp3.sh"
