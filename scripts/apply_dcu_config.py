#!/usr/bin/env python3
"""Apply the validated K100AI/DCU compressed-tensors config to the upstream W8A8 checkpoint.

The ModelScope checkpoint contains the W8A8 weights, but its original config.json
lacks the compressed-tensors quantization metadata used by the validated Hygon
vLLM 0.18.1 runtime. This script verifies the expected upstream config before
replacing it with the exact DCU-adapted config used for the published results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

UPSTREAM_CONFIG_SHA256 = "ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6"
DCU_CONFIG_SHA256 = "b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae"
CONFIG_NAME = "Qwen3.6-35B-A3B-W8A8-DCU.config.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Apply even if the current config hash is not the expected upstream hash.",
    )
    args = ap.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    target = model_dir / "config.json"
    source = Path(__file__).resolve().parents[1] / "configs" / CONFIG_NAME

    if not target.is_file():
        raise SystemExit(f"missing model config: {target}")
    if not source.is_file():
        raise SystemExit(f"missing repository DCU config: {source}")

    current_hash = sha256(target)
    source_hash = sha256(source)
    if source_hash != DCU_CONFIG_SHA256:
        raise SystemExit(
            f"repository DCU config hash mismatch: {source_hash} != {DCU_CONFIG_SHA256}"
        )

    if current_hash == DCU_CONFIG_SHA256:
        print("DCU config is already applied; nothing to do.")
        return

    if current_hash != UPSTREAM_CONFIG_SHA256 and not args.force:
        raise SystemExit(
            "refusing to overwrite an unexpected config.json. "
            f"got {current_hash}, expected upstream {UPSTREAM_CONFIG_SHA256}. "
            "Use --force only after manually verifying the checkpoint."
        )

    # Basic model identity guard before changing anything.
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("model_type") != "qwen3_5_moe":
        raise SystemExit(f"unexpected model_type: {data.get('model_type')!r}")

    backup = model_dir / "config.json.upstream.bak"
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)

    final_hash = sha256(target)
    if final_hash != DCU_CONFIG_SHA256:
        raise SystemExit(f"post-copy hash mismatch: {final_hash}")

    q = json.loads(target.read_text(encoding="utf-8"))["quantization_config"]
    print(f"applied DCU config: {target}")
    print(f"sha256: {final_hash}")
    print(
        "quantization: "
        f"{q['quant_method']} / {q['format']} / ignore={len(q.get('ignore', []))} entries"
    )
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
