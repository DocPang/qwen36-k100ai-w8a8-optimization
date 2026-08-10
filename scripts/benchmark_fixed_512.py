#!/usr/bin/env python3
"""Reproduce the fixed-512 single-request benchmark used for R269 acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests

PROMPT = (
    "K100 single-GPU inference optimization fixed benchmark. Explain kernel "
    "tuning, quantization, speculative decoding, and profiling in a detailed "
    "technical report with numbered sections."
)


def one(base: str, model: str, max_tokens: int, timeout: float) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "ignore_eos": True,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base.rstrip('/')}/v1/completions", json=payload, timeout=timeout
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    text = str(data["choices"][0]["text"])
    tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
    if tokens != max_tokens:
        raise RuntimeError(f"expected {max_tokens} completion tokens, got {tokens}")
    return {
        "completion_tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": tokens / elapsed,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen36-35b-a3b-w8a8-k100ai")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.warmup_tokens > 0:
        warmup = one(args.base, args.model, args.warmup_tokens, args.timeout)
        print("warmup", json.dumps(warmup), flush=True)

    rows = []
    for i in range(args.rounds):
        row = one(args.base, args.model, args.max_tokens, args.timeout)
        row["round"] = i + 1
        rows.append(row)
        print(json.dumps(row), flush=True)

    speeds = [row["tok_s"] for row in rows]
    result = {
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "rows": rows,
        "median_tok_s": statistics.median(speeds),
        "mean_tok_s": statistics.fmean(speeds),
        "hashes": sorted({row["sha256"] for row in rows}),
    }
    print("SUMMARY", json.dumps(result, indent=2), flush=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
