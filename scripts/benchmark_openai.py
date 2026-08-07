#!/usr/bin/env python3
"""Small reproducible steady-state decode benchmark for an OpenAI-compatible vLLM endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time

import requests

DEFAULT_PROMPT = (
    "Explain how a production inference server should validate an optimization "
    "without confusing kernel microbenchmarks with end-to-end gains. Be concrete."
)


def one_run(endpoint: str, model: str, prompt: str, max_tokens: int, timeout: float):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first_content = None
    end = None
    chunks = []
    usage = None
    with requests.post(endpoint, json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            body = raw[6:]
            if body == "[DONE]":
                end = time.perf_counter()
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                reasoning = delta.get("reasoning_content") or ""
                piece = reasoning + text
                if piece:
                    if first_content is None:
                        first_content = time.perf_counter()
                    chunks.append(piece)
    if end is None:
        end = time.perf_counter()
    if first_content is None:
        raise RuntimeError("No streaming content received")
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError("Endpoint did not return completion token usage")
    completion_tokens = int(usage["completion_tokens"])
    decode_s = max(end - first_content, 1e-9)
    output = "".join(chunks)
    return {
        "completion_tokens": completion_tokens,
        "ttft_s": first_content - start,
        "decode_s": decode_s,
        "decode_tok_s": completion_tokens / decode_s,
        "total_s": end - start,
        "sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="qwen36-35b-a3b-w8a8-k100ai")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--output", help="optional JSON output path")
    args = ap.parse_args()

    rows = []
    for i in range(args.rounds):
        row = one_run(args.endpoint, args.model, args.prompt, args.max_tokens, args.timeout)
        row["round"] = i + 1
        rows.append(row)
        print(
            f"round={i+1} completion_tokens={row['completion_tokens']} "
            f"ttft={row['ttft_s']:.3f}s decode={row['decode_s']:.3f}s "
            f"decode={row['decode_tok_s']:.2f} tok/s sha256={row['sha256']}"
        )

    speeds = [r["decode_tok_s"] for r in rows]
    summary = {
        "endpoint": args.endpoint,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "rounds": rows,
        "decode_tok_s_median": statistics.median(speeds),
        "decode_tok_s_mean": statistics.mean(speeds),
        "all_outputs_identical": len({r["sha256"] for r in rows}) == 1,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rounds"}, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
