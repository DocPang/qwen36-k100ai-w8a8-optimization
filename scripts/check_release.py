#!/usr/bin/env python3
"""Static release checks: syntax and accidental private-environment leakage."""
from __future__ import annotations

import pathlib
import py_compile
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in ROOT.glob("patches/*/sitecustomize.py"):
    py_compile.compile(str(p), doraise=True)

forbidden = [
    re.compile(r"/Users/[^/]+/", re.I),
    re.compile(r"/(?:data|srv|opt)/[^\s]+/models?/", re.I),
    re.compile(r"\b(?:2|10|172|192)\.\d+\.\d+\.\d+\b"),
]
violations = []
for p in ROOT.rglob("*"):
    if not p.is_file() or ".git" in p.parts or ".raw" in p.parts:
        continue
    if p.name == "check_release.py":
        continue
    try:
        text = p.read_text(errors="strict")
    except Exception:
        continue
    for rx in forbidden:
        if rx.search(text):
            violations.append((str(p.relative_to(ROOT)), rx.pattern))

if violations:
    for path, pattern in violations:
        print(f"privacy check failed: {path}: {pattern}")
    raise SystemExit(1)
print("release checks passed")
