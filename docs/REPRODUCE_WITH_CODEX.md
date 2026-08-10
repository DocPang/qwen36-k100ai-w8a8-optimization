# Reproduce R269 with Codex

Use the prompt below after cloning this repository on a Linux host with **Hygon K100AI / gfx928**.

```text
Reproduce the current accepted Qwen3.6-35B-A3B W8A8 single-GPU result in this repository.

Hard requirements:
- The accelerator must be Hygon K100AI / gfx928. K100 is a different product.
- Use one GPU only (TP=1).
- Do not modify unrelated running services or GPUs.
- Use the pinned vLLM 0.18.1 / DTK 26.04 image from README.md.
- Use ModelScope checkpoint metax-tech/Qwen3.6-35B-A3B-W8A8 only.
- Do not substitute another quantization or vLLM release while claiming reproduction.
- Treat the first Triton/compile request as warmup, not steady-state throughput.

Procedure:
1. Check that Docker and the K100AI devices are usable.
2. Install ModelScope CLI if necessary: `python3 -m pip install -U modelscope`.
3. Set a free GPU and port, then run:
   `MODEL_DIR=$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU GPU_ID=0 PORT=8000 bash scripts/quickstart_r269.sh`
4. Confirm `/v1/models` is healthy.
5. Inspect container logs and confirm the R269 marker:
   `[K100 R269 split MoE stage2] exact M4 second-GEMM override installed`
6. Run:
   `python3 -m pip install -r requirements.txt`
   `python3 scripts/benchmark_fixed_512.py --base http://127.0.0.1:8000 --model qwen36-35b-a3b-w8a8-k100ai --rounds 6 --max-tokens 512 --out results/reproduction_r269.json`
7. Record every run, mean, median and SHA256. On the validated stack, the formal same-GPU reference is about 106.3 tok/s; GPU7 reached about 107.46 tok/s median.
8. The reference fixed-512 SHA256 is `80c82006a973ecc78fa3fb7a8483b76bc311693bdf277cb296365be0db6c7e00`. If the output differs, report the difference instead of silently treating it as the same reproduction.
9. If a generic webpage benchmark is also used, label it separately. Its random prompts can produce lower or higher MTP acceptance and should not replace the fixed benchmark.
10. Check `Prefix cache hit rate` when investigating cache effects. Prefix Cache primarily affects repeated-prefix prefill/TTFT and should not be assumed to explain decode changes.
11. Read `docs/R269_RELEASE_NOTES.md` for the accepted optimization and quality gates before attempting further changes.
12. Run `python3 scripts/check_release.py` and `bash -n scripts/*.sh` before publishing any modification.

Safety / scientific rules:
- Never claim a kernel microbenchmark as an end-to-end win without full-model A/B.
- Preserve correctness records: success, output hash, and token/logprob checks where applicable.
- Do not steal a production port or GPU for testing.
- Keep changed launch parameters in the benchmark report.
```

The public scripts deliberately take paths, GPU IDs and ports from environment variables so the repository does not depend on the original private infrastructure.
