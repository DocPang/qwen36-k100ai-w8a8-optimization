# Reproduce with Codex

Use this prompt after cloning the repository on a host equipped with **Hygon K100AI**.

```text
Reproduce the Qwen3.6-35B-A3B W8A8 K100AI inference results in this repository.

Hard requirements:
- The accelerator must be Hygon K100AI (gfx928). K100 is a different product; do not treat it as equivalent.
- Use one GPU only (TP=1) for each benchmark service.
- Do not modify unrelated running services or GPUs.
- Do not report a kernel microbenchmark as a win unless full-model steady-state throughput also improves.
- Keep a correctness record (request success, output SHA256, and when comparing the same inference mode, token/logprob checks where available).

Upstream runtime:
- Follow docs/SOURCES.md.
- Use Hygon community vLLM 0.18.1 / DTK 26.04 image:
  harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811

Model:
- Download ModelScope model metax-tech/Qwen3.6-35B-A3B-W8A8.
- Do not download or substitute a different quantization without clearly labeling the experiment.

Procedure:
1. Verify the GPU model is K100AI and the container reports the expected vLLM/DTK stack.
2. Set MODEL_DIR to the downloaded checkpoint.
3. Start scripts/serve_nomtp.sh on one free GPU.
4. Wait for /v1/models, issue one short warm-up request, then run scripts/benchmark_openai.py with max_tokens=512 for at least 3 steady-state rounds.
5. Record throughput and output hashes. R180 should be around 53-55 tok/s on the validated configuration.
6. Stop that test service cleanly.
7. Start scripts/serve_mtp3.sh on one free GPU.
8. Warm up, then run the same fixed benchmark. A conservative expected result is around 85 tok/s; speculative acceptance can make some workloads substantially faster.
9. Compare the result with results/RESULTS.md. Do not claim the 107 tok/s peak as a universal rate.
10. Save all changed files and benchmark output. If any source edit is made, run python -m py_compile on both sitecustomize.py patches before serving.
```

## Why this prompt avoids local infrastructure assumptions

The public launch scripts take `MODEL_DIR`, `GPU_ID`, `PORT`, `CACHE_DIR`, and `IMAGE` from environment variables. No private hostname, IP, user directory, production port, or local mount layout is required.
