# Upstream sources and provenance

This repository intentionally points users back to the original upstream artifacts instead of repackaging them.

## Hygon / SourceFind runtime

Tested environment source:

- Qwen3.6 ModelZoo v1.1 tag, explicitly labeled “新增K100AI支持，及FP8数据类型”:
  <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags>
- Environment revision that pins the vLLM 0.18.1 runtime image/version information:
  <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/commit/8a54c4e3df888c81165e5106657d17865bac3644>
- Current Qwen3.6 ModelZoo page:
  <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md>

The page documents DTK 26.04, vLLM 0.18.1 and the image:

`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`

Validated repository digest for the published results:

`sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811`

The current ModelZoo lists Qwen3.6 support for K100AI, and the v1.1 tag records the addition of K100AI support. Our launch scripts preserve `--disable-custom-all-reduce` from the tested K100AI runtime recipe.

## ModelScope W8A8 checkpoints and DCU adaptation

Historical server records show that two different Qwen3.6-35B-A3B W8A8 checkpoints were downloaded independently on June 10, 2026:

- `Eco-Tech/Qwen3.6-35B-A3B-w8a8`: an Ascend-oriented checkpoint stored as 10 `quant_model_weights-*.safetensors` shards. Shell history retains the explicit ModelScope download command. This checkpoint was **not** used for the final K100AI optimization results.
- `metax-tech/Qwen3.6-35B-A3B-W8A8`: the checkpoint used by this project, stored as 8 `model-*.safetensors` shards. ModelScope download-lock metadata was created at 15:49:23 (+08:00), immediately before files began appearing in the local DCU deployment directory.

The metax-tech provenance is independently confirmed by the local ModelScope `.msc` metadata: its per-file revision hashes are real upload commits in the public metax-tech repository, and representative local shard SHA256 values exactly match the corresponding Git LFS `oid sha256` values.

The **weight shards** in the validated K100AI test environment therefore came from:

`metax-tech/Qwen3.6-35B-A3B-W8A8`

Public page:

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

The server-side download metadata and file revisions match that ModelScope repository. The eight safetensors files were not re-quantized or rewritten after download.

There is an important reproducibility detail: the upstream repository's original `config.json` contains no `quantization_config`. During the June deployment, the checkpoint was used as a local DCU-adapted directory named `Qwen3.6-35B-A3B-W8A8-DCU`, with only `config.json` replaced by a compressed-tensors W8A8 configuration compatible with the tested Hygon vLLM 0.18.1 stack.

The repository includes that exact validated config as:

`configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json`

SHA256 provenance:

- upstream ModelScope config: `ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6`
- validated DCU config: `b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae`

Use `scripts/apply_dcu_config.py` after downloading the ModelScope weights. The public launch scripts deliberately refuse to start if the validated DCU config hash is not present.

## Original Qwen model

Official Qwen3.6-35B-A3B model page:

<https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B>

Official Qwen release information:

<https://qwen.ai/blog?id=qwen3.6-35b-a3b>

## Important naming note

**K100 and K100AI are different accelerator products.** All performance numbers in this repository were obtained on K100AI. `K100_AI.json` is also the upstream vLLM device-config filename used by the tested Hygon runtime.
