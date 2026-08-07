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

## ModelScope W8A8 checkpoint

The exact downloaded model card in the test environment identifies its upstream model ID as:

`metax-tech/Qwen3.6-35B-A3B-W8A8`

Public page:

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

The model card declares Apache License 2.0 and provides the same ModelScope download ID.

## Original Qwen model

Official Qwen3.6-35B-A3B model page:

<https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B>

Official Qwen release information:

<https://qwen.ai/blog?id=qwen3.6-35b-a3b>

## Important naming note

**K100 and K100AI are different accelerator products.** All performance numbers in this repository were obtained on K100AI. `K100_AI.json` is also the upstream vLLM device-config filename used by the tested Hygon runtime.
