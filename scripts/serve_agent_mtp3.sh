#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the DCU-adapted Qwen3.6-35B-A3B-W8A8-DCU directory}"

IMAGE="${IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-k100ai-r202-agent-mtp3}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/r202-agent-mtp3}"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
PATCH_DIR="$ROOT_DIR/patches/r199_agent"
MOE_CONFIG="$ROOT_DIR/configs/E=256,N=512,device_name=K100_AI.json"
DCU_CONFIG_SHA256="b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae"

if command -v sha256sum >/dev/null 2>&1; then
  MODEL_CONFIG_SHA256="$(sha256sum "$MODEL_DIR/config.json" | awk '{print $1}')"
else
  MODEL_CONFIG_SHA256="$(shasum -a 256 "$MODEL_DIR/config.json" | awk '{print $1}')"
fi
if [[ "$MODEL_CONFIG_SHA256" != "$DCU_CONFIG_SHA256" ]]; then
  echo "ERROR: MODEL_DIR is not using the validated DCU config." >&2
  echo "Run: python3 $ROOT_DIR/scripts/apply_dcu_config.py --model-dir $MODEL_DIR" >&2
  echo "Current config SHA256: $MODEL_CONFIG_SHA256" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"

devices=(--device=/dev/kfd --device=/dev/dri)
[[ -e /dev/mkfd ]] && devices+=(--device=/dev/mkfd)

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --shm-size 16g \
  --privileged \
  "${devices[@]}" \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU_ID" \
  -e PYTHONPATH=/opt/k100ai-runtime-patch \
  -e K100AI_NATIVE_INT8_MOE=0 \
  -e K100AI_NATIVE_INT8_MOE_MAX_M=32 \
  -e K100AI_SC_LINEAR_A8W8=1 \
  -e K100AI_SC_LINEAR_A8W8_MAX_M=4 \
  -e K100AI_LM_HEAD_W8A8=1 \
  -e K100AI_LM_HEAD_W8A8_MAX_M=4 \
  -e K100AI_FUSED_QKVZ_BA=1 \
  -e K100AI_HCU_ALIGN_FASTPATH=1 \
  -e K100AI_HCU_GPU_ACCEPT_FASTPATH=1 \
  -e K100AI_LOG_LINEAR_LAYOUT=0 \
  -e K100AI_TRACE_LINEAR_SHAPES=0 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODEL_DIR":/models/qwen36-w8a8:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$PATCH_DIR":/opt/k100ai-runtime-patch:ro \
  -v "$MOE_CONFIG":/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=K100_AI.json:ro \
  "$IMAGE" \
  vllm serve /models/qwen36-w8a8 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --quantization compressed-tensors \
    --served-model-name qwen36-35b-a3b-w8a8-k100ai \
    --generation-config vllm \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    -cc.mode=3 \
    -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
    --cudagraph-capture-sizes 1 4 \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs 32 \
    --speculative-config '{"model":"/models/qwen36-w8a8","method":"qwen3_next_mtp","num_speculative_tokens":3,"quantization":"compressed-tensors"}'

echo "Started $CONTAINER_NAME on GPU $GPU_ID, port $PORT"
echo "Profile: R199 runtime fastpath + MTP3 + prefix cache + multimodal + chunked prefill=${MAX_NUM_BATCHED_TOKENS}"
echo "Follow startup: docker logs -f $CONTAINER_NAME"
