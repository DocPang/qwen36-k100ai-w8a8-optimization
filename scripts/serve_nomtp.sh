#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the ModelScope metax-tech/Qwen3.6-35B-A3B-W8A8 checkpoint directory}"

IMAGE="${IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-k100ai-r180-nomtp}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/r180-nomtp}"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
PATCH_DIR="$ROOT_DIR/patches/r180_nomtp"
MOE_CONFIG="$ROOT_DIR/configs/E=256,N=512,device_name=K100_AI.json"

mkdir -p "$CACHE_DIR"

devices=(--device=/dev/kfd --device=/dev/dri)
[[ -e /dev/mkfd ]] && devices+=(--device=/dev/mkfd)

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
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
  -e K100AI_SC_LINEAR_A8W8=1 \
  -e K100AI_SC_LINEAR_A8W8_MAX_M=4 \
  -e K100AI_LM_HEAD_W8A8=1 \
  -e K100AI_LM_HEAD_W8A8_MAX_M=4 \
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
    --language-model-only \
    --generation-config vllm \
    --disable-custom-all-reduce \
    -cc.mode=3 \
    -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
    --cudagraph-capture-sizes 1 \
    --max-num-seqs 32

echo "Started $CONTAINER_NAME on GPU $GPU_ID, port $PORT"
echo "Follow startup: docker logs -f $CONTAINER_NAME"
