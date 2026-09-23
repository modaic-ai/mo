#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=modaic/mo-1.1-fp8}"
: "${MODEL_REVISION:=60cd2356f6f4e9ec83967aa4f484c26a81f08b88}"
: "${SERVED_MODEL_NAME:=mo}"
: "${VLLM_GPU_MEMORY_UTILIZATION:=0.90}"
: "${VLLM_MAX_MODEL_LEN:=32768}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=65536}"
: "${VLLM_MAX_NUM_SEQS:=128}"
: "${VLLM_MAX_IMAGES_PER_PROMPT:=4}"

args=(
  serve "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --dtype bfloat16
  --tensor-parallel-size 1
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --enable-prefix-caching
  --generation-config vllm
  --limit-mm-per-prompt "{\"image\":${VLLM_MAX_IMAGES_PER_PROMPT},\"video\":0}"
  --max-logprobs 255
  --logprobs-mode raw_logprobs
  --disable-log-stats
  --host 0.0.0.0
  --port 8000
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=(--api-key "${VLLM_API_KEY}")
fi

exec vllm "${args[@]}"
