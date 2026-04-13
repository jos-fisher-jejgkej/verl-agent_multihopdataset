#!/usr/bin/env bash
# =============================================================================
# serve_reranker.sh  –  Deploy a Qwen3-Reranker model via vLLM
#
# Usage:
#   bash scripts/serve_reranker.sh [MODEL] [PORT] [GPU_MEMORY_UTILIZATION] [GPUS]
#
# Examples:
#   bash scripts/serve_reranker.sh                                 # 0.6B, port 8000
#   bash scripts/serve_reranker.sh Qwen/Qwen3-Reranker-4B 8001    # 4B, port 8001
#   bash scripts/serve_reranker.sh Qwen/Qwen3-Reranker-8B 8002 0.85 2  # 8B, 2 GPUs
#
# Supported models:
#   Qwen/Qwen3-Reranker-0.6B  (default, ~1.2 GB VRAM, fits on any GPU)
#   Qwen/Qwen3-Reranker-4B    (~8 GB VRAM, good balance of speed / quality)
#   Qwen/Qwen3-Reranker-8B    (~16 GB VRAM, highest quality)
#
# The server exposes:
#   POST /score     – compute relevance score for (query, doc) pairs
#   POST /rerank    – rerank a list of passages (cohere-style)
#   POST /v1/rerank – same as above (OpenAI-style path)
#   POST /v2/rerank – same as above
#
# =============================================================================

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-Reranker-0.6B}"
PORT="${2:-8000}"
GPU_MEM="${3:-0.80}"
NUM_GPUS="${4:-1}"

HF_OVERRIDES='{"architectures":["Qwen3ForSequenceClassification"],"classifier_from_token":["no","yes"],"is_original_qwen3_reranker":true}'

echo "============================================================"
echo " vLLM Qwen3-Reranker deployment"
echo "  model : ${MODEL}"
echo "  port  : ${PORT}"
echo "  GPUs  : ${NUM_GPUS} x gpu_memory_utilization=${GPU_MEM}"
echo "============================================================"

vllm serve "${MODEL}" \
    --task score \
    --hf-overrides "${HF_OVERRIDES}" \
    --port "${PORT}" \
    --tensor-parallel-size "${NUM_GPUS}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --max-model-len 8192 \
    --trust-remote-code \
    --disable-log-requests
