#!/bin/bash
set -euo pipefail

# Factual reverse-generation probe. It loads the checkpoint once and samples
# several known history-fact ending anchors.

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_reverse}"

if [ ! -d ".venv" ]; then
  command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync --extra gpu
fi
source .venv/bin/activate

MODEL_TAG="${MODEL_TAG:-reverse_d24_ratio8}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
MAX_TOKENS="${MAX_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-40}"
SEED="${SEED:-42}"
DEVICE_TYPE="${DEVICE_TYPE:-}"
STEP="${STEP:-}"

ARGS=(
  --model-tag "$MODEL_TAG"
  --num-samples "$NUM_SAMPLES"
  --max-tokens "$MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --seed "$SEED"
)

if [ -n "$STEP" ]; then
  ARGS+=(--step "$STEP")
fi
if [ -n "$DEVICE_TYPE" ]; then
  ARGS+=(--device-type "$DEVICE_TYPE")
fi

python -m scripts.reverse_fact_probe "${ARGS[@]}" "$@"
