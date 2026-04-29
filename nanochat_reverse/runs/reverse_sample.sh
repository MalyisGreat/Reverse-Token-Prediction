#!/bin/bash
set -euo pipefail

# User-facing reverse-generation wrapper.
# Give it a normal ending anchor; it reverses tokens internally and prints
# readable forward text that leads into the anchor.

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_reverse}"

if [ ! -d ".venv" ]; then
  command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv sync --extra gpu
fi
source .venv/bin/activate

MODEL_TAG="${MODEL_TAG:-reverse_d24_ratio8}"
ANCHOR="${ANCHOR:-the answer is 42.}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_TOKENS="${MAX_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-42}"
DEVICE_TYPE="${DEVICE_TYPE:-}"
STEP="${STEP:-}"
SHOW_REVERSE="${SHOW_REVERSE:-0}"

ARGS=(
  --model-tag "$MODEL_TAG"
  --anchor "$ANCHOR"
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
if [ "$SHOW_REVERSE" = "1" ]; then
  ARGS+=(--show-reverse)
fi

python -m scripts.reverse_generate "${ARGS[@]}" "$@"
