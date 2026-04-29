#!/bin/bash
set -euo pipefail

# Reverse-token nanochat run for a single 8xH100 node.
# It keeps nanochat's high-throughput pretraining stack, but trains the base
# model on BOS + reversed token rows so the causal objective predicts the
# previous token instead of the next token.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_reverse}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ "${WANDB_RUN:-}" = "" ]; then
  WANDB_RUN=dummy
fi

DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODEL_TAG="${MODEL_TAG:-reverse_d${DEPTH}_ratio${TARGET_PARAM_DATA_RATIO}}"
DATA_SHARDS_BOOTSTRAP="${DATA_SHARDS_BOOTSTRAP:-8}"
DATA_SHARDS="${DATA_SHARDS:-170}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-41943040}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
FP8_FLAG="${FP8_FLAG:---fp8}"

echo "nanochat reverse 8xH100 run"
echo "base_dir=$NANOCHAT_BASE_DIR"
echo "model_tag=$MODEL_TAG"
echo "depth=$DEPTH ratio=$TARGET_PARAM_DATA_RATIO nproc=$NPROC_PER_NODE device_batch=$DEVICE_BATCH_SIZE"
echo "data_shards=$DATA_SHARDS save_every=$SAVE_EVERY eval_every=$EVAL_EVERY"

python -m nanochat.report reset

# Download enough local shards to train the tokenizer, then continue the larger
# dataset download while tokenizer training/eval runs.
python -m nanochat.dataset -n "$DATA_SHARDS_BOOTSTRAP"
python -m nanochat.dataset -n "$DATA_SHARDS" &
DATASET_DOWNLOAD_PID=$!

python -m scripts.tok_train
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
  --depth="$DEPTH" \
  --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
  --device-batch-size="$DEVICE_BATCH_SIZE" \
  --model-tag="$MODEL_TAG" \
  --run="$WANDB_RUN" \
  --reverse-training \
  --save-every="$SAVE_EVERY" \
  --eval-every="$EVAL_EVERY" \
  --eval-tokens="$EVAL_TOKENS" \
  --core-metric-every="$CORE_METRIC_EVERY" \
  --sample-every="$SAMPLE_EVERY" \
  $FP8_FLAG

python -m scripts.reverse_generate \
  --model-tag "$MODEL_TAG" \
  --anchor "${SAMPLE_ANCHOR:-the answer is 42.}" \
  --num-samples "${SAMPLE_COUNT:-4}" \
  --max-tokens "${SAMPLE_MAX_TOKENS:-128}"

python -m nanochat.report generate
