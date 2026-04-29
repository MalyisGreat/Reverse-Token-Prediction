#!/bin/bash
set -euo pipefail

# Reverse-token nanochat run for a single 8xH100 node.
# Recommended RunPod template: official RunPod PyTorch on H100 SXM.
# If entering an image manually, prefer:
#   runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204
# It keeps nanochat's high-throughput pretraining stack, but trains the base
# model on BOS + reversed token rows so the causal objective predicts the
# previous token instead of the next token.

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat_reverse}"
mkdir -p "$NANOCHAT_BASE_DIR"
LOG_DIR="${LOG_DIR:-$NANOCHAT_BASE_DIR/logs}"
mkdir -p "$LOG_DIR"

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
RUN_LOG="${RUN_LOG:-$LOG_DIR/${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log}"
DATA_SHARDS_BOOTSTRAP="${DATA_SHARDS_BOOTSTRAP:-8}"
DATA_SHARDS="${DATA_SHARDS:-170}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
EVAL_EVERY="${EVAL_EVERY:-250}"
EVAL_TOKENS="${EVAL_TOKENS:-41943040}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
SAMPLE_EVERY="${SAMPLE_EVERY:--1}"
FP8_FLAG="${FP8_FLAG:---fp8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-}"

exec > >(tee -a "$RUN_LOG") 2>&1
echo "logging to $RUN_LOG"

echo "nanochat reverse H100 run"
echo "base_dir=$NANOCHAT_BASE_DIR"
echo "model_tag=$MODEL_TAG"
echo "depth=$DEPTH ratio=$TARGET_PARAM_DATA_RATIO nproc=$NPROC_PER_NODE device_batch=$DEVICE_BATCH_SIZE"
echo "data_shards=$DATA_SHARDS save_every=$SAVE_EVERY eval_every=$EVAL_EVERY"
if [ "$WANDB_RUN" = "dummy" ]; then
  echo "wandb=disabled (WANDB_RUN=dummy); use logs/checkpoints/reverse_sample.sh to inspect the run"
else
  echo "wandb_run=$WANDB_RUN"
fi
if [ -n "$TOTAL_BATCH_SIZE" ]; then
  echo "total_batch_size=$TOTAL_BATCH_SIZE"
fi

python - <<'PY'
import os
import subprocess
import torch

expected = int(os.environ.get("NPROC_PER_NODE", "8"))
num_nodes = int(os.environ.get("NUM_NODES", "1"))
actual = torch.cuda.device_count()
print(f"cuda_visible_devices={actual} expected_gpus={expected} num_nodes={num_nodes}")
if num_nodes != 1:
    raise SystemExit(
        "This launcher is for a single 8-GPU node. "
        "Do not use a multi-node Instant Cluster without a multi-node torchrun launcher."
    )
if actual < expected:
    raise SystemExit(f"Expected at least {expected} CUDA devices, found {actual}")
try:
    print(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())
except Exception as exc:
    print(f"WARNING: nvidia-smi -L failed: {exc}")
PY

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

TRAIN_EXTRA_ARGS=()
if [ -n "$TOTAL_BATCH_SIZE" ]; then
  TRAIN_EXTRA_ARGS+=(--total-batch-size="$TOTAL_BATCH_SIZE")
fi

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
  "${TRAIN_EXTRA_ARGS[@]}" \
  $FP8_FLAG

RUN_FINAL_SAMPLE="${RUN_FINAL_SAMPLE:-1}"
if [ "$RUN_FINAL_SAMPLE" != "0" ]; then
  MODEL_TAG="$MODEL_TAG" \
  NUM_SAMPLES="${SAMPLE_COUNT:-2}" \
  MAX_TOKENS="${SAMPLE_MAX_TOKENS:-128}" \
    bash runs/reverse_history_probe.sh
fi

python -m nanochat.report generate
