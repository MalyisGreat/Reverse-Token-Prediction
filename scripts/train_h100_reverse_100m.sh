#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-h100_reverse_100m}"
OUT_DIR="${OUT_DIR:-runs_h100_reverse_100m}"
CACHE_DIR="${CACHE_DIR:-cache_h100_reverse_100m}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer_h100_reverse_8192}"
LOG_DIR="${LOG_DIR:-logs}"

DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_FIELD="${TEXT_FIELD:-text}"

TARGET_TOKENS="${TARGET_TOKENS:-3000000000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-512}"

VOCAB_SIZE="${VOCAB_SIZE:-8192}"
TOKENIZER_TRAIN_TEXTS="${TOKENIZER_TRAIN_TEXTS:-300000}"
TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-2}"
RETRAIN_TOKENIZER="${RETRAIN_TOKENIZER:-0}"

D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-12}"
FFN_HIDDEN="${FFN_HIDDEN:-2304}"
DROPOUT="${DROPOUT:-0.0}"

LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"

VAL_BLOCKS="${VAL_BLOCKS:-8192}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-50000}"

AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
FUSED_ADAM="${FUSED_ADAM:-true}"
COMPILE="${COMPILE:-1}"
NO_TENSORBOARD="${NO_TENSORBOARD:-0}"

mkdir -p "$LOG_DIR" "$OUT_DIR" "$CACHE_DIR" "$TOKENIZER_DIR"

TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN))
FINAL_STEPS="$("$PYTHON_BIN" - <<PY
import math
print(math.ceil(int("$TARGET_TOKENS") / int("$TOKENS_PER_STEP")))
PY
)"
ACTUAL_TOKENS=$((FINAL_STEPS * TOKENS_PER_STEP))

echo "H100 reverse-only 100M training"
echo "target_tokens=$TARGET_TOKENS"
echo "tokens_per_step=$TOKENS_PER_STEP"
echo "steps=$FINAL_STEPS"
echo "actual_tokens=$ACTUAL_TOKENS"
echo "model d=$D_MODEL layers=$N_LAYERS heads=$N_HEADS ffn=$FFN_HIDDEN vocab=$VOCAB_SIZE seq=$SEQ_LEN"
echo "batch_size=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS workers=$NUM_WORKERS prefetch=$PREFETCH_FACTOR"

TOKENIZER_PATH="$TOKENIZER_DIR/byte_bpe_vocab${VOCAB_SIZE}.json"
TOKENIZER_FLAGS=()
if [ "$RETRAIN_TOKENIZER" = "1" ]; then
  TOKENIZER_FLAGS+=(--retrain_tokenizer)
fi

if [ ! -f "$TOKENIZER_PATH" ] || [ "$RETRAIN_TOKENIZER" = "1" ]; then
  "$PYTHON_BIN" -u reverse_token_prediction_lab.py \
    --mode tokenizer \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --dataset_split "$DATASET_SPLIT" \
    --text_field "$TEXT_FIELD" \
    --tokenizer_dir "$TOKENIZER_DIR" \
    --vocab_size "$VOCAB_SIZE" \
    --tokenizer_train_texts "$TOKENIZER_TRAIN_TEXTS" \
    --tokenizer_min_frequency "$TOKENIZER_MIN_FREQUENCY" \
    --shuffle_buffer "$SHUFFLE_BUFFER" \
    "${TOKENIZER_FLAGS[@]}"
fi

EXTRA_FLAGS=()
if [ "$COMPILE" = "1" ]; then
  EXTRA_FLAGS+=(--compile)
fi
if [ "$NO_TENSORBOARD" = "1" ]; then
  EXTRA_FLAGS+=(--no_tensorboard)
fi

LOG_PATH="$LOG_DIR/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log"

"$PYTHON_BIN" -u reverse_token_prediction_lab.py \
  --mode train \
  --experiment reverse_only \
  --dataset_name "$DATASET_NAME" \
  --dataset_config "$DATASET_CONFIG" \
  --dataset_split "$DATASET_SPLIT" \
  --text_field "$TEXT_FIELD" \
  --steps_per_experiment "$FINAL_STEPS" \
  --eval_interval "$EVAL_INTERVAL" \
  --save_interval "$SAVE_INTERVAL" \
  --log_interval "$LOG_INTERVAL" \
  --val_blocks "$VAL_BLOCKS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --shuffle_buffer "$SHUFFLE_BUFFER" \
  --learning_rate "$LEARNING_RATE" \
  --min_lr "$MIN_LR" \
  --warmup_steps "$WARMUP_STEPS" \
  --weight_decay "$WEIGHT_DECAY" \
  --d_model "$D_MODEL" \
  --n_layers "$N_LAYERS" \
  --n_heads "$N_HEADS" \
  --ffn_hidden "$FFN_HIDDEN" \
  --dropout "$DROPOUT" \
  --seq_len "$SEQ_LEN" \
  --vocab_size "$VOCAB_SIZE" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --out_dir "$OUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --amp true \
  --amp_dtype "$AMP_DTYPE" \
  --tf32 true \
  --fused_adam "$FUSED_ADAM" \
  --progress_bar false \
  "${EXTRA_FLAGS[@]}" 2>&1 | tee "$LOG_PATH"
