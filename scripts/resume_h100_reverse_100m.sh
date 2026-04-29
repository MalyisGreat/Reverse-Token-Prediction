#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESUME="${RESUME:-runs_h100_reverse_100m/reverse_only/last.pt}"
OUT_DIR="${OUT_DIR:-runs_h100_reverse_100m_continue}"
CACHE_DIR="${CACHE_DIR:-cache_h100_reverse_100m}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer_h100_reverse_8192}"
LOG_DIR="${LOG_DIR:-logs}"
MODEL_NAME="${MODEL_NAME:-h100_reverse_100m_resume}"
EXPERIMENT="${EXPERIMENT:-reverse_only}"
DATA_BIN="${DATA_BIN:-}"
DATA_META="${DATA_META:-}"

ADDITIONAL_TOKENS="${ADDITIONAL_TOKENS:-3000000000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-1024}"
SEQ_LEN_SCHEDULE="${SEQ_LEN_SCHEDULE:-}"

VOCAB_SIZE="${VOCAB_SIZE:-8192}"
D_MODEL="${D_MODEL:-768}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-12}"
N_KV_HEADS="${N_KV_HEADS:-4}"
FFN_HIDDEN="${FFN_HIDDEN:-2560}"
DROPOUT="${DROPOUT:-0.0}"
ATTENTION_PATTERN="${ATTENTION_PATTERN:-local_global}"
SLIDING_WINDOW="${SLIDING_WINDOW:-512}"
GLOBAL_EVERY="${GLOBAL_EVERY:-6}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"

LEARNING_RATE="${LEARNING_RATE:-4e-5}"
MIN_LR="${MIN_LR:-4e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
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

if [ ! -f "$RESUME" ]; then
  echo "Resume checkpoint not found: $RESUME" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR" "$CACHE_DIR"

PREVIOUS_STEPS="$("$PYTHON_BIN" - "$RESUME" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(int(ckpt["step"]))
PY
)"
TOKENS_PER_STEP=$((BATCH_SIZE * GRAD_ACCUM_STEPS * SEQ_LEN))
ADDITIONAL_STEPS="$("$PYTHON_BIN" - <<PY
import math
print(math.ceil(int("$ADDITIONAL_TOKENS") / int("$TOKENS_PER_STEP")))
PY
)"
FINAL_STEPS=$((PREVIOUS_STEPS + ADDITIONAL_STEPS))
ACTUAL_ADDITIONAL_TOKENS=$((ADDITIONAL_STEPS * TOKENS_PER_STEP))
ACTUAL_TOTAL_TOKENS=$((FINAL_STEPS * TOKENS_PER_STEP))

echo "H100 reverse-only 100M resume"
echo "experiment=$EXPERIMENT"
echo "resume=$RESUME"
echo "previous_steps=$PREVIOUS_STEPS"
echo "additional_tokens_target=$ADDITIONAL_TOKENS"
echo "tokens_per_step=$TOKENS_PER_STEP"
echo "additional_steps=$ADDITIONAL_STEPS"
echo "final_absolute_steps=$FINAL_STEPS"
echo "actual_additional_tokens=$ACTUAL_ADDITIONAL_TOKENS"
echo "actual_total_tokens_after_run=$ACTUAL_TOTAL_TOKENS"
echo "model d=$D_MODEL layers=$N_LAYERS heads=$N_HEADS kv_heads=$N_KV_HEADS ffn=$FFN_HIDDEN vocab=$VOCAB_SIZE seq=$SEQ_LEN"
echo "attention pattern=$ATTENTION_PATTERN window=$SLIDING_WINDOW global_every=$GLOBAL_EVERY backend=$ATTENTION_BACKEND"
if [ -n "$SEQ_LEN_SCHEDULE" ]; then
  echo "seq_len_schedule=$SEQ_LEN_SCHEDULE"
fi
if [ -n "$DATA_BIN" ]; then
  echo "data_bin=$DATA_BIN"
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
  --experiment "$EXPERIMENT" \
  --resume "$RESUME" \
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
  --n_kv_heads "$N_KV_HEADS" \
  --ffn_hidden "$FFN_HIDDEN" \
  --dropout "$DROPOUT" \
  --seq_len "$SEQ_LEN" \
  --seq_len_schedule "$SEQ_LEN_SCHEDULE" \
  --attention_pattern "$ATTENTION_PATTERN" \
  --sliding_window "$SLIDING_WINDOW" \
  --global_every "$GLOBAL_EVERY" \
  --attention_backend "$ATTENTION_BACKEND" \
  --vocab_size "$VOCAB_SIZE" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --out_dir "$OUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --data_bin "$DATA_BIN" \
  --data_meta "$DATA_META" \
  --amp true \
  --amp_dtype "$AMP_DTYPE" \
  --tf32 true \
  --fused_adam "$FUSED_ADAM" \
  --progress_bar false \
  "${EXTRA_FLAGS[@]}" 2>&1 | tee "$LOG_PATH"
