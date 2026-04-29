#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
STEPS="${STEPS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-1024}"
LOG_DIR="${LOG_DIR:-logs/bench_h100_attention}"
mkdir -p "$LOG_DIR"

COMMON=(
  --mode train
  --experiment reverse_only
  --toy_data
  --steps_per_experiment "$STEPS"
  --eval_interval "$STEPS"
  --save_interval "$STEPS"
  --log_interval 20
  --val_blocks 64
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$BATCH_SIZE"
  --grad_accum_steps 1
  --num_workers 0
  --d_model 768
  --n_layers 12
  --n_heads 12
  --dropout 0.0
  --seq_len "$SEQ_LEN"
  --vocab_size 8192
  --tokenizer_train_texts 2048
  --out_dir runs_h100_attention_bench
  --cache_dir cache_h100_attention_bench
  --tokenizer_dir tokenizer_h100_attention_bench
  --amp true
  --amp_dtype bfloat16
  --tf32 true
  --fused_adam true
  --compile
  --no_tensorboard
  --progress_bar false
  --rebuild_val
)

run_case() {
  local name="$1"
  shift
  echo "=== $name ==="
  "$PYTHON_BIN" -u reverse_token_prediction_lab.py "${COMMON[@]}" "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

run_case global_mha \
  --ffn_hidden 2304 \
  --n_kv_heads 12 \
  --attention_pattern global \
  --attention_backend sdpa

run_case global_gqa \
  --ffn_hidden 2560 \
  --n_kv_heads 4 \
  --attention_pattern global \
  --attention_backend auto

run_case gemma_local_global \
  --ffn_hidden 2560 \
  --n_kv_heads 4 \
  --attention_pattern local_global \
  --sliding_window 512 \
  --global_every 6 \
  --attention_backend auto
