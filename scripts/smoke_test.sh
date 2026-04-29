#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -u reverse_token_prediction_lab.py \
  --mode train \
  --experiment reverse_only \
  --toy_data \
  --steps_per_experiment 6 \
  --eval_interval 3 \
  --save_interval 3 \
  --val_blocks 8 \
  --batch_size 2 \
  --eval_batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 0 \
  --d_model 64 \
  --n_layers 2 \
  --n_heads 4 \
  --ffn_hidden 128 \
  --seq_len 64 \
  --vocab_size 512 \
  --tokenizer_train_texts 64 \
  --out_dir runs_smoke \
  --cache_dir cache_smoke \
  --tokenizer_dir tokenizer_smoke \
  --amp false \
  --progress_bar false \
  --rebuild_val

"$PYTHON_BIN" -u reverse_token_prediction_lab.py \
  --mode generate \
  --checkpoint runs_smoke/reverse_only/best.pt \
  --prompt " the result was clear." \
  --max_new_tokens 16 \
  --temperature 0.8 \
  --top_k 20 \
  --top_p 0.95 \
  --amp false
