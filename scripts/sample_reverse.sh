#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

CHECKPOINT="${CHECKPOINT:-runs_h100_reverse_100m/reverse_only/best.pt}"
PROMPT="${PROMPT:- the result was clear.}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-120}"
TEMPERATURE="${TEMPERATURE:-0.75}"
TOP_K="${TOP_K:-50}"
TOP_P="${TOP_P:-0.94}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

"$PYTHON_BIN" -u reverse_token_prediction_lab.py \
  --mode generate \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --amp true \
  --amp_dtype "$AMP_DTYPE"
