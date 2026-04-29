#!/bin/bash
set -euo pipefail

# W&B-free monitor for a RunPod training session. It prints the newest log tail
# and the checkpoint directory so a plain terminal/screen session is enough.

DEFAULT_BASE_DIR="$HOME/.cache/nanochat_reverse"
if [ -d /workspace ] && [ -w /workspace ]; then
  DEFAULT_BASE_DIR="/workspace/nanochat_reverse"
fi
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$DEFAULT_BASE_DIR}"
MODEL_TAG="${MODEL_TAG:-reverse_d24_ratio8}"
LOG_DIR="${LOG_DIR:-$NANOCHAT_BASE_DIR/logs}"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
INTERVAL="${INTERVAL:-30}"
LINES="${LINES:-80}"

while true; do
  if command -v tput &> /dev/null; then
    tput clear || true
  fi
  date
  echo
  echo "base_dir=$NANOCHAT_BASE_DIR"
  echo "model_tag=$MODEL_TAG"
  echo

  latest_log="$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)"
  if [ -n "$latest_log" ]; then
    echo "latest_log=$latest_log"
    echo "----- log tail -----"
    tail -n "$LINES" "$latest_log"
  else
    echo "No logs found yet in $LOG_DIR"
  fi

  echo
  echo "----- checkpoints -----"
  if [ -d "$CKPT_DIR" ]; then
    ls -lh "$CKPT_DIR" | tail -n 25
  else
    echo "No checkpoint directory yet: $CKPT_DIR"
  fi

  echo
  echo "Press Ctrl+C to stop watching. Sleeping ${INTERVAL}s..."
  sleep "$INTERVAL"
done
