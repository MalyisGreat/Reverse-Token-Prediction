#!/bin/bash
set -euo pipefail

# Minimal loss/progress view for W&B-free runs.

DEFAULT_BASE_DIR="$HOME/.cache/nanochat_reverse"
if [ -d /workspace ] && [ -w /workspace ]; then
  DEFAULT_BASE_DIR="/workspace/nanochat_reverse"
fi
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$DEFAULT_BASE_DIR}"
MODEL_TAG="${MODEL_TAG:-reverse_d24_ratio8}"
LOG_DIR="${LOG_DIR:-$NANOCHAT_BASE_DIR/logs}"
FOLLOW="${FOLLOW:-1}"

latest_log="$(ls -t "$LOG_DIR"/${MODEL_TAG}_*.log "$LOG_DIR"/*.log 2>/dev/null | head -n 1 || true)"
if [ -z "$latest_log" ]; then
  echo "No log found in $LOG_DIR"
  exit 1
fi

echo "latest_log=$latest_log"
PATTERN='Total number of training tokens|Total batch size|Validation bpb|step [0-9]+/[0-9]+ .*loss:|tok/sec|eta:|Flash Attention|window_pattern|Reverse training enabled'

if [ "$FOLLOW" = "1" ]; then
  grep -E "$PATTERN" "$latest_log" || true
  tail -n 0 -f "$latest_log" | grep --line-buffered -E "$PATTERN"
else
  grep -E "$PATTERN" "$latest_log" || true
fi
