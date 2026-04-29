#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Budget-first defaults. Override TARGET_TOKENS/PREPARE_TOKENS for larger runs.
export TARGET_TOKENS="${TARGET_TOKENS:-1000000000}"
export PREPARE_TOKENS="${PREPARE_TOKENS:-$((TARGET_TOKENS + TARGET_TOKENS / 10 + 50000000))}"
export DATA_DIR="${DATA_DIR:-/workspace/reverse_data}"
export VOCAB_SIZE="${VOCAB_SIZE:-8192}"
export DATA_BIN="${DATA_BIN:-$DATA_DIR/fineweb_edu_vocab${VOCAB_SIZE}_${PREPARE_TOKENS}.bin}"
export DATA_META="${DATA_META:-$DATA_BIN.json}"

export FLASH_ATTN_INSTALL="${FLASH_ATTN_INSTALL:-1}"
export COMPILE="${COMPILE:-1}"
export NO_TENSORBOARD="${NO_TENSORBOARD:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
export ATTENTION_PATTERN="${ATTENTION_PATTERN:-local_global}"
export SLIDING_WINDOW="${SLIDING_WINDOW:-512}"
export GLOBAL_EVERY="${GLOBAL_EVERY:-6}"
export N_KV_HEADS="${N_KV_HEADS:-4}"
export FFN_HIDDEN="${FFN_HIDDEN:-2560}"
export VAL_BLOCKS="${VAL_BLOCKS:-2048}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-20}"

echo "RunPod H100 speedrun"
echo "target_tokens=$TARGET_TOKENS"
echo "prepare_tokens=$PREPARE_TOKENS"
echo "data_bin=$DATA_BIN"

bash scripts/setup_h100_env.sh
source .venv/bin/activate

python - <<'PY'
import torch
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("cuda", torch.version.cuda)
try:
    import flash_attn
except Exception as exc:
    print("flash_attn missing:", exc)
else:
    print("flash_attn", getattr(flash_attn, "__version__", "unknown"))
PY

bash scripts/prepare_token_bin.sh

# Short synthetic attention benchmark before the expensive data-backed run.
STEPS="${BENCH_STEPS:-40}" BATCH_SIZE="${BENCH_BATCH_SIZE:-$BATCH_SIZE}" SEQ_LEN="$SEQ_LEN" bash scripts/bench_h100_attention.sh

bash scripts/train_h100_reverse_100m.sh
