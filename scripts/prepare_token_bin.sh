#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_NAME="${DATASET_NAME:-HuggingFaceFW/fineweb-edu}"
DATASET_CONFIG="${DATASET_CONFIG:-sample-10BT}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
TEXT_FIELD="${TEXT_FIELD:-text}"

VOCAB_SIZE="${VOCAB_SIZE:-8192}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer_h100_reverse_8192}"
TOKENIZER_TRAIN_TEXTS="${TOKENIZER_TRAIN_TEXTS:-300000}"
TOKENIZER_MIN_FREQUENCY="${TOKENIZER_MIN_FREQUENCY:-2}"
RETRAIN_TOKENIZER="${RETRAIN_TOKENIZER:-0}"

PREPARE_TOKENS="${PREPARE_TOKENS:-1100000000}"
PREPARE_CHUNK_TOKENS="${PREPARE_CHUNK_TOKENS:-4000000}"
DATA_DIR="${DATA_DIR:-/workspace/reverse_data}"
DATA_BIN="${DATA_BIN:-$DATA_DIR/fineweb_edu_vocab${VOCAB_SIZE}_${PREPARE_TOKENS}.bin}"
DATA_META="${DATA_META:-$DATA_BIN.json}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

SHUFFLE_BUFFER="${SHUFFLE_BUFFER:-50000}"
TRAIN_SKIP_TEXTS="${TRAIN_SKIP_TEXTS:-10000}"

mkdir -p "$DATA_DIR" "$TOKENIZER_DIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_XET_CACHE"

TOKENIZER_FLAGS=()
if [ "$RETRAIN_TOKENIZER" = "1" ]; then
  TOKENIZER_FLAGS+=(--retrain_tokenizer)
fi

if [ -f "$DATA_BIN" ] && [ -f "$DATA_META" ] && [ "${REBUILD_TOKEN_BIN:-0}" != "1" ]; then
  echo "token bin exists: $DATA_BIN"
  exit 0
fi

echo "Preparing token bin"
echo "data_bin=$DATA_BIN"
echo "prepare_tokens=$PREPARE_TOKENS"

"$PYTHON_BIN" -u reverse_token_prediction_lab.py \
  --mode prepare_tokens \
  --dataset_name "$DATASET_NAME" \
  --dataset_config "$DATASET_CONFIG" \
  --dataset_split "$DATASET_SPLIT" \
  --text_field "$TEXT_FIELD" \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --vocab_size "$VOCAB_SIZE" \
  --tokenizer_train_texts "$TOKENIZER_TRAIN_TEXTS" \
  --tokenizer_min_frequency "$TOKENIZER_MIN_FREQUENCY" \
  --shuffle_buffer "$SHUFFLE_BUFFER" \
  --train_skip_texts "$TRAIN_SKIP_TEXTS" \
  --prepare_tokens "$PREPARE_TOKENS" \
  --prepare_chunk_tokens "$PREPARE_CHUNK_TOKENS" \
  --data_bin "$DATA_BIN" \
  --data_meta "$DATA_META" \
  "${TOKENIZER_FLAGS[@]}"
