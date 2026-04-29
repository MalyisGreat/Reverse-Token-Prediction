#!/bin/bash
set -euo pipefail

# 5xH100 variant of the reverse nanochat speedrun.
#
# Upstream nanochat's headline leaderboard is for 8xH100. This wrapper keeps the
# same d24/ratio8/fp8 speedrun model settings, but fixes the batch geometry for
# five ranks:
#
#   5 GPUs * device_batch 16 * seq 2048 = 163,840 tokens per micro-step
#   total batch 983,040 tokens = 6 gradient accumulation micro-steps
#
# The model code also pads vocab rows to a multiple that is safe for WORLD_SIZE=5,
# so the distributed AdamW/Muon optimizer can shard tensors without shape asserts.

export NPROC_PER_NODE="${NPROC_PER_NODE:-5}"
export DEPTH="${DEPTH:-24}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-983040}"
export MODEL_TAG="${MODEL_TAG:-reverse_d${DEPTH}_ratio${TARGET_PARAM_DATA_RATIO}_5xh100}"
export WANDB_RUN="${WANDB_RUN:-reverse-d${DEPTH}-ratio${TARGET_PARAM_DATA_RATIO}-5xh100}"

# Keep the first run observable and recoverable.
export SAVE_EVERY="${SAVE_EVERY:-500}"
export EVAL_EVERY="${EVAL_EVERY:-250}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:--1}"

bash runs/reverse_speedrun_8xh100.sh
