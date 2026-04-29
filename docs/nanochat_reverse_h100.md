# Nanochat Reverse H100 Run

This repo now vendors upstream `karpathy/nanochat` under `nanochat_reverse/` and adds one focused change:

```text
forward nanochat row:
input:  <|bos|> x0 x1 x2 ...
target:         x0 x1 x2 ...

reverse nanochat row:
input:  <|bos|> xT xT-1 xT-2 ...
target:         xT-1 xT-2 xT-3 ...
```

The model is still a normal causal Transformer. The training rows are reversed after the leading BOS token, so the causal objective learns previous-token prediction.

## 8xH100 Launch

Run from the vendored nanochat folder on a Linux 8xH100 node:

```bash
cd nanochat_reverse
screen -L -Logfile reverse_speedrun.log -S reverse-nanochat \
  bash runs/reverse_speedrun_8xh100.sh
```

Useful overrides:

```bash
WANDB_RUN=reverse-d24 \
MODEL_TAG=reverse_d24_ratio8 \
DEPTH=24 \
TARGET_PARAM_DATA_RATIO=8 \
DEVICE_BATCH_SIZE=16 \
SAVE_EVERY=1000 \
DATA_SHARDS=170 \
bash runs/reverse_speedrun_8xh100.sh
```

Defaults are intentionally close to nanochat's 8xH100 speedrun:

- `torchrun --standalone --nproc_per_node=8`
- `--fp8` on H100
- depth `24`
- target param:data ratio `8`
- device batch size `16`
- local ClimbMix parquet shards, not live HF streaming during training
- checkpoints under `$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG`

CORE eval is disabled by default because a reverse-only model is not a normal forward question-answering model. Training still reports reverse validation BPB.

## Generation

After a checkpoint exists:

```bash
cd nanochat_reverse
source .venv/bin/activate
python -m scripts.reverse_generate \
  --model-tag reverse_d24_ratio8 \
  --anchor "the answer is 42." \
  --num-samples 4 \
  --max-tokens 128
```

The anchor is normal forward text. The script reverses the anchor tokens, samples in reverse order, then flips the full token stream back into readable text.

## Why Keep The Old Trainer?

`reverse_token_prediction_lab.py` is still useful for cheap local experiments and objective ablations. The vendored nanochat path is the performance path for rented H100 time.
