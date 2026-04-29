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

## RunPod Template

Use one single-node `8x H100 SXM` pod if RunPod has it available. Do not use a multi-node Instant Cluster for this launcher; `runs/reverse_speedrun_8xh100.sh` uses single-node `torchrun --standalone --nproc_per_node=8` and will now fail fast if RunPod exposes `NUM_NODES` greater than `1`.

Template choice:

- Best current manual image: `runpod/pytorch:1.0.3-cu1290-torch291-ubuntu2204`
- UI template name: official `RunPod PyTorch` / PyTorch CUDA template
- Older still-valid fallback from RunPod's PyTorch 2.8 guide: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

The repo runs `uv sync --extra gpu`, which installs nanochat's pinned `torch==2.9.1+cu128` stack into `.venv`. That means the RunPod template mainly needs working H100 drivers, SSH/Jupyter access, CUDA-compatible system libraries, and enough disk. Avoid community templates for the paid full run unless there is a specific reason to trust them.

Recommended pod settings:

- GPU: `8x H100 SXM`, not PCIe if SXM is available
- Pod type: normal Pod / single node
- Container disk plus volume: at least `500GB`; use more if keeping many checkpoints
- Template/image: official RunPod PyTorch, or the manual image above
- Ports: defaults are fine; SSH or web terminal is enough for launch

## 8xH100 Launch

Run from the vendored nanochat folder on the 8xH100 node:

```bash
git clone https://github.com/MalyisGreat/Reverse-Token-Prediction.git
cd Reverse-Token-Prediction/nanochat_reverse

WANDB_RUN=dummy \
screen -L -Logfile reverse_8xh100.log -S reverse-nanochat-8x \
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

Loss visibility:

- live console/screen output prints every training step: `loss`, `tok/sec`, `bf16_mfu`, elapsed time, and ETA
- validation BPB prints every `EVAL_EVERY` steps
- the launcher writes a persistent log under `$NANOCHAT_BASE_DIR/logs/`
- set `WANDB_RUN=...` to log `train/loss`, `train/tok_per_sec`, `train/mfu`, `val/bpb`, and timing plots to W&B
- leave `WANDB_RUN` unset, or set `WANDB_RUN=dummy`, to skip W&B entirely

CORE eval is disabled by default because a reverse-only model is not a normal forward question-answering model. Training still reports reverse validation BPB.

Monitor it with:

```bash
MODEL_TAG=reverse_d24_ratio8 bash runs/reverse_watch.sh
```

Sample the latest checkpoint with:

```bash
MODEL_TAG=reverse_d24_ratio8 \
ANCHOR="and that was the strange part." \
bash runs/reverse_sample.sh
```

## 5xH100 Fallback

Run from the vendored nanochat folder on a Linux 5xH100 node:

```bash
cd nanochat_reverse
screen -L -Logfile reverse_5xh100.log -S reverse-nanochat-5x \
  bash runs/reverse_speedrun_5xh100.sh
```

This wrapper keeps the same nanochat d24/ratio8/fp8 speedrun model settings, but adjusts the distributed geometry for five ranks:

- `NPROC_PER_NODE=5`
- `TOTAL_BATCH_SIZE=983040`
- `DEVICE_BATCH_SIZE=16`
- `SAVE_EVERY=500`
- `EVAL_EVERY=250`

The model pads vocabulary rows to a multiple that is safe for `WORLD_SIZE=5`, so distributed AdamW/Muon sharding will not fail on embedding or unembedding tensors.

Loss visibility:

- live console/screen output prints every training step: `loss`, `tok/sec`, `bf16_mfu`, elapsed time, and ETA
- validation BPB prints every `EVAL_EVERY` steps
- the launcher also writes a persistent log under `$NANOCHAT_BASE_DIR/logs/`
- set `WANDB_RUN=...` to log `train/loss`, `train/tok_per_sec`, `train/mfu`, `val/bpb`, and timing plots to W&B

Important wall-clock note: the official 1.65 hour autoresearch round 2 number is for an 8xH100 node. A full d24/ratio8 run on 5 H100s should be expected to take roughly `8/5` as long before overhead, so under 2 hours is not a safe assumption unless you reduce the target ratio or depth.

To trade quality for a tighter wall-clock budget on 5xH100s, run:

```bash
cd nanochat_reverse
TARGET_PARAM_DATA_RATIO=5 MODEL_TAG=reverse_d24_ratio5_5xh100 \
  bash runs/reverse_speedrun_5xh100.sh
```

That is a budget run, not the official speedrun-equivalent data budget.

## RunPod / Autoresearch Notes

For a direct training run, use a normal Linux multi-H100 pod with CUDA/PyTorch support and persistent storage. The script runs `uv sync --extra gpu`, so it installs nanochat's pinned PyTorch/CUDA stack instead of relying on a preloaded image.

Karpathy's `autoresearch` repo is a research loop for discovering training-code improvements through many short trials. It is useful for future ablations, but it is not the fastest way to execute this 8xH100 reverse pretraining run. The speed-critical path here is the vendored nanochat trainer itself: local ClimbMix shards, FP8 on H100, FA3 when available, Muon+AdamW, and torchrun across the visible GPUs.

Checklist for a rental pod:

- choose H100 SXM if possible, not PCIe, for best bandwidth
- attach enough persistent disk for ClimbMix shards, checkpoints, logs, and tokenizer artifacts
- make sure `nvidia-smi` shows all 8 GPUs before launch
- launch from `nanochat_reverse/`, not repo root
- use `WANDB_RUN=...` if you want live loss plots outside the terminal
- monitor `$NANOCHAT_BASE_DIR/logs/*.log` and `$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/`

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

The shorter wrapper is:

```bash
MODEL_TAG=reverse_d24_ratio8 \
ANCHOR="the answer is 42." \
bash runs/reverse_sample.sh
```

The anchor is normal forward text. The script reverses the anchor tokens, samples in reverse order, then flips the full token stream back into readable text.

## Why Keep The Old Trainer?

`reverse_token_prediction_lab.py` is still useful for cheap local experiments and objective ablations. The vendored nanochat path is the performance path for rented H100 time.
