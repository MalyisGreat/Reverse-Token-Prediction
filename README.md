# Reverse Token Prediction

Reverse Token Prediction trains a causal language model on text in the opposite direction. Instead of learning `prefix -> next token`, the core experiment learns `suffix -> previous token`. At inference time, you provide an ending anchor and the model generates the text that plausibly leads into that ending.

This repo packages the working reverse-only MIRROR experiment with H100-oriented launch scripts for a roughly 100M parameter model.

## Core Idea

Normal causal LM:

```text
input:  x0 x1 x2 ...
target: x1 x2 x3 ...
```

Reverse causal LM:

```text
input:  xT xT-1 xT-2 ...
target: xT-1 xT-2 xT-3 ...
```

Displayed generation is reversed back into normal reading order, so a suffix like:

```text
the experiment proved the point.
```

can produce a paragraph that leads into that ending.

## H100 Quick Start

On a Linux H100 box:

```bash
git clone https://github.com/MalyisGreat/Reverse-Token-Prediction.git
cd Reverse-Token-Prediction

bash scripts/setup_h100_env.sh
source .venv/bin/activate

# Short correctness check.
bash scripts/smoke_test.sh

# Full reverse-only 100M run. Defaults target 3B tokens.
bash scripts/train_h100_reverse_100m.sh
```

The H100 launcher defaults to:

- `~106M` parameters
- `vocab_size=8192`
- `seq_len=512`
- `batch_size=256`
- `bfloat16` autocast
- fused AdamW
- optional `torch.compile`
- fixed reverse-only objective
- FineWeb-Edu streaming data

Override anything with environment variables:

```bash
TARGET_TOKENS=5000000000 BATCH_SIZE=384 COMPILE=1 bash scripts/train_h100_reverse_100m.sh
```

If `batch_size=256` is too conservative for your H100, try `384` or `512`. If your dataloader becomes the bottleneck, raise `NUM_WORKERS` and use local dataset cache storage.

## Resume

Resume a checkpoint and add more tokens:

```bash
RESUME=runs_h100_reverse_100m/reverse_only/last.pt \
ADDITIONAL_TOKENS=3000000000 \
bash scripts/resume_h100_reverse_100m.sh
```

The resume script reads the checkpoint step and computes the absolute `--steps_per_experiment` target automatically.

## Sample

```bash
CHECKPOINT=runs_h100_reverse_100m/reverse_only/best.pt \
PROMPT=" the result was clear." \
bash scripts/sample_reverse.sh
```

The prompt is an ending anchor, not a chat message. Good anchors look like full sentence endings:

```text
 the result was clear.
 and that was the strange part.
 therefore the answer is Paris.
```

Bare anchors like `42` are intentionally weak and tend to produce bibliography/date/table-shaped text.

## Files

- `reverse_token_prediction_lab.py` - single-file trainer/evaluator/generator
- `scripts/setup_h100_env.sh` - Python environment setup
- `scripts/train_h100_reverse_100m.sh` - H100 from-scratch reverse-only training
- `scripts/resume_h100_reverse_100m.sh` - H100 continuation training
- `scripts/smoke_test.sh` - toy local correctness check
- `scripts/sample_reverse.sh` - suffix-conditioned generation
- `docs/reverse_training_idea.md` - experiment rationale and expected behavior

## Notes

This is not an instruction/chat model. It is a suffix-conditioned reverse language model. For answer-to-reasoning experiments, fine-tune on data formatted like:

```text
Question: ...
Reasoning: ...
Therefore, the answer is ...
```

Then sample backward from the final answer line and validate the generated trace separately.
