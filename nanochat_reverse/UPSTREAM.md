# Upstream Provenance

This directory vendors `karpathy/nanochat`.

- Upstream repository: https://github.com/karpathy/nanochat
- Upstream commit: `0aaca56`
- Vendored on: 2026-04-28

Local reverse-token additions:

- `scripts/base_train.py` adds `--reverse-training`
- `nanochat/dataloader.py` can reverse packed rows after BOS
- `scripts/base_eval.py` can evaluate reverse BPB with `--reverse-training`
- `scripts/reverse_generate.py` generates readable lead-in text from an ending anchor
- `runs/reverse_speedrun_8xh100.sh` launches the 8xH100 reverse pretraining run
- `runs/reverse_speedrun_5xh100.sh` launches the 5xH100 reverse pretraining run with compatible batch/vocab sharding
