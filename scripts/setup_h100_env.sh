#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt
if [ "${FLASH_ATTN_INSTALL:-0}" = "1" ]; then
  python -m pip install flash-attn --no-build-isolation
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("bf16_supported", torch.cuda.is_bf16_supported())
try:
    import flash_attn
except Exception as exc:
    print("flash_attn_available", False, exc)
else:
    print("flash_attn_available", True, getattr(flash_attn, "__version__", "unknown"))
PY
