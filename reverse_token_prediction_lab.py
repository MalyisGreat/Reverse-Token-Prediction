#!/usr/bin/env python3
"""
mirror5m_lab.py

A single-file experiment harness for testing many dense self-supervision ideas on a
~5M parameter local language model:

  - forward-only causal LM baseline
  - reverse-direction autoregressive training
  - direction-token baseline
  - direction-embedding / direction-adapter variants
  - forward/backward bridge token prediction
  - multi-token prediction
  - intermediate-layer next-token losses
  - latent future-state prediction
  - forward/backward representation alignment
  - layer-to-layer prediction
  - ELECTRA-ish replaced-token detection

The defaults target an NVIDIA RTX 2080 Ti class GPU and an i7-9700K class CPU:
small model, fp16 autocast, PyTorch SDPA attention, 2 dataloader workers, streaming
FineWeb-Edu, and a compact byte-level BPE tokenizer.

Install:
    pip install torch datasets tokenizers tensorboard tqdm numpy

Quick offline smoke test:
    python mirror5m_lab.py --mode sweep --toy_data --sweep_preset tiny \
      --steps_per_experiment 20 --eval_interval 10 --batch_size 4 --num_workers 0

Quick real-data sweep:
    python mirror5m_lab.py --mode sweep --sweep_preset quick

Single experiment:
    python mirror5m_lab.py --mode train --experiment mirror_bridge_mtp2 \
      --steps_per_experiment 1000

Generate from checkpoint:
    python mirror5m_lab.py --mode generate --checkpoint runs_mirror5m/mirror_bridge_mtp2/best.pt \
      --prompt "The cat went on" --max_new_tokens 80
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    load_dataset = None
    _DATASETS_IMPORT_ERROR = exc
else:
    _DATASETS_IMPORT_ERROR = None

try:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except Exception as exc:  # pragma: no cover
    Tokenizer = None
    _TOKENIZERS_IMPORT_ERROR = exc
else:
    _TOKENIZERS_IMPORT_ERROR = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from flash_attn import flash_attn_func
except Exception:  # pragma: no cover
    flash_attn_func = None


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<fwd>", "<rev>"]
PAD, UNK, BOS, EOS, FWD, REV = SPECIAL_TOKENS

SDPA_HAS_ENABLE_GQA = "enable_gqa" in (getattr(F.scaled_dot_product_attention, "__doc__", "") or "")

TOY_CORPUS = [
    "The cat went on a walk and found a quiet garden behind the old library.",
    "Small language models can learn useful structure when the objective is dense and stable.",
    "Forward prediction teaches prefixes to explain the next token.",
    "Reverse prediction teaches suffixes to explain the previous token.",
    "A bridge objective asks left context and right context to meet in the middle.",
    "The student opened the notebook, wrote a hypothesis, and tested it carefully.",
    "Good experiments compare simple baselines against one change at a time.",
    "A tiny transformer should be fast enough to debug before running overnight.",
    "Predictive coding can be translated into small auxiliary losses between hidden states.",
    "The model should generate normally even if it used future context during training.",
] * 10000


# -----------------------------
# Utilities
# -----------------------------


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    v = v.lower().strip()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {v!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_name_seed(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_int(n: int | float) -> str:
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(int(n))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device(args: argparse.Namespace) -> torch.device:
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def autocast_context(device: torch.device, enabled: bool, dtype_name: str = "float16"):
    if device.type == "cuda" and enabled:
        dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type=device.type, enabled=False)


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_int_list(s: str) -> List[int]:
    if s is None or not str(s).strip():
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_seq_len_schedule(schedule: str, total_steps: int, max_seq_len: int) -> List[Tuple[int, int]]:
    if not schedule or not schedule.strip():
        return [(0, max_seq_len)]
    entries: List[Tuple[int, int]] = []
    for item in schedule.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.replace("@", ":").split(":")
        if len(parts) == 1:
            seq_len = int(parts[0])
            start_step = 0
        elif len(parts) == 2:
            seq_len = int(parts[0])
            marker = parts[1].strip()
            if "." in marker:
                start_step = int(float(marker) * total_steps)
            else:
                start_step = int(marker)
        else:
            raise ValueError(f"bad --seq_len_schedule entry {item!r}")
        if seq_len <= 0:
            raise ValueError("--seq_len_schedule sequence lengths must be positive")
        if seq_len > max_seq_len:
            raise ValueError(f"--seq_len_schedule asks for {seq_len}, but --seq_len is only {max_seq_len}")
        entries.append((max(0, start_step), seq_len))
    if not entries:
        return [(0, max_seq_len)]
    return sorted(entries, key=lambda x: x[0])


def seq_len_at_step(args: argparse.Namespace, step: int, total_steps: int) -> int:
    schedule = parse_seq_len_schedule(args.seq_len_schedule, total_steps, args.seq_len)
    current = schedule[0][1]
    for start_step, seq_len in schedule:
        if step >= start_step:
            current = seq_len
        else:
            break
    return current


def crop_batch_to_seq_len(batch: torch.Tensor, seq_len: int) -> torch.Tensor:
    target_len = seq_len + 1
    if batch.size(1) <= target_len:
        return batch
    max_start = batch.size(1) - target_len
    start = int(torch.randint(0, max_start + 1, (1,), device=batch.device).item())
    return batch[:, start : start + target_len]


def lr_at_step(step: int, total_steps: int, base_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def maybe_compile(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    if args.compile and hasattr(torch, "compile"):
        try:
            return torch.compile(model)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[warn] torch.compile failed; continuing uncompiled: {exc}")
    return model


# -----------------------------
# Tokenizer and streaming data
# -----------------------------


def toy_text_stream(seed: int = 0) -> Iterator[str]:
    rng = random.Random(seed)
    data = TOY_CORPUS[:]
    while True:
        rng.shuffle(data)
        for text in data:
            yield text


def hf_text_stream(args: argparse.Namespace, seed: int, skip: int = 0) -> Iterator[str]:
    if args.toy_data:
        stream = toy_text_stream(seed)
        for _ in range(skip):
            next(stream)
        yield from stream
        return

    if load_dataset is None:
        raise RuntimeError(
            "datasets is not installed. Run: pip install datasets\n"
            f"Import error: {_DATASETS_IMPORT_ERROR}"
        )

    ds_kwargs = dict(path=args.dataset_name, split=args.dataset_split, streaming=True)
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    ds = load_dataset(**ds_kwargs)

    if args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=args.shuffle_buffer)

    if skip > 0:
        ds = ds.skip(skip)

    for ex in ds:
        text = ex.get(args.text_field, None)
        if text is None:
            # Fall back to the first string field.
            for v in ex.values():
                if isinstance(v, str):
                    text = v
                    break
        if isinstance(text, str) and text.strip():
            yield text


def train_or_load_tokenizer(args: argparse.Namespace) -> Tokenizer:
    if Tokenizer is None:
        raise RuntimeError(
            "tokenizers is not installed. Run: pip install tokenizers\n"
            f"Import error: {_TOKENIZERS_IMPORT_ERROR}"
        )

    tok_dir = safe_mkdir(args.tokenizer_dir)
    tok_path = tok_dir / f"byte_bpe_vocab{args.vocab_size}.json"
    if tok_path.exists() and not args.retrain_tokenizer:
        tok = Tokenizer.from_file(str(tok_path))
        return tok

    print(f"[{now()}] training byte-level BPE tokenizer: vocab_size={args.vocab_size}")
    tok = Tokenizer(BPE(unk_token=UNK))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.tokenizer_min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    def iterator() -> Iterator[str]:
        src = hf_text_stream(args, seed=args.seed + 100, skip=0)
        for i, text in enumerate(src):
            if i >= args.tokenizer_train_texts:
                break
            yield text

    tok.train_from_iterator(iterator(), trainer=trainer, length=args.tokenizer_train_texts)
    tok.save(str(tok_path))
    print(f"[{now()}] saved tokenizer to {tok_path}")
    return tok


def token_id(tok: Tokenizer, s: str) -> int:
    tid = tok.token_to_id(s)
    if tid is None:
        raise RuntimeError(f"tokenizer is missing special token {s!r}")
    return int(tid)


class PackedTokenDataset(IterableDataset):
    """Streams text, tokenizes it, and yields fixed-length token blocks.

    Each yielded tensor has length seq_len + 1. Standard LM training uses
    input block[:-1] and labels block[1:]. Direction-token training uses
    [<fwd>/<rev>] + block[:-1] and labels block with the first label ignored.
    """

    def __init__(self, args: argparse.Namespace, tok: Tokenizer, seed: int, skip_texts: int = 0):
        super().__init__()
        self.args = args
        self.tok = tok
        self.seed = seed
        self.skip_texts = skip_texts
        self.eos_id = token_id(tok, EOS)
        self.block_len = args.seq_len + 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        local_seed = self.seed + 9973 * worker_id

        # Make every worker see a different shuffle. This may duplicate some examples,
        # but it keeps streaming robust across HF iterable datasets and is fine for
        # short local sweeps.
        text_iter = hf_text_stream(self.args, seed=local_seed, skip=self.skip_texts)

        buf: List[int] = []
        for i, text in enumerate(text_iter):
            if num_workers > 1 and (i % num_workers) != worker_id:
                continue
            ids = self.tok.encode(text).ids
            if not ids:
                continue
            ids.append(self.eos_id)
            buf.extend(ids)
            while len(buf) >= self.block_len:
                block = buf[: self.block_len]
                del buf[: self.block_len]
                yield torch.tensor(block, dtype=torch.long)


def build_or_load_val_blocks(args: argparse.Namespace, tok: Tokenizer) -> torch.Tensor:
    val_dir = safe_mkdir(args.cache_dir)
    val_path = val_dir / f"val_blocks_vocab{args.vocab_size}_seq{args.seq_len}_n{args.val_blocks}.pt"
    if val_path.exists() and not args.rebuild_val:
        return torch.load(val_path, map_location="cpu")

    print(f"[{now()}] building fixed validation set: blocks={args.val_blocks}")
    eos_id = token_id(tok, EOS)
    block_len = args.seq_len + 1
    blocks: List[torch.Tensor] = []
    buf: List[int] = []
    text_iter = hf_text_stream(args, seed=args.seed + 200, skip=args.val_skip_texts)

    for text in text_iter:
        ids = tok.encode(text).ids
        if not ids:
            continue
        ids.append(eos_id)
        buf.extend(ids)
        while len(buf) >= block_len and len(blocks) < args.val_blocks:
            block = torch.tensor(buf[:block_len], dtype=torch.long)
            del buf[:block_len]
            blocks.append(block)
        if len(blocks) >= args.val_blocks:
            break

    if not blocks:
        raise RuntimeError("could not build validation set; no text was loaded")
    val = torch.stack(blocks, dim=0)
    torch.save(val, val_path)
    print(f"[{now()}] saved validation blocks to {val_path}")
    return val


# -----------------------------
# Model
# -----------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, T, Dh]
        t = q.size(-2)
        cos = self.cos[:t].to(dtype=q.dtype, device=q.device)[None, None, :, :]
        sin = self.sin[:t].to(dtype=q.dtype, device=q.device)[None, None, :, :]
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        return q, k


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        max_seq_len: int,
        n_kv_heads: int = 0,
        attention_kind: str = "global",
        sliding_window: int = 0,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_kv_heads <= 0:
            n_kv_heads = n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if attention_kind not in {"global", "local"}:
            raise ValueError(f"unknown attention_kind {attention_kind!r}")
        if attention_backend not in {"sdpa", "flash_attn", "auto"}:
            raise ValueError(f"unknown attention_backend {attention_backend!r}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_repeat = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_dim = n_kv_heads * self.head_dim
        self.qkv = nn.Linear(d_model, d_model + 2 * self.kv_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.attention_kind = attention_kind
        self.sliding_window = int(sliding_window)
        self.attention_backend = attention_backend
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)
        if attention_kind == "local":
            if self.sliding_window <= 0:
                raise ValueError("local attention requires --sliding_window > 0")
            pos = torch.arange(max_seq_len)
            q_pos = pos[:, None]
            k_pos = pos[None, :]
            mask = (k_pos <= q_pos) & (k_pos >= q_pos - (self.sliding_window - 1))
            self.register_buffer("local_mask", mask, persistent=False)
        else:
            self.local_mask = None

    def _use_flash_attn(self, x: torch.Tensor) -> bool:
        if self.attention_backend == "sdpa":
            return False
        if flash_attn_func is None:
            return False
        return x.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split([self.d_model, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        if self._use_flash_attn(x):
            window_size = (-1, -1)
            if self.attention_kind == "local":
                window_size = (self.sliding_window - 1, 0)
            y = flash_attn_func(
                q.transpose(1, 2).contiguous(),
                k.transpose(1, 2).contiguous(),
                v.transpose(1, 2).contiguous(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=window_size,
            )
            return self.out(y.reshape(b, t, c))

        attn_mask = None
        is_causal = True
        if self.attention_kind == "local":
            attn_mask = self.local_mask[:t, :t]
            is_causal = False

        if self.n_kv_heads != self.n_heads and not SDPA_HAS_ENABLE_GQA:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        sdpa_kwargs = {}
        if self.n_kv_heads != self.n_heads and SDPA_HAS_ENABLE_GQA:
            sdpa_kwargs["enable_gqa"] = True
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            **sdpa_kwargs,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


class DirectionAdapter(nn.Module):
    """Tiny direction-specific low-rank residual adapter.

    There are two adapters: index 0 for forward, index 1 for reverse. Keeping
    this small lets reverse prediction have a private subspace without changing
    the inference-time forward path much.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.rank = rank
        if rank <= 0:
            self.down = None
            self.up = None
        else:
            self.down = nn.Parameter(torch.zeros(2, rank, d_model))
            self.up = nn.Parameter(torch.zeros(2, d_model, rank))
            nn.init.normal_(self.down, mean=0.0, std=0.02)
            nn.init.zeros_(self.up)

    def forward(self, x: torch.Tensor, direction: int) -> torch.Tensor:
        if self.rank <= 0 or self.down is None or self.up is None:
            return torch.zeros_like(x)
        d = int(direction)
        z = F.linear(x, self.down[d])
        return F.linear(z, self.up[d])


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_hidden: int,
        dropout: float,
        max_seq_len: int,
        adapter_rank: int,
        n_kv_heads: int,
        attention_kind: str,
        sliding_window: int,
        attention_backend: str,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            max_seq_len,
            n_kv_heads=n_kv_heads,
            attention_kind=attention_kind,
            sliding_window=sliding_window,
            attention_backend=attention_backend,
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_hidden, dropout)
        self.drop = nn.Dropout(dropout)
        self.adapter_attn = DirectionAdapter(d_model, adapter_rank)
        self.adapter_mlp = DirectionAdapter(d_model, adapter_rank)

    def forward(self, x: torch.Tensor, direction: int, use_adapters: bool) -> torch.Tensor:
        a = self.attn(self.norm1(x))
        if use_adapters:
            a = a + self.adapter_attn(x, direction)
        x = x + self.drop(a)
        m = self.mlp(self.norm2(x))
        if use_adapters:
            m = m + self.adapter_mlp(x, direction)
        x = x + self.drop(m)
        return x


class TinyMirrorLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_layers: int = 10,
        n_heads: int = 6,
        n_kv_heads: int = 0,
        ffn_hidden: int = 512,
        max_seq_len: int = 257,
        dropout: float = 0.1,
        adapter_rank: int = 16,
        max_mtp_k: int = 4,
        attention_pattern: str = "global",
        sliding_window: int = 0,
        global_every: int = 6,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        if attention_pattern not in {"global", "local", "local_global"}:
            raise ValueError(f"unknown attention_pattern {attention_pattern!r}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads <= 0 else n_kv_heads
        self.max_seq_len = max_seq_len
        self.max_mtp_k = max_mtp_k
        self.attention_pattern = attention_pattern
        self.sliding_window = sliding_window
        self.global_every = global_every
        self.attention_backend = attention_backend

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dir_emb = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            attention_kind = self._attention_kind_for_layer(layer_idx)
            self.blocks.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    ffn_hidden,
                    dropout,
                    max_seq_len,
                    adapter_rank,
                    n_kv_heads=self.n_kv_heads,
                    attention_kind=attention_kind,
                    sliding_window=sliding_window,
                    attention_backend=attention_backend,
                )
            )
        self.norm = RMSNorm(d_model)

        # Auxiliary heads. They are present for all experiments so checkpoints are compatible.
        self.bridge = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.align_proj = nn.Linear(d_model, d_model, bias=False)
        self.latent_pred = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.layer_pred = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.rtd_head = nn.Linear(d_model, 1)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(max(0, max_mtp_k - 1))]
        )
        self.apply(self._init_weights)

    def _attention_kind_for_layer(self, layer_idx: int) -> str:
        if self.attention_pattern == "global":
            return "global"
        if self.attention_pattern == "local":
            return "local"
        if layer_idx == self.n_layers - 1:
            return "global"
        if self.global_every > 0 and (layer_idx + 1) % self.global_every == 0:
            return "global"
        return "local"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        return h @ self.tok_emb.weight.t()

    def forward(
        self,
        input_ids: torch.Tensor,
        direction: int = 0,
        use_direction_emb: bool = False,
        use_adapters: bool = False,
        return_layers: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(f"input length {input_ids.size(1)} > max_seq_len {self.max_seq_len}")
        if return_layers is None:
            return_layers = []
        return_set = set(int(x) for x in return_layers)
        x = self.tok_emb(input_ids)
        if use_direction_emb:
            d = torch.full((input_ids.size(0), input_ids.size(1)), int(direction), device=input_ids.device, dtype=torch.long)
            x = x + self.dir_emb(d)
        x = self.drop(x)

        layers: Dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            x = block(x, direction=int(direction), use_adapters=use_adapters)
            if i in return_set:
                layers[i] = x
        h = self.norm(x)
        logits = h @ self.tok_emb.weight.t()
        return logits, h, layers

    def bridge_logits(self, left_h: torch.Tensor, right_h: torch.Tensor) -> torch.Tensor:
        z = self.bridge(torch.cat([left_h, right_h], dim=-1))
        z = self.norm(z)
        return z @ self.tok_emb.weight.t()

    def mtp_logits(self, h: torch.Tensor, k: int) -> torch.Tensor:
        if k < 2 or k > self.max_mtp_k:
            raise ValueError(f"k must be in [2, {self.max_mtp_k}], got {k}")
        z = self.mtp_heads[k - 2](h)
        z = self.norm(z)
        return z @ self.tok_emb.weight.t()


# -----------------------------
# Experiments and losses
# -----------------------------


@dataclass
class Experiment:
    name: str
    primary_direction: int = 0
    reverse_weight: float = 0.0
    bridge_weight: float = 0.0
    mtp_weight: float = 0.0
    mtp_k: int = 2
    intermediate_weight: float = 0.0
    latent_weight: float = 0.0
    align_weight: float = 0.0
    layerpred_weight: float = 0.0
    rtd_weight: float = 0.0
    # If >0, use a single causal pass where each row is forward or reversed with
    # this probability. This is a fair-compute direction-token test: no second
    # reverse pass is computed. Only use with direction tokens.
    reverse_mix_prob: float = 0.0
    use_direction_token: bool = False
    use_direction_emb: bool = False
    use_adapters: bool = False
    anneal_reverse: bool = True
    notes: str = ""


def get_experiment_catalog() -> Dict[str, Experiment]:
    return {
        "forward_only": Experiment(
            name="forward_only",
            notes="plain causal LM baseline",
        ),
        "reverse_only": Experiment(
            name="reverse_only",
            primary_direction=1,
            notes="pure reverse causal LM: train on xT..x0 and predict the previous original token",
        ),
        "reverse_mtp2_low": Experiment(
            name="reverse_mtp2_low",
            primary_direction=1,
            mtp_weight=0.04,
            mtp_k=2,
            notes="reverse LM plus weak second-previous-token prediction auxiliary loss",
        ),
        "fwd_token_only": Experiment(
            name="fwd_token_only",
            use_direction_token=True,
            notes="forward-only with a <fwd> prefix token; controls for BOS/prefix benefit",
        ),
        "direction_token_rev002": Experiment(
            name="direction_token_rev002",
            reverse_weight=0.02,
            use_direction_token=True,
            anneal_reverse=True,
            notes="direction-token reverse CE with very weak annealed reverse weight 0.02",
        ),
        "direction_token_rev004": Experiment(
            name="direction_token_rev004",
            reverse_weight=0.04,
            use_direction_token=True,
            anneal_reverse=True,
            notes="direction-token reverse CE with weak annealed reverse weight 0.04",
        ),
        "direction_token_rev006": Experiment(
            name="direction_token_rev006",
            reverse_weight=0.06,
            use_direction_token=True,
            anneal_reverse=True,
            notes="direction-token reverse CE with weak annealed reverse weight 0.06",
        ),
        "direction_token_rev008": Experiment(
            name="direction_token_rev008",
            reverse_weight=0.08,
            use_direction_token=True,
            anneal_reverse=True,
            notes="same idea as direction_token_low; included for explicit weight sweep",
        ),
        "direction_token_rev012": Experiment(
            name="direction_token_rev012",
            reverse_weight=0.12,
            use_direction_token=True,
            anneal_reverse=True,
            notes="direction-token reverse CE with moderate annealed reverse weight 0.12",
        ),
        "direction_token_exit_low": Experiment(
            name="direction_token_exit_low",
            reverse_weight=0.08,
            intermediate_weight=0.03,
            use_direction_token=True,
            anneal_reverse=True,
            notes="best-looking direction-token signal plus weak intermediate-layer CE",
        ),
        "direction_token_mtp2_low": Experiment(
            name="direction_token_mtp2_low",
            reverse_weight=0.08,
            mtp_weight=0.04,
            mtp_k=2,
            use_direction_token=True,
            anneal_reverse=True,
            notes="best-looking direction-token signal plus weak 2-token future prediction",
        ),
        "direction_mix_p05": Experiment(
            name="direction_mix_p05",
            reverse_mix_prob=0.05,
            use_direction_token=True,
            notes="fair-compute mixture: 5% of rows reversed with <rev>, one pass only",
        ),
        "direction_mix_p10": Experiment(
            name="direction_mix_p10",
            reverse_mix_prob=0.10,
            use_direction_token=True,
            notes="fair-compute mixture: 10% of rows reversed with <rev>, one pass only",
        ),
        "direction_mix_p20": Experiment(
            name="direction_mix_p20",
            reverse_mix_prob=0.20,
            use_direction_token=True,
            notes="fair-compute mixture: 20% of rows reversed with <rev>, one pass only",
        ),
        "naive_reverse": Experiment(
            name="naive_reverse",
            reverse_weight=0.35,
            anneal_reverse=False,
            notes="shared model sees reversed sequences, no direction signal",
        ),
        "direction_token": Experiment(
            name="direction_token",
            reverse_weight=0.35,
            use_direction_token=True,
            anneal_reverse=False,
            notes="prepend <fwd>/<rev> token to distinguish direction",
        ),
        "direction_embedding": Experiment(
            name="direction_embedding",
            reverse_weight=0.35,
            use_direction_emb=True,
            anneal_reverse=False,
            notes="learned direction embedding added at every position",
        ),
        "reverse_low": Experiment(
            name="reverse_low",
            reverse_weight=0.08,
            anneal_reverse=True,
            notes="low-weight reversed sequences; tests whether reverse signal only needs to be weak",
        ),
        "direction_token_low": Experiment(
            name="direction_token_low",
            reverse_weight=0.08,
            use_direction_token=True,
            anneal_reverse=True,
            notes="direction-token baseline with weak annealed reverse loss",
        ),
        "direction_embedding_low": Experiment(
            name="direction_embedding_low",
            reverse_weight=0.08,
            use_direction_emb=True,
            anneal_reverse=True,
            notes="direction embedding with weak annealed reverse loss",
        ),
        "dir_adapter_reverse_low": Experiment(
            name="dir_adapter_reverse_low",
            reverse_weight=0.08,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="weak reverse signal with small direction-specific adapters",
        ),
        "mtp2_only_low": Experiment(
            name="mtp2_only_low",
            mtp_weight=0.04,
            mtp_k=2,
            notes="forward LM plus weak 2-token future prediction, no reverse pass",
        ),
        "exit_only_low": Experiment(
            name="exit_only_low",
            intermediate_weight=0.03,
            notes="forward LM plus weak intermediate-layer next-token losses, no reverse pass",
        ),
        "direction_embedding_mtp2_low": Experiment(
            name="direction_embedding_mtp2_low",
            reverse_weight=0.08,
            mtp_weight=0.04,
            mtp_k=2,
            use_direction_emb=True,
            anneal_reverse=True,
            notes="weak direction embedding reverse signal plus weak MTP-k2",
        ),
        "dir_adapter_reverse": Experiment(
            name="dir_adapter_reverse",
            reverse_weight=0.35,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="direction embedding plus tiny direction-specific adapters",
        ),
        "mirror_bridge": Experiment(
            name="mirror_bridge",
            reverse_weight=0.35,
            bridge_weight=0.10,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="reverse CE plus bridge CE from left/right states",
        ),
        "mirror_bridge_low": Experiment(
            name="mirror_bridge_low",
            reverse_weight=0.08,
            bridge_weight=0.02,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="weak reverse CE plus weak bridge CE; should test whether earlier bridge weights were too strong",
        ),
        "mirror_bridge_exit_low": Experiment(
            name="mirror_bridge_exit_low",
            reverse_weight=0.08,
            bridge_weight=0.02,
            intermediate_weight=0.03,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="weak bridge method plus weak intermediate CE",
        ),
        "mirror_bridge_mtp2_low": Experiment(
            name="mirror_bridge_mtp2_low",
            reverse_weight=0.08,
            bridge_weight=0.02,
            mtp_weight=0.04,
            mtp_k=2,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="weak bridge method plus weak 2-token future prediction",
        ),
        "mirror_bridge_mtp2": Experiment(
            name="mirror_bridge_mtp2",
            reverse_weight=0.35,
            bridge_weight=0.10,
            mtp_weight=0.08,
            mtp_k=2,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus 2-token future prediction",
        ),
        "mirror_bridge_mtp4": Experiment(
            name="mirror_bridge_mtp4",
            reverse_weight=0.35,
            bridge_weight=0.10,
            mtp_weight=0.10,
            mtp_k=4,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus 2/3/4-token future prediction",
        ),
        "mirror_bridge_exit": Experiment(
            name="mirror_bridge_exit",
            reverse_weight=0.35,
            bridge_weight=0.10,
            intermediate_weight=0.06,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus intermediate-layer next-token CE",
        ),
        "mirror_bridge_latent": Experiment(
            name="mirror_bridge_latent",
            reverse_weight=0.35,
            bridge_weight=0.10,
            latent_weight=0.05,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus mid-layer -> future final-state prediction",
        ),
        "mirror_bridge_align": Experiment(
            name="mirror_bridge_align",
            reverse_weight=0.35,
            bridge_weight=0.10,
            align_weight=0.03,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus weak forward/reverse cosine alignment",
        ),
        "mirror_bridge_layerpred": Experiment(
            name="mirror_bridge_layerpred",
            reverse_weight=0.35,
            bridge_weight=0.10,
            layerpred_weight=0.05,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus layer-to-layer representation prediction",
        ),
        "mirror_bridge_rtd": Experiment(
            name="mirror_bridge_rtd",
            reverse_weight=0.35,
            bridge_weight=0.10,
            rtd_weight=0.08,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="bridge method plus replaced-token detection on corrupted input",
        ),
        "mirror_full_light": Experiment(
            name="mirror_full_light",
            reverse_weight=0.30,
            bridge_weight=0.08,
            mtp_weight=0.05,
            mtp_k=2,
            intermediate_weight=0.03,
            latent_weight=0.03,
            use_direction_emb=True,
            use_adapters=True,
            anneal_reverse=True,
            notes="small combo of the best-looking dense signals",
        ),
    }


def experiments_for_preset(preset: str) -> List[str]:
    if preset == "tiny":
        return ["forward_only", "reverse_only", "naive_reverse", "mirror_bridge_mtp2"]
    if preset == "quick":
        return [
            "forward_only",
            "naive_reverse",
            "direction_token",
            "direction_embedding",
            "mirror_bridge",
            "mirror_bridge_mtp2",
            "mirror_bridge_exit",
            "mirror_bridge_latent",
            "mirror_full_light",
        ]
    if preset == "diagnostic":
        return [
            "forward_only",
            "fwd_token_only",
            "direction_token",
            "direction_token_low",
            "reverse_low",
            "direction_embedding_low",
            "dir_adapter_reverse_low",
            "mtp2_only_low",
            "exit_only_low",
            "direction_token_exit_low",
            "direction_token_mtp2_low",
            "mirror_bridge_low",
            "mirror_bridge_exit_low",
            "mirror_bridge_mtp2_low",
            "direction_embedding_mtp2_low",
        ]
    if preset == "isolate":
        return [
            "forward_only",
            "fwd_token_only",
            "direction_token_rev002",
            "direction_token_rev004",
            "direction_token_rev006",
            "direction_token_rev008",
            "direction_token_rev012",
            "exit_only_low",
            "mtp2_only_low",
            "direction_token_exit_low",
            "direction_token_mtp2_low",
        ]
    if preset == "fair_compute":
        return [
            "forward_only",
            "fwd_token_only",
            "direction_mix_p05",
            "direction_mix_p10",
            "direction_mix_p20",
            "direction_token_rev004",
            "direction_token_rev008",
        ]
    if preset == "full":
        return list(get_experiment_catalog().keys())
    raise ValueError(f"unknown preset {preset!r}; choose tiny, quick, diagnostic, isolate, fair_compute, or full")


def scheduled_weight(base: float, step: int, total_steps: int, kind: str, args: argparse.Namespace) -> float:
    if base <= 0:
        return 0.0
    p = min(1.0, max(0.0, step / float(max(1, total_steps))))
    if kind == "reverse_anneal":
        # Starts at base and decays to base * reverse_final_frac.
        floor = args.reverse_final_frac
        return base * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p)))
    if kind == "mtp_ramp":
        # Do not ask the tiny model to predict multiple futures immediately.
        start = args.mtp_start_frac
        if p <= start:
            return 0.0
        ramp = min(1.0, (p - start) / max(1e-8, 1.0 - start))
        return base * ramp
    return base


def make_lm_io(
    block: torch.Tensor,
    direction: int,
    use_direction_token: bool,
    fwd_id: int,
    rev_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Create model inputs and labels.

    Returns (input_ids, labels, ignore_first_label).
    For direction-token mode, the first label is ignored so evaluation remains
    standard next-token perplexity: x_1..x_n given prefix and a direction token.
    """
    seq = block if direction == 0 else torch.flip(block, dims=[1])
    if use_direction_token:
        tok = fwd_id if direction == 0 else rev_id
        prefix = torch.full((seq.size(0), 1), tok, dtype=seq.dtype, device=seq.device)
        inp = torch.cat([prefix, seq[:, :-1]], dim=1)
        labels = seq
        return inp, labels, True
    inp = seq[:, :-1]
    labels = seq[:, 1:]
    return inp, labels, False


def make_mixed_direction_lm_io(
    block: torch.Tensor,
    reverse_prob: float,
    fwd_id: int,
    rev_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor]:
    """Single-pass fair-compute forward/reverse mixture.

    Each row is independently kept forward or reversed. A <fwd>/<rev> prefix
    tells the shared decoder which causal direction it is seeing. This tests
    whether reverse supervision is useful without paying for a second model pass.
    """
    if reverse_prob <= 0:
        rev_mask = torch.zeros((block.size(0),), dtype=torch.bool, device=block.device)
    else:
        rev_mask = torch.rand((block.size(0),), device=block.device) < float(reverse_prob)
    seq = block.clone()
    if bool(rev_mask.any()):
        seq[rev_mask] = torch.flip(block[rev_mask], dims=[1])
    prefix_ids = torch.where(
        rev_mask,
        torch.full_like(rev_mask, int(rev_id), dtype=block.dtype),
        torch.full_like(rev_mask, int(fwd_id), dtype=block.dtype),
    ).view(block.size(0), 1)
    inp = torch.cat([prefix_ids, seq[:, :-1]], dim=1)
    labels = seq
    return inp, labels, True, rev_mask.float().mean()


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_first: bool = False) -> torch.Tensor:
    if ignore_first:
        labels = labels.clone()
        labels[:, 0] = -100
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return F.mse_loss(pred, target)


def compute_losses(
    model: TinyMirrorLM,
    batch: torch.Tensor,
    exp: Experiment,
    args: argparse.Namespace,
    step: int,
    total_steps: int,
    special_ids: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Compute all enabled objectives for one microbatch."""
    return_layers: set[int] = set()
    intermediate_layers = parse_int_list(args.intermediate_layers)
    if exp.intermediate_weight > 0:
        return_layers.update(intermediate_layers)
    if exp.latent_weight > 0:
        return_layers.add(args.latent_layer)
    if exp.layerpred_weight > 0:
        return_layers.add(args.layerpred_source_layer)
        return_layers.add(args.layerpred_target_layer)

    # Fair-compute direction-token mixture: one causal pass, some rows reversed.
    # This is the cleanest test of whether reverse examples help under the same
    # GPU budget. It deliberately avoids direction embeddings/adapters because
    # those are batch-global in this simple model.
    if exp.reverse_mix_prob > 0:
        if not exp.use_direction_token:
            raise ValueError("reverse_mix_prob requires use_direction_token=True")
        mix_inp, mix_labels, mix_ignore_first, rev_frac = make_mixed_direction_lm_io(
            batch,
            reverse_prob=exp.reverse_mix_prob,
            fwd_id=special_ids[FWD],
            rev_id=special_ids[REV],
        )
        mix_logits, mix_h, mix_layers = model(
            mix_inp,
            direction=0,
            use_direction_emb=False,
            use_adapters=False,
            return_layers=sorted(return_layers),
        )
        loss_mix = cross_entropy_loss(mix_logits, mix_labels, ignore_first=mix_ignore_first)
        losses: Dict[str, torch.Tensor] = {
            "forward": loss_mix,
            "mixed": loss_mix,
            "rev_frac": rev_frac.to(batch.device),
        }
        total = loss_mix

        # Optional intermediate CE is well-defined for the mixed sequence too.
        # Keep it off in the default fair_compute preset unless explicitly selected.
        if exp.intermediate_weight > 0:
            aux_losses = []
            for layer_idx in intermediate_layers:
                h = mix_layers.get(layer_idx)
                if h is not None:
                    aux_logits = model.lm_logits(h)
                    aux_losses.append(cross_entropy_loss(aux_logits, mix_labels, ignore_first=mix_ignore_first))
            loss_intermediate = torch.stack(aux_losses).mean() if aux_losses else torch.zeros((), device=batch.device, dtype=loss_mix.dtype)
            losses["intermediate"] = loss_intermediate
            total = total + exp.intermediate_weight * loss_intermediate

        losses["total"] = total
        losses["w_reverse"] = torch.tensor(0.0, device=batch.device)
        losses["w_mtp"] = torch.tensor(0.0, device=batch.device)
        return losses

    primary_direction = int(getattr(exp, "primary_direction", 0))
    primary_inp, primary_labels, primary_ignore_first = make_lm_io(
        batch,
        direction=primary_direction,
        use_direction_token=exp.use_direction_token,
        fwd_id=special_ids[FWD],
        rev_id=special_ids[REV],
    )
    primary_logits, primary_h, primary_layers = model(
        primary_inp,
        direction=primary_direction,
        use_direction_emb=exp.use_direction_emb,
        use_adapters=exp.use_adapters,
        return_layers=sorted(return_layers),
    )
    loss_primary = cross_entropy_loss(primary_logits, primary_labels, ignore_first=primary_ignore_first)

    losses: Dict[str, torch.Tensor] = {"primary": loss_primary}
    if primary_direction == 0:
        losses["forward"] = loss_primary
    else:
        losses["reverse"] = loss_primary
    total = loss_primary

    fwd_h = primary_h if primary_direction == 0 else None
    fwd_layers = primary_layers if primary_direction == 0 else {}
    fwd_labels = primary_labels
    fwd_ignore_first = primary_ignore_first

    rev_w = exp.reverse_weight
    if exp.anneal_reverse:
        rev_w = scheduled_weight(exp.reverse_weight, step, total_steps, "reverse_anneal", args)

    need_reverse = primary_direction == 0 and (rev_w > 0 or exp.bridge_weight > 0 or exp.align_weight > 0)
    rev_h: Optional[torch.Tensor] = None
    if need_reverse:
        rev_inp, rev_labels, rev_ignore_first = make_lm_io(
            batch,
            direction=1,
            use_direction_token=exp.use_direction_token,
            fwd_id=special_ids[FWD],
            rev_id=special_ids[REV],
        )
        rev_logits, rev_h, _ = model(
            rev_inp,
            direction=1,
            use_direction_emb=exp.use_direction_emb,
            use_adapters=exp.use_adapters,
            return_layers=[],
        )
        loss_rev = cross_entropy_loss(rev_logits, rev_labels, ignore_first=rev_ignore_first)
        losses["reverse"] = loss_rev
        total = total + float(rev_w) * loss_rev

    if exp.bridge_weight > 0:
        if primary_direction != 0:
            raise ValueError("bridge objective is only defined for forward-primary experiments")
        if exp.use_direction_token:
            # Bridge alignment is defined for standard causal positions. Keep the
            # direction-token baseline simple and do not combine it with bridge.
            loss_bridge = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        else:
            assert rev_h is not None
            t = batch.size(1) - 1  # standard hidden length
            # target token x_i for i=1..t-1. left state is forward h_{i-1};
            # right state is reverse h_{t-1-i}.
            assert fwd_h is not None
            left = fwd_h[:, : t - 1, :]
            right = rev_h[:, : t - 1, :].flip(1)
            labels = batch[:, 1:-1]
            bridge_logits = model.bridge_logits(left, right)
            loss_bridge = cross_entropy_loss(bridge_logits, labels)
        losses["bridge"] = loss_bridge
        total = total + exp.bridge_weight * loss_bridge

    if exp.align_weight > 0:
        if primary_direction != 0:
            raise ValueError("alignment objective is only defined for forward-primary experiments")
        if exp.use_direction_token:
            loss_align = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        else:
            assert rev_h is not None
            t = batch.size(1) - 1
            assert fwd_h is not None
            left = fwd_h[:, : t - 1, :]
            right = rev_h[:, : t - 1, :].flip(1)
            left_p = model.align_proj(left)
            right_p = model.align_proj(right)
            cos = F.cosine_similarity(left_p.float(), right_p.float(), dim=-1)
            loss_align = (1.0 - cos).mean()
        losses["align"] = loss_align
        total = total + exp.align_weight * loss_align

    if exp.mtp_weight > 0:
        primary_seq = batch if primary_direction == 0 else torch.flip(batch, dims=[1])
        mtp_w = scheduled_weight(exp.mtp_weight, step, total_steps, "mtp_ramp", args)
        mtp_losses = []
        block_len = primary_seq.size(1)
        for k in range(2, min(exp.mtp_k, args.max_mtp_k) + 1):
            if exp.use_direction_token:
                # With a prefix token, h[:, p] normally predicts seq[:, p].
                # The k-token target is therefore seq[:, p + k - 1].
                valid = block_len - (k - 1)
                labels_k = primary_seq[:, k - 1 :]
            else:
                # Without a prefix token, h after x_p predicts x_{p+1};
                # the k-token target is x_{p+k}.
                valid = block_len - k
                labels_k = primary_seq[:, k:]
            if valid <= 0:
                continue
            logits_k = model.mtp_logits(primary_h[:, :valid, :], k=k)
            mtp_losses.append(cross_entropy_loss(logits_k, labels_k))
        if mtp_losses:
            loss_mtp = torch.stack(mtp_losses).mean()
        else:
            loss_mtp = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        losses["mtp"] = loss_mtp
        total = total + mtp_w * loss_mtp

    if exp.intermediate_weight > 0:
        aux_losses = []
        for layer_idx in intermediate_layers:
            h = fwd_layers.get(layer_idx)
            if h is None:
                continue
            aux_logits = model.lm_logits(h)
            aux_losses.append(cross_entropy_loss(aux_logits, fwd_labels, ignore_first=fwd_ignore_first))
        if aux_losses:
            loss_intermediate = torch.stack(aux_losses).mean()
        else:
            loss_intermediate = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        losses["intermediate"] = loss_intermediate
        total = total + exp.intermediate_weight * loss_intermediate

    if exp.latent_weight > 0:
        assert fwd_h is not None
        src = fwd_layers.get(args.latent_layer)
        if src is None or src.size(1) < 2:
            loss_latent = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        else:
            pred = model.latent_pred(src[:, :-1, :])
            target = fwd_h[:, 1:, :].detach()
            loss_latent = normalized_mse(pred, target)
        losses["latent"] = loss_latent
        total = total + exp.latent_weight * loss_latent

    if exp.layerpred_weight > 0:
        src = fwd_layers.get(args.layerpred_source_layer)
        tgt = fwd_layers.get(args.layerpred_target_layer)
        if src is None or tgt is None:
            loss_layerpred = torch.zeros((), device=batch.device, dtype=loss_primary.dtype)
        else:
            pred = model.layer_pred(src)
            loss_layerpred = normalized_mse(pred, tgt.detach())
        losses["layerpred"] = loss_layerpred
        total = total + exp.layerpred_weight * loss_layerpred

    if exp.rtd_weight > 0:
        # ELECTRA-ish dense token-level signal. This is intentionally lightweight:
        # corrupt the input tokens with random replacements and ask a binary head
        # to detect replaced positions.
        base_inp = batch[:, :-1]
        corrupt = base_inp.clone()
        replace_mask = torch.rand_like(corrupt.float()) < args.rtd_prob
        # Do not sample special tokens as replacements.
        low = len(SPECIAL_TOKENS)
        rand_ids = torch.randint(low=low, high=model.vocab_size, size=corrupt.shape, device=corrupt.device)
        corrupt = torch.where(replace_mask, rand_ids, corrupt)
        _, corrupt_h, _ = model(
            corrupt,
            direction=0,
            use_direction_emb=exp.use_direction_emb,
            use_adapters=exp.use_adapters,
            return_layers=[],
        )
        rtd_logits = model.rtd_head(corrupt_h).squeeze(-1)
        loss_rtd = F.binary_cross_entropy_with_logits(rtd_logits.float(), replace_mask.float())
        losses["rtd"] = loss_rtd
        total = total + exp.rtd_weight * loss_rtd

    losses["total"] = total
    # Store scheduled weights as tensors for logging convenience.
    losses["w_reverse"] = torch.tensor(float(rev_w), device=batch.device)
    losses["w_mtp"] = torch.tensor(
        float(scheduled_weight(exp.mtp_weight, step, total_steps, "mtp_ramp", args)) if exp.mtp_weight > 0 else 0.0,
        device=batch.device,
    )
    return losses


# -----------------------------
# Training / evaluation
# -----------------------------


def make_model(args: argparse.Namespace, vocab_size: int) -> TinyMirrorLM:
    model = TinyMirrorLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden=args.ffn_hidden,
        max_seq_len=args.seq_len + 1,
        dropout=args.dropout,
        adapter_rank=args.adapter_rank,
        max_mtp_k=args.max_mtp_k,
        attention_pattern=args.attention_pattern,
        sliding_window=args.sliding_window,
        global_every=args.global_every,
        attention_backend=args.attention_backend,
    )
    return model


@torch.no_grad()
def evaluate_primary_ppl(
    model: TinyMirrorLM,
    val_blocks: torch.Tensor,
    exp: Experiment,
    args: argparse.Namespace,
    device: torch.device,
    special_ids: Dict[str, int],
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    bs = args.eval_batch_size
    eval_direction = int(getattr(exp, "primary_direction", 0))
    for start in range(0, val_blocks.size(0), bs):
        batch = val_blocks[start : start + bs].to(device, non_blocking=True)
        inp, labels, ignore_first = make_lm_io(
            batch,
            direction=eval_direction,
            use_direction_token=exp.use_direction_token,
            fwd_id=special_ids[FWD],
            rev_id=special_ids[REV],
        )
        with autocast_context(device, enabled=args.amp, dtype_name=getattr(args, "amp_dtype", "float16")):
            logits, _, _ = model(
                inp,
                direction=eval_direction,
                use_direction_emb=exp.use_direction_emb,
                use_adapters=exp.use_adapters,
                return_layers=[],
            )
            loss = cross_entropy_loss(logits, labels, ignore_first=ignore_first)
        losses.append(float(loss.detach().cpu()))
    val_loss = float(np.mean(losses))
    ppl = float(math.exp(min(20.0, val_loss)))
    model.train()
    return val_loss, ppl


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    args: argparse.Namespace,
    exp: Experiment,
    step: int,
    best_val: float,
    tokenizer_path: str,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "experiment": asdict(exp),
        "step": step,
        "best_val": best_val,
        "tokenizer_path": tokenizer_path,
    }
    torch.save(ckpt, path)


def load_checkpoint_for_training(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: torch.device | str = "cpu",
) -> Tuple[int, float, Dict]:
    ckpt = torch.load(path, map_location=device)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("step", 0)), float(ckpt.get("best_val", float("inf"))), ckpt


def train_one_experiment(
    args: argparse.Namespace,
    tok: Tokenizer,
    val_blocks: torch.Tensor,
    exp: Experiment,
    device: torch.device,
) -> Dict[str, float | str | int]:
    set_seed(args.seed + stable_name_seed(exp.name))
    exp_dir = safe_mkdir(Path(args.out_dir) / exp.name)
    tokenizer_path = str(Path(args.tokenizer_dir) / f"byte_bpe_vocab{args.vocab_size}.json")

    special_ids = {s: token_id(tok, s) for s in SPECIAL_TOKENS}
    vocab_size = tok.get_vocab_size()
    model = make_model(args, vocab_size=vocab_size).to(device)

    if args.channels_last:
        # No-op for sequence models, kept as a harmless flag for experimentation.
        pass

    model = maybe_compile(model, args)

    n_params = count_params(model._orig_mod if hasattr(model, "_orig_mod") else model)
    print(f"\n[{now()}] experiment={exp.name} params={human_int(n_params)} notes={exp.notes}")
    print(
        f"[{now()}] attention pattern={args.attention_pattern} "
        f"heads={args.n_heads} kv_heads={args.n_kv_heads if args.n_kv_heads > 0 else args.n_heads} "
        f"window={args.sliding_window} global_every={args.global_every} backend={args.attention_backend}"
    )

    opt_kwargs = dict(
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if args.fused_adam and device.type == "cuda":
        opt_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    except TypeError:
        # Older PyTorch builds do not accept fused=True. Fall back cleanly.
        opt_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda" and args.amp_dtype == "float16"))

    start_step = 0
    best_val = float("inf")
    if args.resume and Path(args.resume).exists() and args.mode == "train":
        start_step, best_val, _ = load_checkpoint_for_training(args.resume, model, optimizer, scaler, device=device)
        print(f"[{now()}] resumed from {args.resume} at step={start_step} best_val={best_val:.4f}")

    dataset = PackedTokenDataset(args, tok, seed=args.seed + 300, skip_texts=args.train_skip_texts)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    data_iter = iter(loader)

    writer = None
    if SummaryWriter is not None and not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(exp_dir / "tb"))

    progress_iter = range(start_step + 1, args.steps_per_experiment + 1)
    if tqdm is not None and args.progress_bar:
        progress_iter = tqdm(progress_iter, desc=exp.name)  # type: ignore[assignment]

    meters: Dict[str, float] = {}
    last_log_time = time.time()
    tokens_since_log = 0
    final_val_loss = float("nan")
    final_ppl = float("nan")

    model.train()
    for step in progress_iter:
        lr = lr_at_step(step - 1, args.steps_per_experiment, args.learning_rate, args.min_lr, args.warmup_steps)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        micro_losses: Dict[str, List[float]] = {}

        for _micro in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            effective_seq_len = seq_len_at_step(args, step, args.steps_per_experiment)
            batch = crop_batch_to_seq_len(batch, effective_seq_len)
            tokens_since_log += batch.numel()

            with autocast_context(device, enabled=args.amp, dtype_name=getattr(args, "amp_dtype", "float16")):
                losses = compute_losses(model, batch, exp, args, step, args.steps_per_experiment, special_ids)
                loss = losses["total"] / args.grad_accum_steps

            scaler.scale(loss).backward()

            for k, v in losses.items():
                micro_losses.setdefault(k, []).append(float(v.detach().float().cpu()))

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_f = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
        else:
            grad_norm_f = 0.0

        scaler.step(optimizer)
        scaler.update()

        for k, vals in micro_losses.items():
            meters[k] = float(np.mean(vals))
        meters["grad_norm"] = grad_norm_f
        meters["lr"] = lr

        if writer is not None:
            for k, v in meters.items():
                writer.add_scalar(f"train/{k}", v, step)

        if step % args.log_interval == 0 or step == 1:
            dt = max(1e-9, time.time() - last_log_time)
            toks_per_sec = tokens_since_log / dt
            last_log_time = time.time()
            tokens_since_log = 0
            msg = (
                f"[{now()}] {exp.name} step {step:5d}/{args.steps_per_experiment} "
                f"loss={meters.get('total', float('nan')):.4f} "
                f"primary={meters.get('primary', float('nan')):.4f} "
                f"fwd={meters.get('forward', float('nan')):.4f} "
                f"rev={meters.get('reverse', 0.0):.4f} "
                f"bridge={meters.get('bridge', 0.0):.4f} "
                f"mtp={meters.get('mtp', 0.0):.4f} "
                f"mixrev={meters.get('rev_frac', 0.0):.2f} "
                f"seq={seq_len_at_step(args, step, args.steps_per_experiment)} "
                f"lr={lr:.2e} gn={grad_norm_f:.2f} tok/s={human_int(toks_per_sec)}"
            )
            print(msg)

        if step % args.eval_interval == 0 or step == args.steps_per_experiment:
            val_loss, ppl = evaluate_primary_ppl(model, val_blocks, exp, args, device, special_ids)
            final_val_loss, final_ppl = val_loss, ppl
            val_direction = "reverse" if int(getattr(exp, "primary_direction", 0)) == 1 else "forward"
            print(f"[{now()}] {exp.name} VALID step={step} {val_direction}_loss={val_loss:.4f} ppl={ppl:.2f}")
            if writer is not None:
                writer.add_scalar(f"valid/{val_direction}_loss", val_loss, step)
                writer.add_scalar(f"valid/{val_direction}_ppl", ppl, step)

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(exp_dir / "best.pt", model, optimizer, scaler, args, exp, step, best_val, tokenizer_path)
                print(f"[{now()}] {exp.name} new best checkpoint saved: val={best_val:.4f}")

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(exp_dir / "last.pt", model, optimizer, scaler, args, exp, step, best_val, tokenizer_path)

    save_checkpoint(exp_dir / "last.pt", model, optimizer, scaler, args, exp, args.steps_per_experiment, best_val, tokenizer_path)

    if writer is not None:
        writer.flush()
        writer.close()

    result = {
        "experiment": exp.name,
        "best_val_loss": float(best_val),
        "final_val_loss": float(final_val_loss),
        "final_ppl": float(final_ppl),
        "params": int(n_params),
        "eval_direction": "reverse" if int(getattr(exp, "primary_direction", 0)) == 1 else "forward",
        "notes": exp.notes,
    }

    del model, optimizer, scaler, loader, data_iter
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_sweep(args: argparse.Namespace) -> None:
    device = get_device(args)
    print(f"[{now()}] device={device} cuda={torch.cuda.is_available()} amp={args.amp}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        print(f"[{now()}] GPU={torch.cuda.get_device_name(0)}")

    tok = train_or_load_tokenizer(args)
    val_blocks = build_or_load_val_blocks(args, tok)
    catalog = get_experiment_catalog()

    if args.experiments:
        names = [x.strip() for x in args.experiments.split(",") if x.strip()]
    elif args.mode == "train":
        names = [args.experiment]
    else:
        names = experiments_for_preset(args.sweep_preset)

    for name in names:
        if name not in catalog:
            raise ValueError(f"unknown experiment {name!r}. Available: {', '.join(catalog)}")

    results = []
    for name in names:
        exp = catalog[name]
        result = train_one_experiment(args, tok, val_blocks, exp, device)
        results.append(result)

    out_dir = safe_mkdir(args.out_dir)
    csv_path = out_dir / "sweep_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "best_val_loss", "final_val_loss", "final_ppl", "params", "eval_direction", "notes"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[{now()}] sweep complete. Results:")
    for r in sorted(results, key=lambda x: float(x["best_val_loss"])):
        print(
            f"  {r['experiment']:<24} best_val={float(r['best_val_loss']):.4f} "
            f"final_ppl={float(r['final_ppl']):.2f}"
        )
    print(f"[{now()}] wrote {csv_path}")


# -----------------------------
# Generation
# -----------------------------


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = v[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return logits


@torch.no_grad()
def generate(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --mode generate")
    device = get_device(args)
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args_dict = ckpt.get("args", {})
    exp_dict = ckpt.get("experiment", {"name": "forward_only"})
    exp = Experiment(**{**asdict(Experiment(name="tmp")), **exp_dict})

    tok_path = args.tokenizer_path or ckpt.get("tokenizer_path")
    if not tok_path or not Path(tok_path).exists():
        tok_path = str(Path(saved_args_dict.get("tokenizer_dir", args.tokenizer_dir)) / f"byte_bpe_vocab{saved_args_dict.get('vocab_size', args.vocab_size)}.json")
    tok = Tokenizer.from_file(tok_path)
    special_ids = {s: token_id(tok, s) for s in SPECIAL_TOKENS}

    # Reconstruct model dimensions from checkpoint args unless overridden.
    dummy = argparse.Namespace(**{**vars(args), **saved_args_dict})
    model = make_model(dummy, vocab_size=tok.get_vocab_size()).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    prompt_ids = tok.encode(args.prompt).ids
    if not prompt_ids:
        prompt_ids = [special_ids[BOS]]

    generation_direction = int(getattr(exp, "primary_direction", 0))
    if generation_direction == 1:
        prompt_ids = list(reversed(prompt_ids))

    if exp.use_direction_token:
        mode_id = special_ids[REV] if generation_direction == 1 else special_ids[FWD]
        ids = [mode_id] + prompt_ids
    else:
        ids = prompt_ids[:]

    for _ in range(args.max_new_tokens):
        if exp.use_direction_token:
            # Preserve the direction token and crop old context after it.
            ctx = [ids[0]] + ids[-(dummy.seq_len):]
        else:
            ctx = ids[-dummy.seq_len :]
        x = torch.tensor(ctx, dtype=torch.long, device=device)[None, :]
        with autocast_context(device, enabled=args.amp, dtype_name=getattr(args, "amp_dtype", "float16")):
            logits, _, _ = model(
                x,
                direction=generation_direction,
                use_direction_emb=exp.use_direction_emb,
                use_adapters=exp.use_adapters,
                return_layers=[],
            )
            next_logits = logits[0, -1, :] / max(1e-6, args.temperature)
            next_logits = top_k_top_p_filter(next_logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(next_logits.float(), dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if next_id == special_ids[EOS] and args.stop_at_eos:
            break

    out_ids = ids[1:] if exp.use_direction_token else ids
    if generation_direction == 1:
        out_ids = list(reversed(out_ids))
    print(tok.decode(out_ids))


# -----------------------------
# CLI
# -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Single-file 5M parameter MIRROR-AR objective lab")

    # Mode / output
    p.add_argument("--mode", choices=["sweep", "train", "generate", "tokenizer"], default="sweep")
    p.add_argument("--out_dir", type=str, default="runs_mirror5m")
    p.add_argument("--cache_dir", type=str, default="cache_mirror5m")
    p.add_argument("--tokenizer_dir", type=str, default="tokenizer_mirror5m")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--tokenizer_path", type=str, default="")

    # Data
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_config", type=str, default="sample-10BT")
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--toy_data", action="store_true", help="use built-in tiny repeated corpus for smoke tests")
    p.add_argument("--shuffle_buffer", type=int, default=10_000)
    p.add_argument("--train_skip_texts", type=int, default=10_000)
    p.add_argument("--val_skip_texts", type=int, default=2_000)
    p.add_argument("--val_blocks", type=int, default=256)
    p.add_argument("--rebuild_val", action="store_true")

    # Tokenizer
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--tokenizer_train_texts", type=int, default=20_000)
    p.add_argument("--tokenizer_min_frequency", type=int, default=2)
    p.add_argument("--retrain_tokenizer", action="store_true")

    # Model: default is about 5.3M params with vocab 4096.
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--seq_len_schedule", type=str, default="", help="optional curriculum like 512:0,1024:0.25,2048:0.60; max must fit --seq_len")
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_kv_heads", type=int, default=0, help="0 means full multi-head attention; lower values enable GQA")
    p.add_argument("--ffn_hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--adapter_rank", type=int, default=16)
    p.add_argument("--max_mtp_k", type=int, default=4)
    p.add_argument("--attention_pattern", choices=["global", "local", "local_global"], default="global")
    p.add_argument("--sliding_window", type=int, default=0, help="tokens visible to local attention layers")
    p.add_argument("--global_every", type=int, default=6, help="for local_global, every Nth layer is global and the final layer is always global")
    p.add_argument("--attention_backend", choices=["sdpa", "flash_attn", "auto"], default="sdpa")

    # Training: 2080 Ti-friendly defaults.
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--steps_per_experiment", type=int, default=400)
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--min_lr", type=float, default=4e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--amp", type=str2bool, default=True)
    p.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--tf32", type=str2bool, default=True)
    p.add_argument("--fused_adam", type=str2bool, default=False, help="enable if your PyTorch/CUDA build supports fused AdamW")
    p.add_argument("--compile", action="store_true", help="optional; compile overhead is usually not worth it for short sweeps")
    p.add_argument("--channels_last", action="store_true")

    # Objective details
    p.add_argument("--reverse_final_frac", type=float, default=0.15)
    p.add_argument("--mtp_start_frac", type=float, default=0.20)
    p.add_argument("--intermediate_layers", type=str, default="3,6", help="0-based layer indices for intermediate CE")
    p.add_argument("--latent_layer", type=int, default=4)
    p.add_argument("--layerpred_source_layer", type=int, default=3)
    p.add_argument("--layerpred_target_layer", type=int, default=6)
    p.add_argument("--rtd_prob", type=float, default=0.15)

    # Sweep
    p.add_argument("--sweep_preset", choices=["tiny", "quick", "diagnostic", "isolate", "fair_compute", "full"], default="quick")
    p.add_argument("--experiments", type=str, default="", help="comma-separated experiment names; overrides preset")
    p.add_argument("--experiment", type=str, default="mirror_bridge_mtp2", help="used with --mode train")

    # Logging
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--no_tensorboard", action="store_true")
    p.add_argument("--progress_bar", type=str2bool, default=False)

    # Generation
    p.add_argument("--prompt", type=str, default="The cat went on")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--stop_at_eos", type=str2bool, default=True)

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    safe_mkdir(args.out_dir)
    safe_mkdir(args.cache_dir)
    safe_mkdir(args.tokenizer_dir)

    # RTX 2080 Ti is Turing: fp16 is useful, bf16 is not. High matmul precision is
    # harmless on CPU and helps newer GPUs; TF32 is ignored on Turing.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if args.mode == "tokenizer":
        train_or_load_tokenizer(args)
        return

    if args.mode == "generate":
        generate(args)
        return

    run_sweep(args)


if __name__ == "__main__":
    main()
