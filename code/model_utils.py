import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "model"
LOGS = ROOT / "logs"
EVAL = ROOT / "eval"
REPORT = ROOT / "report"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cpu_threads(n=None):
    if n is None:
        n = max(1, min(14, (os.cpu_count() or 8) - 2))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)
    return n


def create_model(tokenizer, block_size=128):
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=block_size,
        n_ctx=block_size,
        n_embd=128,
        n_layer=4,
        n_head=4,
        n_inner=512,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        activation_function="gelu_new",
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )
    config.use_cache = False
    return GPT2LMHeadModel(config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_losses(path):
    import csv

    rows = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows
