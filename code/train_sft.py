import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel

from model_utils import DATA, LOGS, MODEL, save_json, set_cpu_threads, set_seed
from tokenizer import CharTokenizer, load_jsonl


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, block_size):
        self.items = []
        for row in rows:
            prompt_ids = tokenizer.encode(
                tokenizer.specials["user"] + row["prompt"] + tokenizer.specials["assistant"]
            )
            answer_ids = tokenizer.encode(row["response"] + tokenizer.specials["end"])
            if len(prompt_ids) >= block_size:
                prompt_ids = prompt_ids[: block_size - 2]
            answer_ids = answer_ids[: block_size - len(prompt_ids)]
            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            self.items.append((input_ids, labels))
        self.pad_id = tokenizer.pad_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_id):
    max_len = max(len(x) for x, _ in batch)
    input_ids, labels, attention = [], [], []
    for x, y in batch:
        pad = max_len - len(x)
        input_ids.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
        attention.append([1] * len(x) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )


def lr_at(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * max(step, 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    losses = []
    for input_ids, labels, attention in loader:
        loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
        losses.append(float(loss.detach()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=str, default=str(MODEL / "pretrain" / "best"))
    parser.add_argument("--out", type=str, default=str(MODEL / "sft"))
    parser.add_argument("--log", type=str, default=str(LOGS / "sft.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    threads = set_cpu_threads()
    tokenizer = CharTokenizer.load(DATA / "tokenizer.json")
    train_rows = load_jsonl(DATA / "sft" / "train.jsonl")
    val_rows = load_jsonl(DATA / "sft" / "val.jsonl")
    train_ds = SFTDataset(train_rows, tokenizer, args.block_size)
    val_ds = SFTDataset(val_rows, tokenizer, args.block_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_id),
    )

    model = GPT2LMHeadModel.from_pretrained(args.base)
    model.config.use_cache = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "epoch", "train_loss", "val_loss", "lr", "elapsed_sec"])
    print(
        f"threads={threads} train={len(train_ds)} val={len(val_ds)} "
        f"steps={total_steps} params={sum(p.numel() for p in model.parameters())}"
    )
    best_val = float("inf")
    step = 0
    start = time.time()
    running = []
    for epoch in range(1, args.epochs + 1):
        for input_ids, labels, attention in train_loader:
            step += 1
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running.append(float(loss.detach()))
            if step % 10 == 0 or step == 1:
                print(
                    f"step {step}/{total_steps} epoch {epoch} "
                    f"loss {np.mean(running[-20:]):.4f} lr {lr:.2e}"
                )
        val_loss = evaluate(model, val_loader)
        train_loss = float(np.mean(running[-len(train_loader):]))
        writer.writerow([step, epoch, train_loss, val_loss, lr, time.time() - start])
        log_file.flush()
        print(f"EVAL epoch {epoch} train {train_loss:.4f} val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out / "best")
            tokenizer.save(out / "best" / "tokenizer.json")
            save_json(
                out / "best" / "train_info.json",
                {"stage": "sft", "epoch": epoch, "val_loss": val_loss},
            )
    log_file.close()
    model.save_pretrained(out / "final")
    tokenizer.save(out / "final" / "tokenizer.json")
    final_val = evaluate(model, val_loader)
    info = {
        "stage": "sft",
        "epochs": args.epochs,
        "steps": step,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "lr": args.lr,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "final_train_loss_recent": float(np.mean(running[-100:])),
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "params": sum(p.numel() for p in model.parameters()),
        "threads": threads,
        "elapsed_sec": round(time.time() - start, 2),
        "base": args.base,
    }
    save_json(out / "train_info.json", info)
    print("DONE", info)


if __name__ == "__main__":
    main()
