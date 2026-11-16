import argparse
import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel

from model_utils import DATA, LOGS, MODEL, save_json, set_cpu_threads, set_seed
from tokenizer import CharTokenizer, load_jsonl


class DPODataset(Dataset):
    def __init__(self, rows, tokenizer, block_size):
        self.items = []
        self.pad_id = tokenizer.pad_id
        for row in rows:
            prompt_ids = tokenizer.encode(
                tokenizer.specials["user"] + row["prompt"] + tokenizer.specials["assistant"]
            )
            if len(prompt_ids) >= block_size - 2:
                prompt_ids = prompt_ids[: block_size - 2]
            chosen_answer = tokenizer.encode(row["chosen"] + tokenizer.specials["end"])
            rejected_answer = tokenizer.encode(row["rejected"] + tokenizer.specials["end"])
            chosen_answer = chosen_answer[: block_size - len(prompt_ids)]
            rejected_answer = rejected_answer[: block_size - len(prompt_ids)]
            chosen_ids = prompt_ids + chosen_answer
            rejected_ids = prompt_ids + rejected_answer
            chosen_mask = [0] * len(prompt_ids) + [1] * len(chosen_answer)
            rejected_mask = [0] * len(prompt_ids) + [1] * len(rejected_answer)
            self.items.append((chosen_ids, chosen_mask, rejected_ids, rejected_mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def pad_side(items, pad_id):
    max_len = max(len(ids) for ids, _ in items)
    input_ids, attention, response_mask = [], [], []
    for ids, mask in items:
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
        response_mask.append(mask + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        torch.tensor(response_mask, dtype=torch.long),
    )


def collate(batch, pad_id):
    chosen = pad_side([(x[0], x[1]) for x in batch], pad_id)
    rejected = pad_side([(x[2], x[3]) for x in batch], pad_id)
    return chosen, rejected


def sequence_logps(model, input_ids, attention_mask, response_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:]
    token_logps = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = response_mask[:, 1:].float()
    return (token_logps * mask).sum(dim=-1)


def lr_at(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * max(step, 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, ref_model, loader, beta):
    model.eval()
    losses, margins, correct, total = [], [], 0, 0
    for chosen, rejected in loader:
        c_ids, c_attn, c_mask = chosen
        r_ids, r_attn, r_mask = rejected
        pi_c = sequence_logps(model, c_ids, c_attn, c_mask)
        pi_r = sequence_logps(model, r_ids, r_attn, r_mask)
        ref_c = sequence_logps(ref_model, c_ids, c_attn, c_mask)
        ref_r = sequence_logps(ref_model, r_ids, r_attn, r_mask)
        logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        losses.append(float((-F.logsigmoid(logits)).mean().detach()))
        margins.append(float((pi_c - pi_r).mean().detach()))
        correct += int((pi_c > pi_r).sum().item())
        total += int(pi_c.numel())
    model.train()
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "margin": float(np.mean(margins)) if margins else float("nan"),
        "accuracy": correct / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base", type=str, default=str(MODEL / "sft" / "best"))
    parser.add_argument("--out", type=str, default=str(MODEL / "dpo"))
    parser.add_argument("--log", type=str, default=str(LOGS / "dpo.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    threads = set_cpu_threads()
    tokenizer = CharTokenizer.load(DATA / "tokenizer.json")
    train_rows = load_jsonl(DATA / "dpo" / "train.jsonl")
    val_rows = load_jsonl(DATA / "dpo" / "val.jsonl")
    train_ds = DPODataset(train_rows, tokenizer, args.block_size)
    val_ds = DPODataset(val_rows, tokenizer, args.block_size)
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
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(
        ["step", "epoch", "loss", "margin", "accuracy", "val_loss", "val_margin", "val_accuracy", "lr", "elapsed_sec"]
    )
    print(
        f"threads={threads} train={len(train_ds)} val={len(val_ds)} "
        f"steps={total_steps} beta={args.beta} params={sum(p.numel() for p in model.parameters())}"
    )
    best_val = float("inf")
    step = 0
    start = time.time()
    running = []
    for epoch in range(1, args.epochs + 1):
        for chosen, rejected in train_loader:
            c_ids, c_attn, c_mask = chosen
            r_ids, r_attn, r_mask = rejected
            step += 1
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            pi_c = sequence_logps(model, c_ids, c_attn, c_mask)
            pi_r = sequence_logps(model, r_ids, r_attn, r_mask)
            with torch.no_grad():
                ref_c = sequence_logps(ref_model, c_ids, c_attn, c_mask)
                ref_r = sequence_logps(ref_model, r_ids, r_attn, r_mask)
            logits = args.beta * ((pi_c - pi_r) - (ref_c - ref_r))
            loss = -F.logsigmoid(logits).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            margin = float((pi_c - pi_r).mean().detach())
            acc = float((pi_c > pi_r).float().mean().detach())
            running.append((float(loss.detach()), margin, acc))
            if step % 10 == 0 or step == 1:
                recent = running[-20:]
                print(
                    f"step {step}/{total_steps} epoch {epoch} "
                    f"loss {np.mean([x[0] for x in recent]):.4f} "
                    f"margin {np.mean([x[1] for x in recent]):.3f} "
                    f"acc {np.mean([x[2] for x in recent]):.3f} lr {lr:.2e}"
                )
        metrics = evaluate(model, ref_model, val_loader, args.beta)
        recent = running[-len(train_loader):]
        train_loss = float(np.mean([x[0] for x in recent]))
        train_margin = float(np.mean([x[1] for x in recent]))
        train_acc = float(np.mean([x[2] for x in recent]))
        writer.writerow(
            [
                step,
                epoch,
                train_loss,
                train_margin,
                train_acc,
                metrics["loss"],
                metrics["margin"],
                metrics["accuracy"],
                lr,
                time.time() - start,
            ]
        )
        log_file.flush()
        print(
            f"EVAL epoch {epoch} train_loss {train_loss:.4f} "
            f"val_loss {metrics['loss']:.4f} val_acc {metrics['accuracy']:.3f}"
        )
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            model.save_pretrained(out / "best")
            tokenizer.save(out / "best" / "tokenizer.json")
            save_json(
                out / "best" / "train_info.json",
                {"stage": "dpo", "epoch": epoch, **metrics, "beta": args.beta},
            )
    log_file.close()
    model.save_pretrained(out / "final")
    tokenizer.save(out / "final" / "tokenizer.json")
    final_metrics = evaluate(model, ref_model, val_loader, args.beta)
    info = {
        "stage": "dpo",
        "epochs": args.epochs,
        "steps": step,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "lr": args.lr,
        "beta": args.beta,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "final_train_loss_recent": float(np.mean([x[0] for x in running[-100:]])),
        "final_train_margin_recent": float(np.mean([x[1] for x in running[-100:]])),
        "final_train_accuracy_recent": float(np.mean([x[2] for x in running[-100:]])),
        "final_val_loss": final_metrics["loss"],
        "final_val_margin": final_metrics["margin"],
        "final_val_accuracy": final_metrics["accuracy"],
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
