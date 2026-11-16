import argparse
import csv
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from model_utils import DATA, LOGS, MODEL, create_model, save_json, set_cpu_threads, set_seed
from tokenizer import CharTokenizer


def load_token_ids(path, tokenizer):
    text = Path(path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    return np.asarray(ids, dtype=np.int64)


def make_chunks(ids, block_size):
    n = len(ids) // (block_size + 1)
    if n == 0:
        raise ValueError("Not enough tokens for one chunk")
    arr = ids[: n * (block_size + 1)].reshape(n, block_size + 1)
    return torch.from_numpy(arr)


def lr_at(step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * max(step, 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, chunks, batch_size, max_batches=20):
    model.eval()
    losses = []
    n = min(max_batches, max(1, len(chunks) // batch_size))
    for i in range(n):
        batch = chunks[i * batch_size : (i + 1) * batch_size]
        if len(batch) < 2:
            continue
        x = batch[:, :-1]
        y = batch[:, 1:]
        loss = model(input_ids=x, labels=y).loss
        losses.append(float(loss.detach()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=300)
    parser.add_argument("--out", type=str, default=str(MODEL / "pretrain"))
    parser.add_argument("--log", type=str, default=str(LOGS / "pretrain.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    threads = set_cpu_threads()
    tokenizer = CharTokenizer.load(DATA / "tokenizer.json")
    train_ids = load_token_ids(DATA / "pretrain" / "train.txt", tokenizer)
    val_ids = load_token_ids(DATA / "pretrain" / "val.txt", tokenizer)
    train_chunks = make_chunks(train_ids, args.block_size)
    val_chunks = make_chunks(val_ids, args.block_size)

    model = create_model(tokenizer, block_size=args.block_size)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["step", "train_loss", "val_loss", "lr", "tokens", "elapsed_sec"])

    print(f"threads={threads} train_chunks={len(train_chunks)} val_chunks={len(val_chunks)}")
    print(f"params={sum(p.numel() for p in model.parameters())}")
    best_val = float("inf")
    step = 0
    tokens_seen = 0
    start = time.time()
    running = []
    stop_reason = "completed"
    try:
        while step < args.steps:
            perm = torch.randperm(len(train_chunks))
            for pos in range(0, len(perm), args.batch_size):
                if step >= args.steps:
                    break
                if args.max_seconds and time.time() - start > args.max_seconds:
                    stop_reason = "time_limit"
                    break
                idx = perm[pos : pos + args.batch_size]
                batch = train_chunks[idx]
                x = batch[:, :-1]
                y = batch[:, 1:]
                step += 1
                lr = lr_at(step, args.steps, args.lr, args.warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = model(input_ids=x, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tokens_seen += x.numel()
                running.append(float(loss.detach()))
                if step % 10 == 0 or step == 1:
                    avg = float(np.mean(running[-50:]))
                    elapsed = time.time() - start
                    print(
                        f"step {step}/{args.steps} loss {avg:.4f} "
                        f"lr {lr:.2e} tokens {tokens_seen} elapsed {elapsed:.0f}s"
                    )
                if step % args.eval_every == 0:
                    val_loss = evaluate(model, val_chunks, args.batch_size)
                    avg = float(np.mean(running[-args.eval_every :]))
                    elapsed = time.time() - start
                    writer.writerow([step, avg, val_loss, lr, tokens_seen, elapsed])
                    log_file.flush()
                    print(f"EVAL step {step} train {avg:.4f} val {val_loss:.4f}")
                    if val_loss < best_val:
                        best_val = val_loss
                        model.save_pretrained(out / "best")
                        tokenizer.save(out / "best" / "tokenizer.json")
                        save_json(
                            out / "best" / "train_info.json",
                            {
                                "step": step,
                                "val_loss": val_loss,
                                "params": sum(p.numel() for p in model.parameters()),
                            },
                        )
                if args.save_every and step % args.save_every == 0:
                    ckpt = out / f"checkpoint-{step}"
                    model.save_pretrained(ckpt)
                    tokenizer.save(ckpt / "tokenizer.json")
            if stop_reason == "time_limit":
                break
    finally:
        log_file.close()
    model.save_pretrained(out / "final")
    tokenizer.save(out / "final" / "tokenizer.json")
    final_val = evaluate(model, val_chunks, args.batch_size)
    info = {
        "stage": "pretrain",
        "steps": step,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "lr": args.lr,
        "tokens_seen": tokens_seen,
        "elapsed_sec": round(time.time() - start, 2),
        "final_train_loss_recent": float(np.mean(running[-100:])) if running else None,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "params": sum(p.numel() for p in model.parameters()),
        "threads": threads,
        "stop_reason": stop_reason,
    }
    save_json(out / "train_info.json", info)
    print("DONE", info)


if __name__ == "__main__":
    main()
