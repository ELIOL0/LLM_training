import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from generate import generate_text, load_model
from model_utils import DATA, EVAL, LOGS, MODEL, save_json, set_cpu_threads, set_seed
from tokenizer import CharTokenizer, load_jsonl
from train_dpo import DPODataset, collate, sequence_logps


def chunks_from_text(tokenizer, path, block_size=128):
    text = Path(path).read_text(encoding="utf-8")
    ids = np.asarray(tokenizer.encode(text), dtype=np.int64)
    n = len(ids) // (block_size + 1)
    arr = ids[: n * (block_size + 1)].reshape(n, block_size + 1)
    return torch.from_numpy(arr)


@torch.no_grad()
def perplexity(model, chunks, batch_size=16, max_batches=30):
    model.eval()
    losses = []
    n = min(max_batches, max(1, len(chunks) // batch_size))
    for i in range(n):
        batch = chunks[i * batch_size : (i + 1) * batch_size]
        if len(batch) < 2:
            continue
        loss = model(input_ids=batch[:, :-1], labels=batch[:, 1:]).loss
        losses.append(float(loss.detach()))
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, math.exp(mean_loss) if mean_loss == mean_loss else float("nan")


def distinct_ratio(text, n=2):
    chars = [c for c in text if not c.isspace()]
    grams = [tuple(chars[i : i + n]) for i in range(max(0, len(chars) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


def fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def summarize_outputs(rows):
    lengths = [len(r["output"]) for r in rows]
    repeats = [1.0 - distinct_ratio(r["output"]) for r in rows]
    punct = [any(p in r["output"] for p in "。！？.!?") for r in rows]
    return {
        "avg_chars": float(np.mean(lengths)) if lengths else 0.0,
        "repeat_rate": float(np.mean(repeats)) if repeats else 0.0,
        "punctuation_rate": float(np.mean(punct)) if punct else 0.0,
    }


@torch.no_grad()
def dpo_accuracy(policy_path, ref_path, tokenizer):
    policy = GPT2LMHeadModel.from_pretrained(policy_path)
    ref = GPT2LMHeadModel.from_pretrained(ref_path)
    policy.eval()
    ref.eval()
    rows = load_jsonl(DATA / "dpo" / "val.jsonl")
    ds = DPODataset(rows, tokenizer, 128)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate(b, tokenizer.pad_id),
    )
    correct, total, margins = 0, 0, []
    for chosen, rejected in loader:
        c_ids, c_attn, c_mask = chosen
        r_ids, r_attn, r_mask = rejected
        pi_c = sequence_logps(policy, c_ids, c_attn, c_mask)
        pi_r = sequence_logps(policy, r_ids, r_attn, r_mask)
        correct += int((pi_c > pi_r).sum().item())
        total += int(pi_c.numel())
        margins.append(float((pi_c - pi_r).mean().item()))
    return {
        "accuracy": correct / max(1, total),
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "n": total,
    }


def main():
    set_seed(42)
    set_cpu_threads()
    tokenizer = CharTokenizer.load(DATA / "tokenizer.json")
    chunks = chunks_from_text(tokenizer, DATA / "pretrain" / "test.txt", 128)
    prompts = [r["prompt"] for r in load_jsonl(DATA / "sft" / "val.jsonl")[:30]]
    model_paths = {
        "base": MODEL / "pretrain" / "best",
        "sft": MODEL / "sft" / "best",
        "dpo": MODEL / "dpo" / "best",
    }
    metrics = {}
    outputs = {}
    for name, path in model_paths.items():
        print(f"=== {name}: {path} ===")
        model = load_model(path)
        loss, ppl = perplexity(model, chunks)
        rows = []
        for i, prompt in enumerate(prompts):
            formatted = name != "base"
            text = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=80,
                formatted=formatted,
                seed=100 + i,
            )
            rows.append({"prompt": prompt, "output": text})
        outputs[name] = rows
        metrics[name] = {"test_loss": loss, "test_ppl": ppl, **summarize_outputs(rows)}
        del model

    metrics["dpo_preference"] = dpo_accuracy(
        MODEL / "dpo" / "best", MODEL / "sft" / "best", tokenizer
    )
    stage_info = {}
    for name in ["pretrain", "sft", "dpo"]:
        info_path = MODEL / name / "train_info.json"
        if info_path.exists():
            stage_info[name] = json.loads(info_path.read_text(encoding="utf-8"))
    metrics["stage_val_loss"] = {
        "pretrain": stage_info.get("pretrain", {}).get("final_val_loss"),
        "sft": stage_info.get("sft", {}).get("final_val_loss"),
        "dpo": stage_info.get("dpo", {}).get("final_val_loss"),
    }

    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "prompts.jsonl").write_text(
        "\n".join(json.dumps({"prompt": p}, ensure_ascii=False) for p in prompts) + "\n",
        encoding="utf-8",
    )
    for name, rows in outputs.items():
        with (EVAL / f"outputs_{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(EVAL / "metrics.json", metrics)

    lines = [
        "# 评测报告",
        "",
        "## 指标汇总",
        "",
        "| 模型 | 测试 loss | 测试 ppl | 平均字符数 | 重复率 | 有标点比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["base", "sft", "dpo"]:
        m = metrics[name]
        lines.append(
            f"| {name} | {m['test_loss']:.4f} | {m['test_ppl']:.2f} | "
            f"{m['avg_chars']:.1f} | {m['repeat_rate']:.3f} | {m['punctuation_rate']:.3f} |"
        )
    pref = metrics["dpo_preference"]
    lines += [
        "",
        f"DPO 验证集偏好准确率：**{pref['accuracy']:.3f}**（n={pref['n']}，平均 margin={pref['mean_margin']:.3f}）。",
        "",
        "## 各阶段验证 loss",
        "",
        "| 阶段 | 验证 loss |",
        "|---|---:|",
        f"| 预训练 | {fmt(metrics['stage_val_loss'].get('pretrain'))} |",
        f"| SFT | {fmt(metrics['stage_val_loss'].get('sft'))} |",
        f"| DPO | {fmt(metrics['stage_val_loss'].get('dpo'))} |",
        "",
        "注意：上表的 SFT/DPO 数值来自各自阶段的验证集；下面的语言模型测试 loss/ppl "
        "是在原始预训练文本上计算的。SFT/DPO 后模型分布转向指令格式，"
        "在原始文本上的 ppl 会上升，这是正常的分布偏移，不代表训练失败。",
        "",
        "## 生成样例",
        "",
    ]
    for i in range(min(8, len(prompts))):
        lines += [f"### Prompt {i + 1}", "", f"**用户：** {prompts[i]}", ""]
        for name in ["base", "sft", "dpo"]:
            lines += [f"**{name}：** {outputs[name][i]['output']}", ""]
    lines += [
        "## 说明",
        "",
        "- 测试 loss / ppl 在预训练测试集上计算，用于观察语言建模能力。",
        "- 重复率由字符 2-gram 去重比例换算，越高表示重复越明显。",
        "- DPO 偏好准确率表示 DPO 模型给 chosen 的序列 log-prob 高于 rejected 的比例。",
        "- 本作业模型很小、数据为合成数据，指标只用于证明流程完整，不代表真实 LLM 能力。",
    ]
    (EVAL / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
