import json
from pathlib import Path

from tokenizer import CharTokenizer, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    texts = []
    for name in ["train.txt", "val.txt", "test.txt"]:
        texts.append((DATA / "pretrain" / name).read_text(encoding="utf-8"))
    for name in ["train.jsonl", "val.jsonl"]:
        for row in load_jsonl(DATA / "sft" / name):
            texts.append(row["prompt"])
            texts.append(row["response"])
        for row in load_jsonl(DATA / "dpo" / name):
            texts.append(row["prompt"])
            texts.append(row["chosen"])
            texts.append(row["rejected"])
    tokenizer = CharTokenizer.build(texts)
    tokenizer.save(DATA / "tokenizer.json")
    stats = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "unk_id": tokenizer.unk_id,
        "user_id": tokenizer.user_id,
        "assistant_id": tokenizer.assistant_id,
        "end_id": tokenizer.end_id,
        "sample_chars": "".join(tokenizer.itos[7:37]),
    }
    (DATA / "tokenizer_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
