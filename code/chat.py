import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from model_utils import set_cpu_threads
from tokenizer import CharTokenizer


def load_model(path, tokenizer):
    model = GPT2LMHeadModel.from_pretrained(path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


@torch.no_grad()
def answer(model, tokenizer, prompt, max_new_tokens=64, seed=42):
    formatted = tokenizer.specials["user"] + prompt + tokenizer.specials["assistant"]
    ids = tokenizer.encode(formatted)
    max_prompt = max(8, 128 - max_new_tokens)
    ids = ids[-max_prompt:]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    torch.manual_seed(seed)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=20,
        top_p=0.90,
        temperature=0.6,
        repetition_penalty=1.10,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.end_id,
    )
    text = tokenizer.decode(output[0].tolist())
    if "<assistant>" in text:
        text = text.split("<assistant>", 1)[1]
    return text.replace("<end>", "").replace("<pad>", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[1] / "model" / "final"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    set_cpu_threads()
    model_dir = Path(args.model)
    tokenizer = CharTokenizer.load(model_dir / "tokenizer.json")
    model, tokenizer = load_model(model_dir, tokenizer)

    print("本地小模型交互测试。输入 /exit 退出。")
    print("注意：模型只有 886K 参数，回答质量有限，主要用于验证输入-输出流程。")
    turn = 0
    while True:
        try:
            prompt = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("退出。")
            break
        turn += 1
        text = answer(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            seed=42 + turn,
        )
        print(f"模型：{text}")


if __name__ == "__main__":
    main()
