import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from model_utils import set_cpu_threads
from tokenizer import CharTokenizer


def load_model(path):
    model = GPT2LMHeadModel.from_pretrained(path)
    model.eval()
    model.config.use_cache = True
    return model


@torch.no_grad()
def generate_text(model, tokenizer, prompt, max_new_tokens=80, formatted=True, seed=42):
    torch.manual_seed(seed)
    if formatted:
        prompt = tokenizer.specials["user"] + prompt + tokenizer.specials["assistant"]
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
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
    text = tokenizer.decode(output[0].tolist(), skip_special=False)
    if formatted and "<assistant>" in text:
        text = text.split("<assistant>", 1)[1]
    text = text.replace("<end>", "").replace("<pad>", "").replace("<bos>", "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--raw", action="store_true", help="Do not add user/assistant tags")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_cpu_threads()
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer else model_path / "tokenizer.json"
    tokenizer = CharTokenizer.load(tokenizer_path)
    model = load_model(model_path)
    text = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        formatted=not args.raw,
        seed=args.seed,
    )
    print(text)


if __name__ == "__main__":
    main()
