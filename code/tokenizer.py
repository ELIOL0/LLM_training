import json
from collections import OrderedDict
from pathlib import Path


SPECIALS = OrderedDict(
    [
        ("pad", "\x00"),
        ("bos", "\x01"),
        ("eos", "\x02"),
        ("unk", "\x03"),
        ("user", "\x04"),
        ("assistant", "\x05"),
        ("end", "\x06"),
    ]
)


class CharTokenizer:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.specials = SPECIALS
        self.pad_id = self.stoi[SPECIALS["pad"]]
        self.bos_id = self.stoi[SPECIALS["bos"]]
        self.eos_id = self.stoi[SPECIALS["eos"]]
        self.unk_id = self.stoi[SPECIALS["unk"]]
        self.user_id = self.stoi[SPECIALS["user"]]
        self.assistant_id = self.stoi[SPECIALS["assistant"]]
        self.end_id = self.stoi[SPECIALS["end"]]

    @property
    def vocab_size(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts):
        chars = set()
        for text in texts:
            chars.update(text.replace("\r", ""))
        for value in SPECIALS.values():
            chars.discard(value)
        itos = list(SPECIALS.values()) + sorted(chars)
        return cls(itos)

    def encode(self, text, add_bos=False, add_eos=False, max_length=None):
        ids = [self.stoi.get(ch, self.unk_id) for ch in text.replace("\r", "")]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids, skip_special=False):
        if skip_special:
            special_ids = set(self.stoi[v] for v in SPECIALS.values())
            ids = [int(i) for i in ids if int(i) not in special_ids]
            return "".join(self.itos[int(i)] for i in ids if 0 <= int(i) < len(self.itos))
        pieces = []
        special_names = {v: k for k, v in SPECIALS.items()}
        for i in ids:
            i = int(i)
            if i < 0 or i >= len(self.itos):
                continue
            ch = self.itos[i]
            if ch in special_names:
                pieces.append(f"<{special_names[ch]}>")
            else:
                pieces.append(ch)
        return "".join(pieces)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"itos": self.itos, "specials": dict(SPECIALS)}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
