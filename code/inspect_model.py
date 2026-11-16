import argparse
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description="Inspect a .safetensors checkpoint")
    parser.add_argument(
        "--path",
        default=str(Path(__file__).resolve().parents[1] / "model" / "final" / "model.safetensors"),
    )
    parser.add_argument("--tensor", default="", help="Only inspect one tensor name")
    args = parser.parse_args()

    with safe_open(args.path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if args.tensor:
            tensor = f.get_tensor(args.tensor)
            flat = tensor.flatten()
            print("name:", args.tensor)
            print("shape:", tuple(tensor.shape))
            print("dtype:", tensor.dtype)
            print("numel:", tensor.numel())
            print("mean:", float(flat.mean()))
            print("std:", float(flat.std()) if flat.numel() > 1 else 0.0)
            print("min:", float(flat.min()))
            print("max:", float(flat.max()))
            print("first_values:", flat[:10].tolist())
            return

        total = 0
        print("file:", args.path)
        print("num_tensors:", len(keys))
        for key in keys:
            tensor = f.get_tensor(key)
            total += tensor.numel()
            print(f"{key:45s} {str(tuple(tensor.shape)):18s} {tensor.dtype}")
        print("total_params:", total)


if __name__ == "__main__":
    main()
