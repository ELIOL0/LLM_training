from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pre = pd.read_csv(LOGS / "pretrain.csv")
    axes[0].plot(pre["step"], pre["train_loss"], label="train")
    axes[0].plot(pre["step"], pre["val_loss"], label="val")
    axes[0].set_title("Pretraining loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    sft = pd.read_csv(LOGS / "sft.csv")
    axes[1].plot(sft["step"], sft["train_loss"], label="train")
    axes[1].plot(sft["step"], sft["val_loss"], label="val")
    axes[1].set_title("SFT loss")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    dpo = pd.read_csv(LOGS / "dpo.csv")
    axes[2].plot(dpo["step"], dpo["loss"], label="DPO loss")
    axes[2].plot(dpo["step"], dpo["val_loss"], label="val loss")
    axes[2].set_xlabel("step")
    axes[2].set_title("DPO loss")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    out = LOGS / "loss_curves.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    main()
