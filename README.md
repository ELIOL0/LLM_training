# LLM_training

GitHub 仓库：https://github.com/ELIOL0/LLM_training

这是一个 CPU 可运行的小型语言模型训练作业，完整包含：

1. 从零预训练（pretraining）
2. 指令微调（SFT）
3. 偏好对齐（DPO，作为 RLHF 的简化实现）
4. 评测与生成示例

模型参数量约 **886,016**，使用字符级 tokenizer，训练数据为本地生成的中文合成数据。
本项目的目标不是追求生成质量，而是完整跑通 LLM 训练流程。

## 环境

- Windows
- Anaconda 环境：`transformer`
- Python 3.10
- PyTorch 2.14.1 CPU
- Transformers 5.20.2

直接使用环境中的 Python：

```powershell
conda activate transformer
python --version
```

## 目录

```text
LLM_training/
  code/                    训练、评测、生成脚本
  data/                    预训练/SFT/DPO 数据和 tokenizer
  model/
    final/                 最终提交模型
  logs/                    三阶段训练日志和 loss 曲线
  eval/                    评测指标、prompt、生成结果
```

仓库只提交最终模型 `model/final/`；预训练、SFT、DPO 的中间 checkpoint
保留在本地用于训练衔接，不作为仓库内容。
训练说明报告和个人信息不纳入仓库。

## 复现命令

在项目根目录执行：

```powershell
conda activate transformer
python code/generate_data.py
python code/build_tokenizer.py
python code/train_pretrain.py --steps 3500
python code/train_sft.py --epochs 20
python code/train_dpo.py --epochs 3
python code/eval.py
python code/plot_logs.py
```

## 推理

```powershell
conda activate transformer
python code/generate.py --model model/final --prompt "请用一句话介绍春天。"
```

预训练模型可以用原始文本续写：

```powershell
python code/generate.py --model model/pretrain/best --prompt "清晨，小禾沿着河边散步。" --raw
```

## 加载模型

模型使用 HuggingFace `GPT2LMHeadModel` 格式保存，tokenizer 是项目内的字符级 tokenizer：

```python
import sys
sys.path.append("code")

from transformers import GPT2LMHeadModel
from tokenizer import CharTokenizer

tokenizer = CharTokenizer.load("model/final/tokenizer.json")
model = GPT2LMHeadModel.from_pretrained("model/final")
model.eval()
```

## 关键文件

- `model/final/`：最终模型参数
- `logs/pretrain.csv`、`logs/sft.csv`、`logs/dpo.csv`：训练日志
- `logs/loss_curves.png`：loss 曲线
- `eval/eval_report.md`：评测报告
- `data/dataset_card.md`：数据说明

## 说明

- 本作业没有训练 reward model，也没有使用 PPO；RLHF 阶段使用 DPO。
- DPO 直接使用 chosen/rejected 偏好对优化策略模型，是偏好对齐的常用简化方案。
- 数据和模型都很小，生成质量有限，指标仅用于证明训练流程闭环。
