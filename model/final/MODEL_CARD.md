# Model Card

## 模型名称

TinyChineseGPT-886K

## 基本信息

- 架构：GPT-2 风格 decoder-only Transformer
- 参数量：886,016
- 层数：4
- 隐藏维度：128
- 注意力头：4
- FFN 维度：512
- 上下文长度：128
- 词表：596（字符级 tokenizer）
- 初始化：随机初始化，从零训练

## 训练流程

1. 预训练：3,500 步，约 717 万 tokens，验证 loss 0.5297
2. SFT：278 条训练样本，20 epoch，验证 loss 1.0082
3. DPO：278 条偏好对，3 epoch，验证 loss 0.1644，偏好准确率 0.871

## 数据

- 预训练数据：本地生成的中文短故事、场景描写和简单常识句，约 179 万字符
- SFT 数据：介绍、解释词语、写故事、礼貌改写、建议、简单加法、常识问答、心情描写
- DPO 数据：由 SFT 数据构造的 `(prompt, chosen, rejected)` 偏好对

## 用途

仅用于课程作业，演示从零预训练、指令微调和偏好对齐的完整流程。

## 限制

- 模型非常小，生成质量有限，可能出现重复、错字和事实错误。
- 训练数据为合成数据，领域和表达多样性有限。
- 上下文只有 128 字符，不能处理长文本。
- DPO 偏好数据由规则构造，不代表真实人类偏好。
- 未训练 reward model，未使用 PPO。

## 加载方式

```python
import sys
sys.path.append("code")

from transformers import GPT2LMHeadModel
from tokenizer import CharTokenizer

tokenizer = CharTokenizer.load("model/final/tokenizer.json")
model = GPT2LMHeadModel.from_pretrained("model/final")
```

## 文件

- `config.json`：模型配置
- `model.safetensors`：模型权重
- `tokenizer.json`：字符级 tokenizer
- `generation_config.json`：生成配置
- `train_info.json`：DPO 阶段训练信息
