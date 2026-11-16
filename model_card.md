# Model Card

最终提交模型位于 `model/final/`，详细说明见 `model/final/MODEL_CARD.md`。

- 模型：TinyChineseGPT-886K
- 参数量：886,016
- 训练阶段：预训练 → SFT → DPO
- 预训练验证 loss：0.5297
- SFT 验证 loss：1.0082
- DPO 验证 loss：0.1644
- DPO 验证偏好准确率：0.871
- 用途：课程作业流程演示
- 限制：模型很小、数据为合成数据、生成质量有限
