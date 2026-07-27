# 训练与对齐

一个完整模型通常经历多个目标不同的阶段。把它们都称为“训练”会隐藏数据、优化器、行为目标和系统瓶颈的变化。

## 生命周期

| 阶段 | 主要目标 | 常见风险 |
| --- | --- | --- |
| 预训练 | 学习广泛分布与表示 | 数据污染、欠训练、数值失败 |
| 持续预训练 | 适配领域或新语料 | 遗忘、配比失衡 |
| SFT | 学习任务和对话格式 | 模仿错误、风格过拟合 |
| 偏好学习 | 调整回答相对排序 | 奖励投机、偏好偏差 |
| 推理训练 | 提高搜索、验证或长程解题 | 过长输出、验证器过拟合 |
| 压缩与适配 | 降低部署成本 | 能力退化、校准漂移 |

## 贯穿各阶段的变量

- 数据来源、采样权重和可追溯性；
- tokenizer、chat template 与最大序列长度；
- 精度、优化器、学习率、batch 与梯度裁剪；
- checkpoint 初始化、可训练参数和冻结策略；
- 评测集污染、回归门槛与安全边界；
- 训练 FLOPs、失败重跑和最终服务成本。

## 阅读路径

1. [预训练](pretraining.md)：数据暴露、token 预算与持续训练。
2. [监督微调](supervised-finetuning.md)：模板、loss mask、packing 与能力混合。
3. [优化与稳定性](optimization.md)和[优化器家族](optimizer-families.md)：数值路径、AdamW、Muon 与更新尺度。
4. [后训练总览](post-training.md)与[奖励建模和偏好优化](reward-preference.md)：RM、RLHF、DPO、IPO、KTO 与可验证奖励。
5. [参数高效与压缩](peft-compression.md)：LoRA、量化、蒸馏与部署回归。

涉及多步环境和在线 rollout 时，再进入 [Agentic RL](../agentic-rl/index.md)。
