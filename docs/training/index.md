# 训练与对齐

训练不是一条从 loss 单调走向能力的流水线。每个阶段都重新定义数据分布、可训练参数、目标函数和评测边界；如果只记录“用了哪个算法”，就无法解释模型究竟学到了什么。

从监督示范、偏好建模到在线策略优化，并不存在一条永远正确的固定流水线。[后训练与对齐](../landscape/lineages/training-alignment.md)把 InstructGPT、Constitutional AI、DPO 与可验证奖励放回各自的反馈接口；[强化学习](../reinforcement-learning/index.md)进一步展开序贯决策、策略优化和语言模型反馈，[推理、搜索与验证](../landscape/lineages/reasoning-verification.md)则说明 inference-time search 怎样反过来成为训练数据和奖励来源。关键转折可从 [InstructGPT](../landscape/works/instructgpt.md)、[DPO](../landscape/works/dpo.md) 与 [DeepSeek-R1](../landscape/works/deepseek-r1.md) 逐项深读。

## 生命周期

| 阶段 | 直接优化对象 | 数据单位 | 必须保留的锚点 |
| --- | --- | --- | --- |
| 预训练 | next-token 或去噪似然 | 文档与 token | 数据快照、tokenizer、global token |
| 持续预训练 | 目标域似然 | 新域与通用 replay | 原模型能力与旧数据分布 |
| SFT | 示范条件似然 | prompt–response / trajectory | chat template 与 response mask |
| 蒸馏 | teacher 分布或生成行为 | logits、序列、偏好 | teacher 版本与支持集 |
| Reward Modeling | 人类或 verifier 的相对判断 | pair、list、步骤 | 标注语义与校准集 |
| 离线偏好 | 已收集回答的排序 | chosen/rejected 或标签 | reference policy 与数据覆盖 |
| 在线 RL/RLVR | 当前策略的期望回报 | rollout / episode | behavior policy、reference、verifier |
| 推理后训练 | 搜索、验证与长程解题 | 候选、步骤、轨迹 | 预算、公平评测与污染隔离 |
| PEFT / 压缩 | 低成本适配或部署 | adapter / compressed weight | base digest 与回归基线 |

这些阶段不是固定流水线。可靠 verifier 可能让任务直接从 SFT 进入 RLVR；高质量离线偏好可能无需显式 reward model；领域适配也可能只需检索而非继续训练。

## 共同训练契约

任何阶段的可重放 checkpoint 都不只包含权重：

```text
model and trainable-parameter schema
optimizer, scheduler and precision/scaler state
global steps and effective training tokens
RNG states and distributed topology
data snapshot, mixture version and data cursor
tokenizer, template and maximum length
reference / teacher / reward / verifier versions
code revision and evaluation protocol
```

只恢复 model weights 是 warm start，不是严格 resume。恢复后下一批数据、学习率、随机性和累计 token 都应连续。

### Token 归一化

动态长度、packing 和 response-only mask 会让每个 rank 的有效 token 数不同。正确的全局目标是

$$
\mathcal L
=
\frac{\sum_r\sum_t m_{r,t}\ell_{r,t}}
{\sum_r\sum_t m_{r,t}},
$$

而不是简单平均各 rank 的局部 mean。实现与断言见[训练目标实现](../practice/training-objectives.md)。

### 四个策略身份

后训练中经常同时存在：

- $\pi_\theta^{\mathrm{train}}$：正在更新的 policy；
- $\pi_{\text{old}}^{\mathrm{train}}$：冻结的 update 基准；
- $\mu^{\mathrm{rollout}}$：实际产生 token 的 behavior distribution；
- $\pi_{\text{ref}}$：定义偏离成本的冻结 reference。

$\pi_{\text{old}}^{\mathrm{train}}$ 用于 current–old ratio 或 trust-region，$\mu^{\mathrm{rollout}}$ 决定是否需要 off-policy correction，$\pi_{\text{ref}}$ 用于 KL anchor；三者偶尔权重相同，也不能在算法语义上合并。详见[策略身份与训推分布](../reinforcement-learning/training-inference-discrepancy.md)。

## 阅读路径

### 基础训练

1. [预训练](pretraining.md)：目标、有效 token、数据阶段与恢复。
2. [规模律与实验设计](scaling-experiment-design.md)：$6ND$ 的边界、compute-optimal 与 inference-aware 成本。
3. [优化与稳定性](optimization.md)和[优化器家族](optimizer-families.md)：精度、更新尺度、AdamW 与 Muon 的条件。
4. [监督微调](supervised-finetuning.md)：模板、mask、packing 和能力混合。

### 适配与知识迁移

5. [知识蒸馏](distillation.md)：token、sequence 与 on-policy distillation。
6. [参数高效微调](peft.md)：LoRA、QLoRA、DoRA、adapter 与 merge 契约。
7. [参数高效训练与压缩](peft-compression.md)：稳定总览与部署边界。

### 反馈、策略与推理

8. [后训练总览](post-training.md)：根据反馈接口选择方法。
9. [奖励建模](reward-modeling.md)：Bradley–Terry、不可辨识性、偏差与过程奖励。
10. [离线偏好优化](offline-preference.md)：KL 最优策略、DPO 及其假设。
11. [强化学习总览](../reinforcement-learning/index.md)：先建立 value、policy、feedback 与 data regime 的共同坐标。
12. [在线 RL 与可验证奖励](online-rl.md)：PPO、RLOO、GRPO、policy lag 与退化组。
13. [推理后训练](reasoning-posttraining.md)：从搜索和验证生成训练信号。

reward 来源与 optimizer 的关系见[反馈制度](../reinforcement-learning/feedback-regimes.md)，多步环境、工具动作和长时信用分配继续见 [Agentic RL](../agentic-rl/index.md)；数据来源、真实 token share 和重复暴露见[数据工程](../data/index.md)。

## 正确性与失效

- **阶段名替代目标**：两个都称为 DPO 或 GRPO 的运行，可能使用不同 mask、长度归一化、reference 和 reward。
- **恢复权重替代恢复训练**：数据 cursor 或 optimizer state 重置会改变优化轨迹。
- **训练分数替代独立评测**：reward、judge 和 verifier 同源时容易共同投机。
- **FLOPs 替代成本**：失败重跑、数据处理、rollout 和部署查询可能主导总预算。
- **新 recipe 替代受控基线**：新方法必须在相同数据、token、生成预算和调参资源下比较。

## 何时不训练

若问题来自知识新鲜度、权限数据、确定性计算或少量事实更新，检索、工具、路由或上下文管理通常比改权重更可控。训练适合改变稳定分布与行为；它不应替代可验证的外部状态。

## 验证组织

每个训练阶段都应有：

1. 单 batch 与小模型的目标函数手算对照；
2. mask、长度、padding、rank 划分和 resume 不变量；
3. 固定数据与预算下的最简单基线；
4. 目标能力、通用能力、校准、安全和成本的独立切片；
5. 数据、teacher、reference、reward、verifier 与 judge 的版本隔离；
6. 失败样本、退化 slice 和未决假设，而不只保存汇总分数。

从公开配方观察阶段怎样重新组合，可沿 [DeepSeek](../landscape/families/deepseek.md)、[Kimi](../landscape/families/kimi.md) 与 [GLM](../landscape/families/glm.md) 家族进入；各页会把 checkpoint、报告和未知训练细节分开。
