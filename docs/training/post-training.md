# 后训练与偏好学习

后训练把基础模型的条件分布调整为可交互行为。它可以教授接口、改变回答排序、强化可验证解题和工具策略，却不能凭空补回预训练缺失的世界知识，也不能用一个 reward 覆盖正确性、帮助性、安全与成本。

本页是稳定入口：目标推导和实现细节分别位于[监督微调](supervised-finetuning.md)、[知识蒸馏](distillation.md)、[奖励建模](reward-modeling.md)、[离线偏好优化](offline-preference.md)、[在线 RL](online-rl.md)与[推理后训练](reasoning-posttraining.md)。

## 先看反馈接口

| 可获得信号 | 自然起点 | 不应忽略 |
| --- | --- | --- |
| 高质量示范答案 | SFT | mask、模板、数据覆盖 |
| Teacher logits / outputs | 蒸馏 | teacher 错误、support 与 tokenizer |
| 成对 chosen/rejected | Reward model、DPO/IPO | 标注噪声、reference、长度 |
| 单样本好/坏标签 | KTO 类目标 | 类别不平衡与基准效用 |
| 可执行结果奖励 | Online RL / RLVR | verifier 漏洞、rollout 成本 |
| 步骤级判断 | Process reward / supervision | 信用分配与步骤边界 |
| 环境轨迹 | Agentic RL | policy lag、权限与终态 |

方法名称不能替代监督语义。把二元标签强行拼成 pair，或把 infra failure 当成零奖励，会改变所学习的目标。

## 阶段关系

### SFT 作为行为先验

SFT 教授任务格式、对话边界、工具 schema 和基本完成策略。它通常为偏好或在线 RL 提供初始化，但不是必须固定的一次阶段；新环境或新接口可能需要交替补充 SFT 数据。

### Reward model 与偏好

[InstructGPT](https://arxiv.org/abs/2203.02155) 展示了 SFT、pairwise reward model 和 PPO 的代表性流程。reward model 把相对判断外推到新回答，在线 RL 再优化其分数。风险集中在分布外外推和 shortcut，详见[奖励建模](reward-modeling.md)。

离线偏好方法直接在已有回答上调整策略，部署简单、重放稳定；它们没有在线探索，不能自动发现数据中不存在的解法。DPO 与相关目标的假设见[离线偏好优化](offline-preference.md)。

### 在线策略优化

在线方法从当前或稍旧策略采样，能探索新行为，也使数据分布随训练变化。必须区分生成 rollout 的 $\pi_{\text{old}}$ 与约束偏离的 $\pi_{\text{ref}}$，并记录 verifier、policy version、old log-prob 和终止语义。PPO、RLOO 与 group-relative 目标见[在线 RL](online-rl.md)。

### 推理与验证

数学、代码和结构化任务可使用答案检查器、单元测试或形式 verifier。搜索产生候选，verifier 产生选择或奖励，再把结果转成 SFT、偏好、过程监督或 RL 数据。这个闭环比“使用某个 reasoning algorithm”更重要，见[推理后训练](reasoning-posttraining.md)。

## 共同目标与约束

带 reference 的后训练常写成

$$
\max_\pi\;
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right).
$$

$r$ 定义方向，$\pi_{\text{ref}}$ 定义偏离坐标，$\beta$ 定义二者权衡。实际训练还可能加入长度、格式、安全和成本项；这些项必须分别记录，不能把总 reward 当成唯一可解释指标。

## 数据与版本契约

```text
prompt / environment source and split
response generator and exact policy version
teacher / reference / reward / verifier version
tokenizer, chat template and action mask
preference or reward semantics
sampling and decoding configuration
terminal, truncated, invalid and infra-error status
judge / human annotation protocol
```

生成策略或模板变化后，pair 和 old log-prob 的语义也会变化。旧轨迹不能在没有 importance correction 和兼容性验证时被当成 on-policy 数据。

## 正确性与失效

- **单一 reward 抬升即成功**：模型可能投机长度、格式、测试漏洞或评分器偏好。
- **离线方法写成在线 RL 替代品**：二者的探索能力与数据分布不同。
- **旧策略与 reference 混淆**：PPO ratio 和 KL penalty 被错误实现。
- **忽略 response mask**：prompt 或 tool observation 被当成策略动作。
- **judge 与训练同源**：训练和评测共同偏好相同表面特征。
- **只看平均胜率**：事实性、拒答、长度、多样性和能力回归被掩盖。
- **新 recipe 普适化**：在单一数学或代码设置中有效，不代表跨领域稳定。

## 何时不应继续后训练

若失败来自过期知识、检索缺失、权限控制、确定性计算或工具错误，应先修复外部系统。没有独立评测、可靠反馈或可回放数据时，继续优化会放大测量偏差。高风险动作也不能依赖后训练替代执行层权限。

## 验证组织

1. 先验证数据对象、mask 与目标函数的手算小例。
2. 固定生成预算与数据，比较 SFT 或简单 rejection sampling 基线。
3. 分开报告 task success、事实性、偏好、长度、成本、KL 与多样性。
4. 在新策略样本上重新校准 reward/verifier，而不只测历史 pair。
5. 对 reward 突增、长度漂移和格式集中做轨迹审计。
6. 用人工、可执行 verifier 和独立 judge 形成多通道评测。
7. 对通用能力、安全和目标域分别设置停止与回滚门槛。

多步工具与长时环境继续见 [Agentic RL](../agentic-rl/index.md)，统计和 judge 口径见[评测工具](../practice/evaluation-tooling.md)。
