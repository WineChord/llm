# DeepSeek-R1：可验证奖励、冷启动与蒸馏闭环

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 把 reasoning post-training 的多条路线集中到一套公开流程中：在强 base model 上执行可验证奖励 RL，观察长推理行为，再用少量 cold-start 数据改善可读性，通过 rejection sampling 扩充监督数据，最后进行第二阶段 RL 与模型蒸馏。

它不应被压缩成“GRPO 训练出了推理”。GRPO 属于 policy optimizer；accuracy、format 和后续偏好信号属于 reward；长 rollout、采样数和输出上限属于训练与推理预算。

## R1-Zero：先隔离 RL 的作用

R1-Zero 从 DeepSeek-V3-Base 出发，不先做 reasoning SFT，使用 GRPO 与规则奖励。论文描述的主要 reward 包括：

- **accuracy reward**：数学答案检查、代码编译与测试等；
- **format reward**：要求 reasoning 与 answer 遵循指定结构。

示意写成

$$
R_i
=r_{\mathrm{accuracy}}(x,y_i)
+\lambda r_{\mathrm{format}}(y_i).
$$

这里 $\lambda$ 只是解释性记号，不能在没有公开依据时填成“官方精确权重”。GRPO 对同一 prompt 的一组回答构造相对优势：

$$
\hat A_i
=\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\varepsilon},
$$

再结合相对 rollout policy 的 clipped ratio 与相对 reference 的 KL 约束更新。

R1-Zero 展示了训练过程中更长回答、反思和策略变化等现象，也暴露出可读性差、语言混合与行为范围较窄的问题。这些问题直接推动最终 R1 重新引入 cold-start 数据。

## 最小可执行语义

下面用确定性整数 verifier、独立格式奖励和 group-relative loss 表示 R1-Zero 的不可约接口。`old_logp` 定义 rollout ratio，`ref_logp` 定义 KL；全同 reward 组返回零优势。

```python
import re
import torch
def reward(text, gold):
    numbers = re.findall(r"-?\d+", text)
    accuracy = float(numbers and int(numbers[-1]) == gold)
    formatted = float(re.fullmatch(r"<answer>-?\d+</answer>", text) is not None)
    return accuracy + 0.1 * formatted
def group_advantage(rewards, group_size):
    group = rewards.view(-1, group_size)
    mean = group.mean(-1, keepdim=True)
    std = group.std(-1, keepdim=True, unbiased=False)
    advantage = torch.where(std > 1e-6, (group - mean) / std, torch.zeros_like(group))
    return advantage.flatten()
def grpo_loss(logp, old_logp, ref_logp, advantage, mask, clip=0.2, beta=0.01):
    ratio = (logp - old_logp).exp()
    adv = advantage[:, None]
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    log_ref_ratio = ref_logp - logp
    kl = log_ref_ratio.exp() - log_ref_ratio - 1
    token_loss = -surrogate + beta * kl
    return (token_loss * mask).sum() / mask.sum(), ratio
responses = ["<answer>4</answer>", "5", "4", "<answer>9</answer>", "8", "9"]
golds = [4, 4, 4, 9, 9, 9]
rewards = torch.tensor([reward(text, gold) for text, gold in zip(responses, golds)])
advantage = group_advantage(rewards, 3)
assert torch.allclose(advantage.view(-1, 3).mean(-1), torch.zeros(2), atol=1e-6)
assert torch.equal(group_advantage(torch.ones(6), 3), torch.zeros(6))
old_logp = -torch.tensor([[1., .8, .7], [.9, 1.2, .6], [1.1, .7, .5],
                          [.8, .9, .6], [1.2, .9, .7], [.7, .8, .5]])
ref_logp = old_logp - 0.05
mask = torch.tensor([[1., 1., 1.], [1., 1., 0.], [1., 1., 0.],
                     [1., 1., 1.], [1., 1., 0.], [1., 1., 0.]])
logp = old_logp.clone().requires_grad_(True)
loss, ratio = grpo_loss(logp, old_logp, ref_logp, advantage, mask)
loss.backward()
assert torch.allclose(ratio[mask.bool()], torch.ones(int(mask.sum())))
assert torch.all(logp.grad[~mask.bool()] == 0)
assert reward("<answer>4</answer>", 4) > reward("4", 4) > reward("5", 4)
```

这段代码固定了目标语义，不包含模型采样、分布式 rollout、课程、数据去重或生产级 parser。真实 verifier 必须区分错误答案、非法格式、超时和基础设施故障，后者不能直接当作 policy 失败。

## 最终 R1 不是单阶段 RL

公开流程可以按因果关系读成：

1. **Cold start**：用数千条高质量 reasoning examples 改善 R1-Zero 暴露的可读性与输出结构；
2. **Reasoning-oriented RL**：继续使用数学、代码等可验证任务，并加入语言一致性信号；
3. **Rejection sampling**：从阶段性 checkpoint 生成并筛选 reasoning data；
4. **第二次 SFT**：论文报告约 60 万 reasoning 与约 20 万 non-reasoning 样本；
5. **第二次 RL**：在 reasoning reward 外加入帮助性与安全偏好；
6. **Distillation**：用 R1 生成的数据训练多个较小的 dense model。

这条流程同时使用 SFT、规则 reward、偏好 reward、在线 RL 和蒸馏。只比较“GRPO 与某个离线 loss”会遗漏数据生成与训练阶段的大部分差异。

## 论文实际支持什么

- 在一个已经充分预训练的强 base model 上，规则奖励的在线 RL 可以显著改变数学与代码 reasoning 行为；
- R1-Zero 提供了不经 preliminary reasoning SFT 的受控路线；
- 最终多阶段流程相对 R1-Zero 报告了更好的可读性与通用交互，但公开结果不足以把收益隔离归因于其中某一个阶段；
- 论文报告最终 R1 在多个 reasoning benchmark 上具有竞争力；
- 论文所测小模型设置中，使用 R1 数据蒸馏通常优于直接对小模型执行同类 RL。

这些结果共同支持“可验证 reward、在线探索和高质量轨迹蒸馏可以形成能力闭环”，而不是某个 optimizer 单独解释全部收益。

## 论文没有证明什么

- 没有证明 RL 从随机模型或缺乏相关预训练能力的模型中凭空创造 reasoning；
- 没有证明可见 CoT 忠实呈现模型内部因果计算；
- 没有证明 GRPO 对所有 reward 分布、领域和 group size 都稳定；
- 没有证明更长回答天然更正确；
- 没有证明数学与代码 verifier 可以直接推广到开放写作、事实综合和安全；
- 没有证明蒸馏在所有 student 架构和能力维度上都优于直接 RL；
- 没有排除 base model、数据选择、训练预算与 benchmark 污染对结果的贡献。

## 公开实现边界

[DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)公开了模型权重、使用方式、蒸馏模型和评测信息；论文公开了阶段结构、核心目标与大量实验。官方仓库不是端到端训练系统的完整发布。

精确复现仍缺少全部训练 prompt、过滤与 rejection 细节、reward/parser 版本、超参数、rollout 基础设施、policy lag 处理和内部评测。公开结果可以复核目标与模型行为，但不能据此声称重建了生产 recipe。

## 它连接的两条谱系

训练侧，R1 承接 SFT、在线 RL、group-relative advantage 与 distillation，见[从续写到偏好与在线学习](../lineages/training-alignment.md)。推理侧，它把长轨迹、候选验证、搜索数据和 RLVR 连接起来，见[从外显推理到可验证搜索](../lineages/reasoning-verification.md)。

机制细节见[推理后训练](../../training/reasoning-posttraining.md)、[在线 RL](../../training/online-rl.md)、[推理时计算](../../reasoning/test-time-compute.md)、[搜索与验证](../../reasoning/search-verification.md)和[Agentic RL 数学与算法](../../agentic-rl/math-algorithms.md)。

## Reference {#reference}

- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeek-R1 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1)
