# DPO：把偏好模型折叠进策略目标

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) 针对 RLHF 的一个核心摩擦：如果已经有 chosen/rejected 偏好对，是否必须先拟合 scalar reward，再运行在线 RL？DPO 给出的答案不是“偏好不再需要奖励假设”，而是利用 KL 正则化最优策略的形式，把 reward 差直接改写成 policy/reference log-ratio。

## 从 KL 正则化策略开始

固定 prompt $x$，考虑

$$
\max_\pi
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

其最优策略满足

$$
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

因此

$$
r(x,y)
=\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

对同一 prompt 的 chosen/rejected 做差，$\log Z(x)$ 抵消。代入 Bradley–Terry preference model，定义

$$
h_\theta
=\left[
\log\pi_\theta(y_w\mid x)
-\log\pi_{\mathrm{ref}}(y_w\mid x)
\right]
-\left[
\log\pi_\theta(y_l\mid x)
-\log\pi_{\mathrm{ref}}(y_l\mid x)
\right],
$$

得到

$$
\mathcal L_{\mathrm{DPO}}
=-\mathbb E\log\sigma(\beta h_\theta).
$$

reference 不是可随意丢弃的常数：它随回答变化，定义 policy 相对原行为的移动坐标。

## 最小可执行语义

下面的 `aligned_sequence_logp` 接收已经左移对齐的 logits 与 target：`logits[b,t]` 预测 `target[b,t]`。前两个序列是 chosen，后两个是 rejected。mask 保证 prompt/padding 不进入序列概率。

```python
import torch
import torch.nn.functional as F
def aligned_sequence_logp(logits, target, mask):
    token_logp = logits.log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    return (token_logp * mask).sum(-1)
def dpo_loss(policy_logp, ref_logp, batch_size, beta=0.2):
    pi_w, pi_l = policy_logp[:batch_size], policy_logp[batch_size:]
    ref_w, ref_l = ref_logp[:batch_size], ref_logp[batch_size:]
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(margin).mean(), margin
torch.manual_seed(0)
B, T, V = 2, 3, 5
target = torch.tensor([[1, 2, 0], [2, 3, 4], [0, 1, 0], [4, 1, 2]])
mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [1., 1., 0.], [1., 1., 1.]])
reference = torch.zeros(2 * B, T, V)
policy = reference.clone().requires_grad_(True)
ref_score = aligned_sequence_logp(reference, target, mask)
pi_score = aligned_sequence_logp(policy, target, mask)
loss, margin = dpo_loss(pi_score, ref_score, B)
assert torch.allclose(margin, torch.zeros(B))
assert torch.allclose(loss, torch.log(torch.tensor(2.0)))
loss.backward()
assert torch.all(policy.grad[mask == 0] == 0)
updated = (policy - 0.5 * policy.grad).detach()
new_score = aligned_sequence_logp(updated, target, mask)
new_loss, new_margin = dpo_loss(new_score, ref_score, B)
assert new_loss < loss and torch.all(new_margin > 0)
```

这里使用序列 log-probability 的总和。改成 token mean 会改变长度归纳偏置和原始序列概率语义，不能作为不记录的实现细节。

## DPO 简化了什么

- 不需要单独训练并部署 scalar reward model；
- 不需要在训练环中持续向当前 policy rollout；
- 不需要 value model、GAE 或 PPO 的 on-policy 状态；
- 固定 pair 可以稳定重放，目标易于做手算与单元测试。

它由此降低了系统复杂度，也把能力上限更直接地交给离线 pair 的覆盖、质量与 reference support。

## 论文实际支持什么

原论文在 sentiment control、summarization 和 single-turn dialogue 等实验中表明，DPO 可以用简单分类式目标取得有竞争力的偏好优化结果，并在其设置中比所比较的 RLHF 基线更易训练。

这些实验支持“在若干固定偏好数据任务中，显式 reward model 与在线 PPO 不是获得良好结果的必要步骤”。数学推导依赖 KL-regularized optimum 与 Bradley–Terry 假设；实验只能检验由此得到的目标在受测任务中是否可训练、是否有竞争力，不能反过来证明假设普遍成立。

## 论文没有证明什么

- 没有证明离线偏好能发现 pair support 中从未出现的新策略；
- 没有证明 online RL、探索或环境交互已经没有价值；
- 没有证明 Bradley–Terry 能完整描述有 tie、循环偏好或群体分歧的反馈；
- 没有自动解决 pair 污染、生成策略偏差、模板不一致和长度捷径；
- 没有证明远离 reference 后的隐式 reward 仍可靠；
- 没有覆盖多步工具环境中的状态转移和长时信用分配。

当当前 policy 已明显偏离 pair 生成策略，或任务有低成本可执行 reward 时，重新采样偏好、rejection sampling 或在线 RL 可能更合适。

## 公开实现边界

[作者参考实现](https://github.com/eric-mitchell/direct-preference-optimization)公开了论文训练代码；[TRL 的 DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) 提供了持续演进的工程实现。两者不是同一个复现实验环境，默认 loss、长度归一化、数据 collator、reference 处理和后续变体可能不同。

使用任何实现时都应冻结并记录：

```text
policy / reference revisions
tokenizer and chat template
chosen / rejected generator and annotation protocol
prompt and response masks
sum / mean sequence reduction
beta convention and loss variant
truncation, packing and grouped split
```

## 它留下的问题

DPO 把固定偏好数据利用得更简单，但没有回答“怎样主动发现更好的 reasoning trajectory”。数学、代码等任务若有可执行 verifier，可以让当前策略在线生成候选并获得新 reward；这一分支在 [DeepSeek-R1](deepseek-r1.md) 中形成 reasoning RL 与 distillation 闭环。

整条历史见[从续写到偏好与在线学习](../lineages/training-alignment.md)。更完整的假设、IPO/KTO/SimPO 与验证方法见[离线偏好优化](../../training/offline-preference.md)，在线数据与 old/reference 契约见[在线 RL](../../training/online-rl.md)。

## Reference {#reference}

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)
- [TRL 的 DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
