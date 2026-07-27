# InstructGPT：把示范、偏好与在线优化接起来

[InstructGPT](https://arxiv.org/abs/2203.02155) 的影响不在于首次使用 SFT、reward model 或 PPO，而在于把三者组织成一条面向真实用户指令的完整数据闭环。它回答了一个基础问题：当预训练模型已经会续写文本，怎样让有限的人类反馈改变回答排序和交互行为？

## 前一站留下的问题

预训练最小化自然文本上的 token NLL，instruction SFT 则模仿标注者示范：

$$
\mathcal L_{\mathrm{SFT}}
=-\mathbb E_{(x,y)}
\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}).
$$

SFT 能教授回答格式、拒答方式和任务接口，但一个 prompt 往往有多个合理回答。要求标注者写出唯一最佳答案成本高，也无法表达两个都可接受回答之间的细微偏好。

## 三类数据，三个不同目标

论文流程可拆成：

1. 收集 demonstrations，训练 SFT policy；
2. 对同一 prompt 的多个回答排序，训练 reward model；
3. 从当前策略采样回答，用 reward 与 KL 约束执行 PPO；
4. 在 PPO-ptx 变体中混入预训练目标，缓解能力回归。

对 chosen $y_w$ 和 rejected $y_l$，reward model 使用 Bradley–Terry loss：

$$
\mathcal L_{\mathrm{RM}}
=-\mathbb E\log\sigma\left[
r_\phi(x,y_w)-r_\phi(x,y_l)
\right].
$$

policy 的理想化目标为

$$
\max_\theta
\mathbb E_{y\sim\pi_\theta}[r_\phi(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi_\theta\,\|\,\pi_{\mathrm{ref}}\right).
$$

PPO ratio 则相对一轮更新中冻结的 training-side old policy：

$$
\rho_t
=\exp\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right].
$$

$\pi_{\mathrm{old}}$ 与 $\pi_{\mathrm{ref}}$ 角色不同：前者定义 current–old update ratio，后者定义偏离锚点。真实 rollout behavior 还可能因推理引擎与 sampling processor 不同而偏离 old training policy；完整工程中必须单独记录它，并让 action mask 排除 prompt、padding 与非 policy token。

## 最小可执行语义

下面只保留 pairwise reward、采样 KL reward shaping、response mask 与 PPO clipped surrogate。`old_logp` 表示冻结的 training-side old log-prob；只有验证训推一致后，它才可视作实际 rollout 概率。`ref_logp` 只参与 KL。代码不是完整 trainer，也没有实现 value model 或 GAE。

```python
import torch
import torch.nn.functional as F
def masked_sum(x, mask):
    return (x * mask).sum(-1)
def reward_loss(chosen, rejected):
    return -F.logsigmoid(chosen - rejected).mean()
def ppo_loss(logp, old_logp, advantage, mask, clip=0.2):
    ratio = (logp - old_logp).exp()
    adv = advantage[:, None]
    left = ratio * adv
    right = ratio.clamp(1 - clip, 1 + clip) * adv
    objective = torch.minimum(left, right)
    return -(objective * mask).sum() / mask.sum(), ratio
torch.manual_seed(0)
chosen = torch.tensor([1.0, 0.4], requires_grad=True)
rejected = torch.tensor([0.0, 0.2], requires_grad=True)
rm = reward_loss(chosen, rejected)
assert rm < torch.log(torch.tensor(2.0))
old_logp = torch.tensor([[-1.0, -0.8, -0.7], [-0.9, -1.2, -0.6]])
ref_logp = old_logp - torch.tensor([[0.1, 0.2, 9.0], [0.0, -0.1, 9.0]])
mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
task_reward = torch.tensor([1.0, 0.2])
sampled_kl = masked_sum(old_logp - ref_logp, mask)
shaped_reward = task_reward - 0.05 * sampled_kl
advantage = (shaped_reward - shaped_reward.mean()).detach()
logp = old_logp.clone().requires_grad_(True)
pg, ratio = ppo_loss(logp, old_logp, advantage, mask)
loss = rm + pg
loss.backward()
assert torch.allclose(ratio[mask.bool()], torch.ones(4))
assert torch.all(logp.grad[~mask.bool()] == 0)
assert chosen.grad is not None and torch.isfinite(loss)
```

KL 的 sampled log-ratio 是 reward estimator，不是完整分布上的精确 KL；生产实现必须记录采样分布、reference 版本和系数位置。

## 论文实际支持什么

在论文收集的 OpenAI API prompt 与标注协议下：

- 标注者整体更偏好 InstructGPT 输出；
- 论文报告 1.3B InstructGPT 在该偏好评测中可胜过 175B GPT-3；
- 部分真实性与毒性指标改善；
- PPO-ptx 用预训练混合目标缓解部分公开 benchmark 回归；
- held-out labeler 与客户 prompt 被用于减少只记住训练标注者的风险。

这些是特定模型、数据、时间和评测分布上的实证结果，不是无条件能力排序。

## 论文没有证明什么

- 没有证明 1.3B 模型一般比 175B 模型知识更多或推理更强；
- 没有证明单一 reward model 能完整表示正确性、帮助性与安全；
- 没有证明 PPO 是所有后训练任务的最优 optimizer；
- 没有消除 reward hacking、标注者分歧、简单事实错误和分布外退化；
- 没有证明人类偏好等于事实正确或长期社会效用；
- 没有覆盖后来工具调用、长时 Agent 和可执行环境中的信用分配问题。

## 公开实现边界

[官方模型卡仓库](https://github.com/openai/following-instructions-human-feedback)记录了模型与数据说明；[summarize-from-feedback](https://github.com/openai/summarize-from-feedback)公开了更早的摘要反馈研究代码。二者都不是论文所用 InstructGPT 生产训练系统的完整开源实现。

公开论文给出了阶段、目标和主要实验，但没有提供精确重建所需的全部 prompt 数据、标注操作、模型权重、训练基础设施与内部评测。因此教学实现应验证目标语义，不能声称复现论文模型。

## 它推动了什么

三阶段流程在 instruction-following language model 上规模化展示并广泛推广了偏好后训练，也暴露了两个方向：

1. 能否消去显式 reward model 与在线 rollout，直接使用固定偏好对——见 [DPO](dpo.md)；
2. 在答案可执行验证的领域，能否用规则 reward 做在线探索——见 [DeepSeek-R1](deepseek-r1.md)。

整条因果脉络见[从续写到偏好与在线学习](../lineages/training-alignment.md)。目标与系统细节分别见[监督微调](../../training/supervised-finetuning.md)、[奖励建模](../../training/reward-modeling.md)、[在线 RL](../../training/online-rl.md)和[评测协议](../../evaluation/language-model-evaluation.md)。

## Reference {#reference}

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [openai/following-instructions-human-feedback](https://github.com/openai/following-instructions-human-feedback)
- [summarize-from-feedback](https://github.com/openai/summarize-from-feedback)
