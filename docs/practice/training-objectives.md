# 手撕：训练目标

训练目标最容易出错的地方不是公式名称，而是 mask、归一化、行为策略与参考策略。以下实现把每个约定放进函数参数，并用退化输入检查它的语义。

## Token-normalized cross entropy

对有效 token mask $m_t\in\{0,1\}$：

$$
\mathcal L=
\frac{\sum_t m_t\ell_t}{\sum_t m_t}.
$$

先返回 numerator 与 denominator，分布式训练才能在全局求和后相除：

```python
import torch
from torch import nn
import torch.nn.functional as F
def token_ce_parts(logits, labels, mask):
    """logits:[...,V], labels/mask:[...] -> scalar numerator, denominator."""
    if logits.shape[:-1] != labels.shape or labels.shape != mask.shape:
        raise ValueError("incompatible token shapes")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum(), weight.sum()
def token_ce(logits, labels, mask):
    numerator, denominator = token_ce_parts(logits, labels, mask)
    if denominator == 0:
        raise ValueError("batch has no supervised token")
    return numerator / denominator
```

各 rank 的 local mean 不能再等权平均；ragged batch 的正确分布式缩放见[手撕：分布式与容错](distributed-systems.md)。

## AdamW

AdamW 把 weight decay 与自适应梯度更新解耦：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

```python
@torch.no_grad()
def adamw_step(p, grad, m, v, step, lr, betas=(0.9, 0.999), eps=1e-8, wd=0.0):
    """p/grad/m/v: same shape; state tensors use the chosen state dtype."""
    b1, b2 = betas
    m.mul_(b1).add_(grad, alpha=1 - b1)
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)
    m_hat = m / (1 - b1 ** step)
    v_hat = v / (1 - b2 ** step)
    p.mul_(1 - lr * wd)
    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

```python
p = torch.tensor([1.0, -2.0], dtype=torch.float64)
g = torch.tensor([0.2, -0.4], dtype=torch.float64)
m, v = torch.zeros_like(p), torch.zeros_like(p)
adamw_step(p, g, m, v, step=1, lr=0.1, wd=0.01)
ref = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
opt = torch.optim.AdamW([ref], lr=0.1, weight_decay=0.01)
ref.grad = g.clone()
opt.step()
torch.testing.assert_close(p, ref)
```

生产优化器还需定义哪些参数不 decay、状态 dtype、分片与 checkpoint。梯度裁剪发生在 unscale 和 finite check 之后、step 之前。

## LoRA

[LoRA](https://arxiv.org/abs/2106.09685)对冻结线性层 $W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$ 注入低秩增量：

$$
\Delta W=\frac{\alpha}{r}BA,\qquad
A\in\mathbb R^{r\times d_{\text{in}}},
\quad B\in\mathbb R^{d_{\text{out}}\times r}.
$$

```python
class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base, self.scale = base, alpha / rank
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        self.merged = False
    def forward(self, x):
        y = self.base(x)
        return y if self.merged else y + (x @ self.a.T @ self.b.T) * self.scale
    @torch.no_grad()
    def merge(self):
        if not self.merged:
            self.base.weight.add_(self.b @ self.a, alpha=self.scale)
            self.merged = True
    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.base.weight.sub_(self.b @ self.a, alpha=self.scale)
            self.merged = False
```

```python
torch.manual_seed(1)
layer = LoRALinear(nn.Linear(5, 3, bias=False), rank=2, alpha=4)
x = torch.randn(4, 5)
torch.testing.assert_close(layer(x), layer.base(x))
with torch.no_grad():
    layer.b.normal_()
y = layer(x)
layer.merge()
torch.testing.assert_close(y, layer(x))
layer.merge()
layer.unmerge()
torch.testing.assert_close(y, layer(x))
```

$B=0$ 使初始输出精确等于基座。量化基座不能把增量直接原地写进 packed 4-bit 数据；应在可表达精度中合并并重新量化。

## Knowledge distillation

[Knowledge Distillation](https://arxiv.org/abs/1503.02531)用 teacher 的软分布向 student 传递类别关系。温度为 $T$ 时：

$$
q_T=\operatorname{softmax}(z_{\text{teacher}}/T),\quad
p_T=\operatorname{softmax}(z_{\text{student}}/T),
$$

$$
\mathcal L=(1-\alpha)\mathcal L_{\text{hard}}
+\alpha T^2\operatorname{KL}(q_T\Vert p_T).
$$

```python
def kd_loss(student, teacher, labels, mask, temperature=2.0, alpha=0.5):
    """student/teacher:[B,T,V], labels/mask:[B,T] -> scalar loss."""
    if student.shape != teacher.shape or student.shape[:-1] != labels.shape:
        raise ValueError("teacher and student token spaces must align")
    weight = mask.to(student.dtype)
    if weight.sum() == 0:
        raise ValueError("distillation batch has no valid token")
    logp = F.log_softmax(student / temperature, dim=-1)
    q = F.softmax(teacher.detach() / temperature, dim=-1)
    soft = (q * (q.clamp_min(1e-12).log() - logp)).sum(-1)
    hard = F.cross_entropy(
        student.flatten(0, -2), labels.flatten(), reduction="none"
    ).view_as(labels)
    return (((1 - alpha) * hard + alpha * temperature ** 2 * soft) * weight).sum() / weight.sum()
```

相同 logits 时 soft KL 接近零；mask token 的 teacher 内容不应影响梯度。tokenizer 或词表不同，应改做 sequence-level distillation，不要假装逐 token 对齐。

## Bradley–Terry reward model

$$
P(y_w\succ y_l)=\sigma(r_w-r_l),\qquad
\mathcal L=\operatorname{softplus}(-(r_w-r_l)).
$$

```python
def bradley_terry_loss(reward_a, reward_b, target=None):
    """rewards:[B]; target=1 means a wins, 0 loses, 0.5 ties."""
    delta = reward_a - reward_b
    target = torch.ones_like(delta) if target is None else target.to(delta.dtype)
    return F.binary_cross_entropy_with_logits(delta, target)
```

```python
ra = torch.tensor([2.0, -1.0])
rb = torch.tensor([0.0, 3.0])
base = bradley_terry_loss(ra, rb)
torch.testing.assert_close(base, bradley_terry_loss(ra + 17, rb + 17))
```

共同平移不改变损失，说明奖励零点不可辨识；奖励尺度也会改变后续 RL 强度。ties、位置和长度 shortcut 必须在数据与评测中单独处理。

## DPO、IPO 与 SimPO

令

$$
\Delta_\pi=\log\pi(y_w\mid x)-\log\pi(y_l\mid x),\quad
\Delta_{\mathrm{ref}}=\log\pi_{\mathrm{ref}}(y_w\mid x)
-\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

```python
def preference_loss(chosen, rejected, ref_chosen=None, ref_rejected=None,
                    beta=0.1, kind="dpo", gamma=0.0):
    """Inputs:[B] sequence log-probabilities under one explicit reduction."""
    margin = chosen - rejected
    if kind == "simpo":
        return -F.logsigmoid(beta * margin - gamma).mean()
    if ref_chosen is None or ref_rejected is None:
        raise ValueError(f"{kind} requires reference log-probabilities")
    relative = margin - (ref_chosen - ref_rejected)
    if kind == "dpo":
        return -F.logsigmoid(beta * relative).mean()
    if kind == "ipo":
        return (relative - 1 / (2 * beta)).square().mean()
    raise ValueError(f"unknown preference loss: {kind}")
```

sequence log-probability使用 token sum 还是 mean 会改变长度先验，必须在调用前明确；`beta` 与 SimPO 的 margin convention 也应随实验记录。[DPO](https://arxiv.org/abs/2305.18290)、[IPO](https://arxiv.org/abs/2310.12036)与 [SimPO](https://arxiv.org/abs/2405.14734)不是可以只换字符串而保持其他配方不变的同义目标。

## GAE

[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)区分真正 terminal 与 time-limit truncation。前者不 bootstrap；后者从该 transition 的最终 observation 对应的 `next_value[t]` bootstrap，但 trace 不能跨 episode 继续。Packed trajectory 不能用物理相邻的 `value[t + 1]` 代替真实后继状态：

```python
def gae(reward, value, next_value, terminated, truncated, gamma=0.99, lam=0.95):
    """All tensors:[T]; next_value[t] is V of transition t's true successor."""
    tensors = (value, next_value, terminated, truncated)
    if any(tensor.shape != reward.shape for tensor in tensors):
        raise ValueError("GAE transition tensors must have identical [T] shape")
    adv = torch.zeros_like(reward)
    carry = torch.zeros((), dtype=reward.dtype, device=reward.device)
    for t in range(reward.numel() - 1, -1, -1):
        bootstrap = 1.0 - terminated[t].to(reward.dtype)
        boundary = (terminated[t] | truncated[t]).to(reward.dtype)
        delta = reward[t] + gamma * bootstrap * next_value[t] - value[t]
        carry = delta + gamma * lam * (1.0 - boundary) * carry
        adv[t] = carry
    return adv, adv + value
```

```python
reward = torch.tensor([1.0, 10.0])
value = torch.tensor([2.0, 3.0])
next_value = torch.tensor([4.0, 99.0])
terminated = torch.tensor([False, True])
truncated = torch.tensor([True, False])
adv, returns = gae(reward, value, next_value, terminated, truncated, gamma=0.5, lam=1.0)
torch.testing.assert_close(adv, torch.tensor([1.0, 7.0]))
torch.testing.assert_close(returns, torch.tensor([3.0, 10.0]))
```

工具 observation token 不是策略 action；只有模型实际采样的 action token 进入 policy loss。

## PPO

[PPO](https://arxiv.org/abs/1707.06347)中 $\pi_{\mathrm{old}}$ 是采样行为策略，$\pi_{\mathrm{ref}}$ 是 KL anchor，两者作用不同：

```python
def ppo_policy_loss(new_logp, old_logp, advantage, action_mask,
                    clip=0.2, ref_logp=None):
    """All token tensors:[B,T] -> loss and detached diagnostics."""
    weight = action_mask.to(new_logp.dtype)
    if weight.sum() == 0:
        raise ValueError("trajectory has no action token")
    ratio = (new_logp - old_logp).exp()
    raw = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    loss = -(torch.minimum(raw, clipped) * weight).sum() / weight.sum()
    info = {
        "ratio_mean": (ratio * weight).sum().detach() / weight.sum(),
        "clip_fraction": (((ratio - 1).abs() > clip) * weight).sum().detach() / weight.sum(),
    }
    if ref_logp is not None:
        info["sample_log_ratio"] = ((new_logp - ref_logp) * weight).sum().detach() / weight.sum()
    return loss, info
```

```python
logp = torch.randn(2, 4)
mask = torch.tensor([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)
_, info = ppo_policy_loss(logp, logp, torch.ones_like(logp), mask, ref_logp=logp)
torch.testing.assert_close(info["ratio_mean"], torch.tensor(1.0))
torch.testing.assert_close(info["sample_log_ratio"], torch.tensor(0.0))
assert "sample_kl" not in info
```

ratio 必须由 rollout 保存的 exact old log-probability 计算。tokenizer、模板或 action boundary 改变后，旧 logp 不再兼容。

`sample_log_ratio` 估计的是行为分布 $\mu$ 所采 action token 上的

$$
\widehat{\ell}_{\mu}
=
\frac{\sum_t m_t\left[
\log\pi_{\mathrm{new}}(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)
\right]}
{\sum_t m_t},
\qquad a_t\sim\mu.
$$

在普通 PPO rollout 中 $\mu=\pi_{\mathrm{old}}$。它可以为负，也不是一般意义上的 KL；只有采样确实来自 $\pi_{\mathrm{new}}$ 且对其动作分布取期望时，才对应
$\operatorname{KL}(\pi_{\mathrm{new}}\Vert\pi_{\mathrm{ref}})$ 的 Monte Carlo 估计。

## RLOO 与 GRPO advantage

[RLOO](https://arxiv.org/abs/2402.14740)用同一 prompt 的其他样本作为 leave-one-out baseline：

$$
A_i=R_i-\frac{1}{K-1}\sum_{j\ne i}R_j.
$$

[DeepSeekMath](https://arxiv.org/abs/2402.03300)提出的 GRPO 常见标准化形式为：

$$
A_i=\frac{R_i-\bar R}{\operatorname{std}(R)+\epsilon}.
$$

```python
def group_advantage(reward, group, valid=None, kind="grpo", eps=1e-6):
    """reward/group:[N], valid:[N] -> advantage:[N], invalid stays zero."""
    valid = torch.ones_like(reward, dtype=torch.bool) if valid is None else valid
    out = torch.zeros_like(reward)
    for g in group[valid].unique():
        idx = (group == g) & valid
        r, n = reward[idx], int(idx.sum())
        if kind == "rloo":
            if n < 2:
                raise ValueError("RLOO needs at least two valid samples per group")
            out[idx] = r - (r.sum() - r) / (n - 1)
        elif kind == "grpo":
            if eps <= 0:
                raise ValueError("eps must be positive")
            std = r.std(unbiased=False)
            out[idx] = (r - r.mean()) / (std + eps)
        else:
            raise ValueError(f"unknown estimator: {kind}")
    return out
```

```python
reward = torch.tensor([1.0, 2.0, 3.0, 7.0, 7.0])
group = torch.tensor([0, 0, 0, 1, 1])
got = group_advantage(reward, group, kind="grpo", eps=1e-3)
r0 = reward[:3]
expected = (r0 - r0.mean()) / (r0.std(unbiased=False) + 1e-3)
torch.testing.assert_close(got[:3], expected)
torch.testing.assert_close(got[3:], torch.zeros(2))
```

Infra error 不应自动记为零奖励；先用 `valid` 排除并单列失败率。全同 reward 时分子为零，因此输出精确为零；`eps` 始终进入分母，公式与实现保持一致。

## V-trace

[IMPALA](https://arxiv.org/abs/1802.01561)提出的 V-trace 用于行为策略 $\mu$ 与当前策略 $\pi$ 不同的异步 rollout。令 $\rho_t=\pi(a_t)/\mu(a_t)$，V-trace 对 value 与 trace 权重截断：

```python
def vtrace(log_rho, reward, value, terminated, gamma=0.99,
           rho_bar=1.0, c_bar=1.0):
    """Token tensors:[T], value:[T+1] -> value targets, policy advantages."""
    rho = log_rho.exp()
    clipped_rho, clipped_c = rho.clamp(max=rho_bar), rho.clamp(max=c_bar)
    alive = 1.0 - terminated.to(reward.dtype)
    delta = clipped_rho * (reward + gamma * alive * value[1:] - value[:-1])
    correction = torch.zeros((), dtype=reward.dtype, device=reward.device)
    target = torch.empty_like(reward)
    for t in range(reward.numel() - 1, -1, -1):
        correction = delta[t] + gamma * alive[t] * clipped_c[t] * correction
        target[t] = value[t] + correction
    next_target = torch.cat((target[1:], value[-1:]))
    pg_adv = clipped_rho * (reward + gamma * alive * next_target - value[:-1])
    return target, pg_adv
```

当 $\rho=c=1$ 时，它退化为相应的 on-policy multi-step target。每个 action 必须保存行为策略版本和 old logp；“落后了几个 step”不能代替 importance ratio。

## 最小断言矩阵

| 类别 | 断言 |
| --- | --- |
| identity | LoRA 初始等于 base；KD 相同 logits 的 KL 为零；PPO 新旧相同时 ratio 为 1 |
| mask | padding 与 observation token 不改变 loss 或梯度 |
| degenerate | 空 mask 抛错；RLOO 的 $K<2$ 抛错；GRPO 零方差返回零 |
| invariance | BT 共同平移不变；LoRA merge/unmerge 幂等 |
| policy state | old policy、reference policy 和 current policy 分开传入 |
| missingness | timeout、invalid 与 infra error 不静默变成普通零奖励 |

目标推导见[偏好优化](../training/offline-preference.md)与[在线 RL](../training/online-rl.md)，跨 rank 的 loss 与 global norm 见[手撕：分布式与容错](distributed-systems.md)。

## Reference {#reference}

- [LoRA](https://arxiv.org/abs/2106.09685)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO](https://arxiv.org/abs/2310.12036)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740)
