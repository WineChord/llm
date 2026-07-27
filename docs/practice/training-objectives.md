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

## 在线策略优化入口

GAE、PPO、RLOO 与 GRPO 不再在本页维护第二套实现。它们共同依赖 prompt group、action mask、terminal/truncation、old/behavior/reference policy 和 loss reduction；拆开复制会让这些接口在不同页面逐渐漂移。

统一的张量实现、退化断言与方法变体见[手撕 LLM 策略优化](llm-policy-optimization.md)。概念推导分别见 [Advantage 估计与 GAE](../reinforcement-learning/advantage-estimation-gae.md)、[PPO](../reinforcement-learning/trust-region-ppo.md)与 [GRPO](../reinforcement-learning/grpo.md)。本页继续保留 V-trace，因为它展示的是异步 value target 与 off-policy trace correction，而不是另一种 LLM policy-loss 配方。

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
| identity | LoRA 初始等于 base；KD 相同 logits 的 KL 为零；V-trace 在单位 ratio 下退化为 on-policy target |
| mask | padding 不改变 token loss、偏好目标或梯度 |
| degenerate | 空 loss mask 抛错；缺失 reference 的 DPO/IPO 拒绝执行 |
| invariance | BT 共同平移不变；LoRA merge/unmerge 幂等 |
| policy state | preference reference 与 V-trace behavior/current log-ratio 分开传入 |
| missingness | timeout、invalid 与 infra error 不静默变成普通零奖励 |

目标推导见[偏好优化](../training/offline-preference.md)与[在线 RL](../training/online-rl.md)，跨 rank 的 loss 与 global norm 见[手撕：分布式与容错](distributed-systems.md)。

## Reference {#reference}

- [LoRA](https://arxiv.org/abs/2106.09685)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO](https://arxiv.org/abs/2310.12036)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
