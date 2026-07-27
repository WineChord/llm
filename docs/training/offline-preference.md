# 离线偏好优化

离线偏好优化只使用已经收集的回答和偏好，不在训练环中向当前策略采样。它容易重放、部署简单，却只能重新分配数据支持集中的概率；若高质量行为从未出现在候选中，离线目标无法凭空探索出来。

[DPO 深读](../landscape/works/dpo.md)把 KL 正则化、Bradley–Terry 偏好似然和 reference policy 的关系放回原论文推导，并给出带 response mask 的紧凑实现；更长的算法脉络见[后训练与对齐](../landscape/lineages/training-alignment.md)。

## 从 KL 正则化到最优策略

对固定 prompt $x$，考虑

$$
\max_\pi
\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y
\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)},
$$

并满足 $\sum_y\pi(y\mid x)=1$。加入 Lagrange multiplier，令关于 $\pi(y\mid x)$ 的导数为零：

$$
r(x,y)
-\beta\left(
\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+1
\right)
+\lambda=0.
$$

整理得

$$
\pi^*(y\mid x)
=
\frac{1}{Z(x)}
\pi_{\text{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
$$

其中 $Z(x)$ 只依赖 prompt。因此隐式 reward 可写为

$$
r(x,y)
=
\beta\log\frac{\pi^*(y\mid x)}
{\pi_{\text{ref}}(y\mid x)}
+\beta\log Z(x).
$$

对同一 prompt 的 chosen/rejected 做差，$\log Z(x)$ 抵消。这一步把 Bradley–Terry 偏好与 policy/reference log-ratio 连接起来。

## DPO

定义

$$
h_\theta(x,y_w,y_l)
=
\left[
\log\pi_\theta(y_w\mid x)
-\log\pi_{\text{ref}}(y_w\mid x)
\right]
-
\left[
\log\pi_\theta(y_l\mid x)
-\log\pi_{\text{ref}}(y_l\mid x)
\right].
$$

[Direct Preference Optimization](https://arxiv.org/abs/2305.18290) 使用

$$
\mathcal L_{\text{DPO}}
=-\log\sigma\left(\beta h_\theta\right).
$$

这个推导依赖：

- Bradley–Terry 风格相对偏好；
- KL 正则化的最优策略关系；
- policy 与 reference 对回答有可比较 support；
- chosen 与 rejected 条件在同一 prompt 和模板上；
- 离线 pair 足以覆盖希望提高的行为。

DPO 避免显式 reward model 和在线 rollout，不等于没有奖励假设，也不自动解决标签噪声、分布外回答或 benchmark 污染。

### DPO 的可执行语义 {#dpo-semantic-reference}

下面的 reference 接收已经与目标 token 对齐的 logits、token ID 和 response mask，先得到每条回答的序列 log-probability，再计算逐 pair 的 DPO loss。输出保留 batch 维，便于在外层选择样本权重和全局归约；`normalize=True` 明确切换到 token mean，不能在实验中静默改变。

```python
import torch
import torch.nn.functional as F

def response_logp(logits, tokens, response_mask, normalize=False):
    if logits.ndim < 2 or logits.shape[:-1] != tokens.shape or tokens.shape != response_mask.shape:
        raise ValueError("logits, tokens and response mask must share batch and time axes")
    valid = response_mask.bool()
    if torch.any(valid.sum(-1) == 0): raise ValueError("every response needs a target token")
    row_logits = logits.reshape(-1, logits.size(-2), logits.size(-1))
    row_tokens = tokens.reshape(-1, tokens.size(-1))
    row_valid = valid.reshape(-1, valid.size(-1))
    output = []
    for current_logits, current_tokens, current_valid in zip(
            row_logits, row_tokens, row_valid):
        selected_logits = current_logits[current_valid]
        selected_tokens = current_tokens[current_valid]
        logp = F.log_softmax(selected_logits, -1)
        value = logp.gather(-1, selected_tokens[:, None]).sum()
        output.append(value / current_valid.sum() if normalize else value)
    return torch.stack(output).reshape(tokens.shape[:-1])

def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta):
    margin = (pi_w - ref_w) - (pi_l - ref_l)
    return F.softplus(-beta * margin)

z = torch.tensor([[[float("nan")] * 2, [0., 2.], [2., 0.]]])
y, m = torch.tensor([[-100, 1, 0]]), torch.tensor([[False, True, True]])
base = response_logp(z, y, m)
assert torch.isfinite(base).all()
neutral = dpo_loss(base, base, base, base, beta=0.1)
assert torch.allclose(neutral, torch.full_like(neutral, torch.log(torch.tensor(2.))))
assert torch.all(dpo_loss(base + 1, base, base, base, 0.1) < neutral)
rejected = False
try:
    response_logp(z, y, torch.zeros_like(m), normalize=True)
except ValueError:
    rejected = True
assert rejected
```

核心不变量是 prompt 与 padding 在 `log_softmax` 之前被排除，masked `-100` 和 NaN 均不参与目标；每条回答若没有有效 token 则拒绝。policy/reference 还必须采用同一 token space，且 chosen margin 增大时 loss 下降。这里没有实现 chat template、label shift、分布式加权或 reference forward；生产训练必须对四组 log-probability 使用完全相同的序列语义。

## 序列 log-probability

回答概率为

$$
\log\pi(y\mid x)
=\sum_{t=1}^{|y|}
\log\pi(y_t\mid x,y_{<t}).
$$

总和、token mean 或其他长度校正不是等价实现：

- **sum** 对每个 token 累积 log-ratio，长回答的幅度通常更大；
- **mean** 改变目标，使每个回答近似等权，但也改变原始序列概率语义；
- **外加长度项** 把长度偏好显式建模，便于审计。

选择必须作为目标函数的一部分记录。prompt、padding、tool observation 和非策略 token 不得混入回答 log-prob。

## 相关目标

### IPO

[A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) 从更一般的偏好优化框架分析 RLHF 与 DPO，并提出 Identity Preference Optimization。常见参数化下可把目标写成让 $h_\theta$ 接近固定 margin 的平方损失；具体 margin 与温度 convention 绑定，不能只复制公式而混用另一实现的 $\beta$。

IPO 的意义不是“DPO 的无 sigmoid 版本”，而是改变了隐式偏好变换和大 margin pair 的梯度行为。

### KTO

[KTO](https://arxiv.org/abs/2402.01306) 接受单样本 desirable/undesirable 标签，并相对 reference 构造效用。它适合没有天然 pair 的反馈，但需要处理正负类别不平衡、reference 基准和标签语义；将独立好坏标签随意配对不一定更合理。

### SimPO

[SimPO](https://arxiv.org/abs/2405.14734) 去掉显式 reference，并使用平均 log-probability 与目标 margin。其核心形式可写成

$$
\mathcal L_{\text{SimPO}}
=
-\log\sigma\left[
\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)
-\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)
-\gamma
\right].
$$

reference-free 和 length-normalized 是新的建模选择，不是免费简化；它们改变锚点与长度归纳偏置，必须与 DPO 在相同数据和调参预算下比较。

三者可以共用 chosen/rejected 的序列级输入，但不能共用未声明的 reduction：DPO 与 IPO 的 `pi_*`、`ref_*` 通常是 response token log-probability 之和；SimPO 的 `pi_*` 在进入函数前已经按各自 response 长度取 mean。

```python
import torch
import torch.nn.functional as F
def preference_objective(pi_chosen, pi_rejected, ref_chosen, ref_rejected,
                         kind, beta=.1, margin=0.):
    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_gap = pi_chosen - pi_rejected
    if kind == "simpo":
        return F.softplus(-(beta * policy_gap - margin))
    if ref_chosen is None or ref_rejected is None:
        raise ValueError(f"{kind} requires a reference policy")
    relative_gap = policy_gap - (ref_chosen - ref_rejected)
    if kind == "dpo":
        return F.softplus(-beta * relative_gap)
    if kind == "ipo":
        return (relative_gap - 1 / (2 * beta)).square()
    raise ValueError(kind)
zero = torch.tensor([0.])
neutral = preference_objective(zero, zero, zero, zero, "dpo", beta=.5)
better = preference_objective(torch.tensor([1.]), zero, zero, zero, "dpo", beta=.5)
assert better < neutral
ipo = preference_objective(torch.tensor([1.]), zero, zero, zero, "ipo", beta=.5)
torch.testing.assert_close(ipo, zero)
simpo = preference_objective(torch.tensor([2.]), zero, None, None, "simpo", beta=.5, margin=.3)
torch.testing.assert_close(simpo, F.softplus(torch.tensor([-.7])))
```

DPO 的 logistic loss 会持续奖励更大 relative gap；IPO 在 convention 对应的有限 target 处最小；SimPO 去掉 reference，并显式引入长度归一与 margin。函数返回逐 pair loss，样本权重、ties 和分布式分母留在外层。三种目标与 reduction 的完整小张量实验见[训练目标实现](../practice/training-objectives.md#dpo-ipo-simpo)。

## 数据与版本契约

```text
prompt, pair/list IDs and grouped split
chosen / rejected source and generator policy
preference protocol, ties and disagreement
policy and reference exact revisions
tokenizer, chat template and truncation
response mask and sum/mean normalization
beta / margin and implementation convention
data filtering and contamination audit
```

reference 若使用不同 tokenizer 或模板，log-ratio 不再位于同一 action space。chosen/rejected 被不同截断规则处理，也会制造伪偏好。

## 正确性与失效

- **把 $\pi_{\text{ref}}$ 当可省略常数**：它随回答变化，定义偏离坐标。
- **policy/reference 模板不同**：相同文本对应不同 token 条件分布。
- **sum 与 mean 静默切换**：长度偏置和梯度尺度改变。
- **pair 随机切分**：同一 prompt 或候选泄漏到测试。
- **只保留大 margin pair**：可能偏向容易的风格差异，降低信息量。
- **离线数据来自单一旧策略**：新 policy 进入未覆盖区域后无监督。
- **标签强度被忽略**：tie 和高分歧 pair 被当成确定偏好。
- **reference-free 被写成无锚点**：数据分布和 margin 仍是隐式锚点。
- **只报 judge win rate**：长度、格式和 judge 偏好可能共同抬分。

## 何时不应使用离线偏好

当任务需要发现数据中没有的新解、当前策略已远离 pair 生成策略、可执行 reward 能低成本在线获得，或偏好主要针对多步环境终态时，应考虑在线 RL、搜索或重新收集当前策略数据。若只有可靠示范而没有有意义的相对判断，SFT 更直接。

## 验证

1. 用手算 log-prob 检查 DPO logit 的 policy/reference/chosen/rejected 符号。
2. policy 等于 reference 时，$h_\theta=0$。
3. prompt 与 padding token 对 response log-prob 没有贡献。
4. 同时报 sum/mean log-ratio、回答长度和每个 slice 的 margin。
5. 按 prompt、生成器、长度、语言和分歧程度分层评测。
6. 比较 SFT、DPO、IPO/KTO/SimPO 与不训练 baseline，固定数据和调参预算。
7. 在当前 policy 新采样回答上重新收集小规模人工偏好，检查离线外推。
8. 用事实性、可执行成功、安全、校准和多样性验证 judge 胜率。

奖励数据与偏差见[奖励建模](reward-modeling.md)，需要在线探索时见[在线 RL](online-rl.md)，目标函数的最小对照见[训练目标实现](../practice/training-objectives.md)。

## Reference {#reference}

- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO](https://arxiv.org/abs/2310.12036)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
