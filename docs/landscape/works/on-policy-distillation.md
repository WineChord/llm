# On-Policy Distillation：在学生真正会到达的状态上学习

知识蒸馏最初像一次静态模仿：教师先给出答案，学生再拟合这些固定样本。对自回归模型，这个接口有一个逐步放大的缺口——训练时看到的是教师前缀，部署时却必须接住自己的前缀。一次小偏差会把学生带到训练集没有覆盖的状态，后续每一步都可能继续偏离。

On-Policy Distillation（OPD）改变的不是教师是谁，而是 **由谁产生训练轨迹**：学生先按当前策略生成，教师再在学生实际到达的每个前缀上给出稠密分布监督。它因此占据一个很有用的位置：

| 方法 | 轨迹来自 | 信号密度 | 主要错位 |
| --- | --- | --- | --- |
| SFT / off-policy distillation | 数据或教师 | token 级 | 学生部署时会访问别的前缀 |
| outcome RL | 学生 | 序列级 | 信用分配稀疏且方差高 |
| OPD | 学生 | token 级 | 教师未必能可靠评价所有学生状态 |

这个动机并非始于语言模型：[DAgger](https://arxiv.org/abs/1011.0686) 已经把 sequential
prediction 的核心困难表述为“训练分布与当前策略真正诱导出的状态分布不同”。语言模型中的
[MiniLLM](https://arxiv.org/abs/2306.08543) 把 reverse KL 与学生采样连接起来；
[GKD](https://arxiv.org/abs/2306.13649) 把学生自生成序列与多种分布距离放入统一接口；
[Thinking Machines Lab 的 OPD 实验](https://thinkingmachines.ai/blog/on-policy-distillation/)
进一步展示了 sampled-token 版本在推理与持续学习中的训练方式。[Kimi K3](kimi-k3.md#mopd)
把多教师信号接入 RL-style loss，[DeepSeek-V4](deepseek-v4.md#on-policy-distillation) 则选择计算
更昂贵、方差更低的全词表 reverse KL，并为十余个教师设计调度系统。这些实现属于同一谱系，
却不是同一个损失估计器。

## 两个轴必须分开：轨迹分布与目标方向 {#reverse-kl}

给定 prompt $x$，学生策略为 $\pi_\theta$，教师为 $\pi_E$。在学生生成的前缀

$$
y_{<t}\sim \pi_\theta(\cdot\mid x)
$$

上，逐位置目标可以写成

$$
\mathcal L_t
=D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid x,y_{<t})
\;\middle\|\;
\pi_E(\cdot\mid x,y_{<t})
\right).
$$

“on-policy”描述前缀从哪里来；“reverse KL”描述同一前缀上两组 next-token 分布如何比较。二者不能合并成一句“用学生 token 算 KL”：

- 学生采样决定访问哪些状态，缓解 teacher-forcing 的 state-distribution mismatch；
- 全词表 KL 在每个已访问状态比较所有 token，不只比较刚好采中的 token；
- 教师仍可能在学生的异常前缀上失去校准，on-policy 并不会自动保证监督可靠。

把序列分布的 reverse KL 按 chain rule 展开，正好得到学生访问分布上的 token KL：

$$
D_{\mathrm{KL}}\!\left(\pi_\theta(y\mid x)\middle\|\pi_E(y\mid x)\right)
=\mathbb E_{y\sim\pi_\theta}
\left[
\sum_t D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid x,y_{<t})
\middle\|
\pi_E(\cdot\mid x,y_{<t})
\right)
\right].
$$

这解释了为什么“学生 rollout + 教师逐 token 分布”不是临时工程拼接，而是序列级目标的自然分解。
严格等式还要求两侧使用同一序列空间与终止规则，并且教师在学生支持集上不为零。有限 softmax
logits 通常给每个词表项正概率；若部署了 hard top-$k$ 截断、不同 EOS 规则或强制最大长度，
则必须把截断后的序列分布重新写进目标，不能直接沿用上述 chain rule。

## 单样本估计与全词表目标不是一回事

若只对学生实际采到的 $y_t$ 计算

$$
\widehat r_t
=-\left[
\log \pi_\theta(y_t\mid s_t)
-\log \pi_E(y_t\mid s_t)
\right],
$$

就能把 $\widehat r_t$ 当作 token reward 或 advantage，复用已有 policy-gradient
基础设施。它只需教师返回 sampled-token log-prob，通信与存储便宜，但一次采样无法看见词表中
其余候选，梯度估计方差较大；stop-gradient、importance ratio 与 reduction 的选择还会改变它
究竟在估计什么。

DeepSeek-V4 不采用这种简化，而是在每个前缀显式计算

$$
\mathcal L_t
=\sum_{v\in V}p_\theta(v)
\left[\log p_\theta(v)-\log p_E(v)\right].
$$

下面的 reference 同时验证归一化、非负性和零点语义：

```python
import torch
import torch.nn.functional as F
def reverse_kl(student_logits, teacher_logits):
    log_p = F.log_softmax(student_logits, dim=-1)
    log_q = F.log_softmax(teacher_logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(-1)
s = torch.tensor([[2., 0., -1.]], requires_grad=True)
t = torch.tensor([[1., 1., -2.]])
loss = reverse_kl(s, t)
assert loss.shape == (1,) and loss.item() >= 0
torch.testing.assert_close(reverse_kl(t, t), torch.zeros(1), atol=1e-6, rtol=0)
loss.mean().backward()
assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
```

这里不能把 teacher logits `detach` 与否当作唯一关键点：教师通常本来就是冻结的。真正需要审计的是 student probability 是否仍在计算图中、mask 是否只覆盖有效 assistant token、temperature 是否同时作用于两侧、loss 是按 token 还是按 sequence 归一化。

## 多教师不是把 logits 平均一下 {#full-vocabulary-opd}

DeepSeek-V4 先分别训练数学、代码、Agent 与指令遵循等 specialist，再用超过十个教师合并到
一个学生。报告的公式 (29) 给出

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=\sum_{i=1}^{N}w_i
D_{\mathrm{KL}}\!\left(\pi_\theta\middle\|\pi_{E_i}\right). \tag{29}
$$

若 $w_i\ge 0$ 且 $\sum_iw_i=1$，在一个固定前缀上有

$$
\sum_iw_iD_{\mathrm{KL}}(p\|q_i)
=D_{\mathrm{KL}}(p\|\widetilde q)-\log Z,
\qquad
\widetilde q(v)=\frac{\prod_iq_i(v)^{w_i}}{Z}.
$$

所以加权 reverse KL 对应教师分布的 **归一化几何混合**，不是概率的算术平均。一个教师把某个 token 的概率压得极低时，几何混合会强烈抑制它；如果不同教师适用于不同样本，更合理的接口通常是按 domain、task 或路由结果选择权重，而不是让所有专家在所有状态上等权“投票”。

<details>
<summary>展开：多教师 full-vocabulary reverse KL reference</summary>

```python
import torch
import torch.nn.functional as F
def reverse_kl(student_logits, teacher_logits):
    log_p = F.log_softmax(student_logits, -1)
    log_q = F.log_softmax(teacher_logits, -1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)
def multi_teacher_rkl(s_logits, teachers, weights):
    log_p = F.log_softmax(s_logits, -1)
    log_q = torch.stack([F.log_softmax(x, -1) for x in teachers])
    w = torch.as_tensor(weights, dtype=log_p.dtype, device=log_p.device)
    assert w.ndim == 1 and len(teachers) == w.numel() and (w >= 0).all() and w.sum() > 0
    w = w / w.sum()
    mean_log_q = (w[:, None, None] * log_q).sum(0)
    return (log_p.exp() * (log_p - mean_log_q)).sum(-1), mean_log_q
s = torch.tensor([[.2, -.1, 1.]], requires_grad=True)
teachers = [torch.tensor([[2., 0., -1.]]), torch.tensor([[0., 1., -1.]])]
loss, mean_log_q = multi_teacher_rkl(s, teachers, [.75, .25])
weighted = .75 * reverse_kl(s, teachers[0]) + .25 * reverse_kl(s, teachers[1])
torch.testing.assert_close(loss, weighted)
log_z = torch.logsumexp(mean_log_q, -1)
log_mix = mean_log_q - log_z[:, None]
mixed = (s.log_softmax(-1).exp() * (s.log_softmax(-1) - log_mix)).sum(-1)
torch.testing.assert_close(loss, mixed - log_z)
loss.mean().backward()
assert torch.isfinite(s.grad).all()
```

</details>

这个等式也给出一个诊断：如果 teacher routing 错了，full-vocabulary 只会让错误监督更稠密，不会把错误教师变正确。

## 为什么全词表版本首先是系统问题 {#teacher-scheduling}

设一个 mini-batch 中实际送入教师的 token–teacher pair 数为 $M$，词表大小为 $|V|$，教师
末层宽度为 $d_E$。若每个 token 都评估 $N$ 个教师，则 $M=NT$；若按任务为每条样本路由一个
教师，则通常 $M\approx T$。显式保存所有 teacher logits 需要 $O(M|V|)$ 空间；当
$|V|>10^5$、轨迹很长、教师达到万亿参数量级时，问题不是多做一个 softmax，而是 logits
根本不适合跨阶段长期存放。

DeepSeek-V4 的方案把计算顺序改成：

1. 教师权重放在集中式分布存储，需要时以 ZeRO-like shard 加载；
2. teacher forward 后只缓存最后一层 hidden state，而非 $M\times|V|$ logits；
3. 按教师重新排列样本，一次装入一个 teacher head；
4. 由 hidden state 临时重建 logits，立即累计 KL，再释放大词表张量；
5. teacher loading、head 计算和 student training 异步重叠。

缓存规模因而从 $O(M|V|)$ 降到 $O(Md_E)$；只有在每个 token 只路由到一个教师时，才可进一步
写成约 $O(Td_E)$。代价是之后必须调入对应 prediction head，临时重建 full-vocabulary logits
并立即归约。它本质上是一次 loop interchange：把“各教师的全词表输出长期物化”改成“按
teacher index 聚拢样本、一次只驻留一个 head、按需重建并流式消费”。

数值实现仍有三条硬边界：

- 用 `log_softmax` 或 log-sum-exp 计算，不能先构造低精度概率再取对数；
- vocabulary parallel 下，归一化常数与 $\sum_v p_v(\log p_v-\log q_v)$ 都需要跨 shard
  的正确归约；
- padding、tool observation、prompt 和 assistant token 的 mask 必须与 rollout 契约一致，否则“稠密监督”会训练到不该预测的位置。

## OPD 与 RL、SFT、权重合并的关系

### 与 SFT

SFT 的 teacher-forced cross-entropy 相当于在固定数据前缀上逼近目标 token。它擅长把教师支持集之外的新知识或格式先放进学生的可达区域；OPD 更擅长修正学生已经会访问的状态。实践中先做 domain mid-training / SFT，再做 OPD，不只是经验顺序：如果学生几乎永远到不了目标行为附近，reverse KL 的 mode-seeking 倾向也无法凭空创造那条轨迹。

### 与 outcome RL

二者都可从学生 rollout 开始，但反馈带宽不同。outcome reward 告诉整段结果“成或败”，OPD 在每个前缀比较 next-token 分布。前者能超越固定教师、探索教师未覆盖的策略；后者信号稠密，却以教师能力上限和 teacher inference 成本为代价。它们可以串联或混合，不能据一次 benchmark 就互相替代。

### 与参数平均

参数平均要求模型处于足够相容的权重盆地，也无法按输入动态选择专家。多教师 OPD 在行为分布层合并能力，避免直接对齐参数坐标；但它仍可能发生 capability interference，教师权重、样本路由和学生容量都是待测变量。

## 从 MiniLLM 到 V4：连续性与分叉

| 工作 | 学生访问自己的状态 | 监督粒度 | 关键用途 |
| --- | --- | --- | --- |
| MiniLLM | 是 | reverse-KL 的 on-policy 优化 | 小模型生成蒸馏与 exposure bias |
| Thinking Machines OPD | 是 | sampled-token reverse-KL reward | 推理、个性化与持续学习实验 |
| Kimi K3 MOPD | 是 | 多教师 token signal 接入 policy loss | 多领域、多 effort expert 合并 |
| DeepSeek-V4 OPD | 是 | 多教师 full-vocabulary reverse KL | 十余 specialist 合并，强调低方差与调度 |

V4 报告清楚描述了目标与系统，但没有公开所有 teacher 权重、domain routing 规则、每阶段
token 预算、温度、权重 $w_i$ 或完整消融。因此可以复现目标语义，不能据此声称复现最终训练
配方。

## 实验时应记录什么

- rollout checkpoint 与 learner checkpoint 是否相同，允许多大 policy lag；
- teacher 在学生低概率、乱码或工具错误前缀上的 entropy 与校准；
- full-vocabulary、top-$k$ 截断和 sampled-token estimator 的 wall time / 方差对照；
- 每个 domain 的 teacher 路由覆盖率与冲突样本；
- token normalization、序列长度分布和长回答是否被隐式加权；
- held-out 能力的回归，而不只看被蒸馏任务；
- teacher forward、权重换入、head 重建和通信各自的利用率。

OPD 的核心优势可以浓缩为一句话：在学生会犯错的地方给出比终局奖励更稠密的方向。但它的完整工程语义还包括“谁产生状态、哪个教师负责、比较整个词表还是一个样本、以及怎样让教师计算不淹没训练”。

## Reference {#reference}

- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4 官方模型卡与训练、评测口径](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)
- [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)
- [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686)
- [Kimi K3 官方技术报告与实现入口](https://github.com/MoonshotAI/Kimi-K3)
