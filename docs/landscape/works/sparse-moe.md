# Sparse MoE：只激活少数专家以后

Mixture of Experts 的表面承诺很直接：模型可以拥有许多专家，但每个 token 只经过少数几个，于是容量增长快于激活计算。真正决定这项承诺能否成立的，是 [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)、[GShard](https://arxiv.org/abs/2006.16668) 与 [Switch Transformer](https://arxiv.org/abs/2101.03961) 逐步暴露出的路由、容量和通信问题。Sparse MoE 不是一个单独 layer，而是一份从概率分配一直延伸到集群网络的契约。

## 从加权混合到条件执行

早期 adaptive mixture 令 gating network 输出所有专家权重：

$$
y=\sum_{e=1}^E g_e(x)E_e(x),
\qquad g=\operatorname{softmax}(W_rx).
$$

若所有 $E_e$ 都执行，专家可以专业化，却没有节省主要矩阵乘。下面先采用一种常见的 selected-expert renormalization convention：稀疏门控保留 top-$k$，并在所选专家内重新归一化：

$$
\tilde g_e=
\frac{g_e\mathbf 1[e\in\operatorname{TopK}(g)]}
{\sum_{j\in\operatorname{TopK}(g)}g_j},
\qquad
y=\sum_e\tilde g_eE_e(x).
$$

这项归一化固定了本页 reference 的 combine 语义，却不是所有 MoE 的普遍规则。另一类实现保留所选专家的原始 gate scale；例如 Switch top-1 会用选中专家的 router probability 缩放输出，而不是把唯一权重改成 1。比较实现时必须把这一选择与 residual 分支尺度一并记录。

## Capacity 是执行语义，不是附注

每个专家在一个 batch 中可处理的 token slot 常写成

$$
C_e=\left\lceil
\frac{\text{capacity factor}\times Nk}{E}
\right\rceil,
$$

其中 $N$ 是 token 数。超过容量的 route 必须明确丢弃、送共享专家、重新路由或增加 buffer。下面的 reference 保留 top-$k$ 重归一化、容量与 overflow：

```python
import math
import torch
def sparse_routes(logits, k=2, capacity_factor=1.0):
    prob = logits.softmax(-1)
    score, expert = prob.topk(k, dim=-1)
    score = score / score.sum(-1, keepdim=True)
    n, experts = logits.shape
    capacity = math.ceil(capacity_factor * n * k / experts)
    keep = torch.zeros_like(score, dtype=torch.bool)
    load = torch.zeros(experts, dtype=torch.long)
    for token in range(n):
        for slot in range(k):
            e = int(expert[token, slot])
            if load[e] < capacity:
                keep[token, slot] = True
                load[e] += 1
    kept_score = score * keep
    denom = kept_score.sum(-1, keepdim=True)
    kept_score = torch.where(denom > 0, kept_score / denom, kept_score)
    route_fraction = torch.bincount(expert.reshape(-1), minlength=experts)
    route_fraction = route_fraction.to(prob.dtype) / (n * k)
    aux = experts * (route_fraction * prob.mean(0)).sum()
    return expert, kept_score, keep, load, aux
logits = torch.tensor([[5., 1., 0.], [4., 2., 0.], [3., 2., 0.], [0., 1., 5.]])
expert, weight, keep, load, aux = sparse_routes(logits, capacity_factor=.75)
assert torch.all(load <= 2) and (~keep).any()
row_sum = weight.sum(-1)
assert torch.allclose(row_sum[row_sum > 0], torch.ones_like(row_sum[row_sum > 0]))
assert aux.ndim == 0 and torch.isfinite(aux)
```

循环实现是语义 reference，不是高性能 dispatch。生产实现会排序或直方图分桶 token，构造 permutation，执行 grouped GEMM，再反向 combine。

## 为什么 router 会坍缩

主任务梯度可能偏爱少数当前较强专家，强专家收到更多 token 后又学得更快，形成正反馈。Switch-style 辅助均衡常用离散 route fraction $f_e$ 与平均 router probability $p_e$：

$$
\mathcal L_{\mathrm{aux}}
=\alpha E\sum_e f_ep_e.
$$

在 Switch top-1 中，$f_e$ 是分给专家 $e$ 的 token 比例；本页 top-$k$ reference 将它推广为 $Nk$ 个预容量 route 中该专家所占的比例，因此代码的 dispatch 与辅助项使用同一个 $k$。$p_e$ 提供可微信号。这类辅助项不能保证每个微批、每台机器和每条网络链路都均衡；只看全局平均会隐藏瞬时 hot expert。

router z-loss、logit clipping、noise、expert dropout 与初始化也会影响稳定性。调节均衡系数时必须同时看主任务 loss、token drop、expert load 和路由熵，而不是只追求均匀直方图。

## GShard：专家路由进入集群

GShard 将稀疏专家与分片编译结合。expert parallel 通常需要两次 all-to-all：

```text
original token order
  -> dispatch by expert
  -> all-to-all to expert owners
  -> grouped expert compute
  -> all-to-all back
  -> restore token order
```

理论激活 FLOPs 不包含路由元数据、padding capacity、跨机字节和同步等待。expert placement 若跨越慢链路，小消息 all-to-all 会让算力空闲。系统机制见 [MoE 系统](../../systems/moe-systems.md)。

## Switch：Top-1 的取舍

Switch 采用 top-1，使每个 token 只经过一个专家，并以对应 router probability 缩放专家输出，简化 dispatch 与 combine。它减少计算和通信，不提供多个专家输出的组合；在其他数据、容量和硬件下，top-2 仍可能带来质量或稳定性收益。

Switch 还突出 bfloat16、router 精度、初始化和 selective precision 等稳定性细节。MoE 规模扩张不是只把 `num_experts` 调大：router logits 的数值、expert dropout 和容量策略会共同改变有效训练数据。

## 细粒度与共享专家

[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 把专家拆得更细，并设置始终激活的共享专家。共享分支承担通用变换，路由分支学习条件化专业化；更细粒度提供更多组合，也让 grouped GEMM、路由元数据和通信消息更碎。

这类设计要区分：

- 总参数与每 token 激活参数；
- shared expert FLOPs 与 routed expert FLOPs；
- 理论路由组合数与真实专家专业化；
- 训练平均负载与在线 decode 尾延迟。

## Reference {#reference}

- [Sparse MoE 论文](https://arxiv.org/abs/1701.06538)；
- [GShard 论文](https://arxiv.org/abs/2006.16668)；
- [Switch Transformer 论文](https://arxiv.org/abs/2101.03961)与 [Mesh TensorFlow MoE 实现](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)；
- [DeepSeekMoE 论文](https://arxiv.org/abs/2401.06066)与 [deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)。

代码仓库对应具体软件世代，不保证一条命令重建论文全部数据、硬件和训练运行。完整历史位置见[容量与激活计算怎样分开](../lineages/conditional-compute.md)，架构定义见 [Mixture of Experts](../../architecture/moe.md)，紧凑 dispatch 实现见[分布式与容错](../../practice/distributed-systems.md)。
