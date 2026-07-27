# Mixture of Experts

稀疏 Mixture of Experts 将模型总容量与每个 token 的激活计算部分解耦。真正的设计对象不是“有多少专家”，而是路由、容量、通信、负载和学习信号组成的整体系统。

从 early expert mixtures、Sparsely-Gated MoE 到 Switch 与 DeepSeekMoE 的变化见[条件计算与稀疏专家](../landscape/lineages/conditional-compute.md)；关键路由与 capacity 语义的可执行缩影见[稀疏 MoE 深读](../landscape/works/sparse-moe.md)。

## 路由

设 token 表示为 $x$，router logits 为

$$
s=W_rx.
$$

经过 top-$k$ 选择得到集合 $\mathcal T(x)$，MoE 输出可写为

$$
y=\sum_{i\in\mathcal T(x)}p_i(x)E_i(x),
\qquad
p_i=\frac{\exp(s_i)}{\sum_{j\in\mathcal T(x)}\exp(s_j)}.
$$

top-$k$ 是离散选择；未被选中的专家通常不接收主任务梯度。router 的初始化、噪声、精度和 tie-breaking 都会影响早期专家分化。

## 细粒度与共享专家

将一个大 FFN 拆成更多小专家，可以用更丰富的组合表达相近激活参数量；代价是路由元数据、kernel 粒度和通信消息更碎。共享专家对所有 token 激活，承担通用变换；路由专家则学习条件化分工。[DeepSeekMoE](https://arxiv.org/abs/2401.06066)是细粒度与共享专家组合的代表。

## 容量与溢出

若 batch 中共有 $N$ 个 token、$E$ 个专家、每 token 选择 $k$ 个专家，平均负载为

$$
\bar n=\frac{kN}{E}.
$$

训练实现常为每个专家设置 capacity

$$
C=\left\lceil c\bar n\right\rceil,
$$

其中 $c$ 是 capacity factor。负载超过容量时，可以丢 token、路由到备选专家、增加 padding 或使用无固定容量的动态 kernel。每种策略都改变质量、内存和最坏时延。

### 最小语义实现 {#top-k-routing-with-capacity}

`sparse_moe` 接收扁平 token `x:[N,D]`、router weight `[E,D]` 和 expert weights `[E,D,D]`，返回加权输出、实际专家负载、top-$k$ 索引与被保留的 assignment。capacity 的优先级先看 routing rank：全体 token 的 top-1 先于任何 top-2；同一 rank 内再按 token 输入顺序保留。

```python
import torch
import torch.nn.functional as F
def sparse_moe(x, router, experts, top_k=2, capacity=None):
    if x.ndim != 2 or router.ndim != 2 or experts.ndim != 3: raise ValueError("invalid ranks")
    tokens, hidden = x.shape
    expert_count = experts.shape[0]
    if router.shape != (expert_count, hidden): raise ValueError("router/expert mismatch")
    if experts.shape[1:] != (hidden, hidden): raise ValueError("expert shape mismatch")
    if not isinstance(top_k, int) or not 1 <= top_k <= expert_count:
        raise ValueError("invalid top_k")
    if capacity is not None and (not isinstance(capacity, int) or capacity <= 0):
        raise ValueError("capacity must be a positive integer")
    score, index = F.linear(x, router).float().topk(top_k, dim=-1)
    gate = score.softmax(-1).to(x.dtype)
    output, load = torch.zeros_like(x), torch.zeros(expert_count, dtype=torch.long)
    kept = torch.zeros_like(index, dtype=torch.bool)
    for slot in range(top_k):
        for expert_id, expert in enumerate(experts):
            token = torch.where(index[:, slot] == expert_id)[0]
            if capacity is not None:
                token = token[:max(capacity - load[expert_id].item(), 0)]
            if token.numel():
                output[token] += gate[token, slot, None] * F.linear(x[token], expert)
                load[expert_id] += token.numel()
                kept[token, slot] = True
    return output, load, index, kept
x = torch.tensor([[3., 0.], [2., 0.], [0., 3.]])
router, experts = torch.eye(2), torch.eye(2).repeat(2, 1, 1)
_, load, _, kept = sparse_moe(x, router, experts, top_k=2, capacity=1)
assert kept.tolist() == [[True, False], [False, False], [True, False]]
assert load.tolist() == [1, 1]
try: sparse_moe(x, torch.randn(3, 2), experts, top_k=2)
except ValueError: pass
else: raise AssertionError("router/expert mismatch must fail")
```

`kept` 的断言锁定了 top-1 优先于 top-2、同 slot 按 token 顺序的容量语义。丢弃后不重归一化残余 gate；备选路由、其他 token-dropping 优先级和稳定 tie-breaking 都会改变结果。生产实现还要用 permutation 与 all-to-all 取代 Python 循环，并核对反向、跨 rank 容量和负载统计；具体张量变换见[分布式与容错：MoE dispatch 与 combine](../practice/distributed-systems.md#moe-dispatch-combine)。

## 负载均衡

只优化语言建模损失时，router 可能把大量 token 送到少数专家。经典辅助项同时考察路由概率和实际分配，鼓励专家接收接近均匀的负载；但辅助梯度也可能干扰主目标。

[Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)在 top-$k$ 前为每个专家增加动态 bias，根据近期负载更新 bias，而不让均衡信号直接反传到主 router score。它减少一种干扰来源，却没有消除容量、局部拥塞和跨节点流量问题。

均衡至少有三种口径：

- 全局 token 数是否均匀；
- 每个 expert-parallel group 内是否均匀；
- 实际 wall-clock 是否均匀。

token 数相同的专家也可能因序列形状、kernel 或节点拥塞产生不同耗时。

## Expert Parallel

典型前向包含：

```text
local tokens
-> router and permutation
-> all-to-all dispatch
-> local expert GEMM
-> all-to-all combine
-> inverse permutation
```

通信量与 token 表示宽度、top-$k$、dtype 和跨节点比例相关。专家太小会使 GEMM 无法饱和，专家太大又限制分片与负载弹性。实际系统常将 tensor parallel、data parallel 与 expert parallel 组合，并尝试把 dispatch/combine 与其他计算重叠。

## 训练稳定性

- router logits 使用过低精度可能放大 top-$k$ 边界抖动；
- 高学习率会让专家分工快速漂移；
- 某些 token 类别可能长期绑定少数专家，形成过拟合；
- batch 太小或数据源分布剧变，会让负载统计噪声很大；
- shared expert 与 routed expert 的尺度不一致，会改变残差分布；
- checkpoint 恢复后若路由随机性或数据游标改变，专家负载可能突然跳变。

应监控每专家 token 数、路由概率、drop rate、容量利用率、专家输出范数、跨节点流量和最慢专家时延，而不只看总 loss。

## 推理

MoE decode 的激活 FLOPs 可以较低，但全部专家权重仍需驻留、分片或按需加载。小 batch 下每个专家收到的 token 很少，GEMM 利用率下降；大 batch 又增加 KV 和尾延迟。量化、专家缓存与 speculative routing 都要在真实路由分布上评测。

## 比较表

报告 MoE 至少包含：

```text
total and activated parameters
number and size of routed/shared experts
top-k and router normalization
capacity and overflow policy
load-balancing objective or bias
expert-parallel topology
communication dtype and overlap
training token distribution
quality, throughput, tail latency and memory
```

[Switch Transformer](https://arxiv.org/abs/2101.03961)展示了 top-1 稀疏路由，[DeepSeek-V3](https://arxiv.org/abs/2412.19437)则把细粒度专家、无辅助损失均衡和系统配方结合。替代序列架构见[稀疏与替代架构](moe-alternatives.md)，分布式实现见[模型并行](../systems/model-parallelism.md)。

## Reference {#reference}

- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
