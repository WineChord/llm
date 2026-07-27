# 容量与激活计算怎样分开

稠密网络有一个直接但昂贵的规律：增加参数，几乎总会增加每个 token 的计算。Mixture of Experts 的长期目标，是让模型拥有更大的总容量，却只为当前输入激活一小部分参数。这个想法看似只是“多放几个 FFN”，真正困难的历史却围绕路由、负载、通信与稳定性展开——省下的矩阵乘若转化成拥塞和空闲设备，条件计算就只在纸面上稀疏。

## 早期混合专家：先学会分工

[Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/mixex.html)在 1991 年已经使用 gating network 为不同样本分配专家。早期目标主要是让局部模型专业化，并不一定跳过未选专家的计算。它建立了两个一直延续的对象：

$$
g(x)=\operatorname{softmax}(W_rx),\qquad
y=\sum_e g_e(x)E_e(x).
$$

gating 决定“谁负责”，expert 决定“怎样变换”。若所有专家都计算，容量增加仍会带来近似线性的算力增长。

## 稀疏门控：只执行 Top-K

[Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)把 noisy top-$k$ gating 放进超大模型，只执行少数专家：

$$
y=\sum_{e\in\operatorname{TopK}(g(x))}
\tilde g_e(x)E_e(x).
$$

总参数与激活参数由此部分解耦。新的问题立即出现：router 可能把大量 token 送给少数专家，其他专家得不到训练；设备上的容量有限，溢出 token 需要丢弃、回退或重新路由；专家跨设备时还要进行 all-to-all。

这说明 MoE 的“稀疏”至少有三层：

- 数学上只激活 top-$k$ 专家；
- kernel 上不执行未选专家；
- 集群上路由与通信成本没有吞掉稀疏收益。

完整路由实现和 overflow 语义见[Sparse MoE 深读](../works/sparse-moe.md)。

## GShard：路由问题变成分布式编译问题

[GShard](https://arxiv.org/abs/2006.16668)把稀疏专家与自动分片结合，使超大多语言模型能够跨大量设备训练。此时一个 token 的路径不再只是函数选择，而是数据移动：

```text
token -> router -> dispatch permutation -> all-to-all
      -> local expert -> all-to-all -> combine
```

容量因子、expert placement 与网络拓扑共同决定实际吞吐。相同 top-$k$ 和参数量，在 NVLink 域内、机间网络或不均匀流量下可能表现完全不同。MoE 从架构技巧变成模型—系统协同设计。

## Switch：用 Top-1 换取简单性

[Switch Transformer](https://arxiv.org/abs/2101.03961)选择 top-1 路由，减少每 token 专家计算和 combine 复杂度，并研究大规模训练的稳定性。它不证明 top-1 普遍优于 top-2；它展示的是在特定规模和系统约束下，牺牲组合表达可以换取更简单的路由路径。

负载均衡常使用 token fraction $f_e$ 与平均 router probability $p_e$：

$$
\mathcal L_{\text{aux}}
=\alpha E\sum_{e=1}^{E}f_ep_e.
$$

$f_e$ 包含不可微的 top-$k$ 分配，$p_e$ 提供可微信号。辅助目标太弱会坍缩，太强又可能干扰主任务；它解决的是统计负载，不保证每个通信链路和微批都均衡。

## 专业化粒度继续变化

[ST-MoE](https://arxiv.org/abs/2202.08906)继续研究训练稳定性与迁移；[DeepSeekMoE](https://arxiv.org/abs/2401.06066)把专家拆得更细并加入共享专家，使通用计算与路由专业化并存；后续无辅助损失均衡方法尝试减少均衡目标对主梯度的干扰。

这些变化都在重新选择三件事：

1. 一个专家有多大，专业化的粒度是什么；
2. 哪些参数对所有 token 共享，哪些按 token 条件激活；
3. 均衡信号进入主 loss、router bias，还是调度器。

没有一个选择能脱离 batch、序列长度、expert parallel 拓扑和推理流量单独排名。

## 推理把负载问题重新暴露

训练 batch 较大，可以在 token 维度平均路由；在线 decode 每步 token 少、请求动态，专家分布更容易抖动。即使平均负载均衡，某一步热门专家也会拖慢整个 batch。专家权重若跨节点放置，decode 的小消息 all-to-all 尤其难以利用带宽。

因此，MoE 服务需要同时观察：

$$
\text{active FLOPs},\quad
\text{dispatch bytes},\quad
\text{expert occupancy},\quad
\text{tail latency}.
$$

详细系统路径见[MoE 系统](../../systems/moe-systems.md)，训练放置见[模型并行](../../systems/model-parallelism.md)，推理侧还要结合[调度与 Goodput](../../inference/scheduling-goodput.md)。

## 条件计算并未终结稠密模型

MoE 的优势依赖足够大规模、合理路由和高效通信。小模型、低并发、通信受限或强确定性场景中，稠密 FFN 可能更简单、更稳定。比较时至少固定：

- 总参数、激活参数和实际 FLOPs；
- 每专家容量、top-$k$、溢出策略与 token drop；
- 数据与训练 token；
- expert placement、网络和 batch 形态；
- 质量、吞吐、显存与尾延迟。

这条谱系的核心不是“稀疏替代稠密”，而是把容量扩展的成本从矩阵乘转移到路由与系统。机制总览见[Mixture of Experts](../../architecture/moe.md)，邻近的另一条效率路线见[从显式寻址到有限状态](linear-time-sequence-models.md)。

## Reference {#reference}

- [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/mixex.html)
- [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538)
- [GShard](https://arxiv.org/abs/2006.16668)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [ST-MoE](https://arxiv.org/abs/2202.08906)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
