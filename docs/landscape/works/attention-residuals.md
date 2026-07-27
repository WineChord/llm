# Attention Residuals：当网络深度也拥有寻址能力

residual connection 通常被解释成梯度高速公路：

$$
h_{l+1}=h_l+f_l(h_l).
$$

但它同时规定了另一件事——下一层怎样读取整个网络历史。展开递推，

$$
h_l=h_0+\sum_{i=0}^{l-1}f_i(h_i),
$$

所有早期输出都以固定系数 $1$ 混进同一个 state。梯度可以沿 identity path 直达浅层，深层却只能看到
已经相加后的混合物，不能直接选择“这次重新读取第几层的表示”。

[Attention Residuals（AttnRes）](https://arxiv.org/abs/2603.15031)把 sequence attention 的思想转到
depth axis：每个 sublayer 用一个 pseudo-query，在 embedding 与此前 sublayer outputs 之间做
softmax retrieval。Full AttnRes 保留逐层来源，Block AttnRes 则以 block summary 换取可控的显存、
pipeline communication 与推理 I/O。

本页从 PreNorm dilution 与多条 residual-generalization 路线出发，推导 Full/Block AttnRes，再进入
online merge、pipeline cache 和 [Kimi K3](kimi-k3.md) 的系统接入。稳定的架构接口见
[注意力与位置](../../architecture/attention-position.md#attention-residual)。

## Residual 的两面：梯度路径与深度聚合

对标准 residual，

$$
\frac{\partial\mathcal L}{\partial h_l}
=
\frac{\partial\mathcal L}{\partial h_L}
\prod_{j=l}^{L-1}
\left(I+\frac{\partial f_j}{\partial h_j}\right).
$$

展开连乘时始终存在 identity 项，这是深网容易优化的关键。normalization 放在哪一侧，却产生了长期
存在的取舍。[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
分析了 PostNorm 与 PreNorm 的不同梯度行为：PostNorm 每层重整 residual state，深层梯度更敏感；
PreNorm 把 norm 放进 branch，让 identity path 保持干净。

PreNorm 的代价是 residual magnitude 会随层输出累加。AttnRes 论文把其称为
**PreNorm dilution**：当 $\lVert h_l\rVert$ 随深度增长，固定尺度的新 branch output 在总 state 中
所占比例下降；深层若要产生同等影响，可能被迫输出更大幅度。这里的核心不是“加法一定数值爆炸”，而是
所有过去被压进一份 state 后，相对贡献与可检索性同时恶化。论文在其 Kimi Linear 实验中观察到 baseline
的 hidden/output magnitude 随深度增长，而 AttnRes 更有界，gradient norm 也更均匀；这是作者配方内
的诊断证据，不是所有 PreNorm 模型的普遍增长定律。

## 在 AttnRes 之前，大家怎样改 residual

这些工作处理的是不同轴，不能排成简单的替代关系：

| 路线 | 历史状态 | mixing 是否依赖当前输入 | 主要解决的问题 |
| --- | --- | --- | --- |
| DeepNet | 单 residual stream | 否 | 用 DeepNorm scale 与初始化稳定极深网络 |
| DenseFormer | 直接保留多层输出 | 否，学习静态 depth weights | 让后层重用远处表示 |
| MUDDFormer | 直接保留多层输出 | 是，且拆分 Q/K/V/residual | 给不同 substream 动态 dense connection |
| Hyper-Connections | $m$ 个并行 streams | 是，学习 stream mixing | 扩大跨深度 recurrent state |
| mHC | $m$ 个并行 streams | 是，受 manifold constraint | 恢复 identity 性质并稳定扩展 |
| AttnRes | 逐层或逐 block 来源 | 是，softmax over depth | 显式选择早期表示 |

[DeepNet](https://arxiv.org/abs/2203.00555)通过 DeepNorm 的 residual scaling 与配套初始化把
Transformer 扩到 1,000 层；它解决更新尺度与梯度稳定，却仍把历史压在单一递推 state。

[DenseFormer](https://arxiv.org/abs/2402.02622)在每个 block 后做 Depth-Weighted-Average，让后层
直接访问此前 representations；权重是训练得到、推理后固定的 scalars，所以不同 token 在同一深度使用
相同 mixing。[MUDDFormer](https://arxiv.org/abs/2502.12170)进一步按 token 生成动态 dense weights，
并把 Q、K、V 与 residual stream 分开，表达更细，但连接生成与多路 materialization 也更复杂。

[Hyper-Connections](https://arxiv.org/abs/2409.19606)不保存所有历史层，而把 residual state 扩成
$m$ 个并行 streams，以动态 mixing matrix 读写；[mHC](https://arxiv.org/abs/2512.24880)再把 mixing
投影到受约束 manifold，以恢复 identity mapping 并控制大规模训练的不稳定。它们扩大的是递推 state，
AttnRes 改变的是访问模式：保留可寻址的历史来源并做 depth-wise softmax。两类机制理论上可以组合，
但 state、I/O 与初始化都会随之改变。

由此可以用两个问题定位一项 residual 方法：

1. 它保存一份、$m$ 份，还是所有历史层的表示？
2. 它使用固定系数、输入相关门，还是归一化的竞争式 retrieval？

## Full AttnRes：对所有早期层做 softmax

AttnRes 把 self-attention 与 MLP 都视作独立的 layer。令 $h_1$ 为 token embedding，
$f_i(h_i)$ 为第 $i$ 个 sublayer output：

$$
k_i=v_i=
\begin{cases}
h_1,&i=0,\\
f_i(h_i),&1\le i<l.
\end{cases}
$$

第 $l$ 层只有一个 learnable pseudo-query $q_l=w_l\in\mathbb R^d$，depth score 为

$$
s_{i\to l}
=
w_l^\top\operatorname{RMSNorm}(k_i),
$$

$$
\alpha_{i\to l}
=
\frac{\exp(s_{i\to l})}
{\sum_{j=0}^{l-1}\exp(s_{j\to l})},
\qquad
h_l
=
\sum_{i=0}^{l-1}\alpha_{i\to l}v_i.
$$

$w_l$ 本身不依赖 token；每个 token 的历史表示 $k_i$ 不同，所以 attention weight 仍是
input-dependent。RMSNorm 防止某层仅凭更大的输出范数占据 softmax，而 values 保留原始尺度。
softmax 还让输入成为历史 outputs 的凸组合，避免 PreNorm 那样必然累加全部幅度。

[官方论文](https://arxiv.org/abs/2603.15031)要求 pseudo-query 零初始化。训练开始时所有 logits
相等，Full AttnRes 退化成历史来源的均值，先提供平滑对称的路径，再逐步学习选择。随机初始化会在尚未
形成可比较表示时偏爱偶然来源，作者消融中表现更不稳定。

Full form 对 $L$ 个 sublayers 需要 $O(L^2d)$ depth-attention arithmetic 与 $O(Ld)$ live sources。
由于 $L$ 通常远小于 token length，算术未必昂贵；activation recomputation 与 pipeline parallel
下“这些 source 原本可以释放，现在必须留住并跨 stage 发送”才是主要系统阻力。

## Block AttnRes：在精细寻址与状态量之间插值

把 $L$ 个 sublayers 分成 $N$ 个 blocks，每块约 $S=L/N$ 层。第 $n$ 块的完整 summary 为

$$
b_n=\sum_{j\in\mathcal B_n}f_j(h_j),
$$

块内前 $i$ 层的 partial sum 记为 $b_n^i$，并单独保留 $b_0=h_1$。第 $n$ 块第一层只在

$$
[b_0,b_1,\ldots,b_{n-1}]
$$

上做 depth attention；后续层再加入当前 $b_n^{i-1}$。因此跨块保留选择性，块内继续使用便宜的
additive recurrence。

$N$ 给出连续插值：

- $N=L$、每块一层时回到 Full AttnRes；
- $N=1$ 时块内基本回到标准 residual，只把 embedding 单列为来源；
- 中间值以 block 内细节换取 $O(Nd)$ live state 与通信。

论文跨规模实验发现约 8 个 blocks 保留大部分 Full gain，但“8”是作者模型与硬件上的经验点，不是
算法常数。block boundary 还决定表示语义：attention/MLP 是否分别计层、最后一块是否不满、embedding
是否始终单列，都必须写进 checkpoint 与 runtime contract。

## Online softmax 让两阶段计算保持精确

一个 block 内有 $S$ 个 pseudo-queries，它们是参数，不依赖前面 layer 的实时输出。于是 inter-block
部分可以先一次性 batch：

1. Phase 1 用 $S$ 个 queries 共同读取已完成的 $N$ 个 block representations；
2. Phase 2 随 layer 顺序更新当前 partial sum，并把这个新来源与 Phase 1 结果合并。

对一组 depth logits 保存

$$
m=\max_i s_i,
\qquad
z=\sum_i e^{s_i-m},
\qquad
u=\sum_i e^{s_i-m}v_i.
$$

两个不相交 source groups 的 $(m,z,u)$ 可按
[online softmax](https://arxiv.org/abs/1805.02867)精确合并，无需重新读取 Phase 1 的 values。

### Depth attention 与 online merge reference {#attnres-online-merge}

下面把 depth/source 轴放在第 0 维。整组计算与任意二分后的 summary merge 应一致；零 query 时则
退化为 source mean。

```python
import torch

def depth_summary(query, sources, eps=1e-6):
    value = sources.float()
    key = value * value.square().mean(-1, keepdim=True).add(eps).rsqrt()
    score = torch.einsum("d,l...d->l...", query.float(), key)
    maximum = score.amax(0)
    weight = torch.exp(score - maximum)
    normalizer = weight.sum(0)
    numerator = (weight[..., None] * value).sum(0)
    return maximum, normalizer, numerator

def merge_summary(left, right):
    maximum = torch.maximum(left[0], right[0])
    left_scale = torch.exp(left[0] - maximum)
    right_scale = torch.exp(right[0] - maximum)
    normalizer = left_scale * left[1] + right_scale * right[1]
    numerator = left_scale[..., None] * left[2] + right_scale[..., None] * right[2]
    return maximum, normalizer, numerator

def finish(summary):
    return summary[2] / summary[1][..., None]

torch.manual_seed(0)
sources, query = torch.randn(7, 2, 3, 5), torch.randn(5)
whole = finish(depth_summary(query, sources))
merged = finish(merge_summary(
    depth_summary(query, sources[:3]), depth_summary(query, sources[3:]),
))
torch.testing.assert_close(merged, whole)
torch.testing.assert_close(finish(depth_summary(torch.zeros(5), sources)), sources.mean(0))
```

这是 Full AttnRes 的 softmax 真值，也是 Block AttnRes 两阶段 merge 的核心；没有包含 learnable
RMSNorm weight、block state machine、output prenorm 或 fused backward。
[FLA 的 naive reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/naive.py)
给出了带 norm weights、可选 fused output RMSNorm 与 FP32 math 的当前接口，
[fused operator](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/fused.py)
则以 online softmax 和重计算实现 forward/backward。

## Pipeline Parallel：缓存的是已知历史

标准 residual 在 pipeline stage 间只传当前 hidden state。Block AttnRes 的后续 stages 需要全部已完成
block summaries；若每次 stage transition 都重传完整历史，会反复搬运相同数据。

对 $P$ 个 physical stages、每 rank $V$ 个 interleaved virtual stages，令 $C=PV$，每 physical
stage 平均产生 $N_p$ 个 block representations。论文给出的每 token 朴素通信量为

$$
\operatorname{Comm}_{\mathrm{naive}}
=
\frac{C(C-1)}{2}N_pd.
$$

同一 physical rank 在后续 virtual stage 会再次需要此前收到的 blocks，因此可把它们留在本地 cache，
只传从上次访问以来新完成的部分：

$$
\operatorname{Comm}_{\mathrm{cached}}
=
\frac{P(P-1)}{2}N_pd
+(V-1)P^2N_pd.
$$

peak transition 从 $O(PV)$ 降到 $O(P)$，论文称其为约 $V\times$ 改善，并可在 steady-state 1F1B
中与计算重叠。这种 cache 不是模型记忆：它是 pipeline schedule 内、按 micro-batch 隔离的 activation
transport cache。micro-batch ID、virtual stage、block completion 和 backward lifetime 任一错位，
都会把另一条样本或另一时刻的 depth sources 喂给当前层。

论文报告，无 pipeline parallel 时 Block AttnRes 训练 overhead 可忽略，在其 pipeline 配置下
end-to-end overhead 小于 4%。这是作者系统的测量；拓扑、block boundary 与 interleaving schedule
变化后要重新计数。

## 推理：block representations 像一份 depth KV cache

同一 block 内的所有 queries 会重复读取相同已完成 blocks。两阶段调度把 $S$ 次小读取合成一次 batched
inter-block pass，再以便宜的 sequential partial-sum merge 收尾。论文的 per-token、per-layer
residual I/O 模型中：

- standard residual 为 $3d$；
- Full AttnRes 两阶段后为 $(S+N)d$；
- Block AttnRes 为 $(N/S+5)d$。

在论文示例 $L=128,N=8,S=16$ 中，Block 为 $5.5d$，Full 为 $24d$。该表只计算 residual mechanism
本身的 read/write，不含 attention、MLP、collective 或 kernel launch。

长 prefill 还要保存每个 token 的 $N$ 个 block representations，即 $N T d$ elements。
[AttnRes 论文](https://arxiv.org/abs/2603.15031)给出的 128K、8-block 例子约 15 GB；沿 sequence
axis 分到 8 个 TP ranks 后约 1.9 GB/rank，再以 16K chunked prefill 可降到 0.3 GB 以下。这里分片的
是 token rows，不是把某个 token 的 depth sources 拆散。

## Kimi Linear 与 K3 中的接入

原论文先在 [Kimi Linear](https://arxiv.org/abs/2510.26692) 的 48B total / 3B activated 架构上加入
AttnRes，并用相同配方预训练 1.4T tokens。其 scaling-law 结果中，Block AttnRes 达到同 loss 时，
baseline 需要约 $1.25\times$ compute；最终 48B 对照在作者报告的所有下游项上改善，其中
GPQA-Diamond 为 $36.9\to44.4$、HumanEval 为 $59.1\to62.2$。这些是单一模型家族和训练分布中的
author-reported evidence，不等同于任意 checkpoint 的无训练替换收益。

16-layer ablation 给出了机制线索：baseline loss 1.766，DenseFormer 1.767，mHC 1.747，
Full AttnRes 1.737，block-size 4 的 Block AttnRes 1.746。input-dependent query 变体达到 1.731，
优于默认的 layer-only pseudo-query，却会破坏 queries 可提前 batch 的系统优势；input-independent
mixing、移除 RMSNorm、改用 sigmoid kernel 都有不同程度退化。数字说明 softmax、输入相关 keys 与
normalization 在该设置中共同作用，不能只把 gain 归给“更多 skip connections”。

[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)进一步把
Block AttnRes 放进 93-layer hybrid KDA/MLA backbone：8 个、每个 12 个 backbone layers 的 blocks，最后一块
允许不满；连同 embedding 有 9 个跨 block sources。K3 的系统接入又做了两层优化：

- prefill 用 sequence-parallel activations，把 TP all-reduce 拆成 reduce-scatter 与 all-gather，
  在两者之间对 sequence shard 执行 intra-block merge，避免每个 TP rank 复制所有 block states；
- decode 把 batched inter-block pass 放到 side stream，与 main-stream 工作重叠；intra-block merge、
  partial-sum update 与 RMSNorm 则融合进相邻 TP collective。

这些优化依赖 K3 的 block layout、TP/SP 切分与 kernel schedule。只实现论文伪代码，不会自动得到报告
中的 prefill memory 或 decode latency。

## 公开实现边界

截至 2026-07-28，[MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
公开的是论文、图表与 PyTorch-style Block AttnRes pseudocode，没有完整训练栈、K3 pipeline cache
或生产推理 kernel。FLA 主分支已经提供 naive 与 fused AttnRes operators，并测试 source count、
hidden width、dtype、forward/backward、可选 output RMSNorm 等组合；它实现的是 depth-softmax
算子核心，不等于论文中完整的 block orchestration 与跨 stage cache。

因此复现应分三层：

1. 语义层：source 顺序、embedding、partial block、RMSNorm 与 softmax axis；
2. 算子层：FP32 reference、fused forward/backward、online merge 与重计算；
3. 系统层：checkpointing、PP cache、SP/TP collective、side stream 与 block state lifetime。

“官方仓库存在”只能证明第一层伪代码和论文资产已经公开；“FLA 有 fused op”也不能证明第三层已被
开源复刻。

## 尚未解决的问题

- **Full 还是 Block**：更细 source 提高选择能力，却增加 live state 与通信；硬件变化会移动最优点。
- **block boundary**：固定等长切块未利用 attention/MLP 类型、层功能或 learned routing 的差异。
- **query 条件化**：input-dependent query 在论文消融更好，但会失去提前 batch 的关键系统性质。
- **source collapse**：softmax 是否长期只选 embedding、当前 partial 或少数中层，需要按 token/任务分析。
- **checkpoint migration**：零 query 产生平均而非标准 residual sum，不能无损加载已有 PreNorm 权重。
- **与 mHC 组合**：多 streams 与可寻址历史可能互补，也可能把 state/I/O 成本相乘。
- **训练外深度**：AttnRes 没有证明模型可以像 length extrapolation 那样直接增加 layers。
- **因果解释**：depth weight 可视化显示读取偏好，但不自动构成某层功能的因果证明。

验证时至少覆盖：零初始化首步、source permutation、Full/Block 极限、online merge、block 尾部、
activation recompute、不同 PP/VP schedule、prefill/decode 对齐、forward/backward 误差，以及训练中
每层 output magnitude、gradient norm、attention entropy 与 source occupancy。

## Reference {#reference}

- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555)
- [DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging](https://arxiv.org/abs/2402.02622)
- [Hyper-Connections](https://arxiv.org/abs/2409.19606)
- [MUDDFormer: Breaking Residual Bottlenecks via Multiway Dynamic Dense Connections](https://arxiv.org/abs/2502.12170)
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [FLA AttnRes Naive Reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/naive.py)
- [FLA Fused AttnRes Operator](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/attnres/fused.py)
