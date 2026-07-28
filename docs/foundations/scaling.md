# 缩放与计算

模型规模、数据规模与训练计算共同决定预训练损失，但缩放规律是经验拟合，不是无条件的物理定律。

[Scaling Laws 与 Chinchilla](../landscape/works/scaling-laws-chinchilla.md)重建了两项工作的实验问题、拟合口径和工程含义；[规模、数据与上下文](../landscape/lineages/scaling-and-context.md)则把它们放回 GPT-3、数据受限训练和长上下文演进中。

## 训练计算的近似

对 dense decoder-only Transformer，一个常用数量级估算是：

$$
C_{\text{train}}\approx 6ND
$$

其中 $N$ 是非 embedding 参数量，$D$ 是训练 token 数。系数会随注意力、序列长度、激活重计算、稀疏路由和实现而变化；它适合做预算初筛，不应替代 profiler。

## 参数与数据的配比

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) 研究了损失随模型、数据和计算的幂律趋势。[Chinchilla](https://arxiv.org/abs/2203.15556) 重新讨论固定计算预算下参数与 token 的配比，说明许多早期大模型相对欠训练。

“计算最优”依赖目标：

- 只考虑一次预训练，可能偏向更大的模型。
- 若模型需要长期服务，更多训练 token 和更小模型可能降低全生命周期推理成本。
- 数据质量、去重和课程安排改变有效 token 数，原始 token 不能简单视为等价。
- MoE 的总参数、激活参数与通信成本必须分开报告。

把经验 loss 曲面写成

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
$$

并在 $C=kND$ 下消去 $D$，其内部最优点满足

$$
N_\star\propto C^{\frac{\beta}{\alpha+\beta}},
\qquad
D_\star\propto C^{\frac{\alpha}{\alpha+\beta}}.
$$

这说明“参数与 token 同比增长”不是先验原则，而是特定拟合中 $\alpha$ 与 $\beta$ 接近时的结果。只要数据分布、架构族、训练充分程度或计算口径改变，指数和最优分配就可能移动。[Scaling Laws 与 Chinchilla](../landscape/works/scaling-laws-chinchilla.md)给出从单轴拟合到 isoFLOP 曲线的完整推导。

## $6ND$ 何时不够

$6ND$ 把一次 dense Transformer 训练粗略看作每个参数、每个 token 约六次浮点工作，适合早期预算，却省略了多个会随 shape 改变的量：

- attention 在长序列下包含与 $T^2$ 相关的计算和 IO；
- embedding、输出词表与共享参数未必按同一方式计入 $N$；
- activation checkpointing 重算 forward，但不增加训练 token；
- MoE 只有部分参数被激活，却额外产生路由和 all-to-all；
- padding、失败重启、评测与数据处理消耗真实集群时间，却不进入理想 FLOPs；
- 低精度改变的是执行吞吐与字节，不会自动改变算法 FLOPs。

因此预算应同时保留三本账：算法 FLOPs、实际加速器时间和端到端资源成本。前两者相差很大时，先检查利用率、通信、数据空泡和重算，而不是修改 scaling law。

## 推理缩放

推理阶段至少有三种缩放：

1. **模型缩放**：更大的权重与激活。
2. **上下文缩放**：更长 prompt 增加 prefill 计算和 KV Cache。
3. **测试时计算**：采样更多候选、搜索、验证、工具调用或迭代反思。

测试时计算的收益取决于任务是否可验证、候选是否多样、评分器是否可靠。盲目增加 token 可能只放大延迟与错误自信。

训练最优也不等于生命周期最优。若一个 checkpoint 将服务 $Q$ 个请求，可将决策写成：

$$
\min_{N,D,\pi}
C_{\text{train}}(N,D)
+Q\,C_{\text{serve}}(N,\pi)
\quad
\text{s.t.}\quad
\operatorname{Quality}(N,D,\pi)\ge q_0,
$$

其中 $\pi$ 是量化、检索、搜索或 speculative decoding 等部署策略。更小但训练更充分的模型，可能用更多一次性训练换取更低的长期推理成本；反过来，昂贵的测试时搜索也可能抵消模型缩小带来的收益。

## 一条 scaling 曲线怎样才可解释

任何“规模”结论都应同时说明：模型版本、总参数与激活参数、训练 token 口径、上下文长度、精度、硬件、训练阶段、是否包含数据重复，以及数字是公开披露、实现推断还是未知。实验上还需要：

1. 同一架构族内覆盖多个 $N,D$ 组合，而不是只比较最终 checkpoint；
2. 每条 isoFLOP 曲线上有足够点定位内部最优，而非边界最小值；
3. 固定 tokenizer、数据分布与训练终止规则；
4. 为异常运行保留原因，不事后删除破坏拟合的点；
5. 用未参与拟合的规模或 seed 检查外推；
6. 同时报告拟合区间、残差和参数不确定性。

训练 token 的阶段与证据标签见[训练 token 口径](../landscape/training-tokens.md)，预算落到显存和关键路径时见[内存、数值与硬件](../systems/memory-numerics-hardware.md)。

## Reference {#reference}

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
