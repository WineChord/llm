# 稀疏与替代架构

标准 dense Transformer 让每个 token 激活几乎全部参数。稀疏专家和替代序列模型试图把“参数容量”“每 token 计算”和“序列长度成本”解耦。

## Mixture of Experts

在稀疏 MoE 中，router 为 token 选择 top-$k$ 专家：

$$
y=\sum_{i\in\operatorname{TopK}(g(x))}p_i(x)E_i(x)
$$

总参数量可以很大，但每个 token 只激活少数专家。收益伴随新的系统成本：

- all-to-all 通信与专家并行；
- token 分布不均导致 straggler；
- capacity factor、溢出与丢 token 策略；
- router collapse、专家重复和负载均衡损失；
- checkpoint、量化与推理部署复杂度。

[Switch Transformer](https://arxiv.org/abs/2101.03961) 使用 top-1 路由简化稀疏训练；[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 讨论更细粒度专家与共享专家。比较 MoE 时必须同时报告总参数、激活参数、每 token FLOPs、通信拓扑和端到端吞吐。

路由概率、capacity、溢出策略、无辅助损失均衡和 expert parallel 的完整计算路径见 [Mixture of Experts](moe.md)。本页其余部分集中讨论非标准序列混合器与混合结构。

## State Space Models

状态空间模型以状态递推压缩历史，目标是让序列计算随长度近似线性。[Mamba](https://arxiv.org/abs/2312.00752) 让状态参数依赖输入，并设计硬件感知扫描算法。它减少标准 attention 的二次项，但状态容量、并行训练 kernel、检索精度和生态兼容性仍需具体评估。

[RWKV](https://arxiv.org/abs/2305.13048) 结合 Transformer 风格训练与 RNN 风格推理。此类架构的常数、kernel 成熟度和实际任务质量，往往比渐近复杂度更决定可用性。

## 混合架构

attention、SSM、卷积和 MoE 可以按层或分支组合。混合设计常见理由：

- 用 attention 保留精确内容寻址；
- 用递推或卷积降低大部分长序列成本；
- 用 MoE 增加通道容量；
- 在局部窗口、全局 token 和外部检索之间分工。

评估混合架构时，应追踪信息经过哪些状态、哪些状态可缓存、哪些操作阻塞并行，以及部署栈是否真正支持。
