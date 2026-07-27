# 学习路线

LLM 横跨统计学习、模型结构、分布式系统与产品工程。有效的路线不是把论文按发布日期读完，而是先建立因果链，再沿目标岗位加深。

## 共同主线

1. 从[语言建模](../foundations/language-modeling.md)与[概率、损失和梯度](../foundations/probability-objectives.md)理解 token、条件概率、交叉熵和生成。
2. 学习[分词与表示](../foundations/tokenization.md)，明确序列长度、词表与跨语言能力之间的权衡。
3. 推导 [Transformer](../architecture/transformer.md)、[Decoder Block](../architecture/decoder-block.md)与[注意力家族](../architecture/attention-variants.md)。
4. 把[数据流水线](../data/index.md)、[序列构造](../data/sequence-construction.md)、[预训练](../training/pretraining.md)和[优化稳定性](../training/optimization.md)连接起来。
5. 理解[集合通信与分片](../systems/collectives-sharding.md)、[模型并行](../systems/model-parallelism.md)、[KV Cache](../inference/kv-cache.md)与[推理运行时](../inference/runtime.md)。
6. 用[证据与研究方法](evidence.md)校准模型谱系、训练数字和时效性事实。
7. 最后学习[评测设计](../evaluation/metrics.md)与[可靠性、安全](../evaluation/reliability-safety.md)，避免只看单一榜单。

## 四条深入路径

### 模型与算法

重点阅读[注意力变体](../architecture/attention-variants.md)、[长上下文](../architecture/long-context.md)、[MoE](../architecture/moe.md)、状态空间模型、多模态融合、缩放规律和后训练目标。用[模型谱系](../landscape/index.md)区分家族案例与通用机制。每种方法都回答三个问题：它改变了什么归纳偏置，计算图如何变化，收益在哪些条件下成立。

### 训练系统

从显存账本开始，依次学习[集合通信与状态分片](../systems/collectives-sharding.md)、[张量/流水线/上下文/专家并行](../systems/model-parallelism.md)、激活重计算、混合精度、[Kernel](../systems/kernels-performance.md)、检查点与容错。目标不是记缩写，而是能从模型形状和集群拓扑推导通信量与瓶颈。

### 推理与 AI Infra

先分清 prefill 与 decode，再沿[解码](../inference/decoding.md)、[KV Cache](../inference/kv-cache.md)、[推理运行时](../inference/runtime.md)和 [P/D 分离](../inference/disaggregation.md)研究连续批处理、分页内存、chunked prefill、推测解码和量化。评价方案时同时报告吞吐、首 token 时延、逐 token 时延、尾延迟、显存与输出质量。

### 应用与智能体

先建立可靠的离线评测，再引入[检索增强](../applications/rag.md)、工具调用和[智能体](../applications/agents.md)。需要训练多步决策时进入 [Agentic RL](../agentic-rl/index.md)；需要理解仓库级执行系统时阅读 [Coding Agent](../applications/coding-agents.md)。始终把模型概率、外部证据、执行权限和业务终态分开验证。

### 多模态

先读[视觉语言模型](../multimodal/vision-language.md)与[融合训练](../multimodal/architecture-training.md)，再比较连续特征、离散 token、[生成目标](../multimodal/generative-modeling.md)及[音频视频](../multimodal/audio-video.md)。家族案例只用于检验框架，不替代机制。

### 评测与可靠性

从评测协议出发，依次研究[幻觉与事实性](../evaluation/hallucination.md)、[指令遵循](../evaluation/instruction-following.md)与[生产可靠性](../evaluation/production-reliability.md)。目标是能把错误定位到模型、数据、检索、工具、评分器或服务层，而不是笼统归因。

## 学会的标准

能够解释公式不等于能够训练，跑通框架不等于理解机制。每个专题至少完成一次：手算小例子、读原论文、读一个实现、做对照实验、解释失败案例，并把结论写成可核验的证据卡。
