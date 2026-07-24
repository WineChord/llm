# 学习路线

LLM 横跨统计学习、模型结构、分布式系统与产品工程。有效的路线不是把论文按发布日期读完，而是先建立因果链，再沿目标岗位加深。

## 共同主线

1. 从[语言建模](../foundations/language-modeling.md)理解 token、条件概率、交叉熵和生成。
2. 学习[分词与表示](../foundations/tokenization.md)，明确序列长度、词表与跨语言能力之间的权衡。
3. 推导 [Transformer](../architecture/transformer.md) 与[注意力、位置编码](../architecture/attention-position.md)。
4. 把[数据流水线](../data/index.md)、[预训练](../training/pretraining.md)和[优化稳定性](../training/optimization.md)连接起来。
5. 理解[并行训练](../systems/parallelism.md)、[KV Cache](../inference/kv-cache.md)与[服务调度](../inference/serving.md)。
6. 最后学习[评测设计](../evaluation/metrics.md)与[可靠性、安全](../evaluation/reliability-safety.md)，避免只看单一榜单。

## 四条深入路径

### 模型与算法

重点阅读注意力变体、位置外推、MoE、状态空间模型、多模态融合、缩放规律和后训练目标。每种方法都回答三个问题：它改变了什么归纳偏置，计算图如何变化，收益在哪些条件下成立。

### 训练系统

从显存账本开始，依次学习数据并行、张量并行、流水线并行、专家并行、激活重计算、混合精度、检查点与容错。目标不是记缩写，而是能从模型形状和集群拓扑推导通信量与瓶颈。

### 推理与 AI Infra

先分清 prefill 与 decode，再研究 KV Cache、连续批处理、分页内存、chunked prefill、推测解码和量化。评价方案时同时报告吞吐、首 token 时延、逐 token 时延、尾延迟、显存与输出质量。

### 应用与智能体

先建立可靠的离线评测，再引入[检索增强](../applications/rag.md)、工具调用和[智能体](../applications/agents.md)。把模型概率、外部证据、执行权限和业务终态分开验证。

## 学会的标准

能够解释公式不等于能够训练，跑通框架不等于理解机制。每个专题至少完成一次：手算小例子、读原论文、读一个实现、做对照实验、解释失败案例。
