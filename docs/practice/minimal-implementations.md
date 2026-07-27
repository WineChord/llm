# 手撕实现

这里的代码用于固定数学、shape 与状态语义。它比伪代码完整，比生产框架短：只保留不可替代的核心逻辑、退化分支和最小断言，再把性能优化留给专用 kernel 与运行时。

## 阅读约定

每个实现都回答六个问题：

1. 输入、输出与状态 shape 是什么；
2. 公式里的归一化轴和 mask 在哪里；
3. 哪些边界条件必须显式处理；
4. 怎样与手算或框架 reference 对照；
5. 哪些语义可优化、哪些不可改变；
6. 代码在哪个规模以后不再适合直接使用。

代码以 Python 与 PyTorch reference 为主。低效的显式展开若能让语义更清楚，会被有意保留；正文同时指出生产实现应怎样避免复制、同步或无界循环。

## 实现地图

| 专题 | 核心实现 | 主要不变量 |
| --- | --- | --- |
| [Tokenizer](tokenizers.md) | byte BPE、Unigram Viterbi、Unicode | round-trip、merge rank、兼容性 |
| [张量原语](tensor-primitives.md) | softmax、CE、RMSNorm、SwiGLU、RoPE、GQA、online softmax | 数值稳定、mask、head 映射 |
| [Decoder-only Transformer](transformer-from-scratch.md) | embedding、block、LM loss、KV decode、采样 | 因果性、位置、训练—增量一致 |
| [递推与记忆](sequence-models.md) | selective scan、delta rule、segment/kNN memory | chunk 等价、reset、状态容量 |
| [训练目标](training-objectives.md) | AdamW、LoRA、KD、BT、DPO、GAE、PPO、RLOO/GRPO、V-trace | mask、归一化、old/ref policy |
| [分布式与容错](distributed-systems.md) | token loss、布局、collective、MoE dispatch、checkpoint manifest | 全局语义、顺序、原子提交 |
| [推理引擎](inference-engine.md) | KV allocator、prefix reuse、调度、量化、推测解码 | COW、幂等、精确分布 |
| [检索与智能体](retrieval-agents.md) | BM25、RRF、MMR、tool dispatch、事件归约 | 权限前置、去重、终态 |
| [推理时计算](test-time-compute.md) | self-consistency、beam、PUCT、预算分配 | canonical answer、verifier、预算 |
| [多模态](multimodal.md) | patch、对比、VQ、diffusion、flow、RVQ | 坐标、mask、模态归一、采样 |
| [评测工具](evaluation-tooling.md) | pass@$k$、bootstrap、校准、引用与安全指标 | 配对、分层、分母与缺失值 |

这些页面不是独立项目模板。需要训练器、服务端、配置系统或部署脚手架时，应选成熟框架；需要核对某个公式、排查 shape 或验证优化前后等价时，先回到这里。

## 验证阶梯

实现按相同顺序升级：

1. 手算或解析结果；
2. FP64 小张量与退化输入；
3. 框架 reference；
4. 自定义向量化实现；
5. mixed precision；
6. backward 与梯度检查；
7. 分布式切分；
8. 真实 shape 的正确性；
9. 端到端质量；
10. 性能与资源。

第 4 层更快不代表第 1–3 层可以跳过。若最终 logits 偏差过大，应回到第一个分叉点，而不是直接放宽端到端容差。

## 代码边界

- 示例不包含 CLI、配置加载、日志平台和训练框架封装；
- 随机算法显式传 generator 或 seed；
- 归一化、reduction、mask 与 dtype 不使用含糊默认值；
- 无定义的退化情况抛错或返回明确状态；
- “成功”由断言或环境状态判断，不由打印文本声明；
- 性能代码必须保留一个更慢、更直白的 reference。

实验设计见[实验方法](index.md)，从 loss 到系统逐层排错见[调试手册](debugging.md)。
