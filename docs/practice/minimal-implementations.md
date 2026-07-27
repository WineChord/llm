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

## 代码放在哪里

通用原子实现与对应知识点共址：公式刚定义完，最短的主路径就默认展开，读者可以逐行对齐变量、shape 和 mask。完整测试、辅助类、替代实现或较长状态转移按内容命名并默认折叠；折叠标题本身是稳定链接，可以从其他章节直接抵达。

本页以下地图聚合这些入口，专题实践则负责跨机制组装与更完整的验证。为保持实验可独立运行，实践代码可以重述必要的原子逻辑；它应链接对应正文并复用相同不变量。论文深读聚焦工作的特有改动，系统页则说明优化后仍必须满足的语义。

## 实现地图

| 组合实验 | 正文中的语义核 | 主要不变量 |
| --- | --- | --- |
| [Tokenizer](tokenizers.md) | [BPE merge rank](../foundations/tokenization.md#bpe-merge-rank)、[Unigram Viterbi](../foundations/tokenization.md#unigram-viterbi) | round-trip、merge rank、兼容性 |
| [张量原语](tensor-primitives.md) | [token CE](../foundations/probability-objectives.md#token-normalized-cross-entropy)、[Decoder Block](../architecture/decoder-block.md#pre-norm-decoder-block)、[RoPE](../architecture/position-encoding.md#rotary-position-embedding)、[GQA](../architecture/attention-variants.md#grouped-query-attention)、[online softmax](../systems/attention-kernels.md#online-attention-reference) | 数值稳定、mask、head 映射 |
| [Decoder-only Transformer](transformer-from-scratch.md) | [block](../architecture/decoder-block.md#pre-norm-decoder-block)、[增量解码](../inference/decoding.md) | 因果性、位置、训练—增量一致 |
| [递推与记忆](sequence-models.md) | [selective scan](../architecture/state-space-linear-attention.md#selective-scan)、[kNN memory](../architecture/memory-architectures.md#knn-token-memory) | chunk 等价、reset、状态容量 |
| [训练目标](training-objectives.md) | [AdamW](../training/optimizer-families.md#adamw-step)、[LoRA](../training/peft.md#lora-forward-merge-reference)、[蒸馏](../training/distillation.md#masked-temperature-kl)、[奖励模型](../training/reward-modeling.md#bradley-terry-loss-reference)、[DPO](../training/offline-preference.md#dpo-semantic-reference) | mask、归一化、reference 与 behavior |
| [LLM 策略优化](llm-policy-optimization.md) | [GAE](../reinforcement-learning/advantage-estimation-gae.md#boundaries)、[GRPO](../reinforcement-learning/grpo.md#group-std)、[ratio gate](../reinforcement-learning/ratio-clipping-gating.md#ratio-gates-semantic-reference) | action mask、四策略身份、归约与 gate |
| [分布式与容错](distributed-systems.md) | [全局 token loss](../systems/collectives-sharding.md#global-token-reduction-reference)、[tensor parallel](../systems/model-parallelism.md#tensor-parallel-linear-reference)、[MoE dispatch](../systems/moe-systems.md#moe-dispatch-combine-reference)、[checkpoint](../systems/checkpointing.md#checkpoint-manifest-validation-reference) | 全局语义、顺序、原子提交 |
| [推理引擎](inference-engine.md) | [KV COW](../inference/kv-cache.md#kv-tail-copy-on-write-reference)、[runtime](../inference/runtime.md#request-transition-reference)、[量化](../inference/quantization.md#groupwise-quantization-reference)、[推测解码](../inference/speculative-decoding.md#speculative-acceptance-reference) | COW、幂等、精确分布 |
| [检索与智能体](retrieval-agents.md) | [索引](../applications/retrieval-indexing.md)、[重排](../applications/reranking-context.md)、[工具调用](../applications/tool-use.md)、[事件归约](../applications/agent-runtime.md#agent-runtime-reducer-reference) | 权限前置、去重、终态 |
| [推理时计算](test-time-compute.md) | [采样与预算](../reasoning/test-time-compute.md)、[搜索与验证](../reasoning/search-verification.md#puct-selection-backup-reference) | canonical answer、verifier、预算 |
| [多模态](multimodal.md) | [patchify](../multimodal/vision-language.md#vit-patchify)、[坐标](../multimodal/document-gui-grounding.md#box-coordinate-roundtrip)、[mask 与 loss](../multimodal/unified-understanding-generation.md#context-target-mask-and-modality-loss)、[RVQ](../multimodal/audio-language-models.md#residual-vector-quantization)、[video tubelet](../multimodal/video-world-models.md#video-tubelet) | 坐标、mask、模态归一、采样 |
| [评测工具](evaluation-tooling.md) | [语言模型](../evaluation/language-model-evaluation.md)、[统计推断](../evaluation/statistical-inference.md)、[校准](../evaluation/calibration-uncertainty.md)、[judge](../evaluation/generative-judges.md)、[事实性](../evaluation/hallucination.md)、[安全](../evaluation/safety-evaluation.md) | 配对、分层、分母与缺失值 |

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

## Reference {#reference}

- [PyTorch gradcheck documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
