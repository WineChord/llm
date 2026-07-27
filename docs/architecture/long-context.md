# 长上下文

上下文窗口长度包含至少四个不同命题：接口允许输入多长、训练见过多长、注意力能计算多长，以及模型能可靠利用多远的信息。只扩大配置中的最大位置不能同时证明这四点。

## RoPE 的相对位置

[RoPE](https://arxiv.org/abs/2104.09864)在每个二维子空间对 Q/K 施加位置相关旋转。第 $j$ 个频率可写为

$$
\theta_j=b^{-2j/d_h},
$$

位置 $m$ 的旋转为 $R(m\theta_j)$。由于

$$
\left(R(m\theta)q\right)^\top
\left(R(n\theta)k\right)
=q^\top R((n-m)\theta)k,
$$

点积自然依赖相对位置 $n-m$。

不同频率对应不同旋转周期。超出训练长度后，高频维度可能经历模型未见过的相位模式，低频维度又可能变化不足；长上下文扩展因此是频率分配问题，而不只是把 position ID 变大。

## Position Interpolation

若原训练长度为 $L_0$，目标长度为 $L_1$，最简单的位置插值把新位置压回旧范围：

$$
m'=m\frac{L_0}{L_1}.
$$

所有频率统一缩放易实现，却会压缩短距离分辨率。NTK-aware 一类方法按频率维度调整基数或缩放，使高低频受到不同影响；[YaRN](https://arxiv.org/abs/2309.00071)进一步组合频率分区与 attention scale。实现时应以具体公式和版本为准，“NTK”“dynamic”不是统一规范。

## 训练与推理策略

扩长通常需要组合：

1. 位置缩放或新的位置机制；
2. 足够长且结构真实的数据；
3. 长度课程与稳定的 batch/token 预算；
4. memory-efficient attention；
5. context parallel 或 sequence parallel；
6. 服务端 KV 容量、调度和 admission control。

只在短样本上重复 padding 到长长度，不会产生长距离监督。把多条无关文档拼接成长序列也可能让模型学会忽略远处内容。

## NoPE、递推与渐进长度课程

RoPE 扩长需要处理训练外相位；另一条路线是不在 attention 的 $Q,K$ 上加入显式位置变换，而让因果
计算本身携带顺序。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
给出一个混合实例：KDA 的 ShortConv、逐 token decay 与 recurrent update 对排列顺序敏感，周期性的
MLA 层则使用 NoPE 做全局内容寻址。这里的“无显式位置编码”不等于 permutation invariant：
交换两个 token 会改变 KDA 状态到达后续位置时的内容。

这条分工也说明 NoPE 不能单独评价。K3 的 position signal 来自
[KDA recurrence](state-space-linear-attention.md#kda-recurrence)，而不是 NoPE-MLA 自己；把后者移到
纯全注意力网络，或改变 KDA/MLA 的 3:1 插层比例，都是另一个实验。作者报告从短窗口扩到 1M 时无需
RoPE rescaling 或 interpolation；这证明该 checkpoint 的位置参数不需改写，不证明训练外任意长度都
有同等检索、推理与生成质量。

### 8K → 64K → 256K → 1M

K3 把昂贵的长序列计算集中到后期，公开了四阶段窗口课程：

| 阶段 | 窗口 | 主要作用 | 仍需核对 |
| --- | ---: | --- | --- |
| Pre-training I | 8K | 先学习主体语言、视觉与代码分布 | 短序列是否形成局部捷径 |
| Pre-training II | 64K | 在主体预训练中引入更长依赖 | 长样本比例与 packing |
| Cooldown I | 256K | 用较小训练预算适应长状态与全局寻址 | batch/token 预算变化 |
| Cooldown II | 1M | 在目标窗口上直接训练 | 并行、数值稳定与有效利用 |

报告没有公开每一阶段的完整 token allocation 与切换 step，复现时不能用表格补造这些缺失参数。
“渐进”也不是窗口数字自动递增：每次扩长都会改变每 batch 的序列数、梯度噪声、activation memory、
并行通信和长短样本混合，需要同时追踪 loss spike、状态范数、远程证据命中率和实际 step time。

### 长数据必须提供远程监督

自然长文档和视频并不天然是高质量长上下文训练样本。K3 的公开流程包含 exact/fuzzy dedup、视频帧
perceptual hashing、规则与 classifier 质量过滤、文件结构校验，并对稀缺的真实长且连贯样本上采样。
这些步骤分别处理重复、损坏和分布占比，不能互相替代。

仅拼接仍可能让每个子任务局部可解。报告还构造经过排列与串接的多模态文档和子任务，让解答必须读取
散布在整段上下文中的证据。设计这类数据时至少保存 source boundary、证据 span、原始顺序与变换记录，
并做三个反事实：

1. 删除远端证据后答案应不可恢复；
2. 只保留局部邻域不能凭模板猜中；
3. 调换冲突证据的时间或来源，答案应随之改变。

这比“样本 token 数达到 1M”更接近有效监督。它仍需配合下文的 retrieval、reasoning、generation
矩阵，分别证明能找到、能组合、能长期保持一致。K3 的架构、训练与系统如何共同承载这条课程，见
[Kimi K3](../landscape/works/kimi-k3.md)。

## Attention sinks 与滑动窗口

局部滑动窗口把每个 token 的可见范围限制为 $w$，计算和缓存可接近 $O(Tw)$，但远距离信息必须通过层间传播、全局 token 或外部记忆保留。某些流式 attention 在丢弃早期 token 后明显退化，保留少量初始 sink token 可以稳定分布；这是一种缓存策略，不等于模型能完整记住被淘汰内容。

## DeepSeek-V4：压缩历史、稀疏读取与局部窗口

[DeepSeek-V4](../landscape/works/deepseek-v4.md#csa-hca)把 1M context 分给三种互补路径：CSA 以 $4\times$ 时间压缩后做 query-dependent top-$k$；HCA 以 $128\times$ 时间压缩后做 dense global attention；SWA 保留最近 128 token 的未压缩细节。完整机制见[CSA / HCA](../landscape/works/deepseek-compressed-attention.md)。

训练长度按 $4\text{K}\rightarrow16\text{K}\rightarrow64\text{K}\rightarrow1\text{M}$扩展；sparse attention 到 64K 阶段才引入，并先单独 warm up indexer。Flash 在前 1T token 保持 dense attention，Pro 的 dense 阶段更长但报告没有给出精确 token 数。这个顺序避免从第一步就同时学习语言、压缩器和稀疏选择，但也意味着“总训练 token”不能直接告诉我们模型实际见过多少 1M 样本。

报告的长上下文证据必须按协议拆开：

- MRCR / CorpusQA 测量多 needle retrieval 与 corpus-level analysis；
- 聚合表中的 1M 分数与 Figure 9 的 8-needle 单点不是同一统计口径；
- “context window=1M”只证明接口容量，远距离定位、组合推理和长输出仍要分别验证；
- on-disk prefix cache 和 SWA 重放影响服务成本，不属于模型质量分数。

V4 的 learnable sink 不是把某个真实 token 固定成 sink：每个 head 在 softmax 分母中加入独立可学习 logit，使总 token attention mass 可以小于 1，模型因而能近似选择“不从历史读取”。它和 SWA、partial RoPE 一起构成注意力语义，不能只从最大 context 配置推断。

## 系统代价

标准 attention 的 score 计算为 $O(T^2d)$，KV Cache 则随 $T$ 线性增长。训练时可用 IO-aware kernel 避免物化完整 score 矩阵，但不会消除所有二次 FLOPs。跨设备长序列还需要交换 K/V block 或在线 softmax 统计；[Ring Attention](https://arxiv.org/abs/2310.01889)展示了将 block 通信与计算重叠的一条路线。

## 评测矩阵

### 检索

- 单个 needle 的位置从开头移动到末尾；
- 多个相似干扰项；
- 需要精确复述还是语义定位；
- 不同输入长度下保持相同任务难度。

### 推理

- 多跳证据是否分散在远距离位置；
- 中间证据顺序是否改变；
- 答案能否由局部先验猜出；
- 证据冲突时是否识别时间和来源。

### 生成

- 输入长度与输出长度分别扩展；
- 长文结构、引用与实体一致性；
- KV 压缩、量化或 eviction 后的回归；
- TTFT、TPOT、峰值显存与并发。

常见的“needle-in-a-haystack 全绿”只证明一种定位任务，不证明长文推理、记忆或生产稳定性。

## 发布声明

一个可解释的长上下文声明应同时写：

```text
maximum accepted length
maximum trained length and curriculum
position method and scaling parameters
attention pattern and cache policy
evaluation tasks and evidence positions
input/output length matrix
hardware, latency and memory
known failure regions
```

注意力结构见[注意力家族](attention-variants.md)，长序列并行见[模型并行](../systems/model-parallelism.md)，在线容量见[推理运行时](../inference/runtime.md)。

## GLM 的三段长度课程 {#glm-length-curriculum}

GLM-5 的 base model 总训练量为 28.5T tokens，长上下文不是最后一次位置插值，而是 mid-training 中逐段改变序列长度与数据分布：

| 阶段 | 最大长度 | token 量 | 主要任务 |
| --- | ---: | ---: | --- |
| Mid-training I | 32K | 1T | 领域数据、代码与推理适应 |
| Mid-training II | 128K | 500B | 自然与合成长序列 |
| Mid-training III | 200K | 50B | 超长上下文 |

报告最终上下文上限为 202,752；GLM-5.2 [官方配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)才把 `max_position_embeddings` 提高到 1,048,576。两者应按模型 revision 分开记录，不能把“系列最新上限”写成“GLM-5 报告训练长度”。

DSA 不属于表内 50B 阶段的同一 token 账。报告把它放在 mid-training 结束之后：先做 1000-step indexer warm-up，再用 20B tokens 做 sparse adaptation。长度课程与架构适应必须分别登记，不能把 20B 重复计入 50B。

数据侧同时采用自然长文、合成长样本、[NextLong](https://arxiv.org/abs/2501.12766)与 EntropyLong 一类方法，评测侧还需要区分检索、跨段推理和 Agent 工具历史。搜索 Agent 的 keep-recent / discard-all 进一步说明：窗口容量只是物理上限，哪些 observation 留在活动上下文仍是独立算法，见 [GLM Agentic Engineering](../landscape/works/glm-agentic-engineering.md#context-management)。

## Reference {#reference}

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- [DeepSeek-V4 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
- [NextLong: A Training-Free Long-Context Extension Method](https://arxiv.org/abs/2501.12766)
- [GLM-5.2 官方配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
