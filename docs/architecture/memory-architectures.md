# 记忆架构

“记忆”不是单一模块。对序列模型，它至少可能指：

- 当前窗口中的精确 KV；
- 跨 segment 传递的隐藏状态；
- 被压缩的历史表示；
- 外部键值或文档存储；
- 在推理过程中更新的神经参数。

这些记忆的容量、可写性、可寻址性和生命周期完全不同。讨论前应先说明“记住什么、保存在哪里、何时清除、如何验证”。

## 记忆分类

| 类型 | 写入 | 读取 | 容量 | 主要风险 |
| --- | --- | --- | --- | --- |
| KV Cache | 每个 token 精确追加 | attention 内容寻址 | 随窗口线性增长 | 显存、淘汰 |
| Recurrent state | 递推压缩 | 固定状态读取 | 固定或受控 | 信息覆盖 |
| Segment memory | 保存上一段 hidden | 跨段 attention | 与段数策略相关 | stale state、训练边界 |
| Compressed memory | 历史降采样/压缩 | attention 或融合 | 受压缩率控制 | 细节损失 |
| 外部检索 | 写文档或键值 | ANN/精确检索 | 可扩展 | 错检、投毒、延迟 |
| 可塑神经记忆 | 测试时优化参数 | 参数化函数读取 | 固定参数容量 | 更新不稳、跨请求泄漏 |

## Segment recurrence

[Transformer-XL](https://arxiv.org/abs/1901.02860)把上一 segment 的 hidden state 作为当前 segment 的额外 K/V。设当前层输入为 $H_\tau$，历史为

$$
M_\tau
=
\operatorname{stopgrad}(H_{\tau-1}),
$$

则 key/value 来源可以写成

$$
\tilde H_\tau=[M_\tau;H_\tau].
$$

当前 query 只来自 $H_\tau$，因此训练仍按 segment 推进。`stopgrad` 防止计算图跨全部历史无限增长；它也意味着模型不是通过反向传播直接修改旧 segment 表示。

### 相对位置

复用旧 hidden 时，绝对位置会让同一缓存随 segment 改变语义。Transformer-XL 使用相对位置分解，使 query 与记忆中的 key 按相对距离交互。实现必须区分：

- 逻辑历史位置；
- 当前 segment 内位置；
- memory 截断后仍保留的相对距离。

## 压缩记忆

[Compressive Transformer](https://arxiv.org/abs/1911.05507)在细粒度 memory 被淘汰前，把它压缩到更低分辨率：

$$
C_\tau=f_{\mathrm{compress}}(M_{\mathrm{old}}).
$$

当前层可同时读取短期 memory 与 compressed memory。压缩可以是 pooling、卷积或学习映射，并配辅助重建/对齐目标。

压缩率带来直接取舍：

- 更高压缩率降低内存；
- 局部细节、顺序和稀有 token 更易丢失；
- 训练目标若只重建 hidden，不一定保留任务所需证据。

## 外部非参数记忆

[RETRO](https://arxiv.org/abs/2112.04426)从大规模文本库检索近邻 chunk，再通过 cross-attention 注入模型。[Memorizing Transformers](https://arxiv.org/abs/2203.08913)把历史 K/V 写入近邻索引并在后续读取。

抽象地，给定 query $q$，记忆返回

$$
\mathcal N_k(q)
=
\operatorname{TopK}_{(k_i,v_i)\in\mathcal M}
\operatorname{sim}(q,k_i).
$$

若记忆直接提供 token 概率，可与模型分布插值：

$$
p(y\mid x)
=
\lambda p_{\mathrm{LM}}(y\mid x)
+
(1-\lambda)p_{\mathrm{mem}}(y\mid q_x).
$$

外部记忆的优势是知识可更新、可删除、可追踪来源；代价是索引一致性、召回误差、额外延迟以及检索内容的权限与可信度。

## 测试时可塑参数

另一条路线把一个小网络 $f(W;\cdot)$ 当作记忆。读入键值对 $(k_t,v_t)$ 时，在前向过程中更新参数：

$$
W_t
=
W_{t-1}
-
\eta_t\nabla_W
\ell(f(W_{t-1};k_t),v_t),
$$

再用 query 读取

$$
y_t=f(W_t;q_t).
$$

[Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620)将测试时训练层作为序列模型组件。它与普通微调不同：更新发生在模型定义的前向路径中，并应有明确的序列 reset、梯度截断和并行算法。

## 写入策略

不是所有输入都值得进入长期记忆。写入策略可以基于：

- novelty：与已有 key 是否重复；
- surprise：当前记忆对输入的预测误差；
- utility：未来任务是否可能需要；
- trust：来源与权限是否允许持久化；
- cost：写入、索引和后续检索成本；
- expiry：信息是否有时效。

无选择地写入会把冗余、错误和攻击内容长期保存。对可更新外部记忆，删除和溯源与写入同等重要。

## 生命周期与隔离

任何有状态实现都应定义：

1. 状态属于 token、sequence、conversation、用户还是全局；
2. batch 中每个样本何时 reset；
3. prefix cache 是否能跨请求共享；
4. 被取消或失败的请求是否回滚写入；
5. 模型版本变化后旧状态是否兼容；
6. 如何执行过期、删除、权限回收与审计。

生命周期不清会把质量问题升级为隐私和安全问题。

## Shape 与实现契约

### Segment memory

若

$$
H\in\mathbb R^{B\times T\times d},
\qquad
M\in\mathbb R^{B\times T_m\times d},
$$

则拼接后 K/V 长度为 $T_m+T$。padding mask、position bias 与 memory 截断必须同步更新。

### Fast-weight memory

若每个 head 使用

$$
W\in\mathbb R^{B\times H\times d_k\times d_v},
$$

则 batch reorder、beam expansion 和 speculative branch 都必须复制或回滚正确状态。不能把普通 KV Cache 的索引操作直接套用到可写参数。

## 失效模式

- **状态泄漏**：上一个请求的 memory 影响下一个请求。
- **Stale memory**：过时事实持续覆盖新证据。
- **检索投毒**：恶意内容被高相似度反复召回。
- **灾难性覆盖**：相近 key 的新值破坏旧值。
- **无界增长**：索引、KV 或 segment memory 没有淘汰策略。
- **压缩错保真**：重建 loss 良好，但任务关键细节丢失。
- **Reset 不一致**：训练按样本清空，服务却跨会话保留。
- **分支污染**：beam/speculative 被拒绝的路径仍写入状态。
- **版本漂移**：encoder 或模型更新后旧 key 不再可比。

## 验证

| 层级 | 必测项 |
| --- | --- |
| 代数 | segment 拼接与显式长序列参考对齐 |
| Reset | 拼 batch、重排、取消和复用均不串状态 |
| 容量 | 写入量增加时 recall/overwrite 曲线 |
| 时间 | 新旧冲突、过期、删除后的读取行为 |
| 安全 | 不可信内容、跨租户 key、权限撤销 |
| 系统 | memory bytes、检索延迟、写放大、GC |
| 任务 | 精确复制、关联回忆、更新事实、多跳证据 |

## 前沿观察：Titans

[Titans](https://arxiv.org/abs/2501.00663)把短期 attention 与基于在线更新的神经长期记忆组合，并讨论多种集成方式。其[Google Research 页面](https://research.google/pubs/titans-learning-to-memorize-at-test-time/)提供了作者公开摘要。

适合进入观察层的结论是：surprise 驱动的可塑记忆为“固定窗口之外如何持续更新状态”提供了一种具体设计。论文报告的超长上下文和任务收益仍属于给定模型、数据与实现中的作者实验；在独立复现、更多规模和服务栈证据形成前，不应写成通用的无限记忆能力。

状态递推的实现与 recall 压力测试见[序列模型手撕实现](../practice/sequence-models.md)，外部文档记忆见[RAG](../applications/rag.md)，KV 生命周期见[KV Cache](../inference/kv-cache.md)。
