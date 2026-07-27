# 数据工程

训练数据不是模型之外的准备工作，而是目标函数的一部分。来源、过滤、采样、模板和 loss mask 共同决定模型实际看到的分布；任何一层不可追溯，最终 checkpoint 的能力变化就难以解释、复现或撤销。

## 问题地图

一条文档进入训练前会经历多种身份：

```text
source object
  -> captured artifact
  -> parsed document
  -> filtered and deduplicated document
  -> tokenized example
  -> packed training sequence
  -> checkpoint exposure
```

每次转换都应保留稳定标识、父对象、规则版本和拒绝原因。只保存最终 token 文件无法回答“这段内容来自哪里”“为什么保留”“进入过哪些 checkpoint”或“删除请求影响哪些派生物”。

| 层次 | 核心问题 | 对应页面 |
| --- | --- | --- |
| 来源 | 能否获取、使用、更新和删除 | [来源、谱系与快照](sources-provenance.md) |
| 质量 | 哪些内容被保留，误删与漏删如何衡量 | [过滤、去重与污染](filtering-dedup.md) |
| 分布 | 各领域实际贡献多少 token 与梯度 | [数据混合与课程](mixtures-curricula.md) |
| 生成 | 合成样本增加了什么，也复制了什么 | [合成数据](synthetic-data.md) |
| 隐私 | 训练记忆、抽取风险与删除如何验证 | [记忆化、隐私与删除](memorization-privacy.md) |
| 张量 | 文档怎样变成 token、mask 与 position | [序列构造与打包](sequence-construction.md) |
| 反馈 | 偏好、步骤、环境和 verifier 怎样对齐 | [偏好、过程与轨迹数据](feedback-trajectories.md) |
| 治理 | 质量门槛、版本和责任怎样持续维护 | [质量与治理](quality-governance.md) |

## 三个不可替代的契约

### 文档契约

最小字段包括：

```text
document_id
source_id and capture_id
content digest
language, domain and timestamp
license / usage decision
parser and normalization version
filter decisions and dedup cluster
split and deletion status
```

原始对象、解析结果和训练文本应使用不同 ID。否则重新解析后内容变化，却仍被误认为同一个训练样本。

### 采样契约

配置中的样本概率不等于训练中的 token 权重。对每个来源都应同时记录：

- 名义采样概率；
- 实际抽到的文档数和有效 token 数；
- 被截断、mask 或 packing 后的有效 loss token；
- 相对语料规模的重复暴露次数；
- 在各训练阶段和 checkpoint 区间的累计暴露。

精确关系与温度采样见[数据混合与课程](mixtures-curricula.md)。

### 序列契约

tokenizer、模板、截断、packing、attention mask、loss mask 与 position IDs 必须作为同一接口验证。样本内容正确，并不保证训练张量正确；一处边界错位足以让模型稳定优化错误目标。

## 端到端机制

1. **登记与快照**：先定义允许的来源、时间边界与获取方式，再生成不可变原始快照。
2. **解析与规范化**：保留标题、段落、代码和表格等结构，同时隔离模板噪声。
3. **过滤与去重**：将内容安全、质量筛选、exact dedup、near dedup 和 benchmark decontamination 分开记录。
4. **切分**：在会泄漏的同源对象之间先建立 group，再按 group 或时间划分 train/dev/test。
5. **混合与课程**：把目标分布转换为可执行 sampler，并审计真实 token share 与重复暴露。
6. **分词与打包**：固定 tokenizer 和序列规则，生成可重放 shard 与 data cursor。
7. **训练与观测**：按来源记录 loss、梯度信号、吞吐和异常，而不只看总平均。
8. **删除与重建**：从 source ID 沿谱系定位原始、派生、索引、缓存和训练产物，明确哪些能删除、哪些只能通过重训或经验证的 unlearning 处理。

[Dolma](https://arxiv.org/abs/2402.00159) 展示了开放语料从来源、处理到发布的完整记录，[DataComp-LM](https://arxiv.org/abs/2406.11794) 则把数据候选、训练预算和下游评测放进受控比较。它们提供的是可研究的工程范式，不意味着其过滤阈值可直接迁移到所有语言和任务。

## 正确性与常见失效

- **通过率替代质量**：过滤器留下的比例不能证明保留集更有用；必须有 slice-level 人工审计和训练消融。
- **去重替代污染检查**：语料内部近重复与评测答案泄漏是两个问题，阈值和搜索对象不同。
- **样本数替代 token 数**：长文档来源可能以较小样本概率贡献大部分梯度。
- **删除原始文件即完成删除**：派生 shard、检索索引、缓存和已训练 checkpoint 仍可能包含影响。
- **合成标签丢失**：无法区分真实与生成内容后，训练异常和偏差无法回滚。
- **随机种子被当成复现充分条件**：数据源变化、worker 数、shard 顺序和过滤代码都可能改变样本流。

## 何时不需要复杂流水线

固定、公开、规模很小且不再更新的数据集，可以用单个 manifest 和确定性转换脚本代替分布式数据平台。但来源摘要、内容 digest、切分规则、转换版本和训练暴露仍不能省略。复杂度应随数据的规模、变动频率和风险增长，而不是以“数据工程”名义先建一套无法验证的系统。

## 验证组织

数据发布前至少回答：

1. 同一 snapshot 和代码版本能否生成相同文档 ID、数量与 digest？
2. 各来源从名义采样到有效 loss token 的权重怎样变化？
3. 去重、过滤和隐私规则在语言、领域、长度切片上的误删率如何？
4. train/dev/test 是否在文档族、题目族和时间上隔离？
5. 任一来源被撤回时，能否列出全部派生对象和受影响 checkpoint？
6. 训练后收益是否来自目标数据，而不是模板、长度或 benchmark 泄漏？

数据随后进入[预训练](../training/pretraining.md)、[监督微调](../training/supervised-finetuning.md)或[后训练](../training/post-training.md)。目标函数和采样实现的最小验证入口见[训练目标实现](../practice/training-objectives.md)，统计与污染审计入口见[评测工具](../practice/evaluation-tooling.md)。
