# 来源、谱系与快照

数据可追溯性的目标不是给语料附一张静态说明，而是让任何训练 token 都能沿转换链回到来源对象、获取时间和处理决策。只有这样，复现、更新、许可审计、错误定位与删除才共享同一套事实。

## 问题

网页、代码仓库、论文、对话和人工标注的更新方式不同。即使 URL 不变，内容也可能变化；即使正文相同，许可、时间和上下文也可能不同。把“来源名”直接当成样本身份会混淆：

- 来源集合与某次不可变抓取；
- 原始字节与解析后的正文；
- 内容相同但使用条件不同的对象；
- 修正解析器后得到的新派生版本；
- 被撤回来源与已经生成的 token shard。

因此谱系的基本单位应是不可变对象和有版本的转换，而不是目录名。

## 数据契约

### 五层对象

| 层 | 标识 | 必须记录 |
| --- | --- | --- |
| Source | `source_id` | 所有者、入口、许可或使用决策、范围 |
| Capture | `capture_id` | 获取时间、原始 digest、响应元数据、快照位置 |
| Document | `document_id` | parser 版本、结构、语言、时间、父 capture |
| Example | `example_id` | 过滤、去重 cluster、split、模板或任务变换 |
| Sequence | `sequence_id` | tokenizer、截断、packing、mask、shard 与 offset |

内容 digest 应基于明确的字节或规范化定义。若规范化规则变化，新的 digest 不能覆盖旧值；两者需要通过父子关系连接。

### Source manifest

一个可执行 manifest 至少覆盖：

```text
source_id and source class
canonical location and capture method
allowed scope and exclusions
license / terms snapshot and review status
time, language, region and domain
expected update cadence
parser family and required metadata
privacy / deletion contact path
```

“可公开访问”不等于“适合训练”；“具有许可字段”也不自动解决内容中第三方权利、个人数据和地域约束。manifest 记录的是决策输入和当时结论，不替代必要的法律审查。

## 机制

### 先快照，再解析

原始快照应保持不可变，解析器只生成派生对象。这样可以：

- 在不重新抓取的情况下修复 parser；
- 比较新旧 parser 的文档 diff；
- 证明训练时使用的是哪次内容；
- 在来源消失后保留审计依据；
- 将删除操作传播到全部派生层。

[Dolma](https://arxiv.org/abs/2402.00159) 公开了来源组成与处理过程，[OLMo](https://arxiv.org/abs/2402.00838) 进一步把训练数据、代码、模型和评测放在可研究链路中。这里可借鉴的是可追溯性，而不是照搬其来源选择。

数据、代码、中间 checkpoint、日志与许可证分别开放到什么程度，决定了外部研究真正能复查哪一层；这条演进见[开放模型生态](../landscape/lineages/open-model-ecosystem.md)。

### 结构化解析

解析不能只输出一段扁平文本。标题、段落、列表、代码块、表格、链接和时间戳会影响：

- 去除导航与页脚时的边界；
- 代码和自然语言采用不同质量规则；
- 文档级切分和近重复判断；
- 长文档窗口采样；
- 后续引用和事实时效性。

解析失败应有显式状态，如 `unsupported`、`partial`、`empty`、`corrupt`，而不是返回空文本后被当成低质量样本。

### 快照 diff 与增量更新

增量语料不应每次全量混合后失去时间边界。对相邻快照记录：

$$
\Delta_t
=
\{\text{added},\text{changed},\text{unchanged},\text{removed}\}.
$$

`changed` 需要同时比较原始内容和解析结果。网页模板更新可能改变大量原始字节却不改变正文；parser 更新则可能在原始对象不变时改变训练文本。

[FineWeb](https://arxiv.org/abs/2406.17557) 和 [DataComp-LM](https://arxiv.org/abs/2406.11794) 展示了大规模 web 数据的筛选与受控训练比较。面向持续更新的语料，还需额外冻结 crawl、parser 和过滤器版本，避免把数据变化误判为模型配方变化。

### 删除传播

删除图从 `source_id` 或 `document_id` 出发：

```text
source
  -> captures
  -> documents
  -> dedup representatives
  -> examples
  -> token shards and caches
  -> affected training runs
```

去重会带来特殊问题：被删除对象若是 cluster representative，不能简单删除整簇或保留其文本不变。应从其余成员重新选择代表，并重新执行使用条件检查。

## 正确性与失效

- **URL 作为版本**：同一 URL 随时间变化，无法复现。
- **只记录最终文本 hash**：无法区分抓取变化、解析变化和规范化变化。
- **原地修正数据**：旧 checkpoint 与“同名数据集”失去对应关系。
- **谱系只进日志**：日志轮转后无法查询；谱系应是训练资产的一部分。
- **无效来源静默跳过**：数据比例漂移却不触发告警。
- **删除状态不参与构建**：下次重建又把已撤回对象带回。

## 何时可以简化

若数据是一个固定、带官方版本号的小型数据集，可将 Source 与 Capture 合并，但仍应保存官方版本、下载 digest、split、转换版本和使用决策。只要存在周期更新、多来源合并、敏感数据或长期训练，就不应省略对象分层。

## 验证

每次数据快照至少执行：

1. **重建测试**：相同 manifest、原始快照和代码生成相同对象数量与 digest。
2. **差异审计**：新旧快照按来源、语言、领域、长度和 parser 状态报告增删改。
3. **抽样回溯**：从随机 sequence 能逐层回到原始对象，并重建完全相同的 token。
4. **失败注入**：模拟抓取失败、parser 异常和部分 shard 写入，确认不会发布半成品。
5. **删除演练**：给定一个 document ID，列出所有派生物、重建动作和受影响训练运行。
6. **切片核验**：使用条件和时间边界在混合、去重与导出后仍被保留。

过滤决策见[过滤、去重与污染](filtering-dedup.md)，从文档到张量的精确接口见[序列构造与打包](sequence-construction.md)。
