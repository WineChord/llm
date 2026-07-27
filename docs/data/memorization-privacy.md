# 记忆化、隐私与删除

模型记住训练内容并不总是错误：事实、术语和代码模式都依赖记忆。风险在于模型能否在不适当条件下逐字复现稀有内容、个人数据或秘密，以及来源撤回后能否证明影响已经被处理。

## 概念边界

| 概念 | 问题 |
| --- | --- |
| Memorization | 模型行为对特定训练样本有多强依赖 |
| Extraction | 攻击者能否通过查询恢复训练内容 |
| Membership inference | 能否判断某条内容是否参与训练 |
| Privacy leakage | 输出是否暴露不应披露的个人或敏感信息 |
| Data deletion | 原始与派生数据是否停止被继续使用 |
| Machine unlearning | 已训练模型中样本影响是否被移除或降低 |

这些概念相关但不等价。模型逐字复现公开文本是记忆化，却未必构成隐私泄漏；删除源文件可以停止未来训练，却不会自动改变已有 checkpoint；unlearning benchmark 分数也不等于法律意义上的删除证明。

## 威胁与数据契约

隐私评估先冻结威胁模型：

```text
attacker access: weights / logits / text API
query and compute budget
known prefixes or auxiliary data
target data class and rarity
sampling and filtering controls
success criterion and human review
```

数据侧需记录：

- 来源、时间和使用决策；
- 文档族、重复簇和稀有度切片；
- PII/secret 检测规则、版本与人工复核；
- 训练 split、token shard 与累计暴露；
- 删除状态、请求对象和传播结果；
- 哪些 checkpoint、adapter、索引和合成样本可能受影响。

没有训练暴露与谱系记录，就无法区分“模型从公开知识推断”与“模型逐字恢复某个样本”。

## 记忆与抽取机制

重复、稀有、长而独特的字符串通常更容易被逐字记忆。去重能降低重复权重，但不能保证单次出现的敏感内容不被学习。[Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) 展示了从语言模型生成中恢复训练片段的可能性；[Scalable Extraction of Training Data](https://arxiv.org/abs/2311.17035) 进一步研究了更大规模模型中的可抽取记忆。

因此风险不能只用平均验证损失衡量。至少应分开检查：

- 高频公共文本与低频唯一文本；
- 自然语言、代码、结构化记录和长数字串；
- 不带前缀的自由生成与已知前缀攻击；
- greedy、温度采样和大量候选搜索；
- 原始模型、对齐模型、adapter 与量化模型。

后训练可能降低某些直接复述，也可能在特定 prompt 下保留可抽取内容；拒答并不是参数级删除。

## 防护机制

### 数据最小化

优先不收集或不保留不需要的数据。对来源设置范围、时间和字段白名单，比训练后尝试消除影响更可靠。凭据、访问 token、私密通信和精确个人标识不应因“可过滤”而进入候选语料。

### 过滤与去重

PII/secret 检测应组合确定性规则、上下文分类和人工复核，并保留误报/漏报切片。单纯正则会漏掉非标准格式，也会误删教学样例和公共联系信息。去重规则与细节见[过滤、去重与污染](filtering-dedup.md)。

### 训练与访问控制

训练数据、日志、checkpoint 和调试样本都需要最小权限与保留期限。公开模型和受控 API 的攻击面不同；发布决策应基于实际访问能力，而不是只看离线 benchmark。

### 删除与 unlearning

删除流程分两层：

1. **数据层**：原始对象、派生文档、索引、缓存、token shard 和未来构建全部停止使用；
2. **模型层**：识别受影响 checkpoint，选择重训、从未受影响检查点续训，或采用经验证的 unlearning 方法。

[TOFU](https://arxiv.org/abs/2401.06121) 为生成模型的 unlearning 提供了受控 benchmark，并同时考察遗忘与保留能力。任何方法都可能出现“目标答案变差，但相关信息仍可由改写问题恢复”或“忘掉目标的同时损伤邻近知识”，所以不能只报一个 forget-set loss。

## 正确性与失效

- **把 PII 检测通过写成无隐私风险**：检测器有覆盖边界，模型还可能组合多个非敏感片段。
- **删除源文件即宣称模型已遗忘**：已有权重没有变化。
- **只测固定 prompt**：改写、前缀和多次采样可能恢复内容。
- **只测 forget set**：模型可通过统一拒答获得低分，却破坏 retain set。
- **去重后丢失 cluster 谱系**：删除一个来源时无法重选合法代表。
- **把不可抽取当成未记忆**：有限查询预算下未成功，不证明参数中没有影响。
- **保存完整训练文本到日志**：防护流程本身制造新的敏感副本。

## 何时不应依赖 unlearning

若能在训练前排除数据、从未受影响的 checkpoint 重训，或模型尚未发布，优先使用这些更可验证的路径。对于高风险秘密或严格删除要求，不应把一次实验性的 unlearning 分数当作完全擦除证明。模型输出层 guard 可降低暴露概率，但不能替代数据和模型层处理。

## 验证

1. 按访问方式和攻击预算执行 extraction，而不是只跑固定问答。
2. 在 dedup cluster、稀有度、语言和数据类型上分层报告成功率。
3. 删除后同时审计原始、派生、索引、缓存、checkpoint 与 adapter。
4. unlearning 同时报 forget、retain、相关知识、改写攻击和通用能力。
5. 用未参与方法开发的攻击者、prompt 与时间切片做独立验证。
6. 记录负结果的边界：模型版本、查询预算、解码和检测器覆盖。
7. 对训练和评测日志做内容最小化，避免保存新的敏感副本。

来源与删除传播图见[来源、谱系与快照](sources-provenance.md)，训练阶段的可重放状态见[预训练](../training/pretraining.md)。
