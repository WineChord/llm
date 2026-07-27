# 数据混合与课程

多源训练不是把若干数据集拼接后随机打乱。采样单位、文档长度、截断、loss mask 和重复暴露共同决定每个来源的实际梯度权重；课程学习则让这个分布随训练进度变化。

## 问题

按语料规模比例采样会让高资源来源占主导；提高低资源来源概率又可能快速重复并造成记忆。理想配比取决于目标能力、模型规模、token 预算、训练阶段和来源质量，不存在可跨任务照搬的固定比例。

## 数学契约

### 文档概率与 token share

设每次先以概率 $p_i$ 选择来源 $i$，再从中抽一条文档；该来源平均产生 $\mu_i$ 个有效 loss token。长期真实 token share 为

$$
q_i
=\frac{p_i\mu_i}
{\sum_j p_j\mu_j}.
$$

只有当各来源都生成等长、等 mask 比例的训练序列时，才近似有 $q_i=p_i$。因此 sampler 配置必须与训练日志中的实际 token 计数同时保存。

若来源 $i$ 有 $N_i$ 个可用 token，整个训练消费 $T$ 个有效 token，则平均重复暴露约为

$$
E_i=\frac{Tq_i}{N_i}.
$$

$E_i=1$ 不是严格的“每条样本恰好一次”：有放回采样、长度窗口和动态过滤会让样本暴露不均。它仍是发现小来源被过度重复的重要审计量。

### 温度采样

以来源规模 $n_i$ 构造

$$
p_i
=\frac{n_i^\alpha}
{\sum_j n_j^\alpha},
\qquad 0<\alpha\le 1.
$$

$\alpha<1$ 会提高小来源的文档概率，却不保证目标 token share；还需要代入 $\mu_i$ 并检查 $E_i$。温度只是平滑规模差异，不会自动识别数据质量和任务价值。

### 动态课程

阶段 $s$ 的分布可写成

$$
p_i^{(s)}=p_i(\tau_s,\mathcal M_s),
$$

其中 $\tau_s$ 是训练 token 进度，$\mathcal M_s$ 是截至该阶段的损失、质量与覆盖测量。课程必须预先定义切换依据和最大重复暴露，不能在看到 benchmark 结果后任意调整并仍把测试集称为独立评测。

## 配比机制

### 受控基线

先建立三种基线：

1. 按有效 token 规模比例；
2. 统一来源概率；
3. 一个有明确目标和重复上限的人工配比。

任何自适应方法都应与相同模型、token 预算和优化配方下的这些基线比较。

### 验证损失驱动

[DoReMi](https://arxiv.org/abs/2305.10429) 使用小代理模型和相对参考损失学习领域权重，目标是改善最差领域的相对表现。它依赖领域划分、参考模型、代理到目标规模的迁移和 held-out 数据，不应被简化为“按 loss 越大采得越多”。

### 代理实验搜索

[RegMix](https://arxiv.org/abs/2407.01492) 在较小规模上训练多种 mixture，并用回归预测更大训练的候选配比。它把配比当成实验设计问题，但代理规模、目标指标和候选空间仍决定外推质量。数据处理或模型架构变化后，旧回归不再自动有效。

### 阶段式退火

后期提高高质量、目标域或长上下文数据的权重，可以改变训练末端分布；同时也可能：

- 过度重复小集合；
- 降低语言和领域覆盖；
- 把格式偏好误当能力提升；
- 与学习率 decay、长度增长同时变化，导致无法归因。

退火阶段应保留通用 replay，并用独立消融分开数据切换与优化器日程。

## 执行契约

每个训练区间记录：

```text
mixture version and stage
sampling unit and replacement rule
nominal document probabilities
observed documents / raw tokens / loss tokens
truncation and mask ratios
unique coverage and repeated exposure
per-source loss and evaluation slices
RNG, shard order and data cursor
```

数据源耗尽、解析失败或动态过滤不得静默回退到其他来源。任何实际配比偏离都要显式计数和告警。

## 正确性与失效

- **用样本数报告配比**：长文档或长回答来源的有效 token 权重被隐藏。
- **重复度只按 epoch 计算**：有放回采样和窗口切分下，epoch 不能描述每条内容的暴露。
- **自适应权重追逐噪声**：小验证集的随机波动被放大为大幅配比变化。
- **领域标签过粗**：同一来源内部的语言、时间和质量差异被平均。
- **代理模型结论直接外推**：规模、tokenizer 或优化配方改变后最优混合可能迁移。
- **训练与评测共用反馈**：反复根据测试分数调 mixture 会耗尽测试集独立性。
- **课程变量同时改变**：数据、长度和学习率一起切换，收益无法归因。

## 何时使用简单配比

来源数量少、规模接近、目标与训练分布一致且重复暴露很低时，按有效 token 比例可能是最可靠的起点。数据或预算不足以支持多次代理训练时，也应优先使用透明基线与小范围消融，而不是引入无法验证的自适应 sampler。

## 验证

1. 对每个来源同时画 nominal $p_i$、observed document share、raw token share 和 loss-token share。
2. 报告 $E_i$、unique document coverage 与 cluster-level 重复度。
3. 用固定 held-out slice 比较每个阶段前后的收益和遗忘。
4. 在相同总 token、长度分布、学习率和模型初始化下做 mixture 消融。
5. 恢复 checkpoint 后验证 data cursor、累计 token 和下一批来源完全连续。
6. 对低资源来源抽查是否因过滤器或 tokenizer 膨胀而被系统性削弱。

从文档到有效 token 的细节见[序列构造与打包](sequence-construction.md)，训练预算与规模外推见[规模律与实验设计](../training/scaling-experiment-design.md)。

## Reference {#reference}

- [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)
- [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492)
