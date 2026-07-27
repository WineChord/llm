# 知识蒸馏

知识蒸馏让 student 学习 teacher 的概率分布、生成行为或中间判断。它既可压缩模型，也可把难以直接标注的行为迁移到另一模型；若 student 没有变小，蒸馏仍可能用于改变数据效率或部署接口，而不属于模型压缩。

## 问题

普通 SFT 只看到一个目标 token 或一条示范。teacher 分布还包含“哪些替代 token 也合理”“错误选项有多不合理”等信息，但 token-level 蒸馏要求 teacher 与 student 在同一个词表和 prefix 上比较。生成式任务还面临 exposure gap：训练时总在 teacher/真实前缀上学习，不代表 student 偏离后仍会被纠正。

## Token-level 目标

teacher 与 student 的温度分布为

$$
q_T(v\mid h)
=\operatorname{softmax}\left(\frac{z^{\text{teacher}}(h)}{T}\right)_v,
$$

$$
p_T(v\mid h)
=\operatorname{softmax}\left(\frac{z^{\text{student}}(h)}{T}\right)_v.
$$

对有效位置 mask $m_t$，soft distillation loss 为

$$
\mathcal L_{\text{KD}}
=
\frac{T^2}{\sum_t m_t}
\sum_t m_t
D_{\mathrm{KL}}
\left(q_T(\cdot\mid h_t)\,\|\,p_T(\cdot\mid h_t)\right).
$$

$T>1$ 会软化概率；前面的 $T^2$ 用于抵消 softmax 导数随温度缩小的梯度尺度。它不是可随意省略的装饰。常与 hard-label loss 混合：

$$
\mathcal L
=(1-\lambda)\mathcal L_{\text{hard}}
+\lambda\mathcal L_{\text{KD}}.
$$

$\lambda$、$T$、KL 方向和 token 归一化共同定义目标。只写“使用 KD”不足以复现。

## 数据与计算契约

Token-level KD 需要：

```text
teacher and student exact revisions
shared vocabulary or an explicit token mapping
identical prefix tokens and position semantics
teacher/student chat templates
temperature, KL direction and hard/soft weight
response / padding / action mask
teacher-logit dtype, top-k truncation and storage format
```

teacher 必须停止梯度。若只保存 top-$k$ logits，还需记录剩余概率质量如何处理；把未保存 token 直接设为零会改变 KL。

teacher 和 student tokenizer 不同，或模板产生不同 prefix 时，不应强行按位置对齐 logits。可改用 sequence-level 蒸馏、共享字符/字节空间的辅助目标，或先构造明确映射。

## 蒸馏机制

### Teacher-forced token KD

teacher 与 student 都在同一真实或 teacher 序列前缀上计算分布。它稳定、可离线缓存 logits，却没有覆盖 student 自己犯错后到达的 prefix。

### Sequence distillation

teacher 生成完整回答，student 用 SFT 学习筛选后的序列。它不要求逐 token 词表相同，也能利用工具和 verifier 过滤；代价是 teacher 的分布被单条或少量样本压缩，概率信息与多样性可能丢失。

### On-policy distillation

student 先生成 prefix，再由 teacher 对这些 prefix 给出目标。[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) 研究了这类 student-generated output 上的蒸馏，以缓解 teacher-forcing 与部署分布的差距。

它更接近 student 实际状态，也更昂贵且随训练非平稳。生成样本必须记录 student policy version；teacher 查询失败、截断和拒答不能静默变成普通标签。

### 推理与过程蒸馏

可将 teacher 或搜索产生的答案、步骤、验证结果转成：

- 最终答案 SFT；
- chosen/rejected pair；
- step-level process labels；
- student rollout 上的 teacher correction。

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 报告了把推理行为蒸馏到较小模型的实验。该结果属于具体 teacher、数据和模型族；长推理文本本身不是正确性证明，必须用独立 verifier 和新题评测。搜索到训练的闭环见[推理后训练](reasoning-posttraining.md)。

### 剪枝后的恢复

student 也可以是结构化剪枝后的模型。[Minitron](https://arxiv.org/abs/2407.14679) 结合 pruning 与 distillation 恢复质量。此时应把“结构删除带来的损失”和“蒸馏恢复的收益”分别测量。

## 正确性与失效

- **漏掉 $T^2$**：改变不同温度下 soft loss 的梯度尺度。
- **mask 不一致**：teacher 在 prompt 或 padding 上贡献梯度，student 学到错误目标。
- **teacher/student prefix 不同**：位置对齐的 KL 不再比较同一条件分布。
- **teacher logits 未 detach**：无意中更新 teacher 或浪费图内存。
- **词表不同时硬对齐 token ID**：同一 ID 不代表同一符号。
- **只蒸馏 greedy 序列**：teacher 的替代解与不确定性消失。
- **student 只见 teacher prefix**：部署偏离后的状态未训练。
- **teacher 与评测同源污染**：student 看似迁移能力，实际复制答案。
- **温度和 top-$k$ 缓存未记录**：离线 logits 无法解释或重建。

## 何时不应蒸馏

teacher 在目标域不可靠、teacher 查询成本超过直接标注、两者词表和接口差异过大、数据使用条件不允许派生，或目标是引入 teacher 本身没有的新知识时，不应把蒸馏作为默认方案。若只需教授少量格式，高质量 SFT 往往更透明。

## 验证

1. teacher 与 student logits 完全相同时，soft KL 应接近零。
2. 改变 padding、batch 切分和 mask 后，有效 token 梯度应保持一致。
3. 分别报告 hard loss、soft loss、teacher entropy 与 student–teacher KL。
4. 比较 teacher-forced、sequence 与 on-policy 数据在相同 teacher 查询预算下的收益。
5. 在 teacher 未见的新来源、时间和题族上评测，检查答案污染。
6. 同时报 student baseline、teacher 上限、蒸馏后能力、校准、长度和安全回归。
7. 对 tokenizer、模板、logit 截断和 teacher 版本做恢复测试。

目标函数的最小实现与不变量见[训练目标实现](../practice/training-objectives.md)，合成样本的谱系与筛选见[合成数据](../data/synthetic-data.md)。
