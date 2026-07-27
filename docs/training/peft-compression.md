# 参数高效训练与压缩

参数高效微调、知识蒸馏和模型压缩经常同时出现，却解决不同约束：

| 路线 | 改变什么 | 主要节省 | 不自动节省 |
| --- | --- | --- | --- |
| PEFT | 只训练少量新增或选定参数 | 梯度、optimizer state、多任务存储 | 冻结基座的前向计算 |
| 蒸馏 | 训练 student 拟合 teacher | 取决于 student 尺寸 | teacher 生成与验证成本 |
| 剪枝 | 删除权重、通道、层或结构 | 参数或可执行计算 | 无 kernel 支持的稀疏开销 |
| 量化 | 降低权重、激活或 KV 精度 | 内存、带宽、部分算力 | 所有算子与端到端延迟 |

本页保留为稳定总览。PEFT 的数学和 merge 契约见[参数高效微调](peft.md)，teacher–student 目标见[知识蒸馏](distillation.md)，部署量化见[加速与量化](../inference/acceleration.md)。

## 参数高效微调

[LoRA](https://arxiv.org/abs/2106.09685) 冻结权重 $W$，学习低秩增量

$$
W'=W+\frac{\alpha}{r}BA.
$$

训练显存节省主要来自不为全部参数保存梯度与 optimizer state。服务仍需读取基座权重；若 adapter 以独立分支执行，还会增加额外 kernel 或调度。目标层、rank、缩放、dropout 和 base checkpoint 都属于 adapter 身份。

[QLoRA](https://arxiv.org/abs/2305.14314) 用 4-bit 表示冻结基座，并通过反量化计算图训练 LoRA。它降低微调显存，不表示所有训练和推理计算都以 4-bit 执行，也不保证部署引擎能直接消费训练时的量化格式。

## 蒸馏

蒸馏可拟合 teacher 的 token 分布、完整序列、偏好或推理轨迹。token-level 目标需要 teacher/student 的词表与 prefix 对齐；sequence-level 方法不要求逐 token 同构，却引入生成和筛选分布。on-policy distillation 进一步在 student 访问的 prefix 上查询 teacher，减少 teacher-forcing 与部署分布的差距。

蒸馏不是自动压缩：若 student 与 teacher 同尺寸，它更像行为迁移；若 teacher 生成成本很高，总项目成本可能超过直接 SFT。详见[知识蒸馏](distillation.md)。

## 剪枝与稀疏

非结构化稀疏只有在硬件、存储格式和 kernel 能跳过零值时才产生实际加速。结构化剪枝删除通道、head、层或更大块，更容易映射到现有 GEMM，却直接改变模型容量和布局。

[Minitron](https://arxiv.org/abs/2407.14679) 研究了结合剪枝与蒸馏压缩模型的配方。它提供一种公开证据，不意味着相同剪枝率能跨架构保持质量。比较需报告：

- 被剪对象与结构；
- 稠密和稀疏 checkpoint 大小；
- 实际 kernel、硬件和 batch；
- 端到端 latency、throughput 和 peak memory；
- 校准、长上下文、低资源语言与安全回归；
- 恢复训练和蒸馏的额外成本。

## 量化

- PTQ 在训练后估计 scale 或重构低精度权重；
- QAT 在训练中模拟量化误差；
- weight-only 主要减少权重带宽；
- weight–activation 量化还要求低精度 GEMM 和累加路径；
- KV 量化影响长上下文内存与 attention 误差。

checkpoint 文件写成“4-bit”不证明运行时使用低比特 kernel。必须声明权重、activation、accumulator、scale metadata、group size 与 fallback 算子，并在真实服务形状上测量。

## 选择路径

1. **需要多个低存储任务版本**：先评估 LoRA 或其他 adapter。
2. **单卡难以容纳训练基座**：评估 QLoRA，但验证数值和导出路径。
3. **需要更小的独立模型**：选择蒸馏，必要时结合结构化剪枝。
4. **模型尺寸合适但部署受带宽限制**：评估 PTQ/QAT 与实际 kernel。
5. **只缺少时效知识或私有文档**：优先检索，不要为少量事实训练 adapter。

## 正确性与失效

- **可训练参数少等于训练便宜**：基座前向、activation 和通信仍在。
- **adapter merge 只做一次矩阵加法**：dtype、scale、方向、重复 merge 和 base 版本都可能错。
- **量化文件小等于推理快**：fallback dequant 和不支持的 shape 可能更慢。
- **蒸馏分数接近等于能力相同**：长尾、校准和安全边界可能退化。
- **剪枝稀疏率等于硬件加速率**：没有执行支持时只增加索引成本。
- **压缩后沿用旧阈值**：概率、reward 和 guard calibration 会漂移。

## 何时不应组合全部方法

PEFT、蒸馏、剪枝和量化各自引入误差与版本状态。若无法建立逐阶段基线和回归，不应一次叠加全部方法后只看最终汇总分数。先找到真实瓶颈，再选择最小改动。

## 验证

1. 对 adapter 验证初始化等价、merge 前后等价和重复 merge 防护。
2. 对蒸馏分开测 teacher、student baseline、蒸馏收益和生成成本。
3. 对剪枝测结构变化、实际稀疏 kernel 和恢复训练成本。
4. 对量化测真实执行 dtype、kernel coverage、质量和端到端服务形状。
5. 每一步都保留未压缩基线，并对任务、语言、长度、校准和安全分层。
6. 组合方法按单步、两两组合和最终组合做消融，定位不可逆退化。

SFT 的 loss 与数据边界见[监督微调](supervised-finetuning.md)，最小训练目标检查见[训练目标实现](../practice/training-objectives.md)。
