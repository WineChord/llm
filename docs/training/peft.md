# 参数高效微调

Parameter-Efficient Fine-Tuning（PEFT）冻结大部分基座，只训练少量新增参数或选定参数。它主要减少训练梯度、optimizer state 和多任务 checkpoint 存储，不自动减少基座前向、activation 或推理计算。

## 方法地图

| 方法 | 可训练对象 | 主要优点 | 主要约束 |
| --- | --- | --- | --- |
| Adapter | 层间小型 bottleneck 模块 | 模块边界清楚、可组合 | 额外串行计算 |
| Prefix Tuning | 各层可训练 prefix states | 不改主权重 | 占用上下文/缓存语义 |
| Prompt Tuning | 输入侧 soft tokens | 参数极少 | 小模型或复杂迁移可能受限 |
| LoRA | 线性层低秩增量 | 易训练、可独立存储和合并 | rank 与目标层决定容量 |
| QLoRA | 量化冻结基座 + LoRA | 显著降低微调显存 | 数值、kernel 与导出更复杂 |
| DoRA | 权重方向低秩更新 + magnitude | 分离方向和幅度 | 额外归一化与 merge 状态 |
| AdaLoRA | 动态分配低秩预算 | 低预算下按重要性分配 | 调度和重要性估计更复杂 |

[Adapters](https://arxiv.org/abs/1902.00751)、[Prefix Tuning](https://arxiv.org/abs/2101.00190) 与 [Prompt Tuning](https://arxiv.org/abs/2104.08691) 代表不同插入位置；不存在对所有任务都最优的参数接口。

## LoRA 数学

对线性层

$$
y=xW^\top,
\qquad
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}},
$$

[LoRA](https://arxiv.org/abs/2106.09685) 学习

$$
\Delta W=sBA,
\quad
A\in\mathbb R^{r\times d_{\text{in}}},
\quad
B\in\mathbb R^{d_{\text{out}}\times r},
\quad
s=\frac{\alpha}{r}.
$$

前向为

$$
y
=xW^\top+s(xA^\top)B^\top.
$$

常见初始化让 $A$ 随机、$B=0$，于是初始 $\Delta W=0$，模型输出与 base 精确一致。若两侧都随机初始化，就失去这一不变量。

### LoRA 前向与合并 {#lora-forward-merge-reference}

下面沿用 PyTorch `linear` 的权重约定：$W$ 为 `[out, in]`、$A$ 为 `[rank, in]`、$B$ 为 `[out, rank]`，输入最后一维为 `in`。双分支前向与把 $\Delta W$ 合入 base 的单分支前向应在目标 dtype 的误差范围内一致。

```python
import torch
import torch.nn.functional as F

def lora_linear(x, weight, a, b, alpha):
    scale = alpha / a.shape[0]
    return F.linear(x, weight) + scale * F.linear(F.linear(x, a), b)

def merged_weight(weight, a, b, alpha):
    return weight + (alpha / a.shape[0]) * (b @ a)

torch.manual_seed(0)
x, weight = torch.randn(3, 5), torch.randn(4, 5)
a, b0 = torch.randn(2, 5), torch.zeros(4, 2)
base = F.linear(x, weight)
assert torch.equal(lora_linear(x, weight, a, b0, 8), base)
b = torch.randn(4, 2)
branch = lora_linear(x, weight, a, b, 8)
merged = F.linear(x, merged_weight(weight, a, b, 8))
torch.testing.assert_close(branch, merged, rtol=1e-5, atol=1e-6)
assert merged_weight(weight, a, b, 8).shape == weight.shape
```

零初始化 $B$ 保证初始函数不变，merge 等价保证部署时没有漏加或重复增量。reference 刻意不包含 dropout、fan-in/fan-out 特例、量化打包和 adapter 路由；生产实现必须把这些语义写入 adapter 元数据，并把 merge 状态设计为幂等。

### Merge 与 unmerge

部署前可计算

$$
W_{\text{merged}}
=W+sBA.
$$

merge 后必须禁用 adapter 分支，否则增量被加两次。可靠实现至少保存：

```text
base weight digest
adapter target and orientation
rank, alpha, dropout and dtype
merged / unmerged state
merge precision and output digest
```

重复调用 merge 应被拒绝或保持幂等；unmerge 必须基于同一个未量化 base 与完全相同的 $\Delta W$。低精度中先 merge 再 unmerge 可能因舍入无法恢复原权重，因此最好保留不可变 base。

## QLoRA

[QLoRA](https://arxiv.org/abs/2305.14314) 用 NF4 等 4-bit 表示保存冻结基座，在计算时反量化，并训练 LoRA。其配方还包括 double quantization 与 paged optimizer 等组件。

需要区分：

- 存储权重的位宽；
- 反量化后的 compute dtype；
- LoRA 参数和 optimizer state dtype；
- backward 穿过量化基座的数值路径；
- 导出时目标推理引擎支持的格式。

packed 4-bit 权重通常不能像 FP16 矩阵一样原地执行 $W\leftarrow W+sBA$。若需合并，应在可表达的较高精度中重建 base、加入增量，再按部署方案重新量化并评测。

## DoRA 与动态 rank

[DoRA](https://arxiv.org/abs/2402.09353) 将权重分解为 magnitude 与方向。抽象地写，

$$
V=W+sBA,
\qquad
W'=m\odot\frac{V}{\lVert V\rVert_{\text{axis}}}.
$$

归一化 axis 取决于线性层权重约定，必须在实现中明确；错误 axis 不一定报 shape error，却会改变算法。

[AdaLoRA](https://arxiv.org/abs/2303.10512) 用重要性估计和近似 SVD 表示动态分配 rank 预算，并提供[官方实现](https://github.com/QingruZhang/AdaLoRA)。动态 rank 在低预算设置可能有价值，但调度、重要性估计与额外超参数需要与固定 rank LoRA 在相同预算下比较。

## 目标层与容量

只适配 attention 的 query/value 投影、适配所有 attention/MLP 线性层，或同时训练 embedding、norm 和 head，会得到不同容量。选择应根据：

- 新任务与基座分布差异；
- 是否引入新 token 或新模态接口；
- 目标是风格、格式还是深层知识迁移；
- 单任务质量与多 adapter 存储；
- 部署能否高效切换或 batch 多个 adapter。

rank 不是唯一容量度量。目标层数量、矩阵尺寸和训练 token 一起决定可训练参数与表达空间。

## 数据与版本契约

一个 adapter 的身份至少包括：

```text
base model / tokenizer / template digest
PEFT method and implementation version
exact target modules and weight orientation
rank / alpha / dropout / bias policy
quantization config and compute dtype
trainable parameter list
merge state and export format
training data snapshot and evaluation protocol
```

基座权重即使模型名相同，只要 revision 或量化方式不同，也不能假设 adapter 可互换。

## 正确性与失效

- **初始输出不等于 base**：初始化或 scale 实现错误。
- **目标层名称匹配过宽**：无意适配 head、router 或重复模块。
- **fan-in/fan-out 方向错误**：矩阵形状可能可广播，语义却反转。
- **训练 dropout 在 merge 后仍生效**：训练与部署行为不同。
- **重复 merge**：增量被累计多次。
- **在量化 packed 权重上原地 merge**：数值和布局被破坏。
- **只比较可训练参数**：忽略 activation、基座前向与 adapter 服务开销。
- **多 adapter 叠加不重评**：增量之间可能干扰，线性相加不保证行为可组合。
- **训练模板与服务模板不同**：adapter 学到的条件分布无法复现。

## 何时不应使用 PEFT

若任务需要大幅改变 tokenizer、embedding、模型结构或广泛领域知识，全量微调、持续预训练或新模型可能更合适。若服务只部署一个固定任务且有充足训练资源，全量微调的质量与运行时简单性也可能优于动态 adapter。少量时效事实优先使用检索。

## 验证

1. 初始化时 base 与 PEFT 模型输出逐元素一致。
2. 仅声明的参数 `requires_grad`，optimizer 不包含冻结参数。
3. merge 前的双分支输出与 merge 后单分支输出在目标 dtype 容差内一致。
4. 重复 merge、错误 base、rank 不匹配和量化导出必须显式失败。
5. 对固定 rank、动态 rank、全量微调和不训练 baseline 做相同预算比较。
6. 报告训练显存、step time、checkpoint 大小、服务 latency 和真实吞吐。
7. 对任务、通用能力、长度、语言、校准和安全做分层回归。

SFT 的 mask 与数据权重见[监督微调](supervised-finetuning.md)，方法与压缩边界见[参数高效训练与压缩](peft-compression.md)，最小 LoRA 断言见[训练目标实现](../practice/training-objectives.md)。

## Reference {#reference}

- [Adapters](https://arxiv.org/abs/1902.00751)
- [Prefix Tuning](https://arxiv.org/abs/2101.00190)
- [Prompt Tuning](https://arxiv.org/abs/2104.08691)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [AdaLoRA](https://arxiv.org/abs/2303.10512)
- [QingruZhang/AdaLoRA](https://github.com/QingruZhang/AdaLoRA)
