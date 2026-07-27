# 预训练

预训练把数据分布压缩进参数。核心问题不只是把 loss 降下来，而是用明确的 token 预算、数据暴露和数值路径得到可解释、可恢复的模型；训练数周后才发现 mask、配比或 checkpoint 错误，通常无法靠继续训练补救。

## 目标函数

decoder-only 模型对 token 序列 $x_{1:T}$ 优化 causal language modeling：

$$
\mathcal L_{\text{CLM}}
=-\frac{1}{\sum_t m_t}
\sum_{t=1}^{T}
m_t\log p_\theta(x_t\mid x_{<t}),
$$

其中 $m_t$ 排除 padding、损坏位置和不应跨越的边界。encoder 或 encoder-decoder 模型还可使用 masked language modeling、span corruption 与 sequence-to-sequence denoising；目标选择改变可见上下文，不能仅靠推理时 prompt 修正。

三种经典选择可分别从 [GPT-1](../landscape/works/generative-pretraining-gpt.md)、[BERT](../landscape/works/bert.md)与 [T5](../landscape/works/t5.md)进入；它们比较的是 causal、masked 与 span-corruption 信息流，不是简单的版本替代。

对 packed sequence，attention mask 和 loss mask 是不同契约：前者决定能看到什么，后者决定哪些 token 产生梯度。文档可见性、EOS 与 position 的选择见[序列构造与打包](../data/sequence-construction.md)。

### Causal loss 的最小实现 {#causal-loss-semantic-reference}

输入 `logits` 的形状为 $[B,T,V]$，`token_ids` 与 `loss_mask` 为 $[B,T]$。函数在内部完成 next-token shift，返回尚未归一化的 loss 总和与有效 token 数；跨 data-parallel rank 时应分别对这两个标量求和，最后只做一次除法。

```python
import torch
import torch.nn.functional as F

def causal_lm_terms(logits, token_ids, loss_mask):
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2]:
        raise ValueError("expected logits [B,T,V] and token IDs [B,T]")
    if loss_mask.shape != token_ids.shape:
        raise ValueError("loss mask must align with token IDs")
    pred, target = logits[:, :-1].contiguous(), token_ids[:, 1:].contiguous()
    valid = loss_mask[:, 1:].bool()
    count = valid.sum()
    if not valid.any():
        return pred[valid].sum(), count
    return F.cross_entropy(pred[valid], target[valid], reduction="sum"), count

def global_token_mean(local_sums, local_counts):
    count = torch.stack(local_counts).sum()
    if count <= 0: raise ValueError("global batch has no target token")
    return torch.stack(local_sums).sum() / count

z = torch.zeros(1, 4, 3).index_fill(1, torch.tensor([2]), float("nan")).requires_grad_()
ids = torch.tensor([[0, 1, 2, -100]])
mask = torch.tensor([[False, True, True, False]])
loss_sum, count = causal_lm_terms(z, ids, mask)
assert count.item() == 2 and torch.isfinite(loss_sum)
(loss_sum / count).backward()
assert torch.isfinite(z.grad).all() and z.grad[:, 2].abs().sum() == 0
mean = global_token_mean([loss_sum, 3 * loss_sum], [count, count])
assert torch.allclose(mean, 2 * loss_sum / count)
empty_sum, empty_count = causal_lm_terms(z, ids, torch.zeros_like(mask))
assert empty_sum == 0 and empty_count == 0
rejected = False
try:
    global_token_mean([empty_sum], [empty_count])
except ValueError:
    rejected = True
assert rejected
```

这里最重要的不变量是 shift 只发生一次、mask 跟随目标 token，并在交叉熵之前完成布尔选择。单个 rank 可以返回 `(0, 0)`，使 ragged data parallel 仍能归约；只有全局 count 为零才拒绝更新。reference 不负责构造跨文档 attention mask，也没有执行真实 collective；生产实现应以 FP32 累积归约量。

## Batch 与有效 token

样本维度的 global batch 为

$$
B_{\text{global}}
=B_{\text{micro}}N_{\text{data}}N_{\text{accum}},
$$

但优化和数据进度更应以每步有效 token 计：

$$
T_{\text{step}}
=\sum_{r=1}^{N_{\text{data}}}
\sum_{b,t}m_{r,b,t}.
$$

动态长度、packing、截断和数据源差异都会让 $T_{\text{step}}$ 波动。若每个 data-parallel rank 先求局部 mean 再平均，短序列 rank 会被过度加权。全局 loss 必须先归约 numerator 和有效 token denominator；具体断言见[训练目标实现](../practice/training-objectives.md)。

梯度累积也应以 token 语义定义。最后一个 microbatch 较短时，简单平均 microbatch loss 会改变该 step 的目标。

## 数据与序列契约

预训练运行至少绑定：

```text
data snapshot and mixture stage
tokenizer and vocabulary digest
document boundary and normalization rules
sequence length distribution
packing and cross-document attention semantics
sampling unit, replacement and RNG
effective loss tokens per source
```

名义来源概率与真实 token share 不同；重复暴露会随阶段变化。计算与审计方法见[数据混合与课程](../data/mixtures-curricula.md)。

[DataComp-LM](https://arxiv.org/abs/2406.11794) 在固定模型和训练预算下比较数据方案，说明“更多原始数据”与“更好的训练数据”不是同一命题。[OLMo](https://arxiv.org/abs/2402.00838) 则展示了把数据、训练代码、模型和评测同时公开对可研究性的价值。

## 训练配方

### 初始化与参数化

深层 residual 路径、norm 位置、权重 scale 和学习率共同决定初始稳定性。[$\mu$P](https://arxiv.org/abs/2203.03466) 提供跨宽度迁移超参数的一种参数化框架，但只有在模型各参数族遵循相应规则时成立；普通参数化下直接复制其缩放结论无效。

### 优化器与日程

常见配方包含 warmup、峰值学习率、稳定区间和 decay。学习率应按 effective token、global batch、参数化与 optimizer 联合调优，不能仅按参数量复制。AdamW、Muon 和参数路由的条件见[优化器家族](optimizer-families.md)。

### 精度

需要分别声明：

- 参数存储与 master weight dtype；
- forward/backward compute dtype；
- reduction 与 residual accumulation dtype；
- gradient communication 与 optimizer state dtype；
- loss scaling、overflow 和随机舍入状态。

“使用 BF16”或“使用 FP8”不足以复现实验。数值异常诊断见[优化与稳定性](optimization.md)。

## 课程与持续预训练

课程可按质量、语言、领域、难度和序列长度改变分布。每次切换都应记录：

- 切换发生的 global token；
- 新旧 nominal 与 observed token share；
- 每个来源的累计重复暴露；
- 学习率或长度是否同时变化；
- 切换前后的 held-out loss 与能力 slice。

若数据、学习率和 context length 同时切换，收益无法归因。后期提高高质量数据权重是一种配方，不是普遍规律；它可能改善目标能力，也可能牺牲覆盖和增加记忆。

持续预训练适合领域适配，但需保留通用 replay，并分开测量领域增益、通用遗忘、词表变化和输出分布漂移。更换 tokenizer 会改变 embedding、序列长度和历史 PPL 的可比性，不能当成普通数据更新。

## 长上下文阶段

增加训练长度同时改变：

- attention FLOPs 与 activation memory；
- global batch 和 optimizer noise；
- position 分布与 RoPE 等位置参数；
- 文档拼接、截断和长文档采样；
- 通信布局与吞吐。

只对位置编码插值，或只在少量长样本上继续训练，不能证明模型会使用远距离证据。应在长度、证据位置、干扰项、检索跨度和生成长度上分层评测，并与相同总 token 的短上下文基线比较。

## Checkpoint 与恢复

严格 resume 至少保存：

```text
model, optimizer, scheduler and scaler
global step and effective tokens
RNG states for model and data workers
data shard, cursor and mixture stage
distributed topology and parameter layout
tokenizer, config and code revision
```

恢复测试不应只看 loss 大致连续。应在同一 checkpoint 分叉：一条不中断继续，另一条保存后恢复，再比较下一批样本、学习率、梯度和参数更新。硬件拓扑变化时，参数与 optimizer shard 的重排也要单独验证。

## 成本与规模

密集 Transformer 训练常用

$$
C_{\text{train}}\approx 6ND
$$

粗估参数相关 FLOPs，其中 $N$ 是参与稠密计算的参数数，$D$ 是训练 token。它忽略 attention 的长度项、embedding、MoE 路由、重计算、低利用率、通信、失败重跑和数据处理，不能替代真实成本账本。边界、compute-optimal 推导与 inference-aware 总成本见[规模律与实验设计](scaling-experiment-design.md)。

## 正确性与失效

- **loss 正常即数据正确**：模板噪声、跨文档 attention 和来源配比错误仍可得到平滑曲线。
- **step 替代 token**：长度课程后，同样 step 数不再代表同样训练量。
- **局部 mean 等权平均**：长度不均时改变全局目标。
- **只恢复权重**：optimizer、RNG 和 data cursor 重置，训练轨迹发生跳变。
- **验证集跟随训练语料更新**：历史 loss 失去可比性。
- **单指标选择 checkpoint**：PPL 改善可能伴随下游、校准或安全回归。
- **以 $6ND$ 宣称实际消耗**：遗漏系统利用率、attention 与失败成本。

## 何时不应从头预训练

当目标只是少量知识更新、私有文档问答、固定工具调用或窄域接口时，检索、SFT、adapter 或持续预训练通常更经济。数据来源不清、评测不独立或预算不足以完成受控消融时，从头预训练很难得到可解释结论。

## 验证组织

1. 单 batch overfit，并逐 token 检查 shift、mask 和文档边界。
2. 单卡高精度建立参考，再逐步打开低精度、数据并行和模型并行。
3. 在 padding、microbatch、rank 划分变化后验证全局 loss 与梯度不变。
4. 报告每个来源的实际 loss token、重复暴露和 held-out loss。
5. 运行 uninterrupted 与 save/resume 分叉测试。
6. 对数据阶段、长度阶段和学习率阶段做独立消融。
7. 同时报告训练 loss、跨域 PPL、目标能力、记忆化、稳定性、吞吐和总成本。

训练规模的实验设计见[规模律与实验设计](scaling-experiment-design.md)，系统并行见[并行训练](../systems/parallelism.md)。

## Reference {#reference}

- [DataComp-LM](https://arxiv.org/abs/2406.11794)
- [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
