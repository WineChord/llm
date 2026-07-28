# 优化与稳定性

大模型训练中的“算法问题”和“系统问题”经常表现为同一条异常 loss 曲线。诊断必须把数据、数值、优化器、并行通信和 checkpoint 状态同时纳入。

## AdamW

Adam 维护一阶与二阶矩估计：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

包含 bias correction 的 AdamW 更新为：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

$$
\theta_{t+1}
=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

weight decay 不进入自适应梯度分母，这正是 AdamW 与把 $L_2$ 项直接加进 Adam gradient 的差别，见 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。bias、norm scale、embedding 和 router 参数是否 decay 属于参数分组契约；优化器状态通常以 FP32 保存，是训练显存的重要部分。

## 学习率

常见日程包含 warmup、峰值学习率和 cosine/linear decay。峰值与稳定 batch、模型宽深、初始化、数据质量和精度有关。仅从参数量复制学习率配方风险很高。

global batch 还必须说明分母是 sequence、非 padding token 还是可训练 token。若各 microbatch 有效 token 数不同，先对每个 microbatch 求均值再累加，会让短样本获得更高的单 token 权重。严格的 token-normalized gradient 应累计 loss sum 与有效 token count，再在 optimizer step 前统一归一化；分布式场景还要对二者使用一致的 collective。

## 混合精度

- FP16 动态范围较窄，常配 loss scaling。
- BF16 保留与 FP32 相近的 exponent，训练通常更稳，但 mantissa 更短。
- FP8 需要格式选择、缩放因子、amax 历史与 kernel 支持。
- 某些归一化、归约、优化器状态和 residual accumulation 仍需更高精度。

“使用 BF16/FP8”不足以复现实验；必须报告 master weights、梯度、归约、通信、累加和 optimizer state 的精度。

自动混合精度的顺序也有语义：FP16 训练应先反缩放梯度，再检查非有限值和做 global-norm clipping；若先 clip 被放大的梯度，阈值已经改变。global norm 必须跨所有参数和分片计算：

$$
\lVert g\rVert_2
=\sqrt{\sum_{r}\sum_{i\in r}g_i^2},
\qquad
g\leftarrow g\min\left(1,\frac{c}{\lVert g\rVert_2+\epsilon}\right).
$$

在 ZeRO/FSDP 下，每个 rank 只看本地 norm 会得到不同缩放因子。数值格式、通信归约和状态分片的完整接口见[精度与数值](../systems/precision-numerics.md)与[集合通信和状态分片](../systems/collectives-sharding.md)。

## 初始化与参数化

残差分支随深度累积，初始化 scale、norm 位置和 residual scaling 会共同影响稳定性。[Tensor Programs V](https://arxiv.org/abs/2203.03466) 的 $\mu$P 讨论跨宽度迁移超参数的一种参数化框架，但只有满足其参数化条件时才成立。

稳定性监控不应只看 loss。每层 activation RMS、gradient RMS、update-to-weight ratio

$$
\rho_\ell=\frac{\lVert\Delta\theta_\ell\rVert_2}
{\lVert\theta_\ell\rVert_2+\epsilon}
$$

能区分“全局学习率过大”和“少数层异常”。若 loss 尚正常但某层 $\rho_\ell$ 持续上升，通常比等待 NaN 更早暴露精度、初始化或数据问题。

## 路由时滞与动态稳定守卫

大规模 MoE 常提前计算下一批 token 的路由，才能预取远端 expert 权重或 activation。若执行时参数已经从 $\theta_{t-\Delta t}$ 更新到 $\theta_t$，route ID 与当前 hidden state / router score 不再完全匹配。

[DeepSeek-V4](../landscape/works/deepseek-v4.md#training-stability)的 Anticipatory Routing 明确接受这项时滞：当前特征仍由 $\theta_t$ 计算，但执行预先由旧参数决定的 expert ID。常态训练不持续付这笔代价；loss-spike detector 触发后，控制器回滚到近期 checkpoint，临时开启 anticipatory path，让数据和 route 更早进入流水，再在稳定后关闭。报告称只有该模式活跃时增加约 20% overhead，而总训练 overhead 可忽略。

这是一项作者系统上的经验恢复机制，不是异步路由的收敛证明。报告没有披露 $\Delta t$、spike threshold、回滚窗口、误报率或独立消融。复现时至少记录：

```text
feature policy version
route policy version and delay
trigger metric / threshold / cooldown
rollback checkpoint and data cursor
activation duration and recovered steps
expert-load and loss trajectory before/after
```

V4 还把 SwiGLU linear branch clamp 到 $[-10,10]$，gate branch 只做上界 10。hard clamp 限制极端 activation，却在阈值外改变梯度；它与 anticipatory routing 的共同作用尚没有完整理论解释。

## 异常定位顺序

1. 固定单 batch，验证前向 loss 与 token mask。
2. 用单卡高精度建立参考。
3. 逐步打开混合精度、数据并行和模型并行。
4. 比较梯度、参数更新和 optimizer state。
5. 检查 resume 前后数据游标、随机数与 scheduler。

不要用降低学习率掩盖错误 mask、坏数据、通信丢失或 checkpoint 不一致。

症状也能缩短排查路径：

| 症状 | 先查什么 | 容易误判成 |
| --- | --- | --- |
| 单步突然 NaN | 当前 batch、loss scale、归约与 kernel | 学习率长期过大 |
| resume 后立即跳变 | optimizer/scheduler、RNG、数据游标 | 新数据分布 |
| 只有某些 rank 发散 | collective、shard ownership、坏设备 | 普通随机波动 |
| loss 缓慢恶化且 norm 平稳 | 数据混合、重复率、标签或 mask | 数值溢出 |
| 吞吐下降但 loss 正常 | shape、路由负载、通信与重算 | 优化器退化 |

AdamW、Muon、参数分组和更新尺度的系统比较见[优化器家族](optimizer-families.md)，softmax/交叉熵的数值起点见[概率、损失与梯度](../foundations/probability-objectives.md)，可重放 checkpoint 见[Checkpoint、韧性与可观测性](../systems/checkpointing.md)。

## Reference {#reference}

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
