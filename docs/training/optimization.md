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

AdamW 将 weight decay 与自适应梯度更新解耦，见 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。优化器状态通常以 FP32 保存，是训练显存的重要部分。

## 学习率

常见日程包含 warmup、峰值学习率和 cosine/linear decay。峰值与稳定 batch、模型宽深、初始化、数据质量和精度有关。仅从参数量复制学习率配方风险很高。

## 混合精度

- FP16 动态范围较窄，常配 loss scaling。
- BF16 保留与 FP32 相近的 exponent，训练通常更稳，但 mantissa 更短。
- FP8 需要格式选择、缩放因子、amax 历史与 kernel 支持。
- 某些归一化、归约、优化器状态和 residual accumulation 仍需更高精度。

“使用 BF16/FP8”不足以复现实验；必须报告 master weights、梯度、归约、通信、累加和 optimizer state 的精度。

## 初始化与参数化

残差分支随深度累积，初始化 scale、norm 位置和 residual scaling 会共同影响稳定性。[Tensor Programs V](https://arxiv.org/abs/2203.03466) 的 $\mu$P 讨论跨宽度迁移超参数的一种参数化框架，但只有满足其参数化条件时才成立。

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

AdamW、Muon、参数分组和更新尺度的系统比较见[优化器家族](optimizer-families.md)，softmax/交叉熵的数值起点见[概率、损失与梯度](../foundations/probability-objectives.md)。

## Reference {#reference}

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
