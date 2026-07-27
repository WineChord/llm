# 多步回报、资格迹与 GAE

一步 TD 每看到一个 transition 就 bootstrap，Monte Carlo 则等待完整回报。多步方法在两者之间连续插值：向前多看几步可减少对当前 value estimate 的依赖，却也会引入更多采样噪声。理解这条轴，才能分清 n-step return、TD($\lambda$)、eligibility trace 与 GAE 各自在做什么。

本文沿用 transition $(s_t,a_t,r_t,s_{t+1})$，重点保留 n-step、$\lambda$-return 与 eligibility trace 的历史连接。GAE 的完整推导、双 mask、actor/critic target 与语言模型时间轴已独立到[Advantage 估计与 GAE](advantage-estimation-gae.md)。先读[价值函数与 Bellman 递推](values-bellman.md)和[Monte Carlo、TD 与控制](prediction-control.md)，会更容易看清 bootstrap target 从何而来。

## 从一步到 n 步

忽略中途边界时，$n$-step return 为

$$
G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^n V(s_{t+n}).
$$

$n=1$ 是 TD target；若 $n$ 一直延伸到真正 terminal，最后一项消失，就得到 Monte Carlo return。增加 $n$ 通常降低 bootstrap bias、提高采样 variance，但这不是绝对排序：value 是否准确、reward 噪声、horizon 与 mixing speed 都会改变权衡。

终止语义必须进入定义。令

$$
d_t=\mathbf 1[\texttt{terminated}_t],\qquad
b_t=\mathbf 1[\texttt{terminated}_t\lor\texttt{truncated}_t].
$$

- `terminated` 表示任务动力学真正结束，target 不再 bootstrap；
- `truncated` 表示采样窗口或时间上限结束，通常仍从该 transition 的真实 final observation bootstrap；
- 二者都是轨迹边界，因此 trace 不能串进下一条 episode；
- 基础设施故障不是普通 truncation，应排除样本或单独建模。

若第 $j$ 步遇到 truncation，应在该步使用 $V(s_{t+j+1})$ 后停止展开，而不是读取 batch 中物理相邻的下一条轨迹。

## Forward view：$\lambda$-return

设从 $t$ 到边界还有 $H$ 个 transition。有限轨迹的 forward $\lambda$-return 为

$$
G_t^\lambda
=(1-\lambda)\sum_{n=1}^{H-1}\lambda^{n-1}G_t^{(n)}
+\lambda^{H-1}G_t^{(H)}.
$$

最后一个 return 承接剩余权重，所以权重和恰为 $1$。两个端点很重要：

$$
\lambda=0\Rightarrow G_t^\lambda=G_t^{(1)},\qquad
\lambda=1\Rightarrow G_t^\lambda=G_t^{(H)}.
$$

这里的 $H$ 由 terminal 或 truncation 决定。对 truncation，$G_t^{(H)}$ 仍包含边界 final observation 的 bootstrap；对 terminal 则不包含。

定义一步 TD residual

$$
\delta_t
=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

在一条固定 value snapshot 上，forward view 可等价写成

$$
G_t^\lambda-V(s_t)
=\delta_t+\gamma\lambda(1-b_t)
\left[G_{t+1}^\lambda-V(s_{t+1})\right].
$$

展开就是直到边界为止的加权 TD residual。这个等式也是 batch GAE 的来源。

## Backward view：eligibility trace

forward view 看“未来哪些 return 应更新当前状态”；backward view 反过来维护“当前 TD error 应归因给哪些过去特征”。对可微 value function，经典 accumulating trace 为

$$
e_t=\gamma\lambda e_{t-1}+\nabla_w V_w(s_t),
$$

$$
w_{t+1}=w_t+\alpha\delta_t e_t.
$$

在 episode 边界后必须令 $e=0$。Replacing trace 会把重复激活特征截到固定水平，主要用于稀疏二值特征；它不是 accumulating trace 的普遍等价替换。

### 等价边界

“forward view 与 backward view 等价”必须说明是哪一种等价：

1. 固定整条轨迹中的参数、在末尾累计更新时，传统 TD($\lambda$) 与离线 forward view 精确对应；
2. 传统在线 TD($\lambda$) 每步都会改变参数，后续 target 也随之变化，因此一般只在小步长极限下近似原始 forward view；
3. true-online TD($\lambda$) 定义了随时间截断的 online forward view，并以 Dutch trace 和额外修正项保持逐时刻精确等价；该结论首先针对线性函数逼近，不应无条件外推到任意神经网络。

线性 $V_w(s)=w^\top x(s)$ 的常见 true-online 写法为

$$
e_t=\gamma\lambda e_{t-1}
+\left(1-\alpha\gamma\lambda e_{t-1}^\top x_t\right)x_t,
$$

$$
w_{t+1}=w_t+\alpha(\delta_t+V_t-V_{\mathrm{old}})e_t
-\alpha(V_t-V_{\mathrm{old}})x_t.
$$

episode 开始时 $e=0,V_{\mathrm{old}}=0$；每步先用更新前的参数缓存 $V(s_{t+1})$，完成权重更新后再赋给 $V_{\mathrm{old}}$。这不是在普通 trace 上换一个名字。

## 与 GAE 的接口

GAE 把同一 $\lambda$-return 结构用于 policy-gradient advantage：

$$
\widehat A_t
=\delta_t+\gamma\lambda(1-b_t)\widehat A_{t+1}.
$$

本页到这里为止只建立历史与代数接口：$\lambda$-return 如何连接 forward view 与 eligibility trace。GAE 的 advantage 语义、bootstrap/trace 双边界、actor/critic target，以及 token、turn、segment 三条时间轴见[Advantage 估计与 GAE](advantage-estimation-gae.md)；packed tensor 实现与断言见[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)。

## 常见误区

1. **$\lambda=1$ 永远等于无偏 Monte Carlo。** 只有展开到真正 terminal 才成立；在 truncation 上仍会依赖 bootstrap value。
2. **GAE 越长越准确。** critic bias 会随 residual 传播，采样 variance 与非平稳性也会累积。
3. **反向循环就是 eligibility trace。** batch reverse GAE 只是在固定数据上算 target；在线 trace 还会在采样期间更新参数。
4. **传统 TD($\lambda$) 在线时也与 forward view 严格等价。** 一般只有离线版本精确；true-online 方法专门修复了这个边界。
5. **`done` 一个布尔量足够。** 合并 terminal 与 time-limit truncation 会分别丢失 bootstrap 和 trace-reset 语义。

## Reference {#reference}

- Sutton, [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- van Seijen and Sutton, [True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)
- van Seijen et al., [True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Farama Foundation, [Handling Time Limits in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
