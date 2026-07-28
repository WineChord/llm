# 多步回报、资格迹与 GAE

一步 TD 每看到一个 transition 就 bootstrap，Monte Carlo 则等待完整回报。多步方法在两者之间连续插值：向前多看几步可减少对当前 value estimate 的依赖，却也会引入更多采样噪声。理解这条轴，才能分清 n-step return、TD($\lambda$)、eligibility trace 与 GAE 各自在做什么。

本文沿用 transition $(s_t,a_t,r_t,s_{t+1})$，重点保留 n-step、$\lambda$-return 与 eligibility trace 的历史连接。GAE 的完整推导、双 mask、actor/critic target 与语言模型时间轴已独立到 [Advantage 估计与 GAE](advantage-estimation-gae.md)。先读[价值函数与 Bellman 递推](values-bellman.md)和 [Monte Carlo、TD 与控制](prediction-control.md)，会更容易看清 bootstrap target 从何而来。

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

下面直接消费每个 transition 自己的 `next_value`。真正 terminal 丢弃它；truncation 或达到 $n$ 步时用它 bootstrap，因此实现不需要读取可能属于下一条轨迹的 `value[t + 1]`。

```python
import torch
@torch.no_grad()
def n_step_return(reward, next_value, terminated, boundary, start, steps, gamma=.99):
    if not all(x.shape == reward.shape for x in (next_value, terminated, boundary)):
        raise ValueError("transition tensors must align")
    if not 0 <= start < reward.numel() or steps <= 0:
        raise ValueError("invalid start or horizon")
    total = reward.new_zeros(())
    discount = 1.
    for offset, t in enumerate(range(start, min(start + steps, reward.numel()))):
        total += discount * reward[t]
        discount *= gamma
        if terminated[t]:
            return total
        if boundary[t] or offset + 1 == steps or t + 1 == reward.numel():
            return total + discount * next_value[t]
    raise RuntimeError("unreachable horizon")
r, nv = torch.tensor([1., 2., 10.]), torch.tensor([3., 4., float("nan")])
terminated = torch.tensor([False, False, True])
boundary = torch.tensor([False, True, True])
torch.testing.assert_close(n_step_return(r, nv, terminated, boundary, 0, 1, .5), torch.tensor(2.5))
torch.testing.assert_close(n_step_return(r, nv, terminated, boundary, 0, 3, .5), torch.tensor(3.))
assert n_step_return(r, nv, terminated, boundary, 2, 2, .5) == 10
```

这段函数返回单个 state 的 target，不负责 batch padding、value snapshot 或 loss reduction。完整 n-step 与 trace 对照见[强化学习手撕实现](../practice/reinforcement-learning.md#n-step-return)；生产数据若没有真实 truncation final observation，应标为缺失，而不是把相邻样本 value 当作替代。

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

资格迹是采样时随参数更新的状态，而不是事后倒序计算一组固定 target。对线性 $V_w(s)=w^\top x(s)$，下面一步同时更新 TD residual、accumulating trace 与权重；`terminated` 只关闭 bootstrap，`boundary` 在本次更新后重置 trace。

```python
import torch
@torch.no_grad()
def td_lambda_step(weight, trace, feature, reward, next_feature,
                   terminated, boundary, gamma=.99, lam=.95, alpha=.1):
    if not (weight.shape == trace.shape == feature.shape == next_feature.shape):
        raise ValueError("linear value tensors must align")
    value = torch.dot(weight, feature)
    next_value = value.new_zeros(()) if terminated else torch.dot(weight, next_feature)
    delta = reward + gamma * next_value - value
    trace = gamma * lam * trace + feature
    weight = weight + alpha * delta * trace
    if boundary:
        trace.zero_()
    return weight, trace, delta
w, e = torch.zeros(2), torch.zeros(2)
w, e, d0 = td_lambda_step(w, e, torch.tensor([1., 0.]), 0.,
                          torch.tensor([0., 1.]), False, False, .9, 1., .1)
w, e, d1 = td_lambda_step(w, e, torch.tensor([0., 1.]), 1.,
                          torch.tensor([float("nan"), float("nan")]), True, True, .9, 1., .1)
torch.testing.assert_close(w, torch.tensor([.09, .1]))
torch.testing.assert_close(e, torch.zeros(2))
assert d0 == 0 and d1 == 1
```

最终第一维得到更新，正是第二步 TD error 沿 trace 回传到早期特征；边界后 trace 已清零。非线性 critic 需要用 autograd 得到 $\nabla_wV_w(s)$，true-online、replacing 与 off-policy trace 还会改变递推，不能只复用这个 accumulating 版本。

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

下面的反向实现计算有限片段的 $\lambda$-return。`bootstrap[t]` 决定 $G_t^{(1)}$ 是否包含下一状态价值，`boundary[t]` 决定更长 return 是否跨过该 transition；两者分开后，time-limit truncation 可以 bootstrap，却不会串入下一条 packed 轨迹。

```python
import torch
@torch.no_grad()
def lambda_return(reward, next_value, bootstrap, boundary, gamma=.99, lam=.95):
    if not all(x.shape == reward.shape for x in (next_value, bootstrap, boundary)):
        raise ValueError("transition tensors must align")
    if reward.ndim != 1 or reward.numel() == 0 or not boundary[-1]:
        raise ValueError("a finite fragment must be non-empty, one-dimensional and boundary-closed")
    if bootstrap.dtype != torch.bool or boundary.dtype != torch.bool:
        raise ValueError("bootstrap and boundary must be boolean")
    if torch.any(~bootstrap & ~boundary):
        raise ValueError("a transition without bootstrap must close its trajectory")
    out = torch.empty_like(reward)
    carry = reward.new_zeros(())
    for t in range(reward.numel() - 1, -1, -1):
        one_step = reward[t] + gamma * next_value[t] if bootstrap[t] else reward[t]
        carry = one_step if boundary[t] else one_step + gamma * lam * (carry - next_value[t])
        out[t] = carry
    return out
r = torch.tensor([1., 2., 10.])
nv = torch.tensor([3., 4., float("nan")])
boot = torch.tensor([True, True, False])
boundary = torch.tensor([False, True, True])
ret = lambda_return(r, nv, boot, boundary, gamma=.5, lam=.8)
torch.testing.assert_close(ret[0], torch.tensor(1 + .5 * ((1 - .8) * 3 + .8 * 4)))
assert ret[1] == 4 and ret[2] == 10 and not ret.requires_grad
for bad_bootstrap, bad_boundary in [
    (torch.tensor([True, False, True]), torch.tensor([False, False, True])),
    (torch.ones_like(boot), torch.tensor([False, True, False])),
    (boot.long(), boundary.long()),
]:
    try:
        lambda_return(r, nv, bad_bootstrap, bad_boundary)
    except ValueError:
        continue
    raise AssertionError("invalid return boundaries must be rejected")
```

输入只包含真实 transition，不含 padding；若使用二维 batch，应沿每条轨迹独立递推或先按 trajectory ID 分段。这里返回的是 value target，不是 policy advantage；后者还需减去同一冻结 snapshot 的当前 value。

## 与 GAE 的接口

GAE 把同一 $\lambda$-return 结构用于 policy-gradient advantage：

$$
\widehat A_t
=\delta_t+\gamma\lambda(1-b_t)\widehat A_{t+1}.
$$

本页到这里为止只建立历史与代数接口：$\lambda$-return 如何连接 forward view 与 eligibility trace。GAE 的 advantage 语义、bootstrap/trace 双边界、actor/critic target，以及 token、turn、segment 三条时间轴见 [Advantage 估计与 GAE](advantage-estimation-gae.md)；packed tensor 实现与断言见[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)。

## 常见误区

1. <strong>$\lambda=1$ 永远等于无偏 Monte Carlo。</strong>只有展开到真正 terminal 才成立；在 truncation 上仍会依赖 bootstrap value。
2. <strong>GAE 越长越准确。</strong>critic bias 会随 residual 传播，采样 variance 与非平稳性也会累积。
3. <strong>反向循环就是 eligibility trace。</strong>batch reverse GAE 只是在固定数据上算 target；在线 trace 还会在采样期间更新参数。
4. <strong>传统 TD($\lambda$) 在线时也与 forward view 严格等价。</strong>一般只有离线版本精确；true-online 方法专门修复了这个边界。
5. <strong>`done` 一个布尔量足够。</strong>合并 terminal 与 time-limit truncation 会分别丢失 bootstrap 和 trace-reset 语义。

## Reference {#reference}

- Sutton, [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- van Seijen and Sutton, [True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)
- van Seijen et al., [True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Farama Foundation, [Handling Time Limits in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
