# KL 正则化控制：reference policy 到底约束了什么

语言模型 RL 通常不从随机策略开始。预训练或 SFT policy 已经提供流畅性、知识与基本行为；在线优化只希望在目标分布附近改变排序。KL regularization 把这种“不要离开先验太远”的要求写进控制目标。

## 目标

对 prompt $x$、response $y$ 与任务 reward $r(x,y)$，常见目标为

$$
\max_\pi
\mathbb E_{x\sim\mathcal D,\;y\sim\pi(\cdot\mid x)}
\left[
r(x,y)
-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right].
$$

对固定 $x$ 取期望后，第二项就是

$$
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

$\beta$ 的量纲与 reward 尺度绑定。reward 整体乘 $c$ 而 $\beta$ 不变，会改变最优策略；所以 reward normalization、模型版本和 KL coefficient 不能分开记录。

## 闭式最优策略

对每个 $x$，在 $\sum_y\pi(y\mid x)=1$ 下做变分优化，可得

$$
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right),
$$

其中

$$
Z(x)=
\sum_y
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

这条闭式解默认可行策略对 reference policy 绝对连续，即

$$
\pi(y\mid x)>0
\Longrightarrow
\pi_{\mathrm{ref}}(y\mid x)>0.
$$

因此指数倾斜只能在 $\pi_{\mathrm{ref}}$ 的 support 内重新分配概率；reference 概率为零的 response 不会被有限 reward“复活”。原始 softmax LM 通常有全词表 support，但 top-$k$、top-$p$、grammar mask 或其他受限解码会引入零概率，推导和实现都必须使用同一个可行动作集合。

重排后，

$$
r(x,y)
=\beta\log
\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x).
$$

这条关系连接了在线 RLHF 与 [DPO](../training/offline-preference.md)：同一 prompt 下比较 chosen/rejected 时，$\log Z(x)$ 抵消。但闭式关系依赖特定 KL 正则化和 reward/preference 假设；DPO 不是对任意 RL 目标的代数替代。

## Old policy 与 reference policy

PPO 训练中常同时出现三种 policy：

| policy | 角色 |
| --- | --- |
| $\pi_\theta$ | 当前正在更新 |
| $\pi_{\mathrm{old}}$ | 产生 rollout，用于 importance ratio |
| $\pi_{\mathrm{ref}}$ | 长期行为先验，用于 KL regularization |

PPO ratio

$$
\rho_t
=\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\mathrm{old}}(a_t\mid h_t)}
$$

刻画一次 update 相对 behavior policy 的重加权。真正限制更新的是 PPO clipped surrogate、显式 trust region 或其他约束；ratio 本身只是一项统计量。reference KL

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

约束累计训练漂移。两者偶尔从相同权重初始化，也不具备相同语义。

## Sequence KL 与 token KL

自回归分解给出

$$
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\sum_t
\log\frac{\pi(y_t\mid x,y_{<t})}
{\pi_{\mathrm{ref}}(y_t\mid x,y_{<t})}.
$$

在 $y\sim\pi$ 下取期望，sequence forward KL 可分解为策略访问前缀下的 token KL 之和。但工程估计有几种不同对象：

1. **采样 token 的 log-ratio**：对给定 prefix 只读被采样动作，是单动作 Monte Carlo estimator；计算便宜，但有动作采样方差，单样本可为负。在动作确实来自当前 $\pi$ 时，它的条件期望才是该位置的 forward KL；
2. **完整词表 KL**：对给定 prefix 遍历全 vocabulary，可精确计算该位置的条件 KL，没有动作采样方差，但计算和显存代价更高；整条 rollout 的估计仍会随策略访问到的 prefix 变化；
3. **behavior-sampled log-ratio**：若样本来自 $\pi_{\mathrm{old}}\ne\pi_\theta$，一般不是当前策略 KL 的无偏估计；
4. **token mean**：除以 response length 后不再等于 sequence KL，只是长度归一化诊断。

不要把任何平均 log-ratio 都命名为 `kl`。日志应写清采样分布、方向、词表是否完整以及 reduction。

## Reward shaping 与 loss penalty

KL 可以进入 rollout reward：

$$
\tilde r_t
=r_t^{\mathrm{task}}
-\beta
\left[
\log\pi_{\mathrm{old}}(a_t\mid h_t)
-\log\pi_{\mathrm{ref}}(a_t\mid h_t)
\right],
$$

也可以在 learner loss 中对当前 policy 直接加 penalty。两种实现的估计对象、梯度路径和更新时机不同。

若 KL 作为 reward 进入 return，它会影响 critic target 与早期 advantage；若只作为当前 loss penalty，它不会以同样方式传播进 value。组合使用并非一定错误，但必须明确是否重复计费。

## KL 不是安全边界

小平均 KL 不能保证：

- 每个 prompt 的行为都接近 reference；
- tail token 或少数状态没有剧烈变化；
- 不会出现 reward hacking；
- 输出事实性、安全性或工具权限得到保护。

平均 KL 可能被大量普通 token 稀释。至少同时看 prompt-level 分位数、response length、最大 token log-ratio、拒答/格式 slice 与独立任务指标。

## $\beta$ 的自适应

有些系统根据测得 KL 调整 $\beta$：

$$
\beta_{k+1}
=\beta_k
\exp\left(
\eta\,
\operatorname{clip}
\left(
\frac{\widehat D_k-D_{\mathrm{target}}}
{D_{\mathrm{target}}},
-c,c
\right)
\right).
$$

这只是一个示意性的乘法 controller，不是 RLHF 系统统一采用的更新式，更不是理论自动保证。实际实现也可能使用线性比例更新、固定 horizon 或不同 clamp。测量噪声、policy lag 与长短 response 混合都可能让 $\beta$ 振荡。应记录具体公式、controller interval、目标定义、clamp 与 restart 状态。

## 与最大熵 RL 的联系和区别

最大熵目标常写为

$$
\mathbb E_\pi
\left[
\sum_t r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))
\right].
$$

当 reference 在固定有限 support 上均匀时，负 KL 与 entropy 只差常数。若 action mask 或 grammar 让 support 随状态变化，这个“常数”也会随状态变化，不能无条件丢掉。语言模型 reference 又通常高度非均匀，携带语法、知识和风格先验。因此 RLHF 中的 KL 更像 **relative entropy control**，不是单纯鼓励随机性。探索强度仍由 sampling 与 policy entropy 共同决定。

## 验证

1. $\pi=\pi_{\mathrm{ref}}$ 时 sample log-ratio 为零。
2. 共同修改 prompt token 不改变 response-only KL mask。
3. 分别测试 sequence sum 与 token mean，避免长度约定暗中变化。
4. old/ref 权重交换时单元测试必须失败。
5. reward rescale 后重新检查 $\beta$ 与目标 KL。
6. 按 prompt 和长度分层，而不只报告全局 mean。

实现见[训练目标](../practice/training-objectives.md)，偏好闭式关系见[离线偏好优化](../training/offline-preference.md)，PPO 中 old/ref 的区别见[在线 RL](../training/online-rl.md)。

## Reference {#reference}

- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- Stiennon et al., [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Todorov, [Linearly-Solvable Markov Decision Problems](https://proceedings.neurips.cc/paper/2006/hash/d806ca13ca3449af72a1ea5aedbed26a-Abstract.html)
- Peters, Mülling, and Altun, [Relative Entropy Policy Search](https://ojs.aaai.org/index.php/AAAI/article/view/7727)
- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)
