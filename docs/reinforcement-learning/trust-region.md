# Trust Region 与 TRPO：在策略分布中控制一步走多远

Policy gradient 给出局部上升方向，却没有回答一次更新可以走多远。对监督学习，参数稍有变化通常仍在同一固定数据分布上评估；对强化学习，策略变化还会改变下一批状态与动作的分布。一次过大的更新可能让旧轨迹中的 advantage 立刻失去代表性，使训练在“看似沿梯度上升”时反而退化。

Trust region 的核心不是某个固定 clip 区间，而是：只在旧策略附近信任由旧数据构造的局部模型，并用策略分布的距离而非参数的欧氏距离定义“附近”。本页从 performance-difference identity 走到 natural gradient 与 Trust Region Policy Optimization（TRPO），再解释 Fisher、conjugate gradient 和 line search 各自承担什么。策略梯度起点见 [Policy Gradient](policy-gradient.md)，advantage 的估计见 [Advantage 估计与 GAE](advantage-estimation-gae.md)，一阶近似路线见 [PPO](trust-region-ppo.md)。

## 为什么局部策略改进会失真

考虑无限时域折扣目标

$$
J(\pi)
=
\mathbb E_\pi
\left[
\sum_{t=0}^{\infty}\gamma^t r_t
\right].
$$

定义归一化折扣状态访问分布

$$
d^\pi(s)
=(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t
\Pr_\pi(S_t=s).
$$

对旧策略 $\pi$ 与候选新策略 $\pi'$，performance-difference identity 为

$$
J(\pi')-J(\pi)
=
\frac{1}{1-\gamma}
\mathbb E_{
s\sim d^{\pi'},\,
a\sim\pi'
}
\left[
A^\pi(s,a)
\right].
$$

这个等式本身是精确的，但无法直接成为一个只使用旧 rollout 的算法：右侧需要新策略诱导的状态分布 $d^{\pi'}$。若 $\pi'$ 与 $\pi$ 足够接近，可以暂时把 $d^{\pi'}$ 替换成旧分布 $d^\pi$，得到局部 surrogate

$$
L_\pi(\pi')
=
J(\pi)
+\frac{1}{1-\gamma}
\mathbb E_{
s\sim d^\pi,\,
a\sim\pi'
}
\left[
A^\pi(s,a)
\right].
$$

再用旧策略动作样本重写：

$$
L_\pi(\pi')
=
J(\pi)
+\frac{1}{1-\gamma}
\mathbb E_{
s\sim d^\pi,\,
a\sim\pi
}
\left[
\frac{\pi'(a\mid s)}
{\pi(a\mid s)}
A^\pi(s,a)
\right].
$$

importance ratio 修正了同一旧状态上动作分布的变化，却没有修正状态访问分布已经改变这一事实。因此“大幅更新后 ratio 仍可计算”不等于旧数据仍可靠。Trust region 要控制的正是这个局部近似的有效范围。

## 参数距离不是策略距离

直接约束 $\lVert\theta'-\theta\rVert_2$ 很诱人，却不具有良好的参数化不变性。同一个概率分布可以通过不同网络参数化表示；某些参数方向变化很大但几乎不改变输出，另一些很小的参数变化却会让低概率动作暴涨。

KL divergence 直接比较动作分布。令 $\theta'=\theta+\Delta\theta$，在旧参数附近，

$$
\mathbb E_{s\sim d^{\pi_\theta}}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid s)
\;\|\;
\pi_{\theta+\Delta\theta}(\cdot\mid s)
\right)
\right]
\approx
\frac12
\Delta\theta^\top
F(\theta)
\Delta\theta,
$$

其中

$$
F(\theta)
=
\mathbb E_{
s\sim d^{\pi_\theta},\,
a\sim\pi_\theta
}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)^\top
\right]
$$

是策略分布对应的 Fisher information matrix。在正则条件下，它也等于旧策略处 KL Hessian 的相应形式。

普通梯度把参数空间视为各方向尺度相同；natural gradient 则用 $F$ 作为局部度量：

$$
\widetilde\nabla_\theta J
=
F^{-1}g,
\qquad
g=\nabla_\theta J.
$$

可以从一个局部约束问题直接得到这个方向。在线性化 surrogate、二次近似 KL 后，

$$
\begin{aligned}
\max_{\Delta\theta}\quad
&g^\top\Delta\theta,\\
\text{s.t.}\quad
&\frac12\Delta\theta^\top F\Delta\theta\le\delta.
\end{aligned}
$$

其解沿 $F^{-1}g$，并按 trust-region 半径缩放：

$$
\Delta\theta^*
=
\sqrt{
\frac{2\delta}
{g^\top F^{-1}g}
}
F^{-1}g.
$$

这说明 natural gradient 与 trust region 不是两个无关技巧：前者给出策略分布几何中的方向，后者决定在局部模型仍可信时能走多远。

## TRPO 优化的是什么 {#trpo}

TRPO 以旧策略 $\pi_{\theta_{\mathrm{old}}}$ 产生数据，构造近似问题

$$
\begin{aligned}
\max_\theta\quad
&
\widehat{\mathbb E}_t
\left[
r_t(\theta)\widehat A_t
\right],\\
\text{s.t.}\quad
&
\widehat{\mathbb E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\mathrm{old}}}(\cdot\mid s_t)
\;\|\;
\pi_\theta(\cdot\mid s_t)
\right)
\right]
\le\delta,
\end{aligned}
$$

其中

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
}.
$$

目标中的 ratio 与约束中的 KL 分工不同：

- ratio 用旧策略动作样本估计候选策略的局部 surrogate；
- KL 限制整个动作分布在已采样状态上的平均变化；
- advantage 表达旧策略下动作相对 baseline 的局部收益；
- 状态分布仍来自旧策略，这正是更新必须保持局部的原因。

TRPO 并不把这个约束问题交给通用二阶优化器精确求解。神经网络参数量太大，无法显式构造或求逆 Fisher。实际算法使用局部线性—二次模型获得候选方向，再用真实网络上的回溯检查纠正近似误差。

## Fisher-vector product 与 Conjugate Gradient

需要的是

$$
x\approx(F+\xi I)^{-1}g,
$$

而不是完整的 $F^{-1}$。这里 $\xi I$ 是 damping，用来改善有限样本 Fisher 的条件数并抑制近零曲率方向。

Conjugate Gradient（CG）只要求能够计算矩阵—向量积

$$
v\mapsto(F+\xi I)v,
$$

不要求显式存储矩阵。自动微分可以先对平均 KL 求梯度，再对该梯度与向量 $v$ 的内积求一次 directional derivative，从而得到 Hessian-vector product。内存主要由网络前反向图决定，而不是一个参数平方规模的矩阵。

CG 迭代在 Krylov 子空间中逐步逼近线性系统解。实践中要记录：

- 初始与最终 residual norm；
- 实际迭代次数和停止阈值；
- damping 系数；
- 曲率项 $x^\top Fx$ 是否为正且数值稳定；
- 分布式 worker 是否对同一批状态、mask 与 KL reduction 求全局平均；
- 混合精度是否让小曲率方向下溢。

CG 收敛并不等于策略更新可接受。它只解了局部二次模型中的线性系统；advantage 有噪声、Fisher 来自有限 batch、网络又是非线性的，候选完整步长仍可能越出真实 trust region。

## 为什么还需要 Line Search

得到方向 $x$ 后，先按二次模型把它缩放为完整候选步长。随后对系数 $\alpha\in\{1,\beta,\beta^2,\ldots\}$ 做 backtracking：

$$
\theta_{\mathrm{candidate}}
=
\theta_{\mathrm{old}}
+\alpha\Delta\theta.
$$

一个候选通常至少要同时满足：

1. 在当前 rollout 上，实际 surrogate 没有下降；
2. 重新前向计算得到的平均 KL 不超过阈值；
3. loss、KL、log-probability 与参数均为有限数；
4. 改善幅度相对局部模型预测不是异常地小。

若候选失败，就缩小 $\alpha$；所有候选都失败时，保留旧参数。Line search 的角色不是装饰性的学习率调度，而是把“Fisher 二次近似预测可行”重新交给真实模型检查。

这里还存在一个容易忽略的状态问题：每次候选评估必须从完全相同的旧参数出发，不能在失败候选上继续累积更新。优化器动量、随机层、归一化状态和混合精度 scaler 若参与候选评估，也要保证回滚语义明确。

## 理论保证到底保证了什么

TRPO 的动机来自一类策略改进下界：真实性能差可以由旧策略下的 surrogate 与一个策略分布偏离惩罚共同控制。这个结果说明，只要新旧策略在所有相关状态上足够接近，局部 advantage 改善就能压过状态分布漂移带来的误差。

但论文中的理论对象与常见实现之间有几层距离：

- 理论界通常涉及对状态的最大 divergence，而实现常约束样本上的平均 KL；
- 真实 $A^{\pi_{\mathrm{old}}}$ 被有限 rollout、bootstrap 与 learned critic 近似；
- $d^{\pi_{\mathrm{old}}}$ 只由当前采样状态覆盖，罕见状态上的大变化可能不可见；
- Fisher 与 surrogate 都以有限 batch 估计；
- CG 只近似求解，line search 也只测试有限个步长；
- 神经网络、共享 actor–critic 表征和非平稳环境会进一步改变局部模型；
- 约束一次 policy update，不会自动解决 reward misspecification、探索不足或 verifier 漏洞。

因此“TRPO 有单调改进保证”必须带上前提。工程实现更准确的承诺是：它显式测量并拒绝一部分过大的分布变化，而不是无条件保证每轮真实任务回报上升。

平均 KL 也会掩盖尾部。大量几乎不变的 token 或状态可以稀释少数剧烈漂移的位置。可靠报告应同时包含 KL 分位数、每个任务或长度 slice 的 KL、importance ratio 极值与有效样本覆盖，而不是只给一个 batch mean。

## 与 PPO 的关系：共同动机，不同约束

[PPO](trust-region-ppo.md)保留旧策略 ratio 与局部更新思想，却不再通过 Fisher、CG 和 line search 解显式 KL 约束。PPO-Clip 使用

$$
\min
\left(
r_t\widehat A_t,\,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
$$

构造悲观 surrogate，使某些“继续朝有利方向增大 ratio”的样本在越界后不再提供额外收益。它更适合 minibatch 与一阶优化器，也更容易在大型网络上实现。

但 PPO clip 不等价于 trust-region constraint：

- clip 直接作用于采样动作的 ratio，不检查完整动作分布；
- 不同 advantage 符号对应不同的饱和侧；
- 未采样动作仍可能发生显著变化；
- 多个 minibatch epoch 会让策略持续远离产生数据的 old policy；
- $\epsilon$ 不能直接翻译成固定 KL 半径。

因此 PPO 实现仍常监控 approximate KL、clip fraction、ratio 分位数并做 KL early stopping。TRPO 与 PPO 的关系应理解为“显式受约束的二阶局部优化”与“可扩展的一阶悲观 surrogate”，而不是“后者严格实现了前者”。

## 语言模型中的 trust region

语言模型把单个环境动作展开成大量 token 后，策略距离出现多种粒度。对 response $y=(y_1,\ldots,y_T)$，

$$
\log
\frac{\pi_\theta(y\mid x)}
{\pi_{\mathrm{old}}(y\mid x)}
=
\sum_{t=1}^{T}
\log
\frac{
\pi_\theta(y_t\mid x,y_{<t})
}{
\pi_{\mathrm{old}}(y_t\mid x,y_{<t})
}.
$$

逐 token 平均 KL、逐序列 KL 总和与长度归一化 sequence log-ratio 不是同一个约束。长回答会在总和中自然积累更多偏离；按 token 平均又可能掩盖少量关键 action span 的剧烈变化。工具 Agent 还必须排除 prompt、system token、observation 与 padding，只在策略实际选择的 action 上定义 ratio 和 loss。对应的数据字段见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

LLM 后训练至少要分开四个身份：

- $\pi_{\mathrm{old}}^{\mathrm{train}}$ 是训练引擎上冻结的旧策略，定义一次 update 的局部坐标；
- $\mu^{\mathrm{rollout}}$ 是采样引擎、解码规则与 checkpoint 共同形成的真实 behavior distribution；
- $\pi_{\mathrm{ref}}$ 定义长期行为先验或 KL 正则锚点；
- $\pi_\theta$ 是正在更新的 learner。

标准 on-policy TRPO 假定 $\mu^{\mathrm{rollout}}=\pi_{\mathrm{old}}^{\mathrm{train}}$。在异步队列、量化推理、MoE routing 或 sampling processor 存在时，这个等式必须验证，不能由 checkpoint ID 推断。

相对 reference 的 KL 可以限制模型远离初始能力或风格，但不能证明当前 rollout 对 learner 仍是近似 on-policy；相对 old 的小步更新也不能保证长期不偏离 reference。两类 KL 的完整区分见[KL 正则化控制](kl-regularized-control.md)。异步 rollout 使 behavior policy 更旧时，还需进一步检查 support、policy lag 与[Off-policy 校正](off-policy-correction.md)。

## 实现与诊断契约

一个可审计的 TRPO 实现至少应固定：

```text
old policy revision and exact rollout log-probability
advantage/value snapshot and terminal semantics
action mask and KL reduction granularity
KL direction, target radius and damping
Fisher-vector product convention
CG tolerance, maximum iterations and residual
step scaling and backtracking schedule
surrogate-improvement and KL acceptance rules
distributed averaging and numerical precision
```

最小验证包括：

1. 新旧策略相同时，ratio 为 $1$、KL 为 $0$；
2. Fisher-vector product 与小模型显式 Hessian 乘积一致；
3. CG residual 随迭代下降，且 $x^\top Fx$ 非负；
4. 完整候选超过 KL 阈值时，line search 会缩步或拒绝；
5. 失败候选不会污染下一次候选参数；
6. 保持 action-state 输入不变、只修改被 mask 的非动作位置 log-prob 张量时，action-only KL 不变；
7. 平均 KL、最大 slice KL、ratio 分位数与实际 return 同时报告；
8. 固定 rollout 数、环境步数、训练 token 与 wall-clock 后再比较 PPO。

Trust region 最重要的洞见不是“二阶方法一定更好”，而是任何由旧数据驱动的策略更新都有一个可信邻域。TRPO 把这个邻域显式写进优化问题；PPO、KL early stopping、ratio gate 与异步样本筛选则用不同近似管理同一个矛盾。理解各方法测量的距离、丢弃的信息和接受规则，比记住算法缩写更能迁移到新的语言模型训练配方。

## Reference {#reference}

- Kakade, [A Natural Policy Gradient](https://proceedings.neurips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
- Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Achiam et al., [Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)
