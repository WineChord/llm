# 数学与算法

Agentic RL 的数学难点不是公式更多，而是轨迹更长、环境更贵、动作语义更复杂，导致方差、off-policy 偏差与信用分配同时放大。

## Policy Gradient

轨迹概率为

$$
p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T}
\pi_\theta(a_t\mid h_t)T(s_{t+1}\mid s_t,a_t).
$$

若环境转移与 $\theta$ 无关，REINFORCE 得到

$$
\nabla_\theta J=
\mathbb E_\tau\left[
\sum_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\left(G_t-b(h_t)\right)
\right].
$$

baseline $b$ 不改变期望，却可降低方差。长轨迹使 $G_t$ 同时包含大量后续噪声，因此 value model、组内相对奖励和 step-level verifier 都在尝试改善优势估计。

## PPO

[PPO](https://arxiv.org/abs/1707.06347)使用概率比

$$
r_t(\theta)=
\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid h_t)}
$$

与 clipped objective

$$
L^{\text{clip}}=
\mathbb E_t\left[
\min\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)\right].
$$

在语言模型中还常加入对 reference policy 的 KL 惩罚。PPO 适合复用 rollout，但需要 critic、旧策略概率和较复杂的批次管理。

## GRPO

[DeepSeekMath](https://arxiv.org/abs/2402.03300)使用同一问题的成组采样，以组内奖励标准化近似优势：

$$
\hat A_i=
\frac{r_i-\mu_G}{\sigma_G+\epsilon}.
$$

它可省去单独的 value model，但不是“无基线”：组均值本身就是相对基线。若同组样本奖励全部相同，优势信号接近零；若奖励稀疏且采样多样性不足，训练会停滞。

用于多步 agent 时还要决定：整条轨迹共享一个 $\hat A_i$，还是用过程奖励分解到 action span。前者简单但信用粗，后者依赖可信的 step verifier。

## 组相对方法的变体

“GRPO-like”往往掩盖了不同的估计量。几个看似次要的选择会改变哪些 prompt、序列和 token 主导梯度：

| 方法 | 主要变化 | 解决的问题 | 新风险 |
| --- | --- | --- | --- |
| Dr. GRPO | 去掉组标准差与逐响应长度归一化 | prompt 难度和长度偏置 | 梯度尺度变化，需要公平重调 |
| DAPO | 非对称 clip、动态采样、token-level loss、软长度惩罚 | 探索、无信号组与截断噪声 | 重采样成本和选择偏差 |
| GSPO | sequence-level ratio 与 clipping | 序列奖励和 token ratio 错配 | 单个异常 token 影响整条序列 |
| SAPO | 连续可微的 ratio gate | hard clip 丢弃全部梯度 | temperature 与长度归一化敏感 |

[Dr. GRPO](https://arxiv.org/abs/2503.20783)提醒：分母也是算法的一部分。[DAPO](https://arxiv.org/abs/2503.14476)则是一组协同 recipe，不应只抽出“dynamic sampling”当作完整方法。

### GSPO

[GSPO](https://arxiv.org/abs/2507.18071)可用序列平均 log-ratio 定义

$$
\rho_i^{\text{seq}}=
\exp\left(
\frac{1}{L_i}\sum_{t=1}^{L_i}
\left[\log\pi_\theta(y_{i,t})-
\log\pi_{\text{old}}(y_{i,t})\right]
\right).
$$

它把 trust signal 与 sequence reward 对齐。多轮 agent 仍需决定“sequence”是一轮、一个 action span，还是完整 episode；三种选择对应不同信用粒度。

### SAPO

[SAPO](https://arxiv.org/abs/2511.20347)以平滑门替代 hard clipping。对 ratio $r$ 和按优势符号选择的 temperature $\tau$，

$$
f(r)=\frac{4}{\tau}\sigma\left(\tau(r-1)\right).
$$

在 $r=1$ 附近保留 on-policy 梯度，偏离较大时连续衰减。它并不自动消除 off-policy 偏差；序列内 ratio 分散程度和正负优势的 temperature 都必须监控。

## Baseline 与过程奖励

### RLOO 与 ReMax

RLOO 用同组其他样本的平均奖励作为 leave-one-out baseline，避免把当前样本同时放入 baseline。若只想额外生成一个确定性基线，[ReMax](https://arxiv.org/abs/2310.10505)使用

$$
\hat A=
R(y_{\text{sample}})-R(y_{\text{greedy}}).
$$

在随机环境中，两次 rollout 应共享可控初始状态；否则环境噪声会污染策略差异。

### 过程奖励

显式 Process Reward Model 为中间步骤打分；[PRIME](https://arxiv.org/abs/2502.01456)尝试从 outcome label 和 policy rollout 学习隐式过程信号。过程奖励能缩短信号路径，也会引入新的可攻击代理目标。

如果 shaping 形如

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s),
$$

在标准条件下可保持最优策略；一般学习得到的“进度分”不具备这一保证。

## DPO 与离线偏好

[DPO](https://arxiv.org/abs/2305.18290)将偏好对直接转为分类式目标：

$$
\mathcal L_{\text{DPO}}=
-\mathbb E\log\sigma\left(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}
\right).
$$

它避免在线 reward model + RL 循环，适合稳定偏好数据；但对 agent 轨迹，整条序列的偏好无法自然指出哪一次工具调用有问题。离线数据还受 behavior policy 覆盖限制。

## 奖励设计

总奖励常写成

$$
R=
w_sR_{\text{success}}+
w_pR_{\text{process}}-
w_cC_{\text{cost}}-
w_vP_{\text{violation}}.
$$

- $R_{\text{success}}$：测试通过、目标状态或精确答案；
- $R_{\text{process}}$：中间证明、合法工具调用或子目标；
- $C_{\text{cost}}$：token、调用数、时间和外部资源；
- $P_{\text{violation}}$：越权、破坏状态或格式违规。

权重会改变最优策略。对成本惩罚过强，agent 可能提前终止；过程奖励过密，agent 可能优化评分器表面特征而非最终结果。

## 信用分配

### 回报折扣

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k.
$$

小 $\gamma$ 偏好近端收益，却可能让早期规划得不到信用；$\gamma\approx1$ 保留最终目标，却增加方差。

### Reward-to-go 与 GAE

GAE 使用 TD 残差

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
\qquad
\hat A_t^{\text{GAE}}=
\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

它在偏差和方差之间调节，但 value function 必须理解工具状态与长历史。

### 层级分解

把高层子目标 $z_k$ 与低层动作分开：

$$
z_k\sim\pi_H(z\mid h),\qquad
a_t\sim\pi_L(a\mid h,z_k).
$$

层级策略缩短信号路径，却新增子目标边界、终止条件和跨层训练问题。

## On-policy 与异步系统

当 rollout worker 使用参数 $\theta_b$，learner 已更新到 $\theta$，数据就变成 off-policy。重要性比

$$
\frac{\pi_\theta(a\mid h)}{\pi_{\theta_b}(a\mid h)}
$$

可校正期望，但长序列上的比率乘积极不稳定。工程上通常限制 policy lag、给轨迹绑定模型版本、降低每批更新次数，或直接丢弃过旧轨迹。

[V-trace](https://arxiv.org/abs/1802.01561)使用截断重要性权重构造 off-policy actor-critic target；[AReaL](https://arxiv.org/abs/2505.24298)讨论大语言模型 RL 的异步执行与 staleness-aware 训练。版本差只能近似表示 lag，实际还要监控 behavior/current policy 的 KL 或 token ratio。

### 单 rollout 异步优化

成组方法要求同一 prompt 的多个 rollout 都完成，长尾 agent episode 会阻塞整组。[SAO](https://arxiv.org/abs/2607.07508)使用每个 prompt 单条轨迹、learned critic 和 stored rollout log-probability；其双侧重要性区间只保留

$$
1-\epsilon_l<
\exp\left(\log\pi_\theta-\log\pi_{\text{rollout}}\right)
<1+\epsilon_h
$$

内的 token。这样可以流式消费轨迹，却增加 critic 成本、冷启动和丢弃 tail token 的偏差。单 rollout 解决的是调度障碍，不自动解决长时信用。

## 选择方法

| 条件 | 更自然的起点 |
| --- | --- |
| 高质量示范充足 | SFT / imitation |
| 稳定成对偏好，环境昂贵 | DPO 类离线优化 |
| 可验证答案、可成组采样 | RLOO / GRPO |
| 组归一化或长度偏置明显 | Dr. GRPO / DAPO 式消融 |
| sequence reward 与 token ratio 不稳定 | GSPO |
| 需要平滑的 token 级 trust gate | SAPO |
| 需要细粒度 value 与多 epoch 复用 | PPO |
| 长轨迹、明确子目标 | turn-level / 层级 RL + 过程验证 |
| 长尾异步 rollout、每状态单次采样 | critic + SAO 式校正 |
| 线上探索风险高 | 离线数据 + 保守部署 |

算法选择必须与[数据与环境](data-environments.md)和[训练系统](training-systems.md)共同设计。
