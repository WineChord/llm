# Scaling Laws 与 Chinchilla：固定算力该怎样分

Scaling Laws 和 Chinchilla 经常被压成两条口号：“loss 遵循幂律”和“参数与 token 应同比增加”。更有价值的读法，是把它们看成同一个实验问题的两次建模：给定有限计算，怎样用一组不同大小、不同数据量的训练运行估计 loss 曲面，再从曲面选择资源分配。[Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)建立早期系统规律，[Chinchilla 2022](https://arxiv.org/abs/2203.15556)通过新的 isoFLOP 实验改变了 compute-optimal 结论。

## 单轴规律为何有吸引力

Kaplan 等人在固定其他瓶颈的单轴实验中使用

$$
L(X)\approx\left(\frac{X_c}{X}\right)^{\alpha_X},
\qquad X\in\{N,D,C_{\min}\},
$$

描述非 embedding 参数、数据或 compute-efficient 计算扩大时的 loss。取对数后：

$$
\log L=\alpha_X\log X_c-\alpha_X\log X,
$$

即可做线性拟合。

```python
import torch
def fit_power(x, loss):
    design = torch.stack([torch.ones_like(x), -x.log()], dim=1)
    intercept, alpha = torch.linalg.lstsq(design, loss.log()).solution
    x_c = (intercept / alpha).exp()
    return x_c, alpha
x = torch.logspace(2, 6, 12, dtype=torch.float64)
true_x_c, true_alpha = 4e3, 0.27
loss = (true_x_c / x).pow(true_alpha)
x_c, alpha = fit_power(x, loss)
assert torch.allclose(x_c, torch.tensor(true_x_c, dtype=x_c.dtype), atol=1e-8)
assert torch.allclose(alpha, torch.tensor(true_alpha, dtype=alpha.dtype), atol=1e-8)
```

论文也指出 loss 最终会受非零数据熵限制而趋平，但没有把 $L_\infty$ 加入这些单轴拟合。若后续研究采用 $L_\infty+A X^{-\alpha}$，应把它明确标成扩展模型，并拟合或独立估计 $L_\infty$。真实实验也不能只看一条无噪声曲线：参数相关、训练未收敛、早停和小模型异常都会影响指数；应报告 bootstrap 或 profile likelihood 区间，并保留 holdout 运行检验外推。

## Kaplan 结论为何鼓励更大模型

Kaplan 等人的实验与建模认为，在其设置下参数扩展带来的收益较强，compute-efficient training 倾向训练更大模型但不到完全收敛。这一结果与 GPT-3 式规模扩张相互强化：模型参数被视为优先投入方向。

这里的“compute-efficient”是训练目标。模型一旦需要长期服务，参数带来的推理成本并未进入同一优化问题。

## Chinchilla 怎样重新测量

Chinchilla 用更多模型尺寸和训练 token 组合研究

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

在 dense Transformer 常用近似 $C\approx6ND$ 下，固定 $C$ 意味着参数与数据互相制约。下面直接在一条固定计算曲线上寻找最优点：

```python
def predicted_loss(n, d, e=1.5, a=120.0, b=80.0, alpha=.45, beta=.35):
    return e + a / n**alpha + b / d**beta
compute = 6e14
n = torch.logspace(6, 11, 1000, dtype=torch.float64)
d = compute / (6 * n)
curve = predicted_loss(n, d)
i = int(curve.argmin())
assert 0 < i < len(n) - 1
assert curve[i] < curve[0] and curve[i] < curve[-1]
assert torch.allclose(6 * n[i] * d[i], torch.tensor(compute, dtype=n.dtype))
```

这个最优点完全由示例系数决定。论文中的“约 20 token/parameter”是拟合与实验范围的摘要，不应硬编码进所有训练计划。

## 70B / 1.4T 说明什么

Chinchilla 模型用约 70B 参数和 1.4T token，在报告的任务中与显著更大但训练 token 较少的模型竞争。它支持“许多当时模型在固定训练算力下数据不足”的判断，不支持以下外推：

- 任何 70B 模型都能达到相同质量；
- token 数可以忽略数据质量和重复率；
- MoE 的总参数可直接代入 dense $6ND$；
- 后训练、多模态 token 与预训练文本 token 完全同质；
- 生命周期推理成本不会改变最优点。

## 为什么今天仍需重做曲线

现代配方改变了测量条件：更好的数据过滤、重复训练、长上下文阶段、MoE、低精度和多 token prediction 都会移动 loss 曲面。小规模 proxy 的 tokenizer、depth/width ratio 与优化器若不同，也可能不能外推到目标模型。

一个可审计 scaling study 至少需要：

1. 预注册 $N,D,C$ 的精确定义；
2. 多个独立 seed 与 holdout 规模点；
3. 相同数据分布、架构族和收敛判据；
4. 训练 loss、下游能力与生命周期成本分开拟合；
5. 置信区间和外推距离；
6. 对失败或异常运行不做选择性删除。

## Reference {#reference}

- [Scaling Laws 论文](https://arxiv.org/abs/2001.08361)；
- [Chinchilla 论文](https://arxiv.org/abs/2203.15556)与 [DeepMind 官方说明](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)。

Chinchilla 没有公开一套可完整重建论文训练运行的官方训练仓库；第三方拟合脚本不应被写成官方实现。实验方法见[缩放实验设计](../../training/scaling-experiment-design.md)，训练数字口径见[训练 token](../training-tokens.md)，更长的因果脉络见[从规模规律到上下文内适应](../lineages/scaling-and-context.md)。
