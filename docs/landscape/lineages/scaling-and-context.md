# 从规模规律到上下文内适应

大模型时代最容易被压缩成一句话：“模型越大，能力越强。”真正影响研究路线的并不是这句定性判断，而是三个逐步出现的问题：损失能否随资源平滑预测，固定计算预算应怎样分给参数和数据，以及一个不更新参数的模型为什么能从上下文示例中临时改变行为。这三件事互相连接，却分别属于经验规律、资源配置和行为机制。

## 当规模从经验变成可拟合对象

早期神经语言模型已经观察到扩大数据与模型会改善困惑度，但训练昂贵时，研究者更需要知道“再增加十倍资源大概会怎样”。[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)在受控实验中分别用幂律拟合 loss 与非 embedding 参数、数据和 compute-efficient 计算的关系；把相应自变量统记为 $X$，其单轴形式可写成：

$$
L(X)\approx\left(\frac{X_c}{X}\right)^{\alpha_X},
\qquad X\in\{N,D,C_{\min}\}.
$$

论文也指出这些趋势最终会受非零数据熵限制而趋平，但没有把不可约项直接加入上述单轴拟合。真正的转折，是把规模扩张从单个模型发布变成可外推的实验曲线。它也埋下了风险：幂律只在观测范围、数据分布和训练配方内是经验模型；换 tokenizer、数据质量、架构或目标，常数与指数都可能变化。

[Scaling Laws 与 Chinchilla 深读](../works/scaling-laws-chinchilla.md)会展开拟合、固定算力优化和不确定性。机制与实验设计分别见[缩放与计算](../../foundations/scaling.md)和[缩放实验设计](../../training/scaling-experiment-design.md)。

## GPT-3：不更新参数也能临时适应

[GPT-3](https://arxiv.org/abs/2005.14165)把规模曲线与 few-shot prompting 放在同一组实验中。给定示例

$$
(x_1,y_1),\ldots,(x_k,y_k),x_q,
$$

模型直接计算

$$
p_\theta(y_q\mid x_1,y_1,\ldots,x_k,y_k,x_q),
\qquad \theta\ \text{不变}.
$$

这不是普通微调：示例只进入激活和 KV 状态，不进入参数。GPT-3 在大规模通用语言模型上系统展示了 zero-shot、one-shot 与 few-shot prompting，使自然语言说明和示例成为表达任务的主要接口；prompt 格式、示例顺序、label 语义和上下文预算也由此成为模型行为的一部分。

论文展示的是广泛行为证据，不是 ICL 的唯一机制证明。规模增大与 ICL 改善相关，不能直接推出模型内部执行了梯度下降、贝叶斯更新或某一种固定算法。不同数据与任务可能借用不同电路。

## Chinchilla：更大参数不一定是更好的算力使用

[Chinchilla](https://arxiv.org/abs/2203.15556)重新研究固定计算下参数 $N$ 与训练 token $D$ 的分配，用

$$
L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
$$

拟合 under-parameterized 与 under-trained 两侧。对 dense Transformer，常用近似

$$
C\approx 6ND
$$

把二者放进同一计算约束。其结论推动后来模型使用更多数据训练相对较小的参数规模，但“约 20 token/parameter”只是在论文设置中的便捷概括，不是跨数据质量、重复率、MoE、持续预训练和生命周期成本的自然常数。

训练最优也不等于部署最优。若一个 checkpoint 将服务海量请求，总参数更少可能大幅降低长期推理成本；若只训练一次、调用很少，另一种分配可能更合理。资源目标应写成

$$
C_{\text{life}}
=C_{\text{train}}
+n_{\text{request}}C_{\text{inference}}
+C_{\text{storage/operation}},
$$

而不是只优化一次预训练 FLOPs。

## Induction heads：从行为曲线走向局部机制

[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/)在小型受控 Transformer 中研究一种复制模式：若上下文出现 $[A][B]$，后来再见到 $[A]$ 时，attention head 倾向预测 $[B]$。它给出了 ICL 机制的具体候选，也展示了某些电路在训练中出现时 loss 曲线怎样变化。

这类证据很重要，但边界同样重要：

- 在所研究的小型模型和任务中，定位、消融与训练动力学证据支持存在可识别的 induction mechanism；
- 它不证明所有大模型 ICL 都由单一 induction head 完成；
- 语言、回归、分类和长程任务可能需要不同的组合电路；
- 行为上相同的 few-shot 提升，内部原因可能不同。

因此，ICL 应同时保留行为协议与机制假设。行为实验见[上下文学习](../../foundations/in-context-learning.md)，内部归因还需要干预而不只相关性观察。

## 数据质量把缩放曲线变成移动目标

当高质量数据有限，重复训练、合成数据和课程会改变 $D$ 的含义。[Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)研究重复数据条件下的缩放，[DoReMi](https://arxiv.org/abs/2305.10429)与后续数据混合工作则说明 token 来源比例会影响给定预算的收益。

一万亿 token 不是同质质量单位。代码、数学、多语言、重复网页和合成推理轨迹对 loss 与下游能力的贡献不同。模型之间比较训练量时，必须回到[训练 token 口径](../training-tokens.md)和[数据混合与课程](../../data/mixtures-curricula.md)。

## 这条谱系留下的阅读方法

遇到新的 scaling 或 test-time scaling 结论，可以依次问：

1. 自变量是参数、有效 token、训练 FLOPs、采样 token，还是 wall-clock；
2. loss 或能力指标在哪个范围拟合，是否存在不可约项；
3. 数据、架构和训练配方是否固定；
4. 外推区间离观测数据多远，参数区间有多宽；
5. 训练收益是否转化为固定推理预算下的收益；
6. 行为提升是现象证据，还是已有干预支持的机制解释。

这样才能把“规模”从口号还原成可以证伪的实验对象。预训练目标如何形成这一基座见[预训练目标谱系](pretraining-objectives.md)，开放 checkpoint 如何改变缩放研究见[从可下载权重到可研究系统](open-model-ecosystem.md)。
