# 从外显推理到可验证搜索

推理能力的演进不只是回答越来越长。真正发生变化的是：模型能否提出多条候选路径，系统能否判断局部与最终结果，计算预算能否分配给仍有希望的分支，以及搜索所得能否反哺参数。

这条谱系包含三种不同的计算：

- **参数学习**：SFT、偏好优化或 RL 改变 checkpoint；
- **推理预算**：采样、反思、搜索和工具调用在参数固定时增加计算；
- **验证**：规则、测试、reward model 或 process verifier 对候选提供信号。

扩大其中一项不会自动补齐另外两项。

## Chain-of-Thought 打开了一条外显工作带

[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) 用包含中间步骤的 few-shot demonstrations 改变生成轨迹。把 rationale 记为 $z$、答案记为 $a$，可写成

$$
p(a,z\mid x)=p(z\mid x)p(a\mid x,z).
$$

外显步骤为采样、检查与回溯提供了接口，但不能被直接解释为模型内部计算的忠实转录。论文在若干大模型与算术、常识、符号任务上观察到收益；它没有证明所有模型、所有任务都会因更长 rationale 获益。

训练侧若希望模型更常生成这类轨迹，可以使用 reasoning SFT；推理侧则可以只改变 prompt 和 decoding。两者的算力位置与泛化边界不同。

## Self-consistency 从单路径转向候选集合

Greedy decoding 只保留一条高概率路径。[Self-Consistency](https://arxiv.org/abs/2203.11171) 采样多条 reasoning path，并对归一化后的最终答案投票：

$$
a^*
=\arg\max_a
\sum_{i=1}^{N}
\mathbf 1[\operatorname{extract}(y_i)=a].
$$

若单次成功概率为 $p$ 且样本近似独立，oracle coverage 为

$$
P(C_N)=1-(1-p)^N.
$$

真实候选往往高度相关，所以这个公式只提供理想参照。多数票还可能放大共享错误。至少应同时报告：

- pass@1；
- oracle coverage / pass@$N$；
- 实际选择后的 selected@$N$；
- 总生成 token 与延迟；
- 候选答案和解法的有效多样性。

## Verifier 把“生成到”与“选出来”分开

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) 生成多个数学解答，再训练 verifier 排序。若候选集合事件为 $C_N$，选择器只从候选中返回答案，则

$$
P(\text{chosen correct})
=P(C_N)\,
P(\text{choose correct}\mid C_N).
$$

增加采样主要提升第一项；更好的 verifier 主要提升第二项。只报告 oracle pass@$N$ 会把一个不可部署的 oracle 当成真实系统。

二元 outcome verifier 可用

$$
\mathcal L_v
=-\mathbb E\left[
y\log\sigma(v_\phi(x,z))
+(1-y)\log(1-\sigma(v_\phi(x,z)))
\right]
$$

训练。它仍可能学习生成器特有的错误风格，或把长度和格式当捷径。因此应在新生成器、不同长度和对抗格式上校准。

## Process supervision 把信用移到步骤

Outcome verifier 只能在结尾给出信号。[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 对数学解答的中间步骤做监督，并发布了 [PRM800K](https://github.com/openai/prm800k)。若第 $k$ 个步骤标签为 $y_k$：

$$
\mathcal L_{\mathrm{PRM}}
=-\sum_k m_k
\left[
y_k\log p_k+(1-y_k)\log(1-p_k)
\right].
$$

过程监督可以定位首个错误、为搜索提供前缀分数，却引入新的建模选择：

- step 由句子、token、公式还是环境 action 切分；
- 当前步骤正确是否意味着前缀仍可完成；
- 多种正确路径是否都被接受；
- candidate score 使用最小值、乘积、末步还是学习聚合；
- 标签是否偏爱冗长、模板化推理。

论文在代表性 MATH 子集上支持过程监督优于结果监督；[OpenAI 的公开说明](https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/)明确指出，能否推广到数学之外仍未知。

## Search 把 policy 变成 proposal distribution

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) 把生成、评价、选择和回溯组织成树。常见前缀评分可写成

$$
s(h)
=\lambda\log\pi_\theta(h\mid x)
+(1-\lambda)v(h)
-\mu\,\operatorname{cost}(h).
$$

只依赖 policy score 容易重复熟悉的错误；只依赖 verifier 容易 Goodhart。搜索还需要冻结：

```text
state and action granularity
branching / depth / stopping rules
generated-token and verifier budgets
terminal / invalid / timeout semantics
answer extractor and verifier versions
```

[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) 表明，在其模型、数学任务和 verifier 设置中，最合适的推理策略依赖题目难度，自适应计算可以优于固定预算。该结果不能推出小模型加搜索在所有领域都能替代大模型。

推理预算的完整成本与停止规则见[推理时计算](../../reasoning/test-time-compute.md)，beam、PUCT 和 verifier 校准见[搜索与验证](../../reasoning/search-verification.md)。

## 从搜索结果回到训练

搜索得到的不只是一个答案，而是一组带结构的数据：

| 搜索产物 | 可形成的监督 | 丢失或保留的信息 |
| --- | --- | --- |
| 单条成功轨迹 | SFT / distillation | 丢失失败对比与搜索成本 |
| 成功、失败回答对 | DPO / reward modeling | 保留相对结果，弱化步骤信用 |
| 步骤标签 | PRM / process SFT | 保留局部信用，依赖步骤定义 |
| 可执行终局 reward | Online RL / RLVR | 保留当前策略探索，reward 稀疏 |
| 搜索访问与 value | policy/value target | 同时继承搜索器偏差 |

[STaR](https://arxiv.org/abs/2203.14465) 展示了迭代生成、筛选 reasoning rationale 并重新训练的早期路线。搜索数据不能只保留成功样本：失败类型、候选概率、生成策略、预算与 verifier 版本决定它还能支持哪些因果判断。

## RLVR 把可验证结果用于在线更新

对数学答案、代码测试或确定环境终态，可以令

$$
R_i=v_{\mathrm{exec}}(x,y_i).
$$

这定义了 reward provenance，而没有定义 advantage estimator。可以使用 PPO、RLOO、GRPO 或其他 policy-gradient 方法更新。以 group-relative advantage 为例：

$$
A_i
=\frac{R_i-\bar R}
{\operatorname{std}(R)+\varepsilon}.
$$

全组成功或全组失败时，$A_i$ 应为零或该组被显式跳过。改变采样、课程或 group size 可能提高产生混合 reward 的概率，但那是数据策略变化，不是归一化公式凭空获得了信息。

RLVR 的优势是 reward 可重复执行、误差边界相对清楚；它的限制来自 verifier 规格。公开测试可能被针对性优化，格式 parser 可能被绕过，基础设施失败也不能静默记成错误答案。

## DeepSeek-R1 闭合了训练与推理循环

[DeepSeek-R1](../works/deepseek-r1.md) 将几条路线连接起来：

1. R1-Zero 在强 base model 上直接进行规则奖励的 reasoning RL；
2. 可读性和语言混合问题推动 cold-start reasoning data；
3. 第一轮 RL 后通过 rejection sampling 收集成功轨迹；
4. reasoning 与非 reasoning 数据共同 SFT；
5. 第二轮 RL 兼顾推理、帮助性与安全；
6. 高质量轨迹再蒸馏到更小模型。

因此它不是“只靠 GRPO 的单阶段算法”，也不是“从随机初始化创造推理”。GRPO 是 optimizer 侧的一部分，accuracy/format 是 reward 侧的一部分，长轨迹与多样采样则属于 rollout 和推理预算。

## 三个预算必须分别报告

| 预算 | 典型变量 | 常见混淆 |
| --- | --- | --- |
| 训练预算 | training tokens、rollouts、updates、accelerators | 把更多训练当成算法收益 |
| 推理预算 | output tokens、samples、search nodes、tools | 把 oracle pass@$N$ 当单次准确率 |
| 验证预算 | verifier calls、tests、judge tokens、human labels | 忽略 selector 成本和误差 |

可靠比较需要在同一预算下报告候选覆盖、选择准确率、最终成功、长度、成本和失败分解。评测协议见[语言模型评测](../../evaluation/language-model-evaluation.md)，污染与重复暴露见[评测污染](../../evaluation/contamination.md)。

## 证据边界

- CoT 使推理步骤可见，不证明这些步骤忠实反映内部因果机制。
- Self-consistency 改善若干受测推理任务，不保证多数候选独立，也不保证自由文本可稳定聚合。
- Process supervision 在公开数学实验中有效，不证明步骤监督跨领域普遍优于结果监督。
- Search 的收益依赖 proposal diversity、verifier 与预算；节点数相同不代表 token 或工具成本相同。
- RLVR 只在规格可执行的部分提供可靠 reward，不能自动覆盖事实综合、审美与开放式安全判断。
- [DeepSeek-R1](../works/deepseek-r1.md) 展示了一套特定 base model、数据与评测下的公开配方，不构成完整生产训练复现。

训练闭环的机制页见[推理后训练](../../training/reasoning-posttraining.md)，多步环境与动作信用见[Agentic RL 的搜索、过程奖励与验证](../../agentic-rl/search-verification.md)。
