# 推测解码：接受规则、状态与收益边界

标准自回归生成每次 target forward 只确认一个 token，串行深度等于输出长度。推测解码先便宜地产生多个候选，再让 target 一次并行验证；它减少的是 target 的串行调用次数，而不是 target 必须表达的条件分布。

“一次接受多个 token”不自动保证结果正确。需要区分保持 target 分布的精确算法、只保持 greedy 结果的算法，以及改变分布的近似算法。

## 两模型精确采样

设 draft policy 为 $q$，target policy 为 $p$。draft 依次提出

$$
x_1,\ldots,x_\gamma
\sim q.
$$

target 对候选前缀一次计算各位置分布。对候选 $x_i$，精确接受概率为

$$
\alpha_i
=
\min
\left(
1,\,
\frac{p_i(x_i)}{q_i(x_i)}
\right).
$$

若拒绝，不能简单改从 $p_i$ 重采样，而要从正残差分布采样：

$$
p_i'(x)
=
\frac{
\left[p_i(x)-q_i(x)\right]_+
}{
\sum_y\left[p_i(y)-q_i(y)\right]_+
}.
$$

若全部 $\gamma$ 个候选均接受，再从 target 的下一个位置采样一个额外 token。[Leviathan 等人的 speculative decoding](https://arxiv.org/abs/2211.17192)与[并行 speculative sampling](https://arxiv.org/abs/2302.01318)给出了这类保持 target 分布的方法。

关键不是概率公式孤立成立，而是 $p_i$ 和 $q_i$ 必须已经应用同一套当前状态下的 temperature、penalty、grammar、禁用 token 与其他 logit processors。

## 接受长度与速度模型

为形成直觉，假设各位置条件接受率恒为 $a$，一次最多验证 $\gamma$ 个 draft token。每轮期望产出 token 数近似为

$$
\mathbb E[K]
=\sum_{i=0}^{\gamma}a^i
=\frac{1-a^{\gamma+1}}{1-a}.
$$

当 $a\to1$ 时，期望接近 $\gamma+1$；接受率下降时，继续增大 $\gamma$ 很快失去价值。

粗略速度上界为

$$
S
\approx
\frac{
\mathbb E[K]\,T_{\mathrm{target},1}
}{
T_{\mathrm{draft}}(\gamma)
+T_{\mathrm{verify}}(\gamma)
+T_{\mathrm{state}}
}.
$$

$T_{\mathrm{state}}$ 包含候选组织、采样、KV 提交 / 回滚、grammar 状态和跨设备同步。target verify 处理更多 query token，可能从 bandwidth-bound 变得更 compute-efficient，但高并发下也会改变 batch shape 和 KV 水位。

接受率不是充分指标。应同时报告：

- accepted tokens / target invocation；
- draft、verify、sample 与 state 时间；
- target invocations / output token；
- TTFT、TPOT、goodput；
- 额外权重与 KV 显存；
- 不同温度、任务、长度和 batch 的分布。

## Draft 的来源

### 独立小模型

独立 draft 易于理解，也增加第二套权重、KV 和调度。tokenizer / vocab 必须兼容，跨设备部署还可能引入通信。小得过头会降低接受率，太大又失去成本优势。

### 多头候选

[Medusa](https://arxiv.org/abs/2401.10774)在模型上增加多个 decoding heads，预测多个未来位置并构造候选树。它避免独立 draft 模型，却需要额外训练、tree attention 和候选选择；不同模式对精确性与质量的保证不能混用。

### 特征级 Draft

[EAGLE](https://arxiv.org/abs/2401.15077)在特征空间进行自回归草拟；[EAGLE-2](https://arxiv.org/abs/2406.16858)引入动态 draft tree；[EAGLE-3（2025）](https://arxiv.org/abs/2503.01840)继续调整训练和特征使用。[官方实现](https://github.com/SafeAILab/EAGLE)适合核对训练产物与支持范围。

这些 2024–2025 路线说明候选不必来自完整独立语言模型，但会新增 draft head、训练数据、版本和运行时 tree verification。论文速度不能直接外推到不同 target、batch、temperature 或服务引擎。

### 模型原生多 token 预测

若 target 训练时包含 multi-token prediction heads，可以复用模型内部特征产生候选。它减少额外模型驻留，但仍要定义候选分布、验证路径和部署兼容；“模型能预测多个未来 token”不等于服务默认能精确接受它们。

## Tree Verification

单链 draft 只验证一个候选序列；tree 方法在一个 forward 中验证多个分支。若树有节点集合 $\mathcal T$，verify 的成本不是节点数的简单线性函数：

$$
T_{\mathrm{verify}}
=f\left(
|\mathcal T|,
\text{tree mask},
\text{KV layout},
\text{kernel shape}
\right).
$$

宽树提高覆盖率，也增加 attention mask、候选排序、KV 分支和采样成本。tree node 的 parent、position、token 和 cache slot 必须一一对应；任何重排都要同步更新这些数组。

## 状态事务

一次 speculative round 应视为小事务：

```text
snapshot committed state
create provisional draft branch
run target verification
commit accepted prefix and one target token
discard every rejected suffix
advance stream and RNG exactly once
```

需要同步管理：

- target 与 draft KV；
- token IDs 和 position；
- target / draft RNG counter；
- repetition / frequency penalty history；
- grammar automaton state；
- stop-sequence buffer；
- beam / tree parent；
- streaming cursor 与 finish reason。

拒绝后只裁剪 token array、却保留多写入的 KV 或 grammar state，会让后续输出静默偏离。

## 与 Logit Processor 的组合

若原始 target logits 为 $z_i$，实际采样分布是处理器组合

$$
p_i=
\mathcal P_i
\left(
z_i;\,
\text{history},\,
\text{grammar},\,
\text{sampling config}
\right).
$$

接受比率必须使用处理后的 $p_i$ 与 $q_i$。需要特别处理：

- repetition / frequency / presence penalty；
- min length、EOS 与 bad words；
- temperature、top-$k$、top-$p$ 和 min-$p$；
- JSON schema 或 grammar mask；
- 跨 token stop string；
- all-masked grammar state。

若 draft 无法表达 target 的 tokenizer 或约束状态，应关闭精确 speculative，而不是用近似映射伪装兼容。

## Batch 与服务

不同请求会在不同位置拒绝，造成 verify 后输出数和下一轮 shape 不规则。调度器需要决定：

- speculative 与普通 decode 是否混排；
- 每请求 $\gamma$ 是否动态；
- draft 和 target 是否共用 GPU；
- provisional KV 是否进入 block budget；
- graph bucket 怎样覆盖不同 tree；
- 高负载时是否降级为普通 decode。

低并发单请求最容易获得串行深度收益；高并发 target 本身已形成大 batch 时，draft 可能抢占同一 GPU 的算力和显存。服务优化应以 goodput 而非单请求 speedup 决策。

## 正确性契约

1. 明确声明保持的是 target 采样分布、greedy 输出还是近似质量；
2. $p$、$q$ 使用相同 token space 和处理器顺序；
3. 每个候选位置的分布条件于正确的已接受历史；
4. 残差分布非负、归一化，并覆盖浮点舍入边界；
5. 拒绝 suffix 的 KV、RNG、grammar 和 stop state 全部回滚；
6. provisional block 不可被其他请求或 prefix cache 看见；
7. accepted token 只流式发送一次；
8. 关闭 speculative 后，普通 decode 路径仍是独立 reference。

浮点误差可能让 $p-q$ 出现微小负值，应先取正部再归一化；若残差和数值退化，需要定义稳定 fallback。

## 常见失效

- 接受率高却不加速：draft、tree 构造或 state transaction 占主导；
- 温度升高后收益骤降：draft 与 target 分布更难一致；
- grammar 请求出现错误：processor 状态没有随分支复制和回滚；
- p99 变差：少数低接受请求反复支付 draft + verify；
- 高并发吞吐下降：draft 占用 target 的关键资源；
- 显存超预期：两套权重、draft KV 和 provisional tree 未计入；
- 固定 seed 不可重放：不同接受路径消费 RNG 的次数不一致；
- 模型升级后输出异常：draft head 与 target 权重版本不兼容。

## 何时不用

- target 已被大 continuous batch 充分利用；
- 输出很短，初始化和 draft 固定成本占主导；
- draft 接受率低或任务分布剧烈变化；
- tokenizer、vocab、grammar 或 logit processors 不兼容；
- 显存无法容纳额外 draft 权重和 provisional KV；
- 严格重放要求尚未定义 RNG 与状态事务；
- 近似方法无法通过质量和安全回归。

## 验证

精确算法先在小词表上枚举完整序列，比较 speculative 与普通 target sampling 的经验分布；再覆盖：

1. 第一个 token 拒绝、全部接受和中间拒绝；
2. $q(x)=0$、$p(x)=0$ 与极端概率；
3. temperature、top-$k$、top-$p$ 和 repetition penalty；
4. grammar、EOS、跨 token stop 与最大长度；
5. 固定 seed、取消、抢占和 OOM 回滚；
6. paged KV 的 block 边界和 tree COW；
7. mixed batch、不同 $\gamma$ 与 graph fallback；
8. 模型 / draft 版本切换。

性能矩阵至少包含 task、温度、prompt / output 长度、batch、接受长度分布、draft / verify 时间、TTFT、TPOT、goodput、显存与功耗。[TensorRT-LLM 的 speculative decoding 文档](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/speculative-decoding.html)可用于核对一种生产实现的模式与约束，但最终结论必须以目标版本和本地工作负载为准。

候选接受、状态回滚与分页 KV 边界的紧凑 reference 见[手撕：推理引擎](../practice/inference-engine.md)。

## Reference {#reference}

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [并行 speculative sampling](https://arxiv.org/abs/2302.01318)
- [Medusa](https://arxiv.org/abs/2401.10774)
- [EAGLE](https://arxiv.org/abs/2401.15077)
- [EAGLE-2](https://arxiv.org/abs/2406.16858)
- [EAGLE-3（2025）](https://arxiv.org/abs/2503.01840)
- [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)
- [TensorRT-LLM 的 speculative decoding 文档](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/speculative-decoding.html)
