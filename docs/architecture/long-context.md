# 长上下文

上下文窗口长度包含至少四个不同命题：接口允许输入多长、训练见过多长、注意力能计算多长，以及模型能可靠利用多远的信息。只扩大配置中的最大位置不能同时证明这四点。

## RoPE 的相对位置

[RoPE](https://arxiv.org/abs/2104.09864)在每个二维子空间对 Q/K 施加位置相关旋转。第 $j$ 个频率可写为

$$
\theta_j=b^{-2j/d_h},
$$

位置 $m$ 的旋转为 $R(m\theta_j)$。由于

$$
\left(R(m\theta)q\right)^\top
\left(R(n\theta)k\right)
=q^\top R((n-m)\theta)k,
$$

点积自然依赖相对位置 $n-m$。

不同频率对应不同旋转周期。超出训练长度后，高频维度可能经历模型未见过的相位模式，低频维度又可能变化不足；长上下文扩展因此是频率分配问题，而不只是把 position ID 变大。

## Position Interpolation

若原训练长度为 $L_0$，目标长度为 $L_1$，最简单的位置插值把新位置压回旧范围：

$$
m'=m\frac{L_0}{L_1}.
$$

所有频率统一缩放易实现，却会压缩短距离分辨率。NTK-aware 一类方法按频率维度调整基数或缩放，使高低频受到不同影响；[YaRN](https://arxiv.org/abs/2309.00071)进一步组合频率分区与 attention scale。实现时应以具体公式和版本为准，“NTK”“dynamic”不是统一规范。

## 训练与推理策略

扩长通常需要组合：

1. 位置缩放或新的位置机制；
2. 足够长且结构真实的数据；
3. 长度课程与稳定的 batch/token 预算；
4. memory-efficient attention；
5. context parallel 或 sequence parallel；
6. 服务端 KV 容量、调度和 admission control。

只在短样本上重复 padding 到长长度，不会产生长距离监督。把多条无关文档拼接成长序列也可能让模型学会忽略远处内容。

## Attention sinks 与滑动窗口

局部滑动窗口把每个 token 的可见范围限制为 $w$，计算和缓存可接近 $O(Tw)$，但远距离信息必须通过层间传播、全局 token 或外部记忆保留。某些流式 attention 在丢弃早期 token 后明显退化，保留少量初始 sink token 可以稳定分布；这是一种缓存策略，不等于模型能完整记住被淘汰内容。

## 系统代价

标准 attention 的 score 计算为 $O(T^2d)$，KV Cache 则随 $T$ 线性增长。训练时可用 IO-aware kernel 避免物化完整 score 矩阵，但不会消除所有二次 FLOPs。跨设备长序列还需要交换 K/V block 或在线 softmax 统计；[Ring Attention](https://arxiv.org/abs/2310.01889)展示了将 block 通信与计算重叠的一条路线。

## 评测矩阵

### 检索

- 单个 needle 的位置从开头移动到末尾；
- 多个相似干扰项；
- 需要精确复述还是语义定位；
- 不同输入长度下保持相同任务难度。

### 推理

- 多跳证据是否分散在远距离位置；
- 中间证据顺序是否改变；
- 答案能否由局部先验猜出；
- 证据冲突时是否识别时间和来源。

### 生成

- 输入长度与输出长度分别扩展；
- 长文结构、引用与实体一致性；
- KV 压缩、量化或 eviction 后的回归；
- TTFT、TPOT、峰值显存与并发。

常见的“needle-in-a-haystack 全绿”只证明一种定位任务，不证明长文推理、记忆或生产稳定性。

## 发布声明

一个可解释的长上下文声明应同时写：

```text
maximum accepted length
maximum trained length and curriculum
position method and scaling parameters
attention pattern and cache policy
evaluation tasks and evidence positions
input/output length matrix
hardware, latency and memory
known failure regions
```

注意力结构见[注意力家族](attention-variants.md)，长序列并行见[模型并行](../systems/model-parallelism.md)，在线容量见[推理运行时](../inference/runtime.md)。
