# 性能成本模型

性能分析的目标不是给系统贴上“算力不足”或“通信瓶颈”的标签，而是回答三个可验证的问题：

1. 理论上必须完成多少计算与搬运？
2. 哪些工作落在关键路径上，哪些可以被隐藏？
3. 改变 batch、序列长度、并行方式或精度后，瓶颈会移到哪里？

本页建立训练和推理共用的符号、成本模型与验证方法。它给出可用于设计和排障的近似式，而不是替代 profiler 的精确预测器。

## 统一符号

| 符号 | 含义 |
| --- | --- |
| $B$ | batch 中的序列数 |
| $S$ | 本次处理的序列长度 |
| $T$ | decode 前已经缓存的上下文长度 |
| $L$ | Transformer 层数 |
| $H$ | hidden size |
| $A$、$A_{\mathrm{kv}}$ | query head 与 KV head 数 |
| $d_h=H/A$ | head dimension |
| $F$ | MLP intermediate size |
| $V$ | 词表大小 |
| $N$ | 参数量或 token 数，按上下文注明 |
| $p$ | 某个并行维度的 rank 数 |
| $s$ | 一个元素占用的字节数 |

所有 FLOPs 公式还应注明乘加按一次还是两次操作计数。本页采用一次乘法加一次加法等于 $2$ FLOPs 的惯例。

## 参数量先于 FLOPs

对标准投影，GQA attention 每层参数量为

$$
P_{\mathrm{attn}}
=H^2+H^2+2H A_{\mathrm{kv}}d_h
=2H^2\left(1+\frac{A_{\mathrm{kv}}}{A}\right).
$$

前两个 $H^2$ 分别来自 query 和 output projection；后两项来自 key、value projection。MHA 满足 $A_{\mathrm{kv}}=A$，于是得到 $4H^2$。

SwiGLU 有 gate、up 和 down 三个矩阵，因此

$$
P_{\mathrm{mlp}}=3HF.
$$

参数量公式应从真实配置推导：是否使用 bias、是否共享 embedding 与 LM head、MoE 每 token 激活几个 expert，都会改变后续计算量。把模型总参数直接乘常数只适合早期估算。

### 可执行的静态账本 {#static-cost-ledger-reference}

下面把配置映射为单层 attention / SwiGLU 参数量和整模型 KV 主体字节数。输入均为逻辑 shape，输出是用于交叉检查配置的整数账本；它不会把 padding、allocator、workspace 或通信 buffer 偷藏进一个经验系数。

```python
def dense_layer_ledger(hidden, query_heads, kv_heads, intermediate):
    assert hidden % query_heads == 0
    head_dim = hidden // query_heads
    attention = 2 * hidden * hidden + 2 * hidden * kv_heads * head_dim
    mlp = 3 * hidden * intermediate
    return {"attention_parameters": attention, "mlp_parameters": mlp}

def kv_cache_bytes(layers, batch, context, kv_heads, head_dim, element_bytes):
    return 2 * layers * batch * context * kv_heads * head_dim * element_bytes

h, qh, ff = 4096, 32, 11008
mha = dense_layer_ledger(h, qh, qh, ff)
gqa = dense_layer_ledger(h, qh, 8, ff)
assert mha["attention_parameters"] == 4 * h * h
assert gqa["attention_parameters"] < mha["attention_parameters"]
assert kv_cache_bytes(32, 2, 1024, 8, h // qh, 2) == 268_435_456
```

这里的不变量是 head dimension 整除、参数重复关系与 K/V 的系数 $2$。生产容量规划还需从 checkpoint 核对 bias、共享权重和实际 placement，并把返回值作为显存账本的一项，而不是峰值显存本身；含 transient 与分片状态的组合账本见[手撕：分布式与容错](../practice/distributed-systems.md)。

## 训练计算

对 dense Transformer，常见的

$$
C_{\mathrm{train}}\approx 6ND
$$

把每个参数、每个训练 token 的 forward、activation-gradient 与 weight-gradient 粗略合并为 $6$ FLOPs。这里 $N$ 是参与矩阵计算的参数量，$D$ 是训练 token 数。它忽略长上下文 attention、重计算、embedding、稀疏激活与非 GEMM 算子。

若采用 MHA、SwiGLU 且 $F=8H/3$，忽略 embedding、norm 和 bias，则每 token、每层的训练计算可近似拆成

$$
C_{\mathrm{layer,token}}
\approx 72H^2+12SH.
$$

第一项来自线性层，第二项来自 attention 的 $QK^\mathsf T$ 和 $PV$。因此长上下文下，单纯的 $6ND$ 会低估 attention 计算。公开模型的精确训练预算应优先使用其披露的口径；例如 [DeepSeek LLM](https://arxiv.org/abs/2401.02954) 明确区分了模型矩阵计算与序列相关项。

训练 step 的 wall time 不是各资源时间的简单求和。若阶段 DAG 的路径集合为 $\mathcal P$，更合适的表示是：

$$
T_{\mathrm{step}}
\approx
\max_{\pi\in\mathcal P}
\sum_{v\in\pi}T_v
+T_{\mathrm{unmodeled}}.
$$

通信、数据预取和重计算只有在它们与其他工作真实重叠时才会从关键路径消失。trace 中时间区间重叠也不等于完全隐藏：共享 SM、HBM 或网络时，两侧都可能减速。

## 推理计算

### Prefill

prefill 同时处理 $S$ 个 token。线性层计算随 $BS$ 线性增长，而 dense attention score 与 value aggregation 随 $BS^2$ 增长。每层的数量级为

$$
C_{\mathrm{prefill,layer}}
\approx 2BS P_{\mathrm{layer}}
+4BS^2H.
$$

因果 attention 只使用下三角，但 kernel 可能通过 tile mask 跳过部分工作，也可能仍执行被 mask 的 tile；必须说明 FLOPs 是算法 FLOPs 还是硬件实际执行 FLOPs。

### Decode

每条序列生成一个新 token 时，主要成本可写为

$$
C_{\mathrm{decode}}
\approx 2B P_{\mathrm{blocks}}
+4BLTH
+2BHV.
$$

$P_{\mathrm{blocks}}$ 不含 embedding 和 LM head；第二项是新 query 与历史 KV 的 score 和加权求和；最后一项是未切分 LM head 的近似成本。GQA 显著降低 KV cache bytes，却不会按同一比例降低所有 query head 的 attention FLOPs。

decode 往往读取大量权重与 KV，却只为每个请求产生一个 token。低 batch 时，它更可能受 HBM 带宽、launch latency 和跨卡同步限制；prefill 的大矩阵更容易接近计算峰值。

## Roofline 与有效带宽

算术强度定义为

$$
I=\frac{\text{FLOPs}}{\text{bytes moved}}.
$$

若峰值计算吞吐为 $F_{\max}$、有效内存带宽为 $B_{\mathrm{mem}}$，则

$$
F_{\mathrm{achievable}}
\le \min(F_{\max},I B_{\mathrm{mem}}).
$$

“bytes moved”应按目标层级计算。相同算子相对 HBM 可能 compute-bound，相对 shared memory 却可能 bandwidth-bound。缓存命中、融合和重计算会改变分母，因此不能只根据 tensor 的逻辑大小判断。

有效带宽为

$$
B_{\mathrm{effective}}
=\frac{\text{实际有效字节数}}{\text{经过时间}}.
$$

分子必须固定口径：若把重复读取、写回和协议开销遗漏，得到的“带宽利用率”可能超过物理峰值而失去意义。

## 显存账本

训练峰值显存至少包括

$$
M_{\mathrm{peak}}
=M_{\mathrm{parameter}}
+M_{\mathrm{gradient}}
+M_{\mathrm{optimizer}}
+M_{\mathrm{activation}}
+M_{\mathrm{communication}}
+M_{\mathrm{workspace}}
+M_{\mathrm{fragmentation}}.
$$

状态分片后，可将每 rank 的稳态模型内存写成

$$
M_{\mathrm{rank}}
=N\left(
b_{\mathrm{replicated}}
+\frac{b_{\mathrm{sharded}}}{p}
\right)
+M_{\mathrm{activation}}
+M_{\mathrm{transient}}.
$$

$M_{\mathrm{transient}}$ 不能省略：FSDP all-gather 的完整 layer、通信 bucket、量化 workspace 和 allocator 暂存都可能决定 OOM，而稳态参数量仍看似有余量。

等长 batch 的 KV cache 大小为

$$
M_{\mathrm{KV}}
=2LBT A_{\mathrm{kv}}d_hs.
$$

只有 KV head 或 cache 本身确实按 tensor-parallel rank 切分时，才能再除以相应并行度。对不等长请求，应以 $\sum_i T_i$ 代替 $BT$。

## 通信下界

设 collective 消息大小为 $n$ bytes，rank 数为 $p$，启动延迟为 $\alpha$，链路有效带宽为 $B_{\mathrm{link}}$。理想 ring 的近似成本为

$$
T_{\mathrm{all\text{-}gather}}
\approx (p-1)\alpha
+\frac{p-1}{p}\frac{n}{B_{\mathrm{link}}},
$$

$$
T_{\mathrm{reduce\text{-}scatter}}
\approx T_{\mathrm{all\text{-}gather}},
$$

$$
T_{\mathrm{all\text{-}reduce}}
\approx 2(p-1)\alpha
+2\frac{p-1}{p}\frac{n}{B_{\mathrm{link}}}.
$$

这只是无拥塞、同质链路和理想流水下的下界。实际系统还受 topology、channel 数、chunk 大小、NIC 绑定和其他作业干扰。[NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)同时规定所有 rank 的 count、datatype 与调用顺序必须匹配；性能模型不能脱离这一正确性前提。

## MFU、HFU 与 goodput

模型 FLOPs 利用率可定义为

$$
\mathrm{MFU}
=\frac{\text{按模型公式计算的有效 FLOPs}}
{\text{设备数}\times F_{\mathrm{peak}}\times T_{\mathrm{wall}}}.
$$

硬件 FLOPs 利用率则把 activation recomputation、padding 或其他实际执行工作也计入分子。两者回答不同问题：

- MFU 更接近“有多少硬件能力转化为模型训练进度”；
- HFU 更接近“设备实际执行了多少浮点工作”。

报告必须注明 dtype 峰值、稀疏峰值是否使用、乘加口径、重计算和 padding 是否计入。高 HFU 也可能来自无效 padding 或过度重计算。

训练系统最终应观察 token goodput：

$$
\mathrm{goodput}_{\mathrm{train}}
=\frac{\text{通过质量与数值检查的有效训练 token}}
{\text{wall time}}.
$$

推理则应按满足 TTFT、TPOT、端到端延迟和质量 SLO 的请求或 token 计数，而非只报离线峰值吞吐。

## 模型的正确性契约

一份可复用的成本模型至少固定：

- 模型配置、是否共享权重、激活 expert 数；
- batch、真实长度分布、padding 与 packing；
- dtype、累加精度、量化 group 和 scale metadata；
- FLOPs 口径与理论/执行 FLOPs 的区别；
- 每个 tensor 的 global shape、placement 和是否复制；
- 互联层级、collective 算法和有效带宽；
- 编译、warmup、数据输入和 checkpoint 是否计时；
- 质量、收敛或 SLO 约束。

模型预测应输出区间和主导项，而不是伪精确的小数。输入条件改变后必须重新计算，不能把单一硬件上的系数外推为定律。

## 常见失效与何时不用

- **只看参数量**：长上下文 attention、KV cache 和稀疏路由不会被正确刻画。
- **只看峰值 FLOPs**：decode、norm、采样和通信常不受计算峰值控制。
- **把重叠时间相加**：会高估 step time；反过来假设完全重叠又会低估资源争用。
- **忽略瞬时显存**：会在 all-gather、融合 workspace 或 graph capture 时 OOM。
- **用单 shape 外推线上流量**：请求长度、prefix 命中和输出长度共同决定实际成本。
- **用微基准替代端到端**：若目标 kernel 只占关键路径很小比例，局部提速几乎不可见。

模型用于容量规划、方案筛选和解释瓶颈；硬件或实现尚未确定、动态控制流占主导时，应给出上下界并尽早转向实测。

## 验证流程

1. 从配置逐项重建参数量，并与 checkpoint tensor 总数对齐。
2. 用 profiler 核对 GEMM、attention、collective 的真实 shape 与调用次数。
3. 分别测 HBM、节点内互联、节点间网络和存储的有效带宽。
4. 对 batch、序列长度和并行度做至少三个尺度点，检查趋势而非只拟合单点。
5. 对比预测的主导项与 trace critical path；若不一致，优先寻找隐藏同步、重编译、allocator 或数据等待。
6. 同时报告吞吐、分位延迟、显存峰值、数值质量和能耗边界。

大规模训练的并行效率可结合 [Megatron-LM 的扩展研究](https://arxiv.org/abs/2104.04473)校准；训练 token 与模型规模的决策边界可参考 [Chinchilla](https://arxiv.org/abs/2203.15556)。这些论文提供特定模型和硬件条件下的证据，不替代当前工作负载的测量。

把模型落到实际线程束、访存与融合行为见 [GPU 执行模型](gpu-execution.md)，服务侧则应把平均吞吐转换为带 SLO 的[调度与 Goodput](../inference/scheduling-goodput.md)。

## Reference {#reference}

- [DeepSeek LLM](https://arxiv.org/abs/2401.02954)
- [NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [Megatron 的大规模训练研究](https://arxiv.org/abs/2104.04473)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
