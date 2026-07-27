# Prefill–Decode 分离

prefill 与 decode 的硬件行为不同：prefill 以较大的矩阵计算处理输入，decode 反复读取权重和 KV、每次只生成少量 token。将二者放在不同 worker，可以独立扩缩和设并行策略，但会新增 KV 传输与分布式状态。

## 为什么分离

共置引擎中，长 prefill 会干扰正在流式生成的 decode，造成 TPOT 尖峰；decode 又可能让 prefill batch 形状零碎。分离后：

| 阶段 | 优化目标 | 常见资源偏好 |
| --- | --- | --- |
| Prefill | TTFT、prompt tokens/s | 高计算吞吐、大 GEMM、长序列并行 |
| Decode | TPOT、并发与 KV 容量 | 高内存带宽、连续批处理、更多副本 |

[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)展示了针对 TTFT/TPOT SLO 分别规划资源和并行的一条路线。

## 数据路径

```text
gateway
-> prefill scheduler
-> prefill worker
-> KV transfer
-> decode admission
-> decode worker
-> stream
```

prefill 不能只传“第一个 token”。decode 必须获得每层 KV、token/position 状态、采样配置和模型身份。

P/D 边界应传递一个可验证 descriptor，而不是让 decode worker 从裸字节猜布局。下面只保留安装 KV 所需的语义核；descriptor 通过后才允许读取 payload。

```python
def validate_kv_descriptor(descriptor, expected):
    required = {"request_id", "model_revision", "layout", "tokens", "blocks", "checksum"}
    if set(descriptor) != required:
        raise ValueError("descriptor schema mismatch")
    for field in ("model_revision", "layout"):
        if descriptor[field] != expected[field]:
            raise ValueError(f"incompatible {field}")
    if descriptor["tokens"] <= 0 or descriptor["blocks"] <= 0:
        raise ValueError("empty KV transfer")
    if not isinstance(descriptor["checksum"], str) or len(descriptor["checksum"]) < 8:
        raise ValueError("invalid checksum")
    return (descriptor["request_id"], descriptor["tokens"], descriptor["blocks"])
descriptor = {
    "request_id": "req-7", "model_revision": "m-3", "layout": "tp2-b16-fp8",
    "tokens": 128, "blocks": 8, "checksum": "8fe31a90",
}
installed = validate_kv_descriptor(descriptor, {"model_revision": "m-3", "layout": "tp2-b16-fp8"})
assert installed == ("req-7", 128, 8)
assert descriptor["tokens"] == descriptor["blocks"] * 16
assert descriptor["model_revision"] == "m-3"
```

真实 descriptor 还需逐 shard 的 layer/head 范围、dtype/scale、position/RoPE、adapter、字节数与 payload location；校验失败必须拒绝安装。`request_id` 只能提供幂等关联，不能代替 checksum、所有权或租户授权。

## KV 传输量

对 $L$ 层、prompt 长度 $T$、$H_{kv}$ 个 K/V head、head dimension $d_h$ 和元素字节数 $s$：

$$
M_{\text{transfer}}
\approx2LTH_{kv}d_hs.
$$

若模型使用低秩 cache、KV 量化或 TP 分片，传输布局会不同。压缩可以降低网络量，却可能增加转换 kernel、scale 元数据和质量误差。

传输时间下界近似

$$
T_{\text{KV}}\ge
\frac{M_{\text{transfer}}}{B_{\text{effective}}},
$$

但实际还包含注册内存、控制消息、分片聚合、拥塞和 decode 端安装 block 的开销。

## 布局兼容

prefill 与 decode 可以使用不同并行度。例如 prefill 用较大 TP，decode 用更多较小 TP 副本。此时 KV shard 需要重排：

$$
\text{layout}_{P}
\longrightarrow
\text{layout}_{D}.
$$

转换应明确：

- layer/head/block 的分片维度；
- 物理 block size 与逻辑位置；
- dtype、量化 scale 和字节序；
- RoPE 与 position configuration；
- model、adapter 和 cache schema version。

版本不兼容必须拒绝，而不是尽力解析。

## 调度与背压

prefill 完成速度超过 decode 消费速度时，KV 会堆积并占用发送端、网络或接收端显存。全局 admission 需要同时考虑：

$$
\lambda_{\text{prefill}},
\quad
\mu_{\text{decode}},
\quad
M_{\text{KV available}},
\quad
\text{SLO}.
$$

不能等 prefill 完成后才发现 decode 无容量。可在 gateway 预留 decode slot，或让 prefill scheduler 获得 decode 端的实时 block/queue 配额。

## Worker 选择

路由不只看最短队列：

- prefix cache 是否命中；
- 目标 adapter/model 是否已加载；
- decode worker 的 KV 空间；
- prompt 与预期输出长度；
- tenant、优先级和 deadline；
- 网络拓扑与当前传输拥塞；
- 失败域与副本健康。

预测输出长度有误时，系统仍需安全过载与抢占策略。

## 可靠性

### Prefill 成功、KV 传输失败

请求不应进入 decode。可以有界重传，或重新选择 prefill worker；重试必须使用不可变请求 ID，避免安装两份 KV。

### Decode worker 在安装后失败

若没有远端 KV 副本，只能重新 prefill；若缓存复制到共享层，则可迁移但增加常态成本。系统应明确恢复点和可接受重算。

### 模型版本切换

同一请求的 prefill 与 decode 必须使用语义兼容的权重。滚动升级时要按版本划分 worker pool，旧请求排空后再回收旧 decode。

### 取消

客户端取消应传播到 gateway、prefill、传输和 decode，并释放所有 block。只从前端队列删除会留下昂贵的孤儿 KV。

## 何时不值得分离

- 模型或负载很小，共置已满足 SLO；
- prompt 很短，KV 传输固定开销占主导；
- 集群网络不足以承载 cache；
- 流量过低，独立池导致资源闲置；
- 运维复杂度与故障面超过可获得的 goodput。

分离不是默认更快，而是对阶段干扰和资源耦合的一种系统解法。

## 评测

使用真实到达过程与长度分布，至少报告：

- TTFT、TPOT、E2E 的 p50/p95/p99；
- 满足双 SLO 的 goodput；
- prefill/decode 利用率与队列时间；
- KV bytes/request、有效带宽和安装时间；
- prefix cache 冷热状态；
- worker 比例变化与弹性时间；
- 传输失败、版本切换和取消后的资源回收；
- 与同硬件共置基线的比较。

单引擎状态见[推理运行时](runtime.md)，在线容量策略见[调度与服务](serving.md)。

descriptor、block 安装与容量估算的可执行组合实验见[推理引擎手撕实现](../practice/inference-engine.md)。

## Reference {#reference}

- [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079)
