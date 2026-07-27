# KV Cache

自回归 decode 时，历史 token 的 K/V 不需要每步重算，因此缓存每层 K/V。缓存换取计算节省，也成为长上下文与高并发服务的主要显存消耗。

## 容量估算

对 $L$ 层、batch $B$、缓存长度 $T$、K/V head 数 $H_{kv}$、head dimension $d_h$、元素字节数 $s$：

$$
M_{\text{KV}}
=2LBTH_{kv}d_hs
$$

系数 2 对应 K 与 V。实际还要加入 block 元数据、对齐、碎片、量化 scale 和临时 workspace。

## 分页管理

连续预留最大长度会造成内部碎片，也不利于请求动态增长。[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) 把逻辑 KV 序列映射到固定大小物理 block，使分配更接近操作系统分页，并支持共享 prefix block。

block 太大增加尾部浪费，太小增加映射开销和 kernel 复杂度。评估必须用真实长度分布与并发。

## Prefix Cache

系统提示、共享文档或多分支采样可以复用前缀 KV。命中需要 tokenizer、模板、模型、adapter、position 与采样前状态完全兼容。缓存键若忽略权限或租户，会形成数据隔离风险。

## 压缩路线

- 减少 K/V head：MQA、GQA。
- 低精度 KV：FP8、INT8 或更低比特，需要校准误差。
- eviction 或滑动窗口：牺牲部分历史可见性。
- token 合并、摘要或 learned compression：改变可恢复的信息。
- 跨层共享或低秩表示：需要架构支持。

“支持更长上下文”必须同时给出质量、缓存容量、并发、TTFT 与 decode 延迟。

block table、copy-on-write、抢占和 prefix cache 的请求级状态见[推理运行时](runtime.md)；跨 prefill/decode worker 的 cache 迁移见[Prefill–Decode 分离](disaggregation.md)。
