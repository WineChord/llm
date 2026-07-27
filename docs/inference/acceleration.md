# 加速与量化

推理优化必须先定位瓶颈。对 compute-bound prefill 优化 GEMM 与 attention；对 bandwidth-bound decode 减少权重/KV 读取、提高有效 batch；对调度瓶颈则优化队列与内存管理。

## Kernel 与图

- fused kernel 减少中间张量和 launch。
- FlashAttention 降低 attention 的 HBM 访问。
- CUDA Graph 可减少稳定 shape 下的 launch 开销，但动态 batch 和内存地址需要专门管理。
- tensor parallel 增加设备总带宽，也引入每层通信。

## Quantization

weight-only 量化对 decode 常有效，因为每 token 反复读取权重；prefill 的收益取决于低比特 GEMM 吞吐。KV 量化更直接影响长上下文容量。需要按层、通道或 group 记录 scale 粒度，并做任务回归。

代表方法见 [GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978) 与 [SmoothQuant](https://arxiv.org/abs/2211.10438)。

## Speculative Decoding

小 draft model 先提出多个 token，大 target model 并行验证；接受规则设计得当时，输出分布可与 target model 原采样一致。[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) 给出代表性算法。

速度取决于：

- draft 与 target 的 token 接受率；
- target 验证多个 token 的并行效率；
- draft 开销与跨设备通信；
- batch、温度、任务和输出分布；
- tokenizer 与词表兼容性。

## 其他杠杆

- prefix caching 与 prompt 去重；
- prompt 压缩或检索减少无效上下文；
- early exit、路由到小模型或级联系统；
- prefill/decode disaggregation；
- adapter 合并、权重共享与多模型调度。

任何加速报告都应给出未优化基线、相同输出质量、完整请求分布、硬件、并发与尾延迟。

采样与精确接受规则见[解码](decoding.md)，online softmax 与融合 kernel 见[Kernel 与性能](../systems/kernels-performance.md)，真实请求的 shape 与缓存状态见[推理运行时](runtime.md)。
