# 调试手册

调试目标是找到第一个语义分叉点。先固定输入、版本和随机状态，再沿数据 → 目标 → 数值 → 分布式 → kernel → 运行时逐层缩小；不要同时改变多项配置期待异常消失。

## 通用起点

保存一个可重放 failure capsule：

- 原始样本与 token IDs；
- tokenizer、template、model 与 adapter 哈希；
- global step、token count、data cursor 与 RNG；
- 完整 shape、stride、dtype、device 和 placement；
- 解析后的配置；
- 首个异常 tensor 前后的统计；
- 单 rank/reference 与优化路径输出；
- 错误、日志、trace 和环境版本。

先在最小 shape 上复现，再逐项恢复真实长度、batch、精度、并行和融合。

## Loss 异常

### 一开始就不对

按顺序检查：

1. labels 是否正确 shift；
2. padding、prompt、tool observation 与文档边界是否 mask；
3. tokenizer 和 chat template 是否与 checkpoint 匹配；
4. vocabulary、special token 和 weight tying；
5. loss 用 token sum 还是 mean；
6. 初始化、学习率和 optimizer state；
7. logits 是否已有 NaN/Inf。

用[手撕 token CE](training-objectives.md)对一个 batch 计算 numerator、denominator 与逐 token loss。

### 训练中 spike

同时记录：

- 具体数据样本与长度；
- loss、grad norm、update norm 与参数 norm；
- learning rate、weight decay 与 scaler；
- 每层 activation/logit 范围；
- collective 前后 global norm；
- 是否刚切 mixture、context length 或 checkpoint；
- 哪个 rank 首先出现非有限值。

不要直接跳过 spike batch；先确认它是坏数据、真实困难样本、数值溢出还是分布式缩放错误。

### Loss 正常但能力退化

检查训练目标与评测任务是否同分布、chat template 是否漂移、response mask 是否学到用户文本、数据混合是否造成遗忘，以及 benchmark 是否更换 harness。平均 loss 可能掩盖某个域或长度 slice 的退化。

## 梯度与优化器

- `unscale → finite check → global norm → clip → step` 顺序是否正确；
- overflow 时参数和 optimizer moments 是否都不更新；
- AdamW decay 是否错误施加到 norm/bias；
- gradient accumulation 是否除以真实 global token 数；
- mixed precision master weight 与 state dtype；
- activation checkpoint 前后 dropout RNG；
- LoRA merge 状态是否在训练时重复计入；
- frozen 参数是否真的没有 grad 与 optimizer state。

在 FP64 小模型上比较一步手写 AdamW 与框架实现，再扩大到目标 dtype。

## 分布式 Hang

1. 收集所有 rank 的最后一个 collective 序号；
2. 比较 process group、op、count、dtype 与 tensor shape；
3. 检查条件分支是否让部分 rank 跳过调用；
4. 检查 async handle、stream event 与 buffer 生命周期；
5. 检查 dataloader 是否让 rank 步数不同；
6. 检查某 rank 先 OOM 或异常退出；
7. 从两 rank、单 collective 的最小复现开始。

不要通过无限增大 timeout 掩盖顺序错误。[集合通信](../systems/collectives-sharding.md)和[系统韧性](../systems/resilience-observability.md)给出全局契约。

## OOM

先做显存账本：

$$
M_{\mathrm{peak}}=
M_{\mathrm{weights}}+
M_{\mathrm{grad}}+
M_{\mathrm{optimizer}}+
M_{\mathrm{activation}}+
M_{\mathrm{communication}}+
M_{\mathrm{transient}}+
M_{\mathrm{fragmentation}}.
$$

定位 OOM 发生在 forward、backward、optimizer step、all-gather、checkpoint staging、prefill 还是 decode。峰值前后抓 allocator snapshot；只看 `nvidia-smi` 无法解释短暂峰值。

训练侧检查 sequence/batch、重计算、sharding 与 all-gather；推理侧检查 admission、输出上限、KV page、prefix refcount 和取消回收。

## Kernel 不一致

每次只比较一个 kernel：

1. 固定输入和 mask；
2. FP64 或框架 reference；
3. forward；
4. backward；
5. 极端值与全 mask；
6. 非连续 stride、尾块与非整除 shape；
7. mixed precision；
8. graph capture；
9. 真实 shape 性能。

声明 storage、input、accumulator、reduction 与 output dtype。同步仅用于测量，不要误放进热路径。

## 训练与增量推理不一致

优先检查：

- causal mask 在 $T_q\ne T_k$ 时的 offset；
- RoPE absolute position 与扩长参数；
- cache 保存旋转前还是旋转后 K；
- GQA query 到 KV head 映射；
- full prompt、chunked prefill 与逐 token decode；
- padding、position IDs 和 sequence boundary；
- train/eval、dropout 和 norm epsilon；
- quantized cache scale 与 layout；
- tokenizer、template 和 stop token。

[完整 Transformer reference](transformer-from-scratch.md)包含 full 与 cached logits 对照。

## 服务尾延迟

把 TTFT 拆成：

$$
T_{\mathrm{TTFT}}=
T_{\mathrm{queue}}+
T_{\mathrm{tokenize}}+
T_{\mathrm{prefill}}+
T_{\mathrm{transfer/install}}+
T_{\mathrm{sample}}+
T_{\mathrm{network}}.
$$

同时切片：

- prompt/output 长度；
- cache hit/miss；
- prefill/decode 混合；
- compile bucket；
- tenant/priority；
- worker、GPU、机架与网络；
- preemption、migration 和 retry；
- 冷启动与稳态。

吞吐上升而 p99 变差通常是 queue、batch 或干扰变化，不应只调模型 kernel。[Goodput 调度](../inference/scheduling-goodput.md)给出 admission 与 SLO 视角。

## KV 泄漏与错乱

检查：

- block refcount 是否等于 live owner；
- shared partial block 写前是否 COW；
- GPU event 完成前是否复用；
- cancel/finish 是否双重释放；
- cache key 是否绑定 model/tokenizer/template/RoPE/dtype/security domain；
- prefix 命中长度是否与精确 token 一致；
- P/D descriptor、checksum 与 install 是否原子。

用[手撕 allocator](inference-engine.md)建立小规模状态机，再注入取消、共享、OOM 和 stale version。

## RAG 错误

按阶段保存 top candidates：

1. 原文是否正确解析和切分；
2. ACL 与时间过滤是否正确；
3. exact sparse/dense 是否命中；
4. ANN 是否额外丢失；
5. fusion/reranker 是否压低 gold；
6. context 是否截断、去重或错序；
7. claim 是否被证据支持；
8. 引用是否定位到正确 span。

正确答案配错误引用说明模型可能依赖参数记忆；检索命中但答案错则继续检查上下文和生成，不要反复调 embedding。

## Agent 卡住或误操作

- 候选动作是否通过 schema、业务和权限三层校验；
- 相同失败是否在无新证据下循环；
- timeout 后副作用是 failed 还是 unknown；
- idempotency key 是否稳定；
- 工具结果自由文本是否被当作控制命令；
- 成功谓词是否读取环境状态；
- cancel 是否传播并完成资源清理；
- memory 是否召回过期或跨域状态。

重放时替换真实副作用工具为 simulator，不能让调试再次发送、支付或删除。[智能体运行时](../applications/agent-runtime.md)与[智能体安全](../applications/agent-security.md)给出状态边界。

## 多模态

常见首个分叉点：

- resize/crop 后坐标未同步；
- tile、patch 或 tubelet 顺序错误；
- image/audio/video padding mask；
- 模态 token loss 未分别归一；
- VQ latent scale 或 codebook collapse；
- diffusion 参数化与 sampler 不匹配；
- streaming audio state 断裂；
- 视频采样漏掉短事件或泄漏未来帧。

使用[多模态原语](multimodal.md)逐一对照 patch、bbox、mask、VQ、noise 和 flow 边界。

## 修复后的门禁

修复不以“这次没报错”为终点。补充：

- 能在旧代码上失败、新代码上通过的最小回归；
- 同类退化和相邻 shape；
- 生产路径与 reference 的等价性；
- 性能没有不可接受回退；
- 故障注入和恢复；
- 文档中的不变量、限制和版本更新。

实验怎样留证见[实验方法](index.md)，统计差异是否可信见[评测工具](evaluation-tooling.md)。

## Reference {#reference}

- [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [NCCL troubleshooting guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
