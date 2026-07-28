# 音频与视频

音频和视频都包含时间轴，但信号速率、tokenizer、延迟和评测目标不同。把两者放在同一实现页面会掩盖最关键的接口差异。

本页保留为稳定分流入口：

- [音频与语音](audio-language-models.md)：log-Mel、codec、RVQ、语音理解与生成、流式和全双工；
- [视频与世界模型](video-world-models.md)：tubelet、时空位置、长视频理解、视频生成与未来预测；
- [具身智能与动作](embodied-agents.md)：把视觉、语言和状态映射到闭环动作；
- [多模态融合、位置与训练](architecture-training.md)：共享的 projector、resampler、mask 与 token budget；
- [长上下文](../architecture/long-context.md)：长时序输入的有效利用与系统成本。

## 同一条时间轴，不同的采样语义

若信号每秒产生 $r$ 个 token、时长为 $\tau$：

$$
N=r\tau.
$$

降低 $r$ 能减少上下文和 prefill，却可能丢掉音素、瞬时事件、运动和同步信息。任何时序模型都应同时报告：

- 原始时长、采样率/帧率；
- 压缩后 token rate；
- lookahead 与算法延迟；
- 模态 token、prefill 和峰值显存；
- 首包、端到端延迟和实时系数；
- 不同采样策略下的任务质量。

音频通常从 waveform 经声学前端或 neural codec 变成高频 token；视频则先在空间上形成 patch，再沿时间形成 frame/tubelet token。两者的 token rate 即使数值相同，也不表示信息量或生成难度相同。AudioLM 使用语义 token 保持长程结构、声学 codec token 恢复局部细节，说明单一离散层往往不能同时承担内容与波形保真；VideoPoet 则把文本、图像、视频和音频 token 放入自回归任务混合，统一的是序列接口，不是底层采样器。

## 时钟必须显式对齐

多模态序列至少涉及三个时钟：

1. 原始媒体时钟，例如 16 kHz sample 或 30 FPS；
2. encoder/tokenizer 的步长与 receptive field；
3. 模型序列中的 token position。

若第 $i$ 个 token 覆盖原始区间 $[a_i,b_i)$，字幕、事件标签和跨模态 attention 应按区间相交对齐，而不是假设 token index 相同。chunked streaming 还要记录 lookahead $l$，因为模型在时间 $t$ 输出的结果实际可能使用到 $t+l$ 的未来信号。

音视频同步误差可写成

$$
\Delta t=\hat t_{\text{audio event}}-\hat t_{\text{visual event}}.
$$

只评价各模态单独质量，会漏掉口型、碰撞声和剪辑点的偏移。对生成任务，应报告同步误差的分布，而不是只给平均值。

## 理解、生成与流式是三种协议

| 任务 | 模型可见信息 | 主要状态 | 关键指标 |
| --- | --- | --- | --- |
| 离线理解 | 完整片段 | 全局 token 序列 | 事件定位、问答、召回 |
| 自回归生成 | 过去 token 与条件 | codec/visual history | 保真、语义、长时一致性 |
| 流式交互 | 过去 + 有界 lookahead | chunk state、缓冲区 | 首包、实时系数、中断响应 |

训练时看完整未来、部署时只看过去，是数据协议变化，不是简单的速度优化。流式模型需要 causal/chunk mask、state reset 和跨 chunk 等价测试；全双工系统还要区分用户语音、模型语音与回声路径。

## 失效沿时间位置切片

- 均匀降采样漏掉短事件；
- 时间戳、字幕、声音和画面错位；
- chunk 边界破坏连续状态；
- 训练使用完整未来，部署却要求流式；
- token/s 看似很高，但缓冲、codec 或网络主导延迟；
- 长序列能被输入，却不能稳定定位中部事件。

还应专门检查片段开头、中部、结尾，短事件与长事件，以及不同速度/帧率。随机裁切上的平均分无法发现“中部遗忘”或 chunk 边界失败。音频的 codec、RVQ 与流式协议见[音频语言模型](audio-language-models.md)，视频的 tubelet、未来预测与世界状态见[视频与世界模型](video-world-models.md)，最小 token packing 与 causal mask 练习见[多模态手撕实现](../practice/multimodal.md)。

## Reference {#reference}

- [AudioLM: A Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143)
- [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125)
- [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)
- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)
