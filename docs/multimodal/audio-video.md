# 音频与视频

音频和视频都包含时间轴，但信号速率、tokenizer、延迟和评测目标不同。把两者放在同一实现页面会掩盖最关键的接口差异。

本页保留为稳定分流入口：

- [音频与语音](audio-language-models.md)：log-Mel、codec、RVQ、语音理解与生成、流式和全双工；
- [视频与世界模型](video-world-models.md)：tubelet、时空位置、长视频理解、视频生成与未来预测；
- [具身智能与动作](embodied-agents.md)：把视觉、语言和状态映射到闭环动作；
- [多模态融合、位置与训练](architecture-training.md)：共享的 projector、resampler、mask 与 token budget；
- [长上下文](../architecture/long-context.md)：长时序输入的有效利用与系统成本。

## 共同约束

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

## 共同失效

- 均匀降采样漏掉短事件；
- 时间戳、字幕、声音和画面错位；
- chunk 边界破坏连续状态；
- 训练使用完整未来，部署却要求流式；
- token/s 看似很高，但缓冲、codec 或网络主导延迟；
- 长序列能被输入，却不能稳定定位中部事件。

音频与视频的最小 token packing、RVQ、时空位置和 causal mask 练习见[多模态手撕实现](../practice/multimodal.md)。
