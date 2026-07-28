# 多模态评测

多模态系统把图片、文档、音频、视频和文本转成共同决策。评测必须区分模型是否真正感知输入、能否将语言 grounding 到正确区域或时间、是否完成跨模态推理，以及输出生成和系统时延是否满足任务。

## 输入协议

原始资产到模型输入之间有一条不可忽略的处理链：

```text
asset bytes
  -> decoder
  -> orientation / colorspace / resampling
  -> resize / crop / tiling / frame sampling
  -> modality encoder
  -> modality tokens and prompt template
```

每一步都应冻结：

```text
asset ID and cryptographic digest
file/container/codec and decoder revision
image dimensions/orientation/colorspace
audio sample rate/channels/window
video fps/duration/frame timestamps
resize/crop/tile/frame selection
OCR/ASR or auxiliary model versions
modality token budget and truncation
model/checkpoint/template/harness commit
```

只保存 URL 或文件名无法重建实际输入。远程资源更新、EXIF orientation、GIF/video 解码和音频 resample 都可能改变结果。

## 统计对象与状态

多模态 item 可能包含多个资产、区域、时间段和问题。逐 item 保存：

```text
success / wrong
asset unreadable or corrupt
unsupported modality / codec
preprocessing failure
context/token overflow
invalid grounding or generation
timeout / infrastructure error
missing judge
```

能力口径可以排除确认损坏的资产，端到端口径应保留上传、解码和预处理失败。若只在 OCR/ASR 成功的子集报告，得到的是条件能力而非完整多模态能力。

## 能力分解

### 感知

感知评测问“输入中有什么”：

- object、文字、布局、声音、说话人、事件；
- 数量、属性、空间和时间关系；
- 小目标、低对比、遮挡、噪声和长尾语言；
- 单帧信息与跨帧变化。

文本先验可能在不看资产时猜对。应加入：

- text-only baseline；
- asset shuffle 或 mismatch；
- 局部遮挡、裁剪和反事实编辑；
- 同问题不同资产的 paired comparison。

若无图/音/视频时分数几乎不降，benchmark 可能主要测语言先验。

### Grounding

给定预测框 $B_p$ 与真值 $B_g$：

$$
\operatorname{IoU}
=
\frac{|B_p\cap B_g|}
{|B_p\cup B_g|}.
$$

报告不同 IoU threshold 下准确率、mean IoU 和 invalid coordinate。点定位、segmentation、时间段 grounding 和文档区域需要各自坐标系：

```text
pixel / normalized coordinates
page index and rendered dimensions
video timestamp / frame index
audio start/end time
```

resize、letterbox 和 tile 后必须能映射回原始资产；否则模型可能定位正确，评测坐标却错。

### 跨模态推理

推理题应标记答案依赖：

- 纯文本即可；
- 单一模态局部证据；
- 多区域/多页/多帧证据；
- 文本与视觉/音频冲突；
- 需要外部工具或知识。

[MMMU](https://arxiv.org/abs/2311.16502) 面向多学科多模态理解与推理。作为 2023 静态 benchmark，运行时仍需冻结图片资产、prompt 和答案协议，并审计 text-only 可解性与训练污染。

### 生成

图像、音频或视频生成至少分开：

- 条件遵循与实体/属性绑定；
- 感知质量和伪影；
- 时序一致性与运动；
- 文本可读性、布局和身份保持；
- 多样性与 mode collapse；
- 安全、来源和水印要求。

自动 embedding 相似度不能覆盖细节、因果关系和可读文字。人类或 multimodal judge 需冻结显示设备、播放条件、随机顺序和 rubric，并执行候选 swap；judge 自身也要接受感知和注入测试。

### 多模态 Agent

GUI Agent 的 observation 是截图/视频，action 作用于真实环境。[OSWorld](https://arxiv.org/abs/2404.07972) 提供真实计算机环境的公开评测。应同时检查：

- 目标状态；
- click/type 等动作与坐标；
- 等待、加载和窗口遮挡；
- 未授权读取、发送、写入和删除；
- 文本声明与真实界面状态是否一致。

详细终态协议见 [Agent 与工具评测](agent-tool-evaluation.md)。

## 时延与成本拆分

端到端时延可分为

$$
\begin{aligned}
T_{\text{e2e}}
={}&T_{\text{load}}
+T_{\text{decode}}
+T_{\text{preprocess}}
+T_{\text{encode}}\\
&+T_{\text{prefill}}
+T_{\text{generate}}
+T_{\text{tool}}
+T_{\text{postprocess}}.
\end{aligned}
$$

还应报告：

- 图片数量、分辨率、tile 数和 modality tokens；
- 音频/视频时长、采样率、帧数与压缩；
- TTFT/首个可用结果；
- streaming 首段延迟与尾延迟；
- video/audio real-time factor；
- peak memory、缓存和重复 encoder 成本；
- judge 与外部 OCR/ASR/tool 成本。

只报模型 decode tok/s 会漏掉大部分多模态链路。

## 分母与聚合

资产、问题和 region 是嵌套结构。多个问题共享一张图或一个视频时，应以资产或来源 cluster 估计区间，不能把每个问答当完全独立。分层报告：

```text
modality and combination
asset source and time
resolution/duration/token budget
language/domain/difficulty
text-only solvability
perception/grounding/reasoning/generation
```

多模态汇总分不能简单平均 IoU、accuracy、judge score 和 latency；应保留各自单位或使用预注册的任务级 success。

## 攻击与失效

- **文本先验冒充视觉理解**：不看图仍答对。
- **预处理差异**：crop、frame 或 resample 改变证据。
- **坐标映射错误**：tile 坐标未还原。
- **只测清晰短资产**：真实噪声与长输入缺失。
- **OCR/ASR 成功子集作分母**：端到端能力被高估。
- **judge 只读 caption**：没有验证原始资产。
- **模态 prompt injection**：图像文字、音频或短视频帧劫持 Agent/judge。
- **静态图片分数代表 GUI 能力**：交互状态、等待和副作用未测。
- **生成自动分数替代人类**：属性绑定和细节错误被遗漏。
- **只报模型推理时延**：解码、encoder 和后处理隐藏。

## 世界模型与具身闭环

世界模型不能只用下一帧或 feature prediction error 排序。决策链至少拆成：

| 层级 | 评测对象 | 关键反事实 |
| --- | --- | --- |
| 状态 | 任务相关信息是否可恢复 | 隐藏或替换局部观察 |
| 动力学 | 动作条件下的单步、多步预测 | 固定状态、交换动作 |
| 不确定性 | OOD 状态和长 rollout 是否变宽 | 扩大 horizon 与动作范围 |
| 规划 | 模型是否改善候选动作选择 | 与无模型、oracle model 对照 |
| 闭环 | 成功、恢复与 model exploitation | 执行扰动、观测延迟 |

生成式世界还要分开视觉逼真度、动作可控性、状态持久性、几何、实时性和闭环收益。详细协议见[世界模型总览](../world-models/index.md)与[表示预测和生成式世界](../world-models/predictive-generative-worlds.md)。

具身评测的基本单位是带 reset、初态分布、硬件和时间限制的 trial，而不是离线 action token。至少同时报告：

- task success、分阶段进展和恢复；
- collision、constraint violation、intervention 与 unsafe success；
- 新对象、指令、场景、任务和 embodiment 的逐轴泛化；
- 观测到动作的端到端 latency、jitter、控制频率和过期动作；
- 相机漂移、遮挡、丢帧、接触扰动和网络失败；
- 机器人、控制器、action contract 与 reset protocol。

不同机器人和任务集的成功率通常没有共同分母，不应排成一个 VLA 总榜。开放环 action MSE/NLL 可用于回归检查，不能替代闭环 success 与安全；完整矩阵见[规划、闭环评测与安全](../embodied/planning-evaluation-safety.md)。

## 何时拆成单模态评测

若目标是定位 image encoder、ASR 或视频采样器的改动，应先做单组件评测，再做端到端多模态任务。将所有模态一次混合，只能得到总失败，无法判断感知、grounding、推理还是工具链问题。

## 验证与报告卡

1. 同一资产在固定 decoder/preprocess 下产生相同 tensor/token digest。
2. 运行 text-only、asset-shuffle 和局部反事实基线。
3. 对坐标和时间 mapping 做手算边界样例。
4. 将 corrupt/unsupported/timeout 与模型错误分开，并报告 coverage。
5. 以资产/视频/文档 cluster 做 paired interval。
6. 对 judge 执行 swap、人工校准和注入测试。
7. 报告完整时延和成本分解。

```text
task and modality dependency
asset/source/revision/digests
decoder/preprocess/sampling protocol
model/template/harness/judge revisions
asset/item/region/trial units
invalid/unsupported/infra denominators
perception/grounding/reasoning/generation metrics
text-only and counterfactual baselines
latency/memory/token/tool cost breakdown
injection/safety/side-effect audit
cluster confidence intervals and known limits
```

judge 协议见[生成式评测与 LLM Judge](generative-judges.md)，污染与跨模态变体见[评测污染](contamination.md)，最小统计工具见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [MMMU](https://arxiv.org/abs/2311.16502)
- [OSWorld](https://arxiv.org/abs/2404.07972)
- [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982)
- [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466)
- [OpenEQA: Embodied Question Answering in the Era of Foundation Models](https://arxiv.org/abs/2404.05080)
- [SimplerEnv: Simulated Manipulation Policy Evaluation Environments with Real-to-Sim Visual Transfer](https://arxiv.org/abs/2405.05941)
