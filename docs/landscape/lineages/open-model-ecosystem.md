# 从可下载权重到可研究系统

“开放模型”听起来像一个二元标签，实际是一组逐渐展开的研究条件。只提供 API，研究者能观察输出；开放权重后，可以检查表示、微调和本地部署；再开放训练代码、数据、检查点与日志，才可能追问一次能力究竟由哪段数据、哪次训练或哪项系统选择产生。开放模型生态的历史，不只是更多 checkpoint 被上传，而是可研究对象一步步从结果向训练过程倒推。

## API 时代留下的缺口

[GPT-3](https://arxiv.org/abs/2005.14165)展示了规模与 in-context learning 的强大组合，也让一个现实矛盾更加明显：训练成本高到难以独立复现，而 API 又隐藏了权重、数据和系统细节。研究者可以比较行为，却很难定位内部机制、复查训练污染，或验证一个改进是否能在同一基座上重现。

开放生态随后沿两条线推进：一条努力释放更大模型，另一条努力释放更完整的研究过程。两者相关，但并不等价。

## 2022：规模本身成为公共工程

[GPT-NeoX-20B](https://arxiv.org/abs/2204.06745)开放权重、训练与评测代码，并建立在 Megatron 与 DeepSpeed 之上；[OPT](https://arxiv.org/abs/2205.01068)发布从 125M 到 175B 的 decoder-only 模型及训练日志；[BLOOM](https://arxiv.org/abs/2211.05100)则通过 BigScience 协作训练 176B 多语言模型，同时公开 ROOTS 的数据目录、治理与跨机构工程过程，但这不等于所有底层语料都可重新分发。

这几项工作的意义不只在参数量。它们让训练失败、分布式配置、数据语言比例和许可证进入公开讨论，也让后来的量化、微调与推理系统拥有真实的大模型对象。对应的官方入口包括 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)、已归档 metaseq 仓库中的 [OPT 项目](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT)与 [BLOOM](https://huggingface.co/bigscience/bloom)。

不过，“权重可下载”仍不能重建训练。数据过滤、精确顺序、优化器状态和中间 checkpoint 只要缺一部分，因果实验就可能无法复现。

## 2023：小而强的基座改变下游生态

[LLaMA](https://arxiv.org/abs/2302.13971)强调用更多公开数据训练较小模型，在多个基准上达到有竞争力的性能；首代权重采用申请访问与研究用途许可证。[Llama 2](https://arxiv.org/abs/2307.09288)把公开权重的可得性推得更广，但其社区许可证仍不等同于 OSI 定义的 open source。两者共同使低秩适配、量化微调、领域数据和本地推理迅速围绕统一模型族积累。

这段历史常被简化成“开源模型追上闭源模型”，但更准确的转折是实验门槛下降：

- 相同架构和 tokenizer 上可以比较数据与后训练方法；
- 7B—70B 的尺寸梯度允许研究 scale 与资源边界；
- 标准模型格式让推理引擎、量化方法和评测 harness 形成共同接口。

同时，Llama 的许可证与访问条款也提醒人们：open weights、open source 与 unrestricted use 是不同维度。[官方模型仓库](https://github.com/meta-llama/llama)应与具体版本许可证一起阅读。

## 从模型发布到完整科学对象

[OLMo](https://arxiv.org/abs/2402.00838)把开放范围扩展到训练数据、训练和评测代码、中间 checkpoint 与日志；[官方仓库](https://github.com/allenai/OLMo)和 Dolma 数据使研究者能够检查比最终权重更早的决策。这里的目标不只是提供一个可部署模型，而是让训练过程本身成为可复查实验。

与此同时，[Qwen](https://arxiv.org/abs/2309.16609)、[Qwen2](https://arxiv.org/abs/2407.10671)与 [DeepSeek](../deepseek-timeline.md)等家族推动了多语言、代码、数学、MoE、长上下文和多模态权重的开放。它们说明开放生态已经不再等于“英文 dense base model”，但不同家族披露的数据、系统和后训练细节仍有明显差异。

## 用六层而不是一个标签

判断一个发布是否支持目标研究，可以沿六层检查：

| 层 | 可回答的问题 |
| --- | --- |
| 接口 | 能否稳定调用并记录精确版本 |
| 权重 | 能否本地推理、分析与适配 |
| 代码 | 能否重放模型、训练与评测逻辑 |
| 数据 | 能否核查来源、过滤、混合与污染 |
| 过程 | 是否提供中间 checkpoint、日志和失败记录 |
| 权利 | 许可证是否允许目标研究、修改与部署 |

开放程度不是单调的总分。一个模型可能权重开放、数据未知；另一个模型可能完整开放训练过程，却规模较小。选择对象时应先写研究问题，再决定哪一层不可缺少。

## 开放生态怎样反过来改变研究

权重与代码的可得性让几个方向加速形成：

1. [LoRA、QLoRA](../../training/peft.md)等适配方法可以跨尺寸反复验证；
2. [GPTQ、AWQ](../../inference/quantization.md)与推理引擎能在真实权重上比较；
3. [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)等公共评测栈获得统一对象；
4. 数据归因、可解释性和模型合并不再完全依赖 API 黑箱；
5. 官方训练仓库暴露的系统问题又推动[并行训练](../../systems/model-parallelism.md)与[容错](../../systems/checkpointing.md)演进。

开放并不自动保证结论可靠。模型卡可能混合 base 与 instruct 结果，后训练数据可能未披露，社区转换格式也可能偏离原权重。版本、哈希、模板、许可证和评测 harness 仍需一并记录。

## 继续阅读

模型规模与数据配比的变化见[从规模规律到上下文内适应](scaling-and-context.md)，训练数字的可比边界见[训练 token 口径](../training-tokens.md)，完整发布事件怎样拆分可参考[DeepSeek 演化案例](../deepseek-timeline.md)。这条谱系关心的是可研究性，不替代具体模型的机制分析。
