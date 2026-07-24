# 文献与资料

以下入口优先指向原始论文、官方实现或课程。技术事实最后核验于 2026-07-24；持续演进的软件应以具体版本文档为准。

## 基础、数据与缩放

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Kudo & Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226)
- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- Soldaini et al., [Dolma](https://arxiv.org/abs/2402.00159)
- Penedo et al., [The RefinedWeb Dataset](https://arxiv.org/abs/2306.01116)
- Li et al., [DataComp-LM](https://arxiv.org/abs/2406.11794)

## 模型结构

- Su et al., [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)
- Press et al., [Train Short, Test Long / ALiBi](https://arxiv.org/abs/2108.12409)
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- Ainslie et al., [GQA](https://arxiv.org/abs/2305.13245)
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- Gu & Dao, [Mamba](https://arxiv.org/abs/2312.00752)
- Peng et al., [RWKV](https://arxiv.org/abs/2305.13048)

## 多模态

- Radford et al., [CLIP](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al., [Vision Transformer](https://arxiv.org/abs/2010.11929)
- Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198)
- Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597)
- Liu et al., [LLaVA](https://arxiv.org/abs/2304.08485)
- Girdhar et al., [ImageBind](https://arxiv.org/abs/2305.05665)
- Team Chameleon, [Chameleon](https://arxiv.org/abs/2405.09818)
- Yue et al., [MMMU](https://arxiv.org/abs/2311.16502)

## 训练、对齐与压缩

- Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685)
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323)
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978)
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438)

## 分布式训练与推理系统

- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054)
- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053)
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965)
- Yu et al., [Orca](https://arxiv.org/abs/2206.02658)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Agrawal et al., [Sarathi-Serve](https://arxiv.org/abs/2403.02310)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Megatron-LM / Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch FullyShardedDataParallel](https://pytorch.org/docs/stable/fsdp.html)

## 检索、工具、智能体与评测

- Lewis et al., [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- Yao et al., [ReAct](https://arxiv.org/abs/2210.03629)
- Schick et al., [Toolformer](https://arxiv.org/abs/2302.04761)
- Liang et al., [HELM](https://arxiv.org/abs/2211.09110)
- Hendrycks et al., [MMLU](https://arxiv.org/abs/2009.03300)
- Rein et al., [GPQA](https://arxiv.org/abs/2311.12022)
- Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- Mazeika et al., [HarmBench](https://arxiv.org/abs/2402.04249)

## 系统课程

- [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)

引用论文结果时应回到具体版本、表格、实验设置和限制；本页是导航，不替代原文。
