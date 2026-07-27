# 文献

主题页保留与具体主张相邻的来源，并在页末以 `Reference` 汇集本页书目；本页只提供贯穿全站的原始论文、官方实现与规范入口。[技术谱系](landscape/index.md)按问题演进组织阅读路径，并为关键工作提供论文、实现、公式与代码相互对照的深读页。书目格式见[阅读方法](guide/method.md#page-reference)。技术事实最后核验于 2026-07-27，持续演进的软件应同时记录版本或 commit。

## 基础、数据与缩放

对应[基础](foundations/index.md)、[数据](data/index.md)与[缩放实验设计](training/scaling-experiment-design.md)。

- Elman, [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1)
- Hochreiter & Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)
- Bengio et al., [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- Sutskever et al., [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- Bahdanau et al., [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Sennrich et al., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- Devlin et al., [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
- Kudo & Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226)
- Raffel et al., [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html)
- Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)
- Xie et al., [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429)
- Liu et al., [RegMix: Data Mixture as Regression for Language Model Pre-training](https://arxiv.org/abs/2407.01492)
- Lee et al., [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- Penedo et al., [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
- Soldaini et al., [Dolma: An Open Corpus of Three Trillion Tokens](https://arxiv.org/abs/2402.00159)
- Li et al., [DataComp-LM](https://arxiv.org/abs/2406.11794)
- Penedo et al., [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- Groeneveld et al., [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)
- Longpre et al., [The Data Provenance Initiative](https://arxiv.org/abs/2310.16787)

## 结构、位置与长序列

对应[模型结构](architecture/index.md)、[位置编码](architecture/position-encoding.md)、[长上下文](architecture/long-context.md)与[非 Transformer 路线](architecture/state-space-linear-attention.md)。

- Zhang & Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Press et al., [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135)
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- Fedus et al., [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
- Wang et al., [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- Dao & Gu, [Transformers are SSMs](https://arxiv.org/abs/2405.21060)
- Peng et al., [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048)
- Sun et al., [Retentive Network](https://arxiv.org/abs/2307.08621)

## Kimi 家族与 K3

[Kimi 技术谱系](landscape/kimi-timeline.md)区分各节点的报告、权重、代码、API、许可证与发布日期；[K3 工作深读](landscape/works/kimi-k3.md)连接结构、训练、系统和评测；[150 项引用图谱](landscape/kimi-k3-reference-map.md)进一步标明每项文献是直接来源、前身、并行工作、benchmark 还是比较对象。KDA、depth mixing、latent routing 与 expert parallel 分别见[Kimi Linear 与 FlashKDA](landscape/works/kimi-linear-flashkda.md)、[Attention Residuals](landscape/works/attention-residuals.md)、[Stable LatentMoE 与 Quantile Balancing](landscape/works/latentmoe-quantile-balancing.md)和[MoonEP](landscape/works/moonep.md)；视觉与音频分支另见[Kimi 家族的多模态路线](multimodal/kimi.md)。

- Kimi Team, [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/abs/2501.12599)
- Kimi Team, [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491)
- Kimi Team, [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)
- Kimi Team, [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276)
- Kimi Team, [Attention Residuals](https://arxiv.org/abs/2603.15031)
- Kimi Team, [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- Moonshot AI, [Kimi K3 model card and weights](https://huggingface.co/moonshotai/Kimi-K3)
- Moonshot AI, [FlashKDA](https://github.com/MoonshotAI/FlashKDA)
- Moonshot AI, [MoonEP](https://github.com/MoonshotAI/MoonEP)

## 多模态与生成

对应[多模态](multimodal/index.md)、[统一理解与生成](multimodal/unified-understanding-generation.md)、[文档与 GUI](multimodal/document-gui-grounding.md)、[音频语言模型](multimodal/audio-language-models.md)和[视频与世界模型](multimodal/video-world-models.md)。

- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- Alayrac et al., [Flamingo](https://arxiv.org/abs/2204.14198)
- Li et al., [BLIP-2](https://arxiv.org/abs/2301.12597)
- Liu et al., [Visual Instruction Tuning / LLaVA](https://arxiv.org/abs/2304.08485)
- Girdhar et al., [ImageBind](https://arxiv.org/abs/2305.05665)
- Team Chameleon, [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
- Yue et al., [MMMU](https://arxiv.org/abs/2311.16502)
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- Peebles & Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- Kim et al., [OCR-free Document Understanding Transformer / Donut](https://arxiv.org/abs/2111.15664)
- Lee et al., [Pix2Struct](https://arxiv.org/abs/2210.03347)
- Cheng et al., [SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents](https://arxiv.org/abs/2401.10935)
- Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper](https://arxiv.org/abs/2212.04356)
- Borsos et al., [AudioLM](https://arxiv.org/abs/2209.03143)
- Wang et al., [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E](https://arxiv.org/abs/2301.02111)
- Kondratyuk et al., [VideoPoet](https://arxiv.org/abs/2312.14125)
- Driess et al., [PaLM-E](https://arxiv.org/abs/2303.03378)
- Brohan et al., [RT-2](https://arxiv.org/abs/2307.15818)

## 预训练、适配与压缩

对应[预训练](training/pretraining.md)、[参数高效适配](training/peft.md)、[蒸馏](training/distillation.md)与[量化](inference/quantization.md)。

- Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- Wei et al., [Finetuned Language Models Are Zero-Shot Learners / FLAN](https://arxiv.org/abs/2109.01652)
- Wang et al., [Self-Instruct](https://arxiv.org/abs/2212.10560)
- Hu et al., [LoRA](https://arxiv.org/abs/2106.09685)
- Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314)
- Liu et al., [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Agarwal et al., [Generalized Knowledge Distillation for Auto-Regressive Sequence Models](https://arxiv.org/abs/2306.13649)
- Liu et al., [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323)
- Lin et al., [AWQ](https://arxiv.org/abs/2306.00978)
- Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438)

## 强化学习基础与策略优化

对应[强化学习总览](reinforcement-learning/index.md)、[历史与脉络](reinforcement-learning/history.md)、[序贯决策](reinforcement-learning/decision-processes.md)、[价值与 Bellman](reinforcement-learning/values-bellman.md)、[策略优化](reinforcement-learning/policy-gradient.md)和[手撕强化学习](practice/reinforcement-learning.md)。

- Bellman, [Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696)
- Watkins and Dayan, [Q-learning](https://doi.org/10.1007/BF00992698)
- Sutton et al., [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Haarnoja et al., [Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)
- Espeholt et al., [IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)
- Kumar et al., [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
- Ho and Ermon, [Generative Adversarial Imitation Learning](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)
- Sutton, Precup, and Singh, [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction](https://doi.org/10.1016/S0004-3702(99)00052-1)

## 语言模型反馈与策略学习

对应[反馈制度](reinforcement-learning/feedback-regimes.md)、[推理 RL 配方地图](reinforcement-learning/reasoning-rl-recipes.md)、[GAE](reinforcement-learning/advantage-estimation-gae.md)、[PPO](reinforcement-learning/trust-region-ppo.md)、[GRPO](reinforcement-learning/grpo.md)、[Ratio 与 Gate](reinforcement-learning/ratio-clipping-gating.md)、[在线 RL](training/online-rl.md)与[语言模型信用分配](reinforcement-learning/credit-assignment.md)。

- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Azar et al., [A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO](https://arxiv.org/abs/2310.12036)
- Ethayarajh et al., [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- Meng et al., [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740)
- Espeholt et al., [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- DeepSeek-AI et al., [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

### 研究前沿

下列工作适合研究复现与条件化比较，不应直接视为跨模型、跨环境的默认配方。

- Yu et al., [DAPO](https://arxiv.org/abs/2503.14476)
- Liu et al., [Understanding R1-Zero-Like Training / Dr. GRPO](https://arxiv.org/abs/2503.20783)
- Cui et al., [Process Reinforcement through Implicit Rewards / PRIME](https://arxiv.org/abs/2502.01456)
- Fu et al., [AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298)
- Yue et al., [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- MiniMax Team, [MiniMax-M1 / CISPO](https://arxiv.org/abs/2506.13585)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Yao et al., [On the Rollout-Training Mismatch in Modern RL Systems](https://www.opt-ml.org/papers/2025/paper116.pdf)
- Ma et al., [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers / R3](https://arxiv.org/abs/2510.11370)
- Ling Team, [Every Step Evolves / IcePop](https://arxiv.org/abs/2510.18855)
- Gao et al., [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)
- Qwen Team, [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
- Hou et al., [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
- Li et al., [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378)

## 推理、搜索与智能体

对应[推理](reasoning/index.md)、[搜索与验证](reasoning/search-verification.md)、[工具使用](applications/tool-use.md)、[Agent 运行时](applications/agent-runtime.md)与[安全边界](applications/agent-security.md)。

- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- Zelikman et al., [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- Yao et al., [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- Lewis et al., [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- Karpukhin et al., [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- Khattab & Zaharia, [ColBERT](https://arxiv.org/abs/2004.12832)
- Nakano et al., [WebGPT](https://arxiv.org/abs/2112.09332)
- Yao et al., [ReAct](https://arxiv.org/abs/2210.03629)
- Schick et al., [Toolformer](https://arxiv.org/abs/2302.04761)
- Wang et al., [Voyager](https://arxiv.org/abs/2305.16291)

## 训练与推理系统

对应[训练系统](systems/index.md)、[并行策略](systems/model-parallelism.md)、[注意力 kernel](systems/attention-kernels.md)与[推理服务](inference/index.md)。

- Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053)
- Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054)
- Huang et al., [GPipe](https://arxiv.org/abs/1811.06965)
- Qi et al., [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)
- Liu et al., [Ring Attention](https://arxiv.org/abs/2310.01889)
- Jacobs et al., [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509)
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- Agrawal et al., [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- Zhong et al., [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)

### 官方实现与规范

- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch DTensor](https://pytorch.org/docs/stable/distributed.tensor.html)
- [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [veRL](https://github.com/volcengine/verl)
- [slime](https://github.com/THUDM/slime)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [TRL](https://github.com/huggingface/trl)

## 评测、可靠性与安全

对应[评测](evaluation/index.md)、[统计推断](evaluation/statistical-inference.md)、[校准](evaluation/calibration-uncertainty.md)、[生成式评审](evaluation/generative-judges.md)、[污染](evaluation/contamination.md)与[安全评测](evaluation/safety-evaluation.md)。

- Liang et al., [Holistic Evaluation of Language Models / HELM](https://arxiv.org/abs/2211.09110)
- Biderman et al., [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782)
- Magnusson et al., [PALOMA: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523)
- Chen et al., [Evaluating Large Language Models Trained on Code / HumanEval](https://arxiv.org/abs/2107.03374)
- Zhou et al., [Instruction-Following Evaluation for Large Language Models / IFEval](https://arxiv.org/abs/2311.07911)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- Wang et al., [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
- Min et al., [FActScore](https://arxiv.org/abs/2305.14251)
- Wei et al., [Long-Form Factuality in Large Language Models / SAFE](https://arxiv.org/abs/2403.18802)
- Kuhn et al., [Semantic Uncertainty](https://arxiv.org/abs/2302.09664)
- Lin et al., [TruthfulQA](https://arxiv.org/abs/2109.07958)
- Manakul et al., [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
- Mazeika et al., [HarmBench](https://arxiv.org/abs/2402.04249)
- Souly et al., [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260)
- Chao et al., [JailbreakBench](https://arxiv.org/abs/2404.01318)
- Röttger et al., [XSTest](https://arxiv.org/abs/2308.01263)
- Wallace et al., [The Instruction Hierarchy](https://arxiv.org/abs/2404.13208)

## Agent 与多模态评测

对应[Agent 与工具评测](evaluation/agent-tool-evaluation.md)和[多模态评测](evaluation/multimodal-evaluation.md)。

- Jimenez et al., [SWE-bench](https://arxiv.org/abs/2310.06770)
- Zhou et al., [WebArena](https://arxiv.org/abs/2307.13854)
- Xie et al., [OSWorld](https://arxiv.org/abs/2404.07972)
- Yao et al., [tau-bench](https://arxiv.org/abs/2406.12045)
- Debenedetti et al., [AgentDojo](https://arxiv.org/abs/2406.13352)
- Liu et al., [AgentBench](https://arxiv.org/abs/2308.03688)
- Qin et al., [ToolLLM / ToolBench](https://arxiv.org/abs/2307.16789)
- Yue et al., [MMMU](https://arxiv.org/abs/2311.16502)
- Li et al., [MMBench](https://arxiv.org/abs/2307.06281)

引用论文结果时，应回到具体版本、表格、数据切分、预算和限制；本页是阅读索引，不替代原文。
