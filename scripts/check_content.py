#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md"))
FILES = [ROOT / "README.md", *DOCS]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\((https://[^)]+)\)")
ARXIV = re.compile(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}")
TABLE_SEPARATOR = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "macOS home path": re.compile(r"/Users/[^/\s]+/"),
    "Windows home path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
}
PRIVATE_TRACES = (
    "the user asked",
    "the user wanted",
    "according to the prompt",
    "according to our conversation",
    "private instruction",
    "用户要求我",
    "根据用户指令",
    "根据本次对话",
    "聊天记录",
    "内部工作流",
    "迁移自",
    "提炼自",
    "来源项目",
    "源仓库",
)
REQUIRED_PAGES = (
    "guide/architecture.md",
    "guide/coverage.md",
    "foundations/probability-objectives.md",
    "foundations/tokenization.md",
    "foundations/in-context-learning.md",
    "data/sources-provenance.md",
    "data/filtering-dedup.md",
    "data/mixtures-curricula.md",
    "data/sequence-construction.md",
    "data/feedback-trajectories.md",
    "data/memorization-privacy.md",
    "architecture/decoder-block.md",
    "architecture/attention-variants.md",
    "architecture/position-encoding.md",
    "architecture/long-context.md",
    "architecture/moe.md",
    "architecture/state-space-linear-attention.md",
    "architecture/memory-architectures.md",
    "guide/evidence.md",
    "landscape/index.md",
    "landscape/training-tokens.md",
    "multimodal/native-generation.md",
    "multimodal/vision-language.md",
    "multimodal/document-gui-grounding.md",
    "multimodal/unified-understanding-generation.md",
    "multimodal/generative-modeling.md",
    "multimodal/audio-language-models.md",
    "multimodal/video-world-models.md",
    "multimodal/embodied-agents.md",
    "training/supervised-finetuning.md",
    "training/reward-preference.md",
    "training/optimizer-families.md",
    "training/scaling-experiment-design.md",
    "training/distillation.md",
    "training/peft.md",
    "training/reward-modeling.md",
    "training/offline-preference.md",
    "training/online-rl.md",
    "training/reasoning-posttraining.md",
    "systems/performance-model.md",
    "systems/gpu-execution.md",
    "systems/collectives-sharding.md",
    "systems/model-parallelism.md",
    "systems/kernels-performance.md",
    "systems/attention-kernels.md",
    "systems/moe-systems.md",
    "systems/precision-numerics.md",
    "systems/resilience-observability.md",
    "reasoning/test-time-compute.md",
    "reasoning/search-verification.md",
    "inference/decoding.md",
    "inference/runtime.md",
    "inference/disaggregation.md",
    "inference/cache-reuse.md",
    "inference/scheduling-goodput.md",
    "inference/quantization.md",
    "inference/speculative-decoding.md",
    "inference/benchmarking-reliability.md",
    "applications/retrieval-indexing.md",
    "applications/reranking-context.md",
    "applications/grounded-generation.md",
    "applications/tool-use.md",
    "applications/agent-runtime.md",
    "applications/memory-planning.md",
    "applications/agent-security.md",
    "applications/coding-agents.md",
    "agentic-rl/index.md",
    "agentic-rl/rl-foundations.md",
    "agentic-rl/math-algorithms.md",
    "agentic-rl/trajectory-contract.md",
    "agentic-rl/search-verification.md",
    "agentic-rl/training-systems.md",
    "reinforcement-learning/index.md",
    "evaluation/benchmark-registry.md",
    "evaluation/language-model-evaluation.md",
    "evaluation/statistical-inference.md",
    "evaluation/calibration-uncertainty.md",
    "evaluation/generative-judges.md",
    "evaluation/agent-tool-evaluation.md",
    "evaluation/multimodal-evaluation.md",
    "evaluation/hallucination.md",
    "evaluation/instruction-following.md",
    "evaluation/safety-evaluation.md",
    "evaluation/contamination.md",
    "evaluation/production-reliability.md",
    "practice/minimal-implementations.md",
    "practice/tensor-primitives.md",
    "practice/transformer-from-scratch.md",
    "practice/tokenizers.md",
    "practice/sequence-models.md",
    "practice/training-objectives.md",
    "practice/reinforcement-learning.md",
    "practice/distributed-systems.md",
    "practice/inference-engine.md",
    "practice/test-time-compute.md",
    "practice/retrieval-agents.md",
    "practice/multimodal.md",
    "practice/evaluation-tooling.md",
)
LINEAGE_PAGES = (
    "landscape/lineages/counts-to-learned-state.md",
    "landscape/lineages/transduction-to-attention.md",
    "landscape/lineages/pretraining-objectives.md",
    "landscape/lineages/scaling-and-context.md",
    "landscape/lineages/open-model-ecosystem.md",
    "landscape/lineages/conditional-compute.md",
    "landscape/lineages/linear-time-sequence-models.md",
    "landscape/lineages/multimodal-generation.md",
    "landscape/lineages/training-alignment.md",
    "landscape/lineages/reasoning-verification.md",
    "landscape/lineages/distributed-training-systems.md",
    "landscape/lineages/inference-serving.md",
    "landscape/lineages/retrieval-agents.md",
    "landscape/lineages/evaluation.md",
)
WORK_PAGES = (
    "landscape/works/lstm.md",
    "landscape/works/seq2seq-and-neural-alignment.md",
    "landscape/works/attention-is-all-you-need.md",
    "landscape/works/generative-pretraining-gpt.md",
    "landscape/works/bert.md",
    "landscape/works/t5.md",
    "landscape/works/scaling-laws-chinchilla.md",
    "landscape/works/sparse-moe.md",
    "landscape/works/s4-mamba.md",
    "landscape/works/instructgpt.md",
    "landscape/works/dpo.md",
    "landscape/works/deepseek-r1.md",
    "landscape/works/sao-compactionrl.md",
    "landscape/works/megatron-zero.md",
    "landscape/works/flashattention.md",
    "landscape/works/vllm-pagedattention.md",
    "landscape/works/clip.md",
    "landscape/works/visual-language-bridges.md",
    "landscape/works/diffusion-dit-flow.md",
    "landscape/works/rag.md",
    "landscape/works/react-toolformer.md",
    "landscape/works/helm-arena.md",
)
RL_PAGES = (
    "reinforcement-learning/history.md",
    "reinforcement-learning/decision-processes.md",
    "reinforcement-learning/values-bellman.md",
    "reinforcement-learning/prediction-control.md",
    "reinforcement-learning/multistep-traces.md",
    "reinforcement-learning/function-approximation.md",
    "reinforcement-learning/exploration-entropy.md",
    "reinforcement-learning/models-planning-hierarchy.md",
    "reinforcement-learning/offline-imitation.md",
    "reinforcement-learning/constraints-multiagent.md",
    "reinforcement-learning/policy-gradient.md",
    "reinforcement-learning/actor-critic.md",
    "reinforcement-learning/trust-region-ppo.md",
    "reinforcement-learning/off-policy-correction.md",
    "reinforcement-learning/language-model-policy.md",
    "reinforcement-learning/kl-regularized-control.md",
    "reinforcement-learning/feedback-regimes.md",
    "reinforcement-learning/rlhf-pipeline.md",
    "reinforcement-learning/critic-free-baselines.md",
    "reinforcement-learning/rlvr.md",
    "reinforcement-learning/verifiers-reward-shaping.md",
    "reinforcement-learning/credit-assignment.md",
    "reinforcement-learning/evaluation-debugging.md",
)
REFERENCE_HEADING = "## Reference {#reference}"
REFERENCE_EXEMPT_PAGES = {
    "agentic-rl/reading-list.md",
    "changelog.md",
    "glossary.md",
    "references.md",
}
LEGACY_REFERENCE_HEADINGS = {
    "## 一手来源",
    "## 原作与实现",
    "## 原作与实现边界",
    "## 原始工作与实现边界",
    "## 原作与代码",
}
GENERIC_REFERENCE_LABELS = {
    "论文",
    "原论文",
    "项目页",
    "官方实现",
    "作者实现",
    "公开实现",
    "官方仓库",
    "官方文档",
    "官方博客",
    "技术报告",
    "报告",
    "文档",
    "参考库",
    "仓库",
}
errors: list[str] = []
reference_page_count = 0


def table_cell_count(line: str) -> int:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith(r"\|"):
        body = body[:-1]
    return len(re.split(r"(?<!\\)\|", body))


def unfenced_lines(text: str) -> list[str]:
    visible: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            visible.append(line)
    return visible


for relative in (*REQUIRED_PAGES, *LINEAGE_PAGES, *WORK_PAGES, *RL_PAGES):
    if not (ROOT / "docs" / relative).is_file():
        errors.append(f"docs/{relative}: 缺少知识架构核心页面")

for relative in LINEAGE_PAGES:
    path = ROOT / "docs" / relative
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if len(text) < 2000:
        errors.append(f"docs/{relative}: 技术谱系展开不足")
    if len(re.findall(r"\]\(https://", text)) < 2:
        errors.append(f"docs/{relative}: 技术谱系缺少足够的一手论文入口")
    if len(re.findall(r"\]\((?!https?://)[^)]+\.md(?:#[^)]+)?\)", text)) < 2:
        errors.append(f"docs/{relative}: 技术谱系缺少机制与工作页链接")

for relative in WORK_PAGES:
    path = ROOT / "docs" / relative
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if len(text) < 2400:
        errors.append(f"docs/{relative}: 关键工作深读不足")
    if len(re.findall(r"\]\(https://", text)) < 2:
        errors.append(f"docs/{relative}: 关键工作缺少足够的一手论文或官方实现")
    if "```python" not in text or not re.search(r"\bassert\b|torch\.testing\.", text):
        errors.append(f"docs/{relative}: 缺少带断言的可执行 reference")
    if len(re.findall(r"\]\((?!https?://)[^)]+\.md(?:#[^)]+)?\)", text)) < 2:
        errors.append(f"docs/{relative}: 关键工作缺少前后谱系与机制链接")

for relative in RL_PAGES:
    path = ROOT / "docs" / relative
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    if len(text) < 3000:
        errors.append(f"docs/{relative}: 强化学习机制展开不足")
    if len(re.findall(r"\]\(https://", text)) < 2:
        errors.append(f"docs/{relative}: 强化学习机制缺少足够的一手来源")
    if len(re.findall(r"\]\((?!https?://)[^)]+\.md(?:#[^)]+)?\)", text)) < 2:
        errors.append(f"docs/{relative}: 强化学习机制缺少上下游链接")

for path in FILES:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    visible_lines = unfenced_lines(text)
    if path.suffix == ".md" and not any(line.startswith("# ") for line in lines):
        errors.append(f"{path.relative_to(ROOT)}: 缺少一级标题")
    if path.parent != ROOT and len(text.strip()) < 240:
        errors.append(f"{path.relative_to(ROOT)}: 页面内容过短")
    open_fence = 0
    for number, line in enumerate(lines, 1):
        if line.startswith("```"):
            open_fence = 0 if open_fence else number
        if (
            not open_fence
            and TABLE_SEPARATOR.match(line)
            and number > 1
            and table_cell_count(lines[number - 2]) != table_cell_count(line)
        ):
            errors.append(
                f"{path.relative_to(ROOT)}:{number}: "
                "Markdown 表头与分隔行列数不一致"
            )
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                errors.append(f"{path.relative_to(ROOT)}:{number}: 检测到 {name} 模式")
        lowered = line.lower()
        for trace in PRIVATE_TRACES:
            if trace in lowered:
                errors.append(f"{path.relative_to(ROOT)}:{number}: 检测到非公开工作流表述")
    if open_fence:
        errors.append(f"{path.relative_to(ROOT)}:{open_fence}: 代码块未闭合")
    if re.search(r"\b(?:TODO|TBD)\b|待补充|敬请期待", text, re.IGNORECASE):
        errors.append(f"{path.relative_to(ROOT)}: 不得发布占位内容")
    if path in DOCS:
        relative = path.relative_to(ROOT / "docs").as_posix()
        legacy = sorted(LEGACY_REFERENCE_HEADINGS.intersection(visible_lines))
        if legacy:
            errors.append(
                f"docs/{relative}: 使用旧来源标题，应统一为 {REFERENCE_HEADING}："
                + "、".join(legacy)
            )
        reference_indexes = [
            index
            for index, line in enumerate(visible_lines)
            if line == REFERENCE_HEADING
        ]
        requires_reference = (
            path.name != "index.md"
            and relative not in REFERENCE_EXEMPT_PAGES
        )
        if requires_reference and len(reference_indexes) != 1:
            errors.append(
                f"docs/{relative}: 知识页必须恰有一个 {REFERENCE_HEADING}"
            )
        if reference_indexes:
            reference_page_count += 1
            if len(reference_indexes) != 1:
                errors.append(
                    f"docs/{relative}: {REFERENCE_HEADING} 不得重复"
                )
            else:
                reference_index = reference_indexes[0]
                section_lines = visible_lines[reference_index + 1 :]
                if any(line.startswith("## ") for line in section_lines):
                    errors.append(
                        f"docs/{relative}: Reference 必须是最后一个二级章节"
                    )
                section = "\n".join(section_lines)
                sources = REFERENCE_LINK.findall(section)
                minimum = 2 if relative in (*LINEAGE_PAGES, *WORK_PAGES) else 1
                if len(sources) < minimum:
                    errors.append(
                        f"docs/{relative}: Reference 至少需要 {minimum} 个 HTTPS Markdown 链接"
                    )
                normalized = [
                    url.split("#", 1)[0].rstrip("/") for _, url in sources
                ]
                if len(normalized) != len(set(normalized)):
                    errors.append(
                        f"docs/{relative}: Reference 存在重复目标"
                    )
                normalized_labels = [
                    re.sub(r"[*_`]", "", label).strip().casefold()
                    for label, _ in sources
                ]
                if len(normalized_labels) != len(set(normalized_labels)):
                    errors.append(
                        f"docs/{relative}: Reference 存在无法区分的重复链接标题"
                    )
                without_links = REFERENCE_LINK.sub("", section)
                if re.search(r"https?://", without_links):
                    errors.append(
                        f"docs/{relative}: Reference 不得包含裸 URL"
                    )
                for label, _ in sources:
                    clean_label = re.sub(r"[*_`]", "", label).strip()
                    if (
                        clean_label in GENERIC_REFERENCE_LABELS
                        or len(clean_label) < 4
                    ):
                        errors.append(
                            f"docs/{relative}: Reference 链接标题缺少可独立识别的信息："
                            f"{label}"
                        )
    for target in LINK.findall(text):
        target = target.split("#", 1)[0]
        if not target or target.startswith(("https://", "mailto:")):
            continue
        if target.startswith("http://"):
            errors.append(f"{path.relative_to(ROOT)}: 外部链接必须使用 HTTPS：{target}")
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            errors.append(f"{path.relative_to(ROOT)}: 内部链接不存在：{target}")

source_count = len({
    url
    for path in DOCS
    for url in ARXIV.findall(path.read_text(encoding="utf-8"))
})
if source_count < 120:
    errors.append(f"docs/: 原始论文链接不足：{source_count} < 120")

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(
    f"内容检查通过：{len(DOCS)} 个页面，{reference_page_count} 个页级 Reference，"
    f"{source_count} 个论文链接"
)
