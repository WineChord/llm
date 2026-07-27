#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md"))
FILES = [ROOT / "README.md", *DOCS]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
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
    "practice/distributed-systems.md",
    "practice/inference-engine.md",
    "practice/test-time-compute.md",
    "practice/retrieval-agents.md",
    "practice/multimodal.md",
    "practice/evaluation-tooling.md",
)
errors: list[str] = []


def table_cell_count(line: str) -> int:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith(r"\|"):
        body = body[:-1]
    return len(re.split(r"(?<!\\)\|", body))


for relative in REQUIRED_PAGES:
    if not (ROOT / "docs" / relative).is_file():
        errors.append(f"docs/{relative}: 缺少知识架构核心页面")

for path in FILES:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
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
print(f"内容检查通过：{len(DOCS)} 个页面，{source_count} 个论文链接")
