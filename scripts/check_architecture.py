#!/usr/bin/env python3
"""Validate navigation coverage and the internal knowledge graph."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
CONFIG = ROOT / "mkdocs.yml"
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
H1 = re.compile(r"(?m)^# (.+?)\s*$")
SECTION = re.compile(r"(?m)^#{2,3}[ \t]+(.+?)\s*$")
EXPLICIT_ANCHOR = re.compile(r"\{#([A-Za-z0-9_-]+)\}")
FENCED_CODE = re.compile(r"(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$")
NAV_TARGET = re.compile(r":\s+([A-Za-z0-9_./-]+\.md)\s*$", re.MULTILINE)
SPECIAL_OUTBOUND_EXEMPT = {
    "index.md",
    "changelog.md",
    "glossary.md",
    "references.md",
}
SPECIAL_INBOUND_EXEMPT = {
    "index.md",
    "changelog.md",
}
REQUIRED_CLUSTERS = {
    "foundation": {
        "foundations/tokenization.md",
        "foundations/in-context-learning.md",
        "foundations/probability-objectives.md",
    },
    "data": {
        "data/sources-provenance.md",
        "data/filtering-dedup.md",
        "data/mixtures-curricula.md",
        "data/sequence-construction.md",
        "data/memorization-privacy.md",
    },
    "architecture": {
        "architecture/decoder-block.md",
        "architecture/attention-variants.md",
        "architecture/position-encoding.md",
        "architecture/long-context.md",
        "architecture/moe.md",
        "architecture/state-space-linear-attention.md",
        "architecture/memory-architectures.md",
    },
    "training": {
        "training/supervised-finetuning.md",
        "training/optimizer-families.md",
        "training/distillation.md",
    },
    "reinforcement-learning": {
        "reinforcement-learning/index.md",
        "reinforcement-learning/decision-processes.md",
        "reinforcement-learning/values-bellman.md",
        "reinforcement-learning/prediction-control.md",
        "reinforcement-learning/multistep-traces.md",
        "reinforcement-learning/advantage-estimation-gae.md",
        "reinforcement-learning/function-approximation.md",
        "reinforcement-learning/policy-gradient.md",
        "reinforcement-learning/actor-critic.md",
        "reinforcement-learning/trust-region.md",
        "reinforcement-learning/trust-region-ppo.md",
        "reinforcement-learning/off-policy-correction.md",
        "reinforcement-learning/language-model-policy.md",
        "reinforcement-learning/feedback-regimes.md",
        "reinforcement-learning/rlhf-pipeline.md",
        "reinforcement-learning/critic-free-baselines.md",
        "reinforcement-learning/grpo.md",
        "reinforcement-learning/ratio-clipping-gating.md",
        "reinforcement-learning/training-inference-discrepancy.md",
        "reinforcement-learning/reasoning-rl-recipes.md",
        "reinforcement-learning/rlvr.md",
        "reinforcement-learning/credit-assignment.md",
        "training/reward-modeling.md",
        "training/offline-preference.md",
        "training/online-rl.md",
        "practice/reinforcement-learning.md",
        "practice/llm-policy-optimization.md",
    },
    "systems": {
        "systems/performance-model.md",
        "systems/gpu-execution.md",
        "systems/collectives-sharding.md",
        "systems/model-parallelism.md",
        "systems/kernels-performance.md",
        "systems/attention-kernels.md",
        "systems/precision-numerics.md",
        "systems/resilience-observability.md",
    },
    "reasoning": {
        "reasoning/test-time-compute.md",
        "reasoning/search-verification.md",
    },
    "inference": {
        "inference/decoding.md",
        "inference/runtime.md",
        "inference/disaggregation.md",
        "inference/cache-reuse.md",
        "inference/scheduling-goodput.md",
        "inference/quantization.md",
        "inference/speculative-decoding.md",
    },
    "multimodal": {
        "multimodal/vision-language.md",
        "multimodal/document-gui-grounding.md",
        "multimodal/generative-modeling.md",
        "multimodal/audio-language-models.md",
        "multimodal/video-world-models.md",
        "multimodal/embodied-agents.md",
    },
    "applications": {
        "applications/retrieval-indexing.md",
        "applications/reranking-context.md",
        "applications/grounded-generation.md",
        "applications/tool-use.md",
        "applications/agent-runtime.md",
        "applications/agent-security.md",
    },
    "agentic-rl": {
        "agentic-rl/rl-foundations.md",
        "agentic-rl/trajectory-contract.md",
        "agentic-rl/search-verification.md",
    },
    "evaluation": {
        "evaluation/language-model-evaluation.md",
        "evaluation/statistical-inference.md",
        "evaluation/calibration-uncertainty.md",
        "evaluation/generative-judges.md",
        "evaluation/agent-tool-evaluation.md",
        "evaluation/safety-evaluation.md",
        "evaluation/contamination.md",
    },
    "implementation": {
        "practice/minimal-implementations.md",
        "practice/transformer-from-scratch.md",
        "practice/training-objectives.md",
        "practice/reinforcement-learning.md",
        "practice/llm-policy-optimization.md",
        "practice/distributed-systems.md",
        "practice/inference-engine.md",
        "practice/retrieval-agents.md",
        "practice/evaluation-tooling.md",
    },
    "lineages": {
        "landscape/lineages/counts-to-learned-state.md",
        "landscape/lineages/transduction-to-attention.md",
        "landscape/lineages/pretraining-objectives.md",
        "landscape/lineages/scaling-and-context.md",
        "landscape/lineages/open-model-ecosystem.md",
        "landscape/lineages/conditional-compute.md",
        "landscape/lineages/linear-time-sequence-models.md",
        "landscape/lineages/multimodal-generation.md",
        "landscape/lineages/training-alignment.md",
        "landscape/lineages/reasoning-policy-optimization.md",
        "landscape/lineages/reasoning-verification.md",
        "landscape/lineages/distributed-training-systems.md",
        "landscape/lineages/inference-serving.md",
        "landscape/lineages/retrieval-agents.md",
        "landscape/lineages/evaluation.md",
        "landscape/kimi-timeline.md",
        "landscape/kimi-k3-reference-map.md",
        "landscape/deepseek-v4-reference-map.md",
        "landscape/glm-timeline.md",
        "landscape/glm-5-reference-map.md",
    },
    "deep-reads": {
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
        "landscape/works/dapo.md",
        "landscape/works/vapo.md",
        "landscape/works/sao-compactionrl.md",
        "landscape/works/kimi-linear-flashkda.md",
        "landscape/works/attention-residuals.md",
        "landscape/works/latentmoe-quantile-balancing.md",
        "landscape/works/kimi-k3.md",
        "landscape/works/moonep.md",
        "landscape/works/deepseek-v4.md",
        "landscape/works/deepseek-compressed-attention.md",
        "landscape/works/manifold-hyper-connections.md",
        "landscape/works/on-policy-distillation.md",
        "landscape/works/tilelang-mega-moe.md",
        "landscape/works/glm-5.md",
        "landscape/works/glm-5-architecture.md",
        "landscape/works/indexcache.md",
        "landscape/works/slime-async-agentic-rl.md",
        "landscape/works/glm-agentic-engineering.md",
        "landscape/works/megatron-zero.md",
        "landscape/works/flashattention.md",
        "landscape/works/vllm-pagedattention.md",
        "landscape/works/clip.md",
        "landscape/works/visual-language-bridges.md",
        "landscape/works/diffusion-dit-flow.md",
        "landscape/works/rag.md",
        "landscape/works/react-toolformer.md",
        "landscape/works/helm-arena.md",
    },
}
REQUIRED_EDGES = {
    "reinforcement-learning/index.md": {
        "reinforcement-learning/advantage-estimation-gae.md",
        "reinforcement-learning/trust-region.md",
        "reinforcement-learning/grpo.md",
        "reinforcement-learning/ratio-clipping-gating.md",
        "reinforcement-learning/training-inference-discrepancy.md",
        "reinforcement-learning/reasoning-rl-recipes.md",
    },
    "reinforcement-learning/advantage-estimation-gae.md": {
        "reinforcement-learning/multistep-traces.md",
        "practice/llm-policy-optimization.md",
    },
    "reinforcement-learning/trust-region-ppo.md": {
        "reinforcement-learning/trust-region.md",
        "reinforcement-learning/training-inference-discrepancy.md",
    },
    "reinforcement-learning/grpo.md": {
        "reinforcement-learning/critic-free-baselines.md",
        "landscape/works/dapo.md",
    },
    "landscape/works/dapo.md": {
        "reinforcement-learning/grpo.md",
        "practice/llm-policy-optimization.md",
    },
    "landscape/works/vapo.md": {
        "reinforcement-learning/advantage-estimation-gae.md",
        "practice/llm-policy-optimization.md",
    },
    "landscape/works/kimi-k3.md": {
        "architecture/state-space-linear-attention.md",
        "architecture/attention-position.md",
        "architecture/moe.md",
        "architecture/long-context.md",
        "training/distillation.md",
        "agentic-rl/training-systems.md",
        "systems/model-parallelism.md",
        "inference/cache-reuse.md",
        "inference/speculative-decoding.md",
        "evaluation/safety-evaluation.md",
        "landscape/works/kimi-linear-flashkda.md",
        "landscape/works/attention-residuals.md",
        "landscape/works/latentmoe-quantile-balancing.md",
        "landscape/works/moonep.md",
    },
    "landscape/works/kimi-linear-flashkda.md": {
        "landscape/works/kimi-k3.md",
        "architecture/state-space-linear-attention.md",
    },
    "landscape/works/attention-residuals.md": {
        "landscape/works/kimi-k3.md",
        "architecture/attention-position.md",
    },
    "landscape/works/latentmoe-quantile-balancing.md": {
        "landscape/works/kimi-k3.md",
        "architecture/moe.md",
    },
    "landscape/works/moonep.md": {
        "landscape/works/kimi-k3.md",
        "systems/moe-systems.md",
    },
    "landscape/kimi-timeline.md": {
        "landscape/works/kimi-k3.md",
        "landscape/kimi-k3-reference-map.md",
        "multimodal/kimi.md",
    },
    "multimodal/kimi.md": {
        "landscape/works/kimi-k3.md",
        "landscape/kimi-timeline.md",
    },
    "landscape/works/deepseek-v4.md": {
        "architecture/attention-variants.md",
        "architecture/long-context.md",
        "architecture/moe.md",
        "training/distillation.md",
        "inference/quantization.md",
        "systems/moe-systems.md",
        "evaluation/agent-tool-evaluation.md",
        "landscape/works/deepseek-compressed-attention.md",
        "landscape/works/manifold-hyper-connections.md",
        "landscape/works/on-policy-distillation.md",
        "landscape/works/tilelang-mega-moe.md",
        "landscape/deepseek-v4-reference-map.md",
    },
    "landscape/works/deepseek-compressed-attention.md": {
        "landscape/works/deepseek-v4.md",
        "architecture/attention-variants.md",
        "architecture/long-context.md",
        "inference/kv-cache.md",
    },
    "landscape/works/manifold-hyper-connections.md": {
        "landscape/works/deepseek-v4.md",
        "architecture/decoder-block.md",
        "training/optimizer-families.md",
        "systems/model-parallelism.md",
    },
    "landscape/works/on-policy-distillation.md": {
        "landscape/works/deepseek-v4.md",
        "landscape/works/kimi-k3.md",
    },
    "landscape/works/tilelang-mega-moe.md": {
        "landscape/works/deepseek-v4.md",
        "landscape/works/deepseek-compressed-attention.md",
        "landscape/works/manifold-hyper-connections.md",
        "systems/gpu-execution.md",
        "systems/moe-systems.md",
        "agentic-rl/training-systems.md",
    },
    "landscape/deepseek-timeline.md": {
        "landscape/works/deepseek-v4.md",
        "landscape/deepseek-v4-reference-map.md",
    },
    "landscape/works/glm-5.md": {
        "architecture/attention-variants.md",
        "architecture/long-context.md",
        "training/pretraining.md",
        "training/distillation.md",
        "agentic-rl/training-systems.md",
        "agentic-rl/data-environments.md",
        "inference/quantization.md",
        "evaluation/agent-tool-evaluation.md",
        "landscape/works/glm-5-architecture.md",
        "landscape/works/indexcache.md",
        "landscape/works/slime-async-agentic-rl.md",
        "landscape/works/glm-agentic-engineering.md",
        "landscape/glm-5-reference-map.md",
    },
    "landscape/works/glm-5-architecture.md": {
        "landscape/works/glm-5.md",
        "architecture/attention-variants.md",
        "training/optimizer-families.md",
        "inference/speculative-decoding.md",
    },
    "landscape/works/indexcache.md": {
        "landscape/works/glm-5.md",
        "landscape/works/glm-5-architecture.md",
        "architecture/attention-variants.md",
    },
    "landscape/works/slime-async-agentic-rl.md": {
        "landscape/works/glm-5.md",
        "agentic-rl/training-systems.md",
        "reinforcement-learning/training-inference-discrepancy.md",
    },
    "landscape/works/glm-agentic-engineering.md": {
        "landscape/works/glm-5.md",
        "agentic-rl/data-environments.md",
        "evaluation/agent-tool-evaluation.md",
    },
    "landscape/glm-timeline.md": {
        "landscape/works/glm-5.md",
        "landscape/glm-5-reference-map.md",
    },
}


def normalize_target(source: Path, target: str) -> Path | None:
    clean = target.split("#", 1)[0]
    if not clean or clean.startswith(("https://", "mailto:")):
        return None
    return (source.parent / clean).resolve()


def main() -> int:
    errors: list[str] = []
    config_text = CONFIG.read_text(encoding="utf-8")
    nav_block = config_text.split("\nnav:\n", 1)
    if len(nav_block) != 2:
        print("mkdocs.yml: 缺少 nav", file=sys.stderr)
        return 1
    nav_paths = NAV_TARGET.findall(nav_block[1])
    nav_counter = Counter(nav_paths)
    docs = sorted(DOCS_ROOT.rglob("*.md"))
    docs_by_relative = {
        path.relative_to(DOCS_ROOT).as_posix(): path.resolve() for path in docs
    }

    for relative, count in sorted(nav_counter.items()):
        if relative not in docs_by_relative:
            errors.append(f"mkdocs.yml: 导航目标不存在：{relative}")
        if count > 1:
            errors.append(f"mkdocs.yml: 导航目标重复 {count} 次：{relative}")
    for relative in sorted(set(docs_by_relative) - set(nav_counter)):
        errors.append(f"docs/{relative}: 页面未进入导航")

    for cluster, required in REQUIRED_CLUSTERS.items():
        missing = sorted(required - set(docs_by_relative))
        if missing:
            errors.append(f"{cluster}: 缺少核心页面：{', '.join(missing)}")

    title_to_pages: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[Path, int] = defaultdict(int)
    incoming: dict[Path, int] = defaultdict(int)
    edges: set[tuple[str, str]] = set()
    known = set(docs_by_relative.values())
    relative_by_path = {
        path: relative for relative, path in docs_by_relative.items()
    }

    for relative, path in docs_by_relative.items():
        text = path.read_text(encoding="utf-8")
        prose = FENCED_CODE.sub("", text)
        titles = H1.findall(prose)
        if len(titles) != 1:
            errors.append(
                f"docs/{relative}: 一级标题数量应为 1，实际为 {len(titles)}"
            )
        elif titles:
            title_to_pages[titles[0].strip()].append(relative)
        sections = [
            re.sub(r"[ \t]+\{#[A-Za-z0-9_-]+\}[ \t]*$", "", title)
            .strip()
            .rstrip("#")
            .strip()
            for title in SECTION.findall(prose)
        ]
        duplicate_sections = sorted(
            title for title, count in Counter(sections).items() if count > 1
        )
        for title in duplicate_sections:
            errors.append(f"docs/{relative}: 二/三级标题重复“{title}”")
        anchors = EXPLICIT_ANCHOR.findall(prose)
        for anchor, count in sorted(Counter(anchors).items()):
            if count > 1:
                errors.append(
                    f"docs/{relative}: 显式 fragment 重复 {count} 次：{anchor}"
                )
        for label, raw_target in LINK.findall(prose):
            if not label.strip():
                errors.append(f"docs/{relative}: 链接标签为空")
            target = normalize_target(path, raw_target)
            if target is None:
                continue
            outgoing[path] += 1
            if target in known:
                incoming[target] += 1
                edges.add((relative, relative_by_path[target]))

    for source, targets in REQUIRED_EDGES.items():
        for target in targets:
            if (source, target) not in edges:
                errors.append(
                    f"docs/{source}: 缺少架构要求的正文链接：{target}"
                )

    for title, pages in sorted(title_to_pages.items()):
        if len(pages) > 1:
            errors.append(f"一级标题重复“{title}”：{', '.join(pages)}")

    for relative, path in docs_by_relative.items():
        if relative not in SPECIAL_OUTBOUND_EXEMPT and outgoing[path] == 0:
            errors.append(f"docs/{relative}: 缺少正文内部链接")
        if relative not in SPECIAL_INBOUND_EXEMPT and incoming[path] == 0:
            errors.append(f"docs/{relative}: 没有其他正文页面指向该页")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "架构检查通过："
        f"{len(docs)} 个页面全部进入导航，"
        f"{sum(outgoing.values())} 条内部链接形成知识图"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
