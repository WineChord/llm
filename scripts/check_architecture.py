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
        "foundations/probability-objectives.md",
        "data/sequence-construction.md",
    },
    "architecture": {
        "architecture/decoder-block.md",
        "architecture/attention-variants.md",
        "architecture/long-context.md",
        "architecture/moe.md",
    },
    "training": {
        "training/supervised-finetuning.md",
        "training/reward-preference.md",
        "training/optimizer-families.md",
    },
    "systems": {
        "systems/collectives-sharding.md",
        "systems/model-parallelism.md",
        "systems/kernels-performance.md",
    },
    "inference": {
        "inference/decoding.md",
        "inference/runtime.md",
        "inference/disaggregation.md",
    },
    "multimodal": {
        "multimodal/vision-language.md",
        "multimodal/generative-modeling.md",
        "multimodal/audio-video.md",
    },
    "agentic-rl": {
        "agentic-rl/rl-foundations.md",
        "agentic-rl/trajectory-contract.md",
        "agentic-rl/search-verification.md",
    },
    "evaluation": {
        "evaluation/language-model-evaluation.md",
        "practice/minimal-implementations.md",
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
    known = set(docs_by_relative.values())

    for relative, path in docs_by_relative.items():
        text = path.read_text(encoding="utf-8")
        titles = H1.findall(text)
        if len(titles) != 1:
            errors.append(
                f"docs/{relative}: 一级标题数量应为 1，实际为 {len(titles)}"
            )
        elif titles:
            title_to_pages[titles[0].strip()].append(relative)
        for label, raw_target in LINK.findall(text):
            if not label.strip():
                errors.append(f"docs/{relative}: 链接标签为空")
            target = normalize_target(path, raw_target)
            if target is None:
                continue
            outgoing[path] += 1
            if target in known:
                incoming[target] += 1

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
