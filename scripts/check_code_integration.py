#!/usr/bin/env python3
"""Validate that semantic references live with their canonical explanations."""

from __future__ import annotations

import argparse
import ast
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BLOCK = re.compile(
    r"^```python[^\n]*\n(?P<code>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
DETAILS = re.compile(
    r"<details(?P<attrs>[^>]*)>(?P<body>.*?)</details>",
    re.IGNORECASE | re.DOTALL,
)
SUMMARY = re.compile(
    r"^\s*<summary\s+id=\"(?P<id>[a-z][a-z0-9-]*)\">"
    r"(?P<label>.*?)"
    r"<span\s+class=\"code-disclosure__meta\">"
    r"Python\s+·\s+(?P<lines>\d+)\s+行"
    r"</span>\s*</summary>\s*"
    r"<div\s+class=\"code-disclosure__body\"\s+markdown=\"1\">\s*"
    r"(?P<payload>.*?)"
    r"</div>\s*$",
    re.IGNORECASE | re.DOTALL,
)
FORBIDDEN_SCAFFOLDING = (
    "argparse",
    "click.",
    "FastAPI",
    "Flask(",
    "uvicorn",
    "logging.basicConfig",
    'if __name__ == "__main__"',
    "if __name__ == '__main__'",
)
MAX_OPEN_LINES = 35
MAX_BLOCK_LINES = 80
MAX_LINE_LENGTH = 100
CANONICAL_PAGES = (
    "foundations/probability-objectives.md",
    "foundations/tokenization.md",
    "data/sequence-construction.md",
    "data/filtering-dedup.md",
    "data/mixtures-curricula.md",
    "data/feedback-trajectories.md",
    "architecture/decoder-block.md",
    "architecture/attention-variants.md",
    "architecture/position-encoding.md",
    "architecture/state-space-linear-attention.md",
    "architecture/memory-architectures.md",
    "architecture/moe.md",
    "training/offline-preference.md",
    "training/pretraining.md",
    "training/supervised-finetuning.md",
    "training/reward-modeling.md",
    "training/peft.md",
    "training/optimizer-families.md",
    "training/distillation.md",
    "systems/performance-model.md",
    "systems/gpu-execution.md",
    "systems/attention-kernels.md",
    "systems/collectives-sharding.md",
    "systems/model-parallelism.md",
    "systems/checkpointing.md",
    "systems/moe-systems.md",
    "systems/precision-numerics.md",
    "inference/decoding.md",
    "inference/kv-cache.md",
    "inference/cache-reuse.md",
    "inference/runtime.md",
    "inference/disaggregation.md",
    "inference/speculative-decoding.md",
    "inference/quantization.md",
    "inference/scheduling-goodput.md",
    "reinforcement-learning/advantage-estimation-gae.md",
    "reinforcement-learning/decision-processes.md",
    "reinforcement-learning/values-bellman.md",
    "reinforcement-learning/prediction-control.md",
    "reinforcement-learning/function-approximation.md",
    "reinforcement-learning/exploration-entropy.md",
    "reinforcement-learning/policy-gradient.md",
    "reinforcement-learning/actor-critic.md",
    "reinforcement-learning/trust-region-ppo.md",
    "reinforcement-learning/off-policy-correction.md",
    "reinforcement-learning/grpo.md",
    "reinforcement-learning/ratio-clipping-gating.md",
    "reinforcement-learning/trust-region.md",
    "reinforcement-learning/multistep-traces.md",
    "reinforcement-learning/critic-free-baselines.md",
    "reinforcement-learning/language-model-policy.md",
    "reinforcement-learning/kl-regularized-control.md",
    "reinforcement-learning/training-inference-discrepancy.md",
    "applications/tool-use.md",
    "applications/agent-runtime.md",
    "applications/retrieval-indexing.md",
    "applications/reranking-context.md",
    "applications/grounded-generation.md",
    "agentic-rl/trajectory-contract.md",
    "evaluation/language-model-evaluation.md",
    "evaluation/statistical-inference.md",
    "evaluation/calibration-uncertainty.md",
    "evaluation/generative-judges.md",
    "evaluation/agent-tool-evaluation.md",
    "evaluation/hallucination.md",
    "evaluation/safety-evaluation.md",
    "reasoning/search-verification.md",
    "reasoning/test-time-compute.md",
    "multimodal/architecture-training.md",
    "multimodal/vision-language.md",
    "multimodal/document-gui-grounding.md",
    "multimodal/unified-understanding-generation.md",
    "multimodal/generative-modeling.md",
    "multimodal/audio-language-models.md",
    "multimodal/video-world-models.md",
    "multimodal/foundations/signals-tokenization.md",
    "multimodal/foundations/alignment-fusion.md",
    "multimodal/foundations/position-time-masks.md",
    "multimodal/foundations/data-training-systems.md",
    "multimodal/vision/representation-grounding.md",
    "multimodal/vision/spatial-3d.md",
    "multimodal/image-generation/history-autoregressive-gan.md",
    "multimodal/image-generation/autoencoders-tokenizers.md",
    "multimodal/image-generation/diffusion-score.md",
    "multimodal/image-generation/latent-dit-flow.md",
    "multimodal/image-generation/control-editing-evaluation.md",
    "multimodal/audio/representations-understanding.md",
    "multimodal/audio/generation-streaming.md",
    "multimodal/video/understanding-long-context.md",
    "multimodal/video/generation.md",
    "multimodal/omni/any-to-any.md",
    "world-models/dynamics-planning.md",
    "world-models/predictive-generative-worlds.md",
    "embodied/state-action-policies.md",
    "embodied/vla-data-lineage.md",
    "embodied/planning-evaluation-safety.md",
)


def nonblank_lines(code: str) -> int:
    return sum(bool(line.strip()) for line in code.splitlines())


def containing_disclosure(
    offset: int,
    disclosures: list[tuple[int, int, re.Match[str]]],
) -> re.Match[str] | None:
    for start, end, match in disclosures:
        if start <= offset < end:
            return match
    return None


def route_for(relative: str) -> str:
    path = Path(relative)
    return path.with_suffix("").as_posix() + "/"


def validate_disclosure_match(
    relative: str,
    disclosure: re.Match[str],
    ids: set[str],
    errors: list[str],
) -> None:
    attrs = disclosure.group("attrs")
    body = disclosure.group("body")
    if re.search(r"\bopen(?:\s|=|$)", attrs, re.IGNORECASE):
        errors.append(f"docs/{relative}: code-disclosure 必须默认折叠")
    if re.search(r"<details\b", body, re.IGNORECASE):
        errors.append(f"docs/{relative}: code-disclosure 不得嵌套")
    summary = SUMMARY.fullmatch(body)
    if summary is None:
        errors.append(
            f"docs/{relative}: 折叠代码必须使用规范 summary、meta 与 body"
        )
        return
    identifier = summary.group("id")
    if identifier in ids:
        errors.append(f"docs/{relative}: 折叠代码 fragment 重复：#{identifier}")
    ids.add(identifier)
    label = re.sub(r"<[^>]+>", "", summary.group("label")).strip()
    if len(label) < 4 or label in {"完整实现", "展开代码", "更多代码"}:
        errors.append(
            f"docs/{relative}: 折叠摘要必须描述具体内容：#{identifier}"
        )
    enclosed = BLOCK.findall(summary.group("payload"))
    if not enclosed:
        errors.append(f"docs/{relative}: 折叠区 #{identifier} 缺少 Python 代码")
    actual_lines = sum(nonblank_lines(code) for code in enclosed)
    declared_lines = int(summary.group("lines"))
    if actual_lines != declared_lines:
        errors.append(
            f"docs/{relative}: #{identifier} 行数标记为 "
            f"{declared_lines}，实际为 {actual_lines}"
        )


def validate_supplemental_disclosures(errors: list[str]) -> int:
    canonical = set(CANONICAL_PAGES)
    total = 0
    for path in sorted(DOCS.rglob("*.md")):
        relative = path.relative_to(DOCS).as_posix()
        if relative in canonical:
            continue
        text = path.read_text(encoding="utf-8")
        disclosures = [
            (match.start(), match.end(), match)
            for match in DETAILS.finditer(text)
            if "code-disclosure" in match.group("attrs")
        ]
        total += len(disclosures)
        ids: set[str] = set()
        for _, _, disclosure in disclosures:
            validate_disclosure_match(relative, disclosure, ids, errors)
        if path.parent == DOCS / "practice":
            for index, block in enumerate(BLOCK.finditer(text), 1):
                count = nonblank_lines(block.group("code"))
                folded = (
                    containing_disclosure(block.start(), disclosures) is not None
                )
                if count > MAX_OPEN_LINES and not folded:
                    errors.append(
                        f"docs/{relative}: Python 代码块 {index} 应默认折叠 "
                        f"({count} > {MAX_OPEN_LINES})"
                    )
    return total


class DisclosureParser(HTMLParser):
    """Collect generated disclosure structure without browser dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[dict[str, object]] = []
        self.disclosures: list[dict[str, object]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "details" and "code-disclosure" in classes:
            entry: dict[str, object] = {
                "open": "open" in attributes,
                "summary": 0,
                "summary_id": None,
                "code": 0,
                "nested": bool(self.stack),
            }
            self.stack.append(entry)
            self.disclosures.append(entry)
            return
        if not self.stack:
            return
        current = self.stack[-1]
        if tag == "summary":
            current["summary"] = int(current["summary"]) + 1
            current["summary_id"] = attributes.get("id")
        elif tag == "code":
            current["code"] = int(current["code"]) + 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "details" and self.stack:
            self.stack.pop()


def validate_source(errors: list[str]) -> tuple[int, int, dict[str, set[str]]]:
    total_blocks = 0
    total_disclosures = 0
    identifiers: dict[str, set[str]] = {}
    for relative in CANONICAL_PAGES:
        path = DOCS / relative
        if not path.is_file():
            errors.append(f"docs/{relative}: 缺少正文代码归属页")
            continue
        text = path.read_text(encoding="utf-8")
        disclosures = [
            (match.start(), match.end(), match)
            for match in DETAILS.finditer(text)
            if "code-disclosure" in match.group("attrs")
        ]
        total_disclosures += len(disclosures)
        ids: set[str] = set()
        for _, _, disclosure in disclosures:
            validate_disclosure_match(relative, disclosure, ids, errors)
        blocks = list(BLOCK.finditer(text))
        total_blocks += len(blocks)
        if not blocks:
            errors.append(f"docs/{relative}: 正文缺少 Python semantic reference")
            continue
        open_blocks = [
            block
            for block in blocks
            if containing_disclosure(block.start(), disclosures) is None
        ]
        if not open_blocks:
            errors.append(f"docs/{relative}: 核心实现没有默认展开")
        if not re.search(r"\bassert\b|torch\.testing\.", text):
            errors.append(f"docs/{relative}: 正文实现缺少可执行断言")
        if "/practice/" not in text:
            errors.append(f"docs/{relative}: 缺少指向组合实验或完整测试的入口")
        page_identifiers: set[str] = set()
        for index, block in enumerate(blocks, 1):
            code = block.group("code")
            count = nonblank_lines(code)
            folded = containing_disclosure(block.start(), disclosures) is not None
            if count > MAX_BLOCK_LINES:
                errors.append(
                    f"docs/{relative}: Python 代码块 {index} 过长 "
                    f"({count} > {MAX_BLOCK_LINES})"
                )
            if not folded and count > MAX_OPEN_LINES:
                errors.append(
                    f"docs/{relative}: Python 代码块 {index} 应默认折叠 "
                    f"({count} > {MAX_OPEN_LINES})"
                )
            longest = max((len(line) for line in code.splitlines()), default=0)
            if longest > MAX_LINE_LENGTH:
                errors.append(
                    f"docs/{relative}: Python 代码块 {index} 存在过长行 "
                    f"({longest} > {MAX_LINE_LENGTH})"
                )
            try:
                ast.parse(code)
            except SyntaxError as error:
                errors.append(
                    f"docs/{relative}: Python 代码块 {index}: {error.msg}"
                )
            for marker in FORBIDDEN_SCAFFOLDING:
                if marker in code:
                    errors.append(
                        f"docs/{relative}: Python 代码块 {index} "
                        f"包含项目脚手架：{marker}"
                    )
            page_identifiers.update(
                re.findall(r"(?m)^(?:def|class)\s+([A-Za-z_]\w*)", code)
            )
        identifiers[relative] = page_identifiers
    return total_blocks, total_disclosures, identifiers


def validate_generated(
    site_dir: Path,
    identifiers: dict[str, set[str]],
    errors: list[str],
) -> int:
    search_path = site_dir / "search" / "search_index.json"
    if not search_path.is_file():
        errors.append(f"{search_path}: 缺少搜索索引")
        return 0
    search_docs = json.loads(search_path.read_text(encoding="utf-8")).get(
        "docs",
        [],
    )
    checked = 0
    for relative in CANONICAL_PAGES:
        route = route_for(relative)
        output = site_dir / route / "index.html"
        if not output.is_file():
            errors.append(f"{route}: 缺少生成页面")
            continue
        parser = DisclosureParser()
        parser.feed(output.read_text(encoding="utf-8"))
        for disclosure in parser.disclosures:
            checked += 1
            if disclosure["open"]:
                errors.append(f"{route}: 折叠代码在生成页面中默认打开")
            if disclosure["summary"] != 1 or not disclosure["summary_id"]:
                errors.append(f"{route}: 折叠代码缺少唯一、可深链的 summary")
            if disclosure["code"] < 1:
                errors.append(f"{route}: 折叠区没有生成代码块")
            if disclosure["nested"]:
                errors.append(f"{route}: 生成页面含嵌套 code-disclosure")
        searchable = " ".join(
            str(item.get("text", ""))
            for item in search_docs
            if str(item.get("location", "")).startswith(route)
        )
        for identifier in sorted(identifiers.get(relative, set())):
            if identifier not in searchable:
                errors.append(
                    f"{route}: 搜索索引未收录代码标识符 {identifier}"
                )
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    supplemental_disclosures = validate_supplemental_disclosures(errors)
    blocks, disclosures, identifiers = validate_source(errors)
    disclosures += supplemental_disclosures
    generated = 0
    if args.site_dir:
        site_dir = args.site_dir
        if not site_dir.is_absolute():
            site_dir = ROOT / site_dir
        generated = validate_generated(site_dir, identifiers, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    suffix = (
        f"，生成页面检查 {generated} 个折叠区"
        if args.site_dir
        else ""
    )
    print(
        f"正文代码检查通过：{len(CANONICAL_PAGES)} 个机制页，"
        f"{blocks} 个 Python 代码块，{disclosures} 个默认折叠区{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
