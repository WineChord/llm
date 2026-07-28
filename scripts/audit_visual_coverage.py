#!/usr/bin/env python3
"""Inventory visual evidence across every documentation page."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"(?ms)^(```+|~~~+).*?^\1\s*$")
HTML = re.compile(r"<[^>]+>")
LINK_TARGET = re.compile(r"\]\([^)]+\)")
PAPER_FIGURE = re.compile(r'<figure\b[^>]*class="[^"]*\bpaper-figure\b', re.I)
LOCAL_RASTER = re.compile(
    r"!\[[^\]]*\]\((?!https?://)[^)\s]+\.(?:png|jpe?g|webp)(?:[?#][^)]*)?\)",
    re.I,
)
EXTERNAL_LINK = re.compile(r"\[[^\]]+\]\(https://[^)\s]+\)")
DISPLAY_MATH = re.compile(r"(?ms)^\$\$\s*.*?^\$\$\s*$")
INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)")
TABLE_ROW = re.compile(r"(?m)^\s*\|.+\|\s*$")
HEADING = re.compile(r"(?m)^#{2,3}\s+")
REFERENCE = re.compile(r"(?m)^## Reference(?:\s+\{[^}]+\})?\s*$")
TEXT_FIRST_NAMES = {
    "changelog.md",
    "glossary.md",
    "references.md",
    "reading-list.md",
}
TEXT_FIRST_DIRS = {"guide"}


def visible_text(source: str) -> str:
    source = FENCE.sub(" ", source)
    source = DISPLAY_MATH.sub(" ", source)
    source = INLINE_MATH.sub(" ", source)
    source = LINK_TARGET.sub("]", source)
    source = HTML.sub(" ", source)
    source = re.sub(r"(?m)^[#>|*+\-\d.\s]+", " ", source)
    return re.sub(r"\s+", " ", source).strip()


def page_kind(path: Path) -> str:
    relative = path.relative_to(DOCS)
    if relative.name in TEXT_FIRST_NAMES or relative.parts[0] in TEXT_FIRST_DIRS:
        return "reference-or-navigation"
    if relative.name == "index.md":
        return "section-index"
    return "substantive"


def inspect(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    figure_count = len(PAPER_FIGURE.findall(source))
    raster_count = len(LOCAL_RASTER.findall(source))
    text = visible_text(source)
    kind = page_kind(path)
    signals = {
        "visible_characters": len(text),
        "h2_h3": len(HEADING.findall(source)),
        "display_math": len(DISPLAY_MATH.findall(source)),
        "inline_math": len(INLINE_MATH.findall(source)),
        "code_fences": len(re.findall(r"(?m)^```", source)) // 2,
        "table_rows": len(TABLE_ROW.findall(source)),
        "external_links": len(EXTERNAL_LINK.findall(source)),
        "has_reference": bool(REFERENCE.search(source)),
    }
    score = 0
    score += signals["visible_characters"] >= 4500
    score += signals["h2_h3"] >= 7
    score += signals["external_links"] >= 10
    score += signals["display_math"] >= 8
    score += signals["table_rows"] >= 12
    if figure_count or raster_count:
        status = "figure-backed"
    elif kind == "substantive" and score >= 2:
        status = "review-candidate"
    else:
        status = "text-first"
    return {
        "page": path.relative_to(ROOT).as_posix(),
        "kind": kind,
        "status": status,
        "paper_figures": figure_count,
        "local_rasters": raster_count,
        "signals": signals,
        "review_score": score,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--report", type=Path)
args = parser.parse_args()

records = [inspect(path) for path in sorted(DOCS.rglob("*.md"))]
counts = Counter(str(record["status"]) for record in records)
summary = {
    "schema_version": 1,
    "pages": len(records),
    "status_counts": dict(sorted(counts.items())),
    "paper_figure_placements": sum(
        int(record["paper_figures"]) for record in records
    ),
}
payload = {
    "summary": summary,
    "pages": records,
}
if args.report:
    report = args.report if args.report.is_absolute() else ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
print(
    "Visual coverage audit passed: "
    f"{summary['pages']} pages, "
    f"{counts['figure-backed']} figure-backed, "
    f"{counts['review-candidate']} review candidates, "
    f"{summary['paper_figure_placements']} paper-figure placements"
)
