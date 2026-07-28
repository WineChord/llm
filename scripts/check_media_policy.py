#!/usr/bin/env python3
"""Reject synthetic vector artwork, remote hotlinks, and image text overlays."""

from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TEXT_SUFFIXES = {".css", ".html", ".js", ".md"}
FORBIDDEN_SOURCE = (
    ("inline SVG", re.compile(r"<svg\b|data:image/svg\+xml", re.IGNORECASE)),
    (
        "diagram source block",
        re.compile(r"(?m)^```(?:mermaid|plantuml|graphviz)\b", re.IGNORECASE),
    ),
    ("canvas artwork", re.compile(r"<canvas\b", re.IGNORECASE)),
    (
        "background image or generated gradient",
        re.compile(
            r"background(?:-image)?\s*:\s*(?:url\(|(?:repeating-)?(?:linear|radial|conic)-gradient\()",
            re.IGNORECASE,
        ),
    ),
)
REMOTE_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?:https?:)?//", re.IGNORECASE)
REMOTE_HTML_IMAGE = re.compile(
    r"<img\b[^>]*\bsrc\s*=\s*[\"']\s*(?:https?:)?//",
    re.IGNORECASE,
)


errors: list[str] = []
for path in sorted(DOCS.rglob("*")):
    if not path.is_file():
        continue
    label = path.relative_to(ROOT)
    if path.suffix.lower() == ".svg":
        errors.append(f"{label}: local SVG assets are not allowed")
        continue
    if path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    text = path.read_text(encoding="utf-8")
    for name, pattern in FORBIDDEN_SOURCE:
        if pattern.search(text):
            errors.append(f"{label}: contains forbidden {name}")
    if REMOTE_MARKDOWN_IMAGE.search(text) or REMOTE_HTML_IMAGE.search(text):
        errors.append(f"{label}: remote image hotlinks are not allowed")

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("Media policy check passed: no synthetic vector or overlay artwork")
