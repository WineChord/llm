#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").rglob("*.md"))
FILES = [ROOT / "README.md", *DOCS]
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ARXIV = re.compile(r"https://arxiv\.org/abs/\d{4}\.\d{4,5}")
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
    "private instruction",
    "用户要求我",
    "根据用户指令",
    "内部工作流",
)
errors: list[str] = []

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

source_count = sum(len(ARXIV.findall(path.read_text(encoding="utf-8"))) for path in DOCS)
if source_count < 30:
    errors.append(f"docs/: 原始论文链接不足：{source_count} < 30")

if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"内容检查通过：{len(DOCS)} 个页面，{source_count} 个论文链接")
