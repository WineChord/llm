#!/usr/bin/env python3
"""Enforce the compact, auditable contract for hand-written references."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / "docs" / "practice"
WORKS = ROOT / "docs" / "landscape" / "works"
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)
MAX_NONBLANK_LINES = 60
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
REQUIRED = {
    "tensor-primitives.md": 6,
    "transformer-from-scratch.md": 5,
    "training-objectives.md": 10,
    "reinforcement-learning.md": 9,
    "llm-policy-optimization.md": 10,
    "distributed-systems.md": 8,
    "inference-engine.md": 8,
    "retrieval-agents.md": 8,
    "evaluation-tooling.md": 8,
    "tokenizers.md": 4,
    "sequence-models.md": 6,
    "test-time-compute.md": 6,
    "multimodal.md": 10,
}


def main() -> int:
    errors: list[str] = []
    total = 0
    for name, minimum in REQUIRED.items():
        path = PRACTICE / name
        if not path.is_file():
            errors.append(f"docs/practice/{name}: 缺少核心手撕实现页")
            continue
        text = path.read_text(encoding="utf-8")
        blocks = BLOCK.findall(text)
        if len(blocks) < minimum:
            errors.append(
                f"docs/practice/{name}: Python 实现不足 "
                f"({len(blocks)} < {minimum})"
            )
        if not re.search(r"验证|断言|测试|不变量", text):
            errors.append(f"docs/practice/{name}: 缺少验证或不变量说明")
        if not re.search(r"\bassert\b|torch\.testing\.", text):
            errors.append(f"docs/practice/{name}: 缺少可执行断言")
        for index, code in enumerate(blocks, 1):
            total += 1
            nonblank = sum(bool(line.strip()) for line in code.splitlines())
            if nonblank > MAX_NONBLANK_LINES:
                errors.append(
                    f"docs/practice/{name}: Python 代码块 {index} "
                    f"过长 ({nonblank} > {MAX_NONBLANK_LINES})"
                )
            try:
                ast.parse(code)
            except SyntaxError as error:
                errors.append(
                    f"docs/practice/{name}: Python 代码块 {index}: "
                    f"{error.msg}"
                )
            for marker in FORBIDDEN_SCAFFOLDING:
                if marker in code:
                    errors.append(
                        f"docs/practice/{name}: Python 代码块 {index} "
                        f"包含项目脚手架：{marker}"
                    )
    work_pages = sorted(WORKS.glob("*.md"))
    if not work_pages:
        errors.append("docs/landscape/works: 缺少关键工作深读页")
    for path in work_pages:
        text = path.read_text(encoding="utf-8")
        blocks = BLOCK.findall(text)
        relative = path.relative_to(ROOT)
        if not blocks:
            errors.append(f"{relative}: 缺少关键机制的可执行 reference")
        if not re.search(r"\bassert\b|torch\.testing\.", text):
            errors.append(f"{relative}: 缺少可执行断言")
        for index, code in enumerate(blocks, 1):
            total += 1
            nonblank = sum(bool(line.strip()) for line in code.splitlines())
            if nonblank > MAX_NONBLANK_LINES:
                errors.append(
                    f"{relative}: Python 代码块 {index} "
                    f"过长 ({nonblank} > {MAX_NONBLANK_LINES})"
                )
            try:
                ast.parse(code)
            except SyntaxError as error:
                errors.append(
                    f"{relative}: Python 代码块 {index}: {error.msg}"
                )
            for marker in FORBIDDEN_SCAFFOLDING:
                if marker in code:
                    errors.append(
                        f"{relative}: Python 代码块 {index} "
                        f"包含项目脚手架：{marker}"
                    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"手撕实现检查通过：{len(REQUIRED)} 个专题、"
        f"{len(work_pages)} 个关键工作，"
        f"{total} 个紧凑 Python 代码块"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
