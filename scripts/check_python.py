#!/usr/bin/env python3
from pathlib import Path
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)
errors: list[str] = []
count = 0

for path in FILES:
    for index, code in enumerate(BLOCK.findall(path.read_text(encoding="utf-8")), 1):
        count += 1
        try:
            ast.parse(code)
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: Python 代码块 {index}: {error.msg}")

if count == 0:
    errors.append("未找到 Python 代码块")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f"Python 检查通过：{count} 个代码块")
