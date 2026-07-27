#!/usr/bin/env python3
"""Execute reference snippets page-by-page in a shared namespace."""

from __future__ import annotations

from pathlib import Path
import re
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / "docs" / "practice"
BLOCK = re.compile(r"^```python[^\n]*\n(.*?)^```$", re.MULTILINE | re.DOTALL)
PAGES = (
    "tensor-primitives.md",
    "transformer-from-scratch.md",
    "training-objectives.md",
    "distributed-systems.md",
    "inference-engine.md",
    "retrieval-agents.md",
    "evaluation-tooling.md",
    "tokenizers.md",
    "sequence-models.md",
    "test-time-compute.md",
    "multimodal.md",
)


def main() -> int:
    try:
        import torch
    except ImportError:
        print(
            "运行 reference 需要 PyTorch；静态检查请使用 check_snippets.py",
            file=sys.stderr,
        )
        return 2
    torch.set_num_threads(1)
    total = 0
    for name in PAGES:
        path = PRACTICE / name
        namespace = {"__name__": "__reference_snippet__"}
        blocks = BLOCK.findall(path.read_text(encoding="utf-8"))
        for index, code in enumerate(blocks, 1):
            total += 1
            try:
                exec(
                    compile(
                        code,
                        f"docs/practice/{name}:block-{index}",
                        "exec",
                        dont_inherit=True,
                    ),
                    namespace,
                )
            except Exception:
                print(
                    f"docs/practice/{name}: Python 代码块 {index} 运行失败",
                    file=sys.stderr,
                )
                traceback.print_exc()
                return 1
        print(f"PASS docs/practice/{name}: {len(blocks)} 个代码块")
    print(f"Reference 运行检查通过：{len(PAGES)} 个页面，{total} 个代码块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
