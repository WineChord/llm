# LLM

一份系统化的大语言模型知识库，连接模型原理、数据、训练、系统、推理、应用与可靠性。

**在线阅读：[www.wineandchord.com/llm](https://www.wineandchord.com/llm/)**

[![Deploy site](https://github.com/WineChord/llm/actions/workflows/pages.yml/badge.svg)](https://github.com/WineChord/llm/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-5c6bc0.svg)](LICENSE)

## 内容

- 从语言建模、分词、缩放规律到 Transformer、稀疏模型与替代架构。
- 从数据工程、预训练、后训练到分布式训练、推理服务与硬件效率。
- 覆盖多模态、检索增强、工具调用、智能体、评测、可靠性与安全。
- 区分稳定原理、工程经验、实验结果与时效性事实，优先引用论文和官方资料。

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_content.py
python scripts/check_python.py
mkdocs serve
```

浏览器打开 `http://127.0.0.1:8000/llm/`。生产构建使用：

```bash
mkdocs build --strict
```

## 目录

- `docs/`：网站正文
- `mkdocs.yml`：站点与导航配置
- `scripts/`：内容、链接与代码检查
- `.github/workflows/pages.yml`：GitHub Pages 发布流程

## License

[MIT](LICENSE)
