# LLM

一份系统化的大语言模型知识库，连接模型原理、数据、训练、系统、推理、智能体与可靠性。

**在线阅读：[www.wineandchord.com/llm](https://www.wineandchord.com/llm/)**

[![Deploy site](https://github.com/WineChord/llm/actions/workflows/pages.yml/badge.svg)](https://github.com/WineChord/llm/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-5c6bc0.svg)](LICENSE)

## 内容

- 从语言建模、分词、缩放规律到 Transformer、稀疏模型与替代架构。
- 以技术谱系连接重要问题、代表工作、论文实现与后续分叉，并提供关键工作深读。
- 从数据工程、预训练、后训练到分布式训练、推理服务与硬件效率。
- 覆盖长上下文、MoE、多模态生成、AI Infra、检索增强、Coding Agent、Agentic RL、评测与生产可靠性。
- 以 [DeepSeek-V4](docs/landscape/works/deepseek-v4.md) 等系统深读连接架构公式、训练目标、kernel、状态恢复、评测协议与完整引用图谱。
- 以 [GLM-5](docs/landscape/works/glm-5.md) 贯通稀疏注意力、预训练课程、异步 Agentic RL、可执行环境、异构部署与逐项证据审计。
- 在机制正文中直接给出可执行语义核与关键断言，并用实践页组织组合实验和完整测试。
- 区分稳定原理、工程经验、实验结果与时效性事实，优先引用论文和官方资料。

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_content.py
python scripts/check_architecture.py
python scripts/check_python.py
python scripts/check_snippets.py
python scripts/check_code_integration.py
python scripts/check_rendering.py
mkdocs build --strict
python scripts/check_links.py --site-dir site
python scripts/check_code_integration.py --site-dir site
python scripts/check_rendering.py --site-dir site --browser
mkdocs serve
```

浏览器打开 `http://127.0.0.1:8000/llm/`。

安装 PyTorch 后，可额外运行 `python scripts/run_reference_snippets.py`，
逐页执行正文语义核、组合实验、关键工作 reference 与断言。

## 目录

- `docs/`：网站正文
- `mkdocs.yml`：站点与导航配置
- `scripts/`：内容、链接与代码检查
- `.github/workflows/pages.yml`：GitHub Pages 发布流程

## License

[MIT](LICENSE)
