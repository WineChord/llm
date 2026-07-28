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
- 以[模型家族](docs/landscape/families/index.md)分开版本、论文、权重、代码、API、产品与许可证，再把 DeepSeek、Kimi、GLM 的具体工作接回通用机制。
- 在机制正文中直接给出可执行语义核与关键断言，并用实践页组织组合实验和完整测试。
- 区分稳定原理、工程经验、实验结果与时效性事实，优先引用论文和官方资料。

## 本地预览

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_content.py
python scripts/check_paper_figures.py
python scripts/check_architecture.py
python scripts/check_python.py
python scripts/check_snippets.py
python scripts/check_code_integration.py
python scripts/check_rendering.py
mkdocs build --strict
python scripts/check_paper_figures.py --site-dir site
python scripts/check_links.py --site-dir site
python scripts/check_code_integration.py --site-dir site
python scripts/check_rendering.py --site-dir site --browser \
  --visual-artifacts-dir visual-audit
mkdocs serve
```

浏览器打开 `http://127.0.0.1:8000/llm/`。

安装 PyTorch 后，可额外运行 `python scripts/run_reference_snippets.py`，
逐页执行正文语义核、组合实验、关键工作 reference 与断言。

## 论文图表

`docs/assets/papers/` 中的裁图由 manifest 绑定到固定版本的源文件、
页码、裁剪框、像素尺寸、摘要和许可证。现有 schema v1 继续受支持；
新增来源使用
[`schemas/paper-figure-manifest-v2.schema.json`](schemas/paper-figure-manifest-v2.schema.json)，
并可在持有本地 PDF 时离线复现：

```bash
python scripts/render_paper_figures.py \
  --manifest docs/assets/papers/<source>/manifest.json \
  --pdf /absolute/path/to/report.pdf
```

生成器不会下载论文；本地 PDF 的 SHA-256 必须先与 manifest 一致。
`--output-dir` 可输出独立副本，`--write-assets` 才会替换登记文件，
`--asset` 可将范围收窄到单个图表。

## 目录

- `docs/`：网站正文
- `mkdocs.yml`：站点与导航配置
- `scripts/`：内容、链接与代码检查
- `.github/workflows/pages.yml`：GitHub Pages 发布流程

## License

正文与站点代码采用 [MIT](LICENSE)。`docs/assets/papers/` 下的第三方论文图表保留各自的来源、版权与许可证记录。
