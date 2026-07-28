#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).with_name("check_typography.py")
SPEC = importlib.util.spec_from_file_location("check_typography", SCRIPT)
assert SPEC and SPEC.loader
TYPOGRAPHY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TYPOGRAPHY
SPEC.loader.exec_module(TYPOGRAPHY)


class TypographyTest(unittest.TestCase):
    def format(self, text: str) -> str:
        return TYPOGRAPHY.transform(text, Path("case.md"))[0]

    def test_markdown_link_boundaries(self) -> None:
        source = "则应从[Kimi-VL](https://example.com/a?q=x)理解。"
        expected = "则应从 [Kimi-VL](https://example.com/a?q=x) 理解。"
        self.assertEqual(self.format(source), expected)

    def test_code_and_url_internals_are_protected(self) -> None:
        source = "令`foo_bar`处理，见https://example.com/a_b?q=x了解。"
        expected = "令 `foo_bar` 处理，见 https://example.com/a_b?q=x 了解。"
        self.assertEqual(self.format(source), expected)

    def test_math_is_unchanged(self) -> None:
        source = r"由公式\(x_i=\operatorname{Norm}(y)\)可知。"
        self.assertEqual(self.format(source), source)

    def test_mixed_script_and_number_spacing(self) -> None:
        source = "使用DeepSeek-R1处理128K上下文，准确率为90%。"
        expected = "使用 DeepSeek-R1 处理 128K 上下文，准确率为 90%。"
        self.assertEqual(self.format(source), expected)

    def test_fullwidth_punctuation_spacing(self) -> None:
        source = "中文， English； 测试 （说明），“Kimi” 模型。"
        expected = "中文，English；测试（说明），“Kimi”模型。"
        self.assertEqual(self.format(source), expected)

    def test_parenthesized_reference_and_fullwidth_digit(self) -> None:
        source = "报告式(２)给出结果，Equation (3)进一步展开。"
        expected = "报告式 (2) 给出结果，Equation (3) 进一步展开。"
        self.assertEqual(self.format(source), expected)

    def test_units_percent_and_math_parenthesis(self) -> None:
        source = r"耗时10ms，显存80GB，准确率90 %；TD($\lambda$)用于更新。"
        expected = r"耗时 10 ms，显存 80 GB，准确率 90%；TD($\lambda$) 用于更新。"
        self.assertEqual(self.format(source), expected)

    def test_markdown_prefix_and_english_apostrophe(self) -> None:
        source = "- “成功”由状态判断；Agents’ Last Exam 保持原名。"
        self.assertEqual(self.format(source), source)

    def test_adjacent_strong_sentence_stays_renderable(self) -> None:
        source = "1. **“reward 是标签。”**正文继续。"
        expected = "1. <strong>“reward 是标签。”</strong>正文继续。"
        self.assertEqual(self.format(source), expected)

    def test_html_wrappers_preserve_markup(self) -> None:
        source = "从<strong>Kimi</strong>理解<span>MoE</span>模型。"
        expected = "从 <strong>Kimi</strong> 理解 <span>MoE</span> 模型。"
        self.assertEqual(self.format(source), expected)

    def test_table_padding_is_structural(self) -> None:
        source = "| 问题 | 是否可验证？ | PPO |"
        self.assertEqual(self.format(source), source)

    def test_fenced_code_is_unchanged(self) -> None:
        source = "正文Kimi\n```python\n中文Kimi\n```\n"
        expected = "正文 Kimi\n```python\n中文Kimi\n```\n"
        self.assertEqual(self.format(source), expected)

    def test_text_after_display_math_is_checked(self) -> None:
        source = "$$\nx_i = 1\n$$\n继续Kimi\n"
        expected = "$$\nx_i = 1\n$$\n继续 Kimi\n"
        self.assertEqual(self.format(source), expected)

    def test_rendered_inline_boundaries(self) -> None:
        bad = TYPOGRAPHY.RenderedTypographyParser(Path("bad.html"))
        bad.feed("<p>从<a href='/x'>Kimi</a>理解</p>")
        self.assertEqual(len(bad.findings), 2)
        good = TYPOGRAPHY.RenderedTypographyParser(Path("good.html"))
        good.feed(
            "<p>从 <a href='/x'>Kimi</a> 理解</p>"
            "<p>中文。\nEnglish soft break</p><pre>中文Kimi</pre>"
        )
        self.assertEqual(good.findings, [])


if __name__ == "__main__":
    unittest.main()
