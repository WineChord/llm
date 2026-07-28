#!/usr/bin/env python3
"""Validate reader-visible Chinese mixed-script typography in Markdown."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    ROOT / "README.md",
    ROOT / "mkdocs.yml",
    *sorted((ROOT / "docs").rglob("*.md")),
]
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
REFERENCE_DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s+\S+")
BLOCK_PREFIX = re.compile(
    r"^\s{0,3}(?:(?:#{1,6}|[-+*]|\d+[.)]|>)\s+)+"
)
URL = re.compile(
    r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"
)
EMAIL = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)
ENTITY = re.compile(r"&(?:#[0-9]+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);")
ATTR_BLOCK = re.compile(
    r"\{(?:[#.][A-Za-z0-9_-]+|"
    r"(?:width|height|loading|decoding|target|rel|title)="
    r"|[ \t])*[^}\n]*\}"
)
OPEN_PUNCTUATION = frozenset("（《【「『“")
CLOSE_PUNCTUATION = frozenset("，。！？；：、）》】」』”")
MARKUP_DELIMITERS = ("***", "___", "**", "__", "~~", "==", "^^", "*", "_")
ADJACENT_STRONG = re.compile(
    r"\*\*(?P<body>(?:[^*]|\*(?!\*))+?)\*\*(?=[\u3400-\u9fffA-Za-z0-9])"
)
VOID_HTML_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
)
BLOCK_HTML_TAGS = frozenset(
    {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
HIDDEN_HTML_TAGS = frozenset(
    {"math", "mjx-container", "pre", "script", "style", "svg", "template"}
)
NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
UNITS = (
    "tokens/s",
    "token/s",
    "requests/s",
    "request/s",
    "TFLOP/s",
    "GFLOP/s",
    "PFLOP/s",
    "TiB/s",
    "GiB/s",
    "MiB/s",
    "KiB/s",
    "TB/s",
    "GB/s",
    "MB/s",
    "KB/s",
    "TFLOPS",
    "GFLOPS",
    "PFLOPS",
    "Tbps",
    "Gbps",
    "Mbps",
    "Kbps",
    "TiB",
    "GiB",
    "MiB",
    "KiB",
    "TB",
    "GB",
    "MB",
    "KB",
    "THz",
    "GHz",
    "MHz",
    "kHz",
    "Hz",
    "µs",
    "μs",
    "ns",
    "us",
    "ms",
    "min",
    "px",
    "rem",
)
NUMBER_UNIT = re.compile(
    rf"(?<![A-Za-z0-9_.:/-])(?P<number>{NUMBER})"
    rf"(?P<unit>{'|'.join(map(re.escape, UNITS))})(?![A-Za-z0-9])"
)
SPACED_PERCENT_OR_DEGREE = re.compile(
    rf"(?P<number>{NUMBER})(?P<spaces>[ \t]+)(?P<mark>[%％°])"
)


@dataclass(frozen=True)
class Glyph:
    char: str
    before: int
    source: int | None


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str
    reason: str


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    reason: str
    excerpt: str


class RenderedTypographyParser(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.path = path
        self.findings: list[Finding] = []
        self.hidden_depth = 0
        self.hidden_mode: str | None = None
        self.last_char: str | None = None
        self.pending_space: str | None = None

    def reset_boundary(self) -> None:
        self.last_char = None
        self.pending_space = None

    def add_finding(self, reason: str, text: str) -> None:
        line, column = self.getpos()
        self.findings.append(
            Finding(
                path=self.path,
                line=line,
                column=column + 1,
                reason=reason,
                excerpt=" ".join(text.split())[:180],
            )
        )

    def consume(self, text: str) -> None:
        for char in text:
            if char.isspace():
                if self.last_char is not None:
                    if char in {"\r", "\n"}:
                        self.pending_space = "softbreak"
                    elif self.pending_space != "softbreak":
                        self.pending_space = "literal"
                continue
            if self.last_char is not None:
                if self.pending_space:
                    if self.pending_space == "literal" and (
                        self.last_char in OPEN_PUNCTUATION
                        or self.last_char in CLOSE_PUNCTUATION
                        or char in OPEN_PUNCTUATION
                        or char in CLOSE_PUNCTUATION
                    ):
                        self.add_finding(
                            "rendered fullwidth punctuation has adjacent whitespace",
                            text,
                        )
                elif needs_mixed_script_space(self.last_char, char):
                    self.add_finding(
                        "rendered Chinese and Latin or numeric text need a space",
                        text,
                    )
            self.last_char = char
            self.pending_space = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.lower()
        if tag in BLOCK_HTML_TAGS:
            self.reset_boundary()
        if self.hidden_depth:
            self.hidden_depth += 1
            return
        classes = {
            value
            for name, raw_value in attrs
            if name == "class" and raw_value
            for value in raw_value.split()
        }
        if tag == "code":
            self.consume("A")
            self.hidden_depth = 1
            self.hidden_mode = "code"
        elif tag in HIDDEN_HTML_TAGS or "arithmatex" in classes:
            self.reset_boundary()
            self.hidden_depth = 1
            self.hidden_mode = "reset"

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.hidden_depth:
            self.hidden_depth -= 1
            if self.hidden_depth == 0:
                if self.hidden_mode == "reset":
                    self.reset_boundary()
                self.hidden_mode = None
        if tag in BLOCK_HTML_TAGS:
            self.reset_boundary()

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() in BLOCK_HTML_TAGS:
            self.reset_boundary()

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.consume(data)


def is_han(char: str) -> bool:
    point = ord(char)
    return (
        0x3400 <= point <= 0x4DBF
        or 0x4E00 <= point <= 0x9FFF
        or 0xF900 <= point <= 0xFAFF
        or 0x20000 <= point <= 0x3134F
    )


def is_latin_or_digit(char: str) -> bool:
    return char.isascii() and char.isalnum()


def needs_mixed_script_space(left: str, right: str) -> bool:
    return (
        is_han(left)
        and (is_latin_or_digit(right) or right in {"%", "°"})
    ) or (
        (is_latin_or_digit(left) or left in {"%", "°"})
        and is_han(right)
    )


def find_closing_bracket(text: str, start: int, end: int) -> int | None:
    depth = 0
    escaped = False
    for index in range(start, end):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
    return None


def find_closing_parenthesis(text: str, start: int, end: int) -> int | None:
    depth = 0
    escaped = False
    quote: str | None = None
    for index in range(start, end):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def find_inline_code_end(text: str, start: int, end: int) -> int | None:
    tick_end = start
    while tick_end < end and text[tick_end] == "`":
        tick_end += 1
    delimiter = text[start:tick_end]
    closing = text.find(delimiter, tick_end, end)
    if closing < 0:
        return None
    return closing + len(delimiter)


def find_math_end(text: str, start: int, end: int) -> int | None:
    if text.startswith(r"\(", start):
        closing = text.find(r"\)", start + 2, end)
        return None if closing < 0 else closing + 2
    if text.startswith(r"\[", start):
        closing = text.find(r"\]", start + 2, end)
        return None if closing < 0 else closing + 2
    if text[start] != "$":
        return None
    delimiter = "$$" if text.startswith("$$", start) else "$"
    cursor = start + len(delimiter)
    while cursor < end:
        closing = text.find(delimiter, cursor, end)
        if closing < 0:
            return None
        if closing == 0 or text[closing - 1] != "\\":
            return closing + len(delimiter)
        cursor = closing + len(delimiter)
    return None


def parse_link(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int, int] | None:
    label_start = start + 2 if text.startswith("![", start) else start + 1
    bracket_start = start + 1 if text.startswith("![", start) else start
    if bracket_start >= end or text[bracket_start] != "[":
        return None
    label_end = find_closing_bracket(text, bracket_start, end)
    if label_end is None or label_end + 1 >= end:
        return None
    target_start = label_end + 1
    if text[target_start] == "(":
        target_end = find_closing_parenthesis(text, target_start, end)
        if target_end is None:
            return None
        return label_start, label_end, target_end + 1
    if text[target_start] == "[":
        reference_end = find_closing_bracket(text, target_start, end)
        if reference_end is None:
            return None
        return label_start, label_end, reference_end + 1
    return None


def html_tag(text: str, start: int, end: int) -> tuple[int, bool] | None:
    if text.startswith("<!--", start):
        closing = text.find("-->", start + 4, end)
        return (end if closing < 0 else closing + 3), False
    closing = text.find(">", start + 1, end)
    if closing < 0:
        return None
    body = text[start + 1 : closing].strip()
    if not body or not re.match(r"/?[A-Za-z][A-Za-z0-9:-]*", body):
        return None
    closing_tag = body.startswith("/")
    self_closing = body.endswith("/")
    name = body.lstrip("/").split(None, 1)[0].rstrip("/").lower()
    wraps_visible_text = not closing_tag and not self_closing and name not in VOID_HTML_TAGS
    return closing + 1, wraps_visible_text


def project(
    text: str,
    start: int = 0,
    end: int | None = None,
    initial_before: int | None = None,
) -> list[Glyph]:
    if end is None:
        end = len(text)
    glyphs: list[Glyph] = []
    cursor = start
    pending_before = initial_before

    def emit(char: str, source: int | None, fallback_before: int) -> None:
        nonlocal pending_before
        glyphs.append(
            Glyph(
                char=char,
                before=pending_before if pending_before is not None else fallback_before,
                source=source,
            )
        )
        pending_before = None

    while cursor < end:
        if text.startswith("<!--", cursor):
            tag = html_tag(text, cursor, end)
            cursor = end if tag is None else tag[0]
            continue
        url_match = URL.match(text, cursor)
        if url_match:
            emit("A", None, cursor)
            cursor = url_match.end()
            continue
        email_match = EMAIL.match(text, cursor)
        if email_match:
            emit("A", None, cursor)
            cursor = email_match.end()
            continue
        entity_match = ENTITY.match(text, cursor)
        if entity_match:
            emit("§", None, cursor)
            cursor = entity_match.end()
            continue
        if text[cursor] == "`":
            code_end = find_inline_code_end(text, cursor, end)
            if code_end is not None:
                emit("A", None, cursor)
                cursor = code_end
                continue
        if text[cursor] == "$" or text.startswith((r"\(", r"\["), cursor):
            math_end = find_math_end(text, cursor, end)
            if math_end is not None:
                emit("§", None, cursor)
                cursor = math_end
                continue
        if text.startswith("[^", cursor):
            footnote_end = text.find("]", cursor + 2, end)
            if footnote_end >= 0:
                emit("§", None, cursor)
                cursor = footnote_end + 1
                continue
        if text[cursor] == "[" or text.startswith("![", cursor):
            parsed = parse_link(text, cursor, end)
            if parsed:
                label_start, label_end, link_end = parsed
                wrapper_before = (
                    pending_before if pending_before is not None else cursor
                )
                nested = project(
                    text,
                    label_start,
                    label_end,
                    initial_before=wrapper_before,
                )
                glyphs.extend(nested)
                pending_before = None
                cursor = link_end
                continue
        if text[cursor] == "<":
            if text.startswith(("<http://", "<https://", "<mailto:"), cursor):
                closing = text.find(">", cursor + 1, end)
                if closing >= 0:
                    emit("A", None, cursor)
                    cursor = closing + 1
                    continue
            tag = html_tag(text, cursor, end)
            if tag:
                tag_end, wraps_visible_text = tag
                if wraps_visible_text and pending_before is None:
                    pending_before = cursor
                cursor = tag_end
                continue
        attr_match = ATTR_BLOCK.match(text, cursor)
        if attr_match:
            cursor = attr_match.end()
            continue
        delimiter = next(
            (
                candidate
                for candidate in MARKUP_DELIMITERS
                if text.startswith(candidate, cursor)
            ),
            None,
        )
        if delimiter:
            closing = text.find(delimiter, cursor + len(delimiter), end)
            if closing > cursor + len(delimiter):
                wrapper_before = (
                    pending_before if pending_before is not None else cursor
                )
                nested = project(
                    text,
                    cursor + len(delimiter),
                    closing,
                    initial_before=wrapper_before,
                )
                glyphs.extend(nested)
                pending_before = None
                cursor = closing + len(delimiter)
                continue
        if text[cursor] == "\\" and cursor + 1 < end:
            emit(text[cursor + 1], cursor + 1, cursor)
            cursor += 2
            continue
        emit(text[cursor], cursor, cursor)
        cursor += 1
    return glyphs


def line_edits(line: str) -> list[Edit]:
    prefix = BLOCK_PREFIX.match(line)
    start = prefix.end() if prefix else 0
    glyphs = project(line, start=start)
    edits: dict[tuple[int, int], Edit] = {}
    for glyph_index, (left, right) in enumerate(zip(glyphs, glyphs[1:])):
        if needs_mixed_script_space(left.char, right.char):
            key = (right.before, right.before)
            edits[key] = Edit(
                start=right.before,
                end=right.before,
                replacement=" ",
                reason="Chinese and Latin or numeric text need a space",
            )
        if left.char == ")" and is_han(right.char):
            opening = next(
                (
                    candidate
                    for candidate in range(glyph_index - 1, -1, -1)
                    if glyphs[candidate].char in {"(", " ", "，", "。", "；", "："}
                ),
                None,
            )
            if (
                opening is not None
                and glyphs[opening].char == "("
                and (
                    any(
                        is_latin_or_digit(glyph.char)
                        or "\uff10" <= glyph.char <= "\uff19"
                        or glyph.char == "§"
                        for glyph in glyphs[opening + 1 : glyph_index + 1]
                    )
                    or (
                        opening > 0
                        and is_latin_or_digit(glyphs[opening - 1].char)
                    )
                )
            ):
                key = (right.before, right.before)
                edits[key] = Edit(
                    start=right.before,
                    end=right.before,
                    replacement=" ",
                    reason="parenthesized Latin or numeric text needs an exterior space",
                )
        if is_han(left.char) and right.char == "(":
            closing = next(
                (
                    candidate
                    for candidate in range(glyph_index + 2, len(glyphs))
                    if glyphs[candidate].char in {")", " ", "，", "。", "；", "："}
                ),
                None,
            )
            if (
                closing is not None
                and glyphs[closing].char == ")"
                and any(
                    is_latin_or_digit(glyph.char)
                    or "\uff10" <= glyph.char <= "\uff19"
                    for glyph in glyphs[glyph_index + 2 : closing]
                )
            ):
                key = (right.before, right.before)
                edits[key] = Edit(
                    start=right.before,
                    end=right.before,
                    replacement=" ",
                    reason="parenthesized Latin or numeric text needs an exterior space",
                )
    for glyph in glyphs:
        if (
            glyph.source is not None
            and "\uff10" <= glyph.char <= "\uff19"
        ):
            replacement = chr(ord("0") + ord(glyph.char) - ord("\uff10"))
            key = (glyph.source, glyph.source + 1)
            edits[key] = Edit(
                start=glyph.source,
                end=glyph.source + 1,
                replacement=replacement,
                reason="reader-visible numbers must use halfwidth digits",
            )
    visible = "".join(glyph.char for glyph in glyphs)
    for match in NUMBER_UNIT.finditer(visible):
        unit = glyphs[match.start("unit")]
        key = (unit.before, unit.before)
        edits[key] = Edit(
            start=unit.before,
            end=unit.before,
            replacement=" ",
            reason="numbers and ordinary units need a space",
        )
    for match in SPACED_PERCENT_OR_DEGREE.finditer(visible):
        for glyph in glyphs[match.start("spaces") : match.end("spaces")]:
            if glyph.source is None or line[glyph.source] not in {" ", "\t"}:
                continue
            key = (glyph.source, glyph.source + 1)
            edits[key] = Edit(
                start=glyph.source,
                end=glyph.source + 1,
                replacement="",
                reason="percent and degree marks stay attached to numbers",
            )
    index = 0
    while index < len(glyphs):
        if glyphs[index].char not in {" ", "\t"}:
            index += 1
            continue
        run_start = index
        while index < len(glyphs) and glyphs[index].char in {" ", "\t"}:
            index += 1
        if run_start == 0 or index == len(glyphs):
            continue
        left = glyphs[run_start - 1].char
        right = glyphs[index].char
        if left == "|" or right == "|":
            continue
        if (
            left in OPEN_PUNCTUATION
            or left in CLOSE_PUNCTUATION
            or right in OPEN_PUNCTUATION
            or right in CLOSE_PUNCTUATION
        ):
            for glyph in glyphs[run_start:index]:
                if glyph.source is None or line[glyph.source] not in {" ", "\t"}:
                    continue
                key = (glyph.source, glyph.source + 1)
                edits[key] = Edit(
                    start=glyph.source,
                    end=glyph.source + 1,
                    replacement="",
                    reason="fullwidth punctuation must not have adjacent spaces",
                )
    return sorted(edits.values(), key=lambda item: (item.start, item.end))


def apply_edits(line: str, edits: list[Edit]) -> str:
    result = line
    for edit in reversed(edits):
        result = result[: edit.start] + edit.replacement + result[edit.end :]
    return result


def repair_adjacent_strong(line: str) -> tuple[str, list[Edit]]:
    edits: list[Edit] = []
    for match in ADJACENT_STRONG.finditer(line):
        body = match.group("body")
        if not body or body[-1] not in CLOSE_PUNCTUATION:
            continue
        edits.append(
            Edit(
                start=match.start(),
                end=match.end(),
                replacement=f"<strong>{body}</strong>",
                reason="adjacent strong emphasis must remain renderable",
            )
        )
    return apply_edits(line, edits), edits


def transform(text: str, path: Path) -> tuple[str, list[Finding]]:
    transformed: list[str] = []
    findings: list[Finding] = []
    fence_delimiter: str | None = None
    display_math = False
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.rstrip("\r\n")
        fence_match = FENCE.match(stripped)
        if fence_match:
            delimiter = fence_match.group(1)
            if fence_delimiter is None:
                fence_delimiter = delimiter[0]
            elif delimiter[0] == fence_delimiter:
                fence_delimiter = None
            transformed.append(line)
            continue
        if fence_delimiter is not None:
            transformed.append(line)
            continue
        if display_math:
            transformed.append(line)
            if stripped.strip() in {"$$", r"\]"}:
                display_math = False
            continue
        if stripped.strip() in {"$$", r"\["}:
            display_math = True
            transformed.append(line)
            continue
        if REFERENCE_DEFINITION.match(stripped):
            transformed.append(line)
            continue
        repaired, emphasis_edits = repair_adjacent_strong(stripped)
        edits = line_edits(repaired)
        for edit in [*emphasis_edits, *edits]:
            findings.append(
                Finding(
                    path=path,
                    line=line_number,
                    column=edit.start + 1,
                    reason=edit.reason,
                    excerpt=stripped,
                )
            )
        fixed = apply_edits(repaired, edits)
        ending = line[len(stripped) :]
        transformed.append(fixed + ending)
    if text and not text.endswith(("\n", "\r")) and transformed:
        transformed[-1] = transformed[-1].rstrip("\r\n")
    return "".join(transformed), findings


def selected_files(arguments: list[str]) -> list[Path]:
    if not arguments:
        return [path for path in DEFAULT_FILES if path.exists()]
    files = []
    for argument in arguments:
        path = Path(argument)
        if not path.is_absolute():
            path = ROOT / path
        files.append(path)
    return files


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_site(site_dir: Path) -> tuple[list[Finding], int]:
    html_files = sorted(site_dir.rglob("*.html"))
    findings: list[Finding] = []
    for path in html_files:
        parser = RenderedTypographyParser(path)
        parser.feed(path.read_text(encoding="utf-8"))
        parser.close()
        findings.extend(parser.findings)
    return findings, len(html_files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--format-stdin", metavar="PATH")
    parser.add_argument("--site-dir", type=Path)
    args = parser.parse_args()
    if args.format_stdin:
        source = sys.stdin.read()
        formatted, _ = transform(source, Path(args.format_stdin))
        sys.stdout.write(formatted)
        return 0
    if args.site_dir:
        site_dir = args.site_dir
        if not site_dir.is_absolute():
            site_dir = ROOT / site_dir
        findings, page_count = check_site(site_dir)
        if findings:
            for finding in findings:
                print(
                    f"{display_path(finding.path)}:{finding.line}:{finding.column}: "
                    f"{finding.reason}\n  {finding.excerpt}"
                )
            print(f"Rendered typography check failed: {len(findings)} finding(s)")
            return 1
        print(f"Rendered typography check passed: {page_count} HTML files")
        return 0
    findings: list[Finding] = []
    changed = 0
    for path in selected_files(args.files):
        source = path.read_text(encoding="utf-8")
        formatted, file_findings = transform(source, path)
        findings.extend(file_findings)
        if args.fix and formatted != source:
            path.write_text(formatted, encoding="utf-8")
            changed += 1
    if args.fix:
        print(f"Typography fixes applied: {changed} files, {len(findings)} edits")
        return 0
    if findings:
        for finding in findings:
            print(
                f"{display_path(finding.path)}:{finding.line}:{finding.column}: "
                f"{finding.reason}\n  {finding.excerpt}"
            )
        print(f"Typography check failed: {len(findings)} finding(s)")
        return 1
    print(f"Typography check passed: {len(selected_files(args.files))} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
