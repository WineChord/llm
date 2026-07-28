#!/usr/bin/env python3
"""Validate Markdown structure and mathematics from source through browser output."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable, Optional, Sequence


EXPECTED_MATHJAX_VERSION = "3.2.2"
BACKTICK = "`"
BARE_COMMANDS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "theta",
    "lambda",
    "mu",
    "pi",
    "rho",
    "sigma",
    "tau",
    "phi",
    "psi",
    "omega",
    "widehat",
    "hat",
    "tilde",
    "bar",
    "overline",
    "underline",
    "frac",
    "dfrac",
    "sqrt",
    "sum",
    "prod",
    "nabla",
    "partial",
    "left",
    "right",
    "operatorname",
    "mathrm",
    "mathbf",
    "mathbb",
    "mathcal",
    "mathsf",
    "mathit",
    "mathtt",
)
BARE_FUNCTIONS = ("log", "exp", "sin", "cos", "tan", "tanh", "softmax")
TEXT_COMMAND = re.compile(
    r"\\(?:text|operatorname|mathrm|mathbf|mathit|mathtt)\{[^{}]*\}"
)
SNIPPET = re.compile(r'(?m)^[ \t]*--8<-- "([^"]+)"[ \t]*$')
VISIBLE_MARKDOWN_PATTERNS = (
    ("strong emphasis", re.compile(r"\*\*[^*]+\*\*|__[^_]+__")),
    ("strikethrough", re.compile(r"~~[^~]+~~")),
    ("inline link", re.compile(r"(?<!!)\[[^\]]+\]\([^)]+\)")),
)


@dataclass(frozen=True)
class Expression:
    path: Path
    line: int
    kind: str
    tex: str


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def mask_range(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] != "\n":
            chars[index] = " "


def fence_marker(line: str) -> Optional[str]:
    stripped = line.lstrip(" ")
    if not stripped:
        return None
    char = stripped[0]
    if char not in (BACKTICK, "~"):
        return None
    length = 0
    while length < len(stripped) and stripped[length] == char:
        length += 1
    return char * length if length >= 3 else None


def mask_markdown_code(text: str, path: Path, errors: list[str]) -> list[str]:
    chars = list(text)
    offset = 0
    active_char: Optional[str] = None
    active_length = 0
    active_line = 0
    for line in text.splitlines(keepends=True):
        marker = fence_marker(line)
        if active_char is not None:
            mask_range(chars, offset, offset + len(line))
            if marker and marker[0] == active_char and len(marker) >= active_length:
                active_char = None
                active_length = 0
                active_line = 0
            offset += len(line)
            continue
        if marker:
            active_char = marker[0]
            active_length = len(marker)
            active_line = line_number(text, offset)
            mask_range(chars, offset, offset + len(line))
            offset += len(line)
            continue
        cursor = 0
        while cursor < len(line):
            if line[cursor] != BACKTICK:
                cursor += 1
                continue
            run = 1
            while cursor + run < len(line) and line[cursor + run] == BACKTICK:
                run += 1
            marker = BACKTICK * run
            end = line.find(marker, cursor + run)
            if end < 0:
                cursor += run
                continue
            mask_range(chars, offset + cursor, offset + end + run)
            cursor = end + run
        offset += len(line)
    if active_char is not None:
        errors.append(f"{path}:{active_line}: unclosed fenced code block")
    return chars


def unescaped_dollars(line: str) -> list[int]:
    result = []
    for index, char in enumerate(line):
        if char != "$":
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2 == 0:
            result.append(index)
    return result


def validate_tex(expression: Expression, errors: list[str]) -> None:
    tex = expression.tex
    label = f"{expression.path}:{expression.line}"
    if not tex.strip():
        errors.append(f"{label}: empty {expression.kind} expression")
        return
    braces: list[int] = []
    for index, char in enumerate(tex):
        if char not in "{}":
            continue
        slashes = 0
        cursor = index - 1
        while cursor >= 0 and tex[cursor] == "\\":
            slashes += 1
            cursor -= 1
        if slashes % 2:
            continue
        if char == "{":
            braces.append(index)
        elif braces:
            braces.pop()
        else:
            errors.append(f"{label}: unmatched closing brace in TeX")
            break
    if braces:
        errors.append(f"{label}: unmatched opening brace in TeX")
    left_count = len(re.findall(r"\\left\b", tex))
    right_count = len(re.findall(r"\\right\b", tex))
    if left_count != right_count:
        errors.append(
            f"{label}: \\left/\\right count differs "
            f"({left_count} versus {right_count})"
        )
    environments: list[str] = []
    for match in re.finditer(r"\\(begin|end)\{([^{}]+)\}", tex):
        operation, name = match.groups()
        if operation == "begin":
            environments.append(name)
        elif not environments or environments[-1] != name:
            errors.append(f"{label}: unmatched \\end{{{name}}}")
            break
        else:
            environments.pop()
    if environments:
        errors.append(f"{label}: unclosed environment {environments[-1]}")
    heuristic = tex
    while True:
        updated = TEXT_COMMAND.sub("", heuristic)
        if updated == heuristic:
            break
        heuristic = updated
    command_pattern = re.compile(
        r"(?<![\\A-Za-z])(" + "|".join(BARE_COMMANDS) + r")(?![A-Za-z])"
    )
    match = command_pattern.search(heuristic)
    if match:
        errors.append(
            f"{label}: probable missing backslash before {match.group(1)!r}"
        )
    function_pattern = re.compile(
        r"(?<![\\A-Za-z])(" + "|".join(BARE_FUNCTIONS) + r")(?=\\|\s*\()"
    )
    match = function_pattern.search(heuristic)
    if match:
        errors.append(
            f"{label}: probable missing backslash before function "
            f"{match.group(1)!r}"
        )


def scan_markdown(path: Path, text: str) -> tuple[list[Expression], list[str]]:
    errors: list[str] = []
    code_mask = mask_markdown_code(text, path, errors)
    outside_math = code_mask.copy()
    visible = "".join(code_mask)
    expressions: list[Expression] = []
    for match in re.finditer(r"\\[()]", visible):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "legacy \\(...\\) delimiter; use $...$"
        )
    for match in re.finditer(r"(?m)^\s*\\[\[\]]\s*$", visible):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "legacy display delimiter; use a standalone $$ line"
        )
    for match in re.finditer(r"(?m)^\$\$\n\$\$$", visible):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "separate adjacent display blocks with a blank line"
        )
    offset = 0
    display_start: Optional[int] = None
    display_line = 0
    display_tex: list[str] = []
    for line in visible.splitlines(keepends=True):
        body = line[:-1] if line.endswith("\n") else line
        dollars = unescaped_dollars(body)
        standalone = body.strip() == "$$"
        if standalone:
            delimiter_at = body.index("$$")
            absolute = offset + delimiter_at
            if delimiter_at != 0:
                errors.append(
                    f"{path}:{line_number(text, absolute)}: "
                    "display $$ delimiters must start at column 1"
                )
            if display_start is None:
                display_start = absolute
                display_line = line_number(text, absolute)
                display_tex = []
            else:
                expression = Expression(
                    path,
                    display_line,
                    "display",
                    "".join(display_tex).strip(),
                )
                expressions.append(expression)
                mask_range(outside_math, display_start, offset + len(body))
                display_start = None
                display_tex = []
            offset += len(line)
            continue
        if display_start is not None:
            if dollars:
                errors.append(
                    f"{path}:{line_number(text, offset + dollars[0])}: "
                    "nested dollar delimiter inside display math"
                )
            display_tex.append(line)
            offset += len(line)
            continue
        adjacent = [
            index
            for index in dollars
            if (index + 1 in dollars) or (index - 1 in dollars)
        ]
        if adjacent:
            errors.append(
                f"{path}:{line_number(text, offset + adjacent[0])}: "
                "display $$ delimiters must occupy their own lines"
            )
            offset += len(line)
            continue
        if len(dollars) % 2:
            errors.append(
                f"{path}:{line_number(text, offset + dollars[-1])}: "
                "unclosed inline dollar delimiter or unescaped currency sign"
            )
            offset += len(line)
            continue
        for index in range(0, len(dollars), 2):
            start, end = dollars[index], dollars[index + 1]
            expression = Expression(
                path,
                line_number(text, offset + start),
                "inline",
                body[start + 1 : end].strip(),
            )
            expressions.append(expression)
            mask_range(outside_math, offset + start, offset + end + 1)
        offset += len(line)
    if display_start is not None:
        errors.append(f"{path}:{display_line}: unclosed display math delimiter")
    for expression in expressions:
        validate_tex(expression, errors)
    plain = "".join(outside_math)
    for match in re.finditer(r"\\[A-Za-z]+", plain):
        errors.append(
            f"{path}:{line_number(text, match.start())}: "
            "TeX command appears outside math or code delimiters"
        )
    if len(re.findall(r"(?i)<details(?:\s[^>]*)?>", plain)) != len(
        re.findall(r"(?i)</details\s*>", plain)
    ):
        errors.append(f"{path}: unbalanced <details> elements")
    return expressions, errors


def canonicalize_legacy_math(path: Path, text: str) -> tuple[str, int]:
    errors: list[str] = []
    mask = mask_markdown_code(text, path, errors)
    visible = "".join(mask)
    output: list[str] = []
    replacements = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        masked_line = visible[offset : offset + len(line)]
        ending = "\n" if line.endswith("\n") else ""
        body = line[:-1] if ending else line
        masked_body = masked_line[:-1] if ending else masked_line
        if masked_body in ("\\[", "\\]") and body == masked_body:
            output.append("$$" + ending)
            replacements += 1
            offset += len(line)
            continue
        cursor = 0
        while cursor < len(line):
            if (
                cursor + 1 < len(line)
                and masked_line[cursor] == "\\"
                and masked_line[cursor + 1] in "()"
            ):
                output.append("$")
                cursor += 2
                replacements += 1
            else:
                output.append(line[cursor])
                cursor += 1
        offset += len(line)
    return "".join(output), replacements


def expand_snippets(root: Path, text: str, seen: frozenset[Path]) -> str:
    def replace(match: re.Match[str]) -> str:
        target = (root / match.group(1)).resolve()
        if target in seen or not target.is_file():
            return match.group(0)
        payload = target.read_text(encoding="utf-8")
        return expand_snippets(root, payload, seen | {target})

    return SNIPPET.sub(replace, text)


class GeneratedPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.math_depth = 0
        self.math_tag: Optional[str] = None
        self.math_kind = ""
        self.math_buffer: list[str] = []
        self.skip_depth = 0
        self.visible_buffer: list[str] = []
        self.expressions: list[str] = []
        self.kinds: list[str] = []

    @staticmethod
    def classes(attrs: Sequence[tuple[str, Optional[str]]]) -> list[str]:
        return (dict(attrs).get("class") or "").split()

    def handle_starttag(
        self, tag: str, attrs: Sequence[tuple[str, Optional[str]]]
    ) -> None:
        classes = self.classes(attrs)
        if tag == "article" and "md-content__inner" in classes:
            self.article_depth += 1
        if not self.article_depth:
            return
        if self.math_tag is not None:
            self.math_depth += 1
        elif "arithmatex" in classes:
            self.math_tag = tag
            self.math_kind = "display" if tag == "div" else "inline"
            self.math_depth = 1
            self.math_buffer = []
        elif tag in ("pre", "code", "script", "style"):
            self.skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.math_tag is not None:
            self.math_depth -= 1
            if self.math_depth == 0:
                payload = "".join(self.math_buffer).strip()
                if payload.startswith(r"\(") and payload.endswith(r"\)"):
                    payload = payload[2:-2].strip()
                elif payload.startswith(r"\[") and payload.endswith(r"\]"):
                    payload = payload[2:-2].strip()
                self.expressions.append(payload)
                self.kinds.append(self.math_kind)
                self.math_tag = None
                self.math_kind = ""
                self.math_buffer = []
        elif self.article_depth and tag in ("pre", "code", "script", "style"):
            self.skip_depth = max(0, self.skip_depth - 1)
        if tag == "article" and self.article_depth:
            self.article_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.math_tag is not None:
            self.math_buffer.append(data)
        elif self.article_depth and not self.skip_depth:
            self.visible_buffer.append(data)

    @property
    def visible_text(self) -> str:
        return " ".join(self.visible_buffer)


def generated_path(site_dir: Path, relative: Path) -> Path:
    if relative == Path("index.md"):
        return site_dir / "index.html"
    if relative.name == "index.md":
        return site_dir / relative.parent / "index.html"
    return site_dir / relative.with_suffix("") / "index.html"


def compare_generated_site(
    root: Path,
    site_dir: Path,
    source_paths: list[Path],
    errors: list[str],
) -> int:
    total = 0
    docs_root = root / "docs"
    for source in source_paths:
        try:
            relative = source.relative_to(docs_root)
        except ValueError:
            continue
        output = generated_path(site_dir, relative)
        if not output.is_file():
            errors.append(f"{relative}: generated page is missing at {output}")
            continue
        expanded = expand_snippets(
            root,
            source.read_text(encoding="utf-8"),
            frozenset({source.resolve()}),
        )
        expected, source_errors = scan_markdown(relative, expanded)
        errors.extend(source_errors)
        parser = GeneratedPageParser()
        parser.feed(output.read_text(encoding="utf-8"))
        total += len(parser.expressions)
        expected_tex = [item.tex for item in expected]
        expected_kinds = [item.kind for item in expected]
        if expected_tex != parser.expressions or expected_kinds != parser.kinds:
            errors.append(
                f"{relative}: source and generated HTML math differ "
                f"({len(expected_tex)} versus {len(parser.expressions)})"
            )
        if re.search(r"\\(?:[A-Za-z]+|[\[\]()])", parser.visible_text):
            errors.append(f"{relative}: generated page leaks raw TeX into prose")
        for label, pattern in VISIBLE_MARKDOWN_PATTERNS:
            match = pattern.search(parser.visible_text)
            if match:
                excerpt = " ".join(match.group(0).split())
                errors.append(
                    f"{relative}: generated page leaks raw Markdown "
                    f"{label}: {excerpt!r}"
                )
    return total


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_string: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def serve_directory(directory: Path) -> Iterable[str]:
    class Handler(QuietHandler):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def chrome_binary(explicit: Optional[str]) -> Optional[str]:
    candidates = [
        explicit,
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    return next(
        (
            candidate
            for candidate in candidates
            if candidate and Path(candidate).is_file()
        ),
        None,
    )


def chromedriver_binary(browser: str) -> Optional[str]:
    result = subprocess.run(
        [browser, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"\b(\d+)\.", result.stdout + result.stderr)
    browser_major = match.group(1) if match else None
    candidates = [shutil.which("chromedriver")]
    cache = Path.home() / ".cache" / "selenium" / "chromedriver"
    if cache.is_dir():
        candidates.extend(
            str(path)
            for path in sorted(cache.glob("*/*/chromedriver"), reverse=True)
        )
    return next(
        (
            candidate
            for candidate in candidates
            if candidate
            and Path(candidate).is_file()
            and (
                browser_major is None
                or any(
                    part == browser_major
                    or part.startswith(browser_major + ".")
                    for part in Path(candidate).parts[-3:]
                )
            )
        ),
        None,
    )


def browser_audit(
    site_dir: Path,
    explicit_chrome: Optional[str],
    visual_artifacts_dir: Optional[Path],
    errors: list[str],
) -> tuple[int, str, int, int]:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError:
        errors.append("browser audit requires Selenium")
        return 0, "unknown", 0, 0
    binary = chrome_binary(explicit_chrome)
    if binary is None:
        errors.append("browser audit could not find Chrome or Chromium")
        return 0, "unknown", 0, 0
    pages = sorted(site_dir.glob("**/index.html"))
    options = Options()
    options.binary_location = binary
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.set_capability("pageLoadStrategy", "eager")
    options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
    driver_path = chromedriver_binary(binary)

    def start_driver():
        service = Service(executable_path=driver_path) if driver_path else Service()
        current = webdriver.Chrome(service=service, options=options)
        current.set_page_load_timeout(45)
        current.set_script_timeout(30)
        return current, WebDriverWait(current, 30)

    driver, wait = start_driver()
    rendered = 0
    version = "unknown"
    visits = 0
    disclosure_interactions = 0
    audit_records: list[dict[str, object]] = []

    def wait_for_paint() -> None:
        driver.execute_async_script(
            """
            const done = arguments[arguments.length - 1];
            Promise.resolve(document.fonts ? document.fonts.ready : undefined)
              .then(() => requestAnimationFrame(() =>
                requestAnimationFrame(() => done(true))));
            """
        )

    def set_color_scheme(scheme: str) -> None:
        driver.execute_cdp_cmd(
            "Emulation.setEmulatedMedia",
            {
                "media": "",
                "features": [
                    {
                        "name": "prefers-color-scheme",
                        "value": scheme,
                    }
                ],
            },
        )
        wait_for_paint()

    def audit_paper_figure(
        index: int,
        *,
        deep_link: bool,
    ) -> dict[str, object]:
        driver.execute_async_script(
            """
            const index = arguments[0];
            const deepLink = arguments[1];
            const done = arguments[arguments.length - 1];
            const figure = document.querySelectorAll(
              'figure.paper-figure'
            )[index];
            if (!figure) {
              done(false);
              return;
            }
            const root = document.documentElement;
            const previousScrollBehavior = root.style.scrollBehavior;
            root.style.scrollBehavior = 'auto';
            if (deepLink && figure.id) {
                history.replaceState(
                  null,
                  '',
                  location.pathname + location.search
                );
                location.hash = encodeURIComponent(figure.id);
                figure.scrollIntoView({
                  block: 'start',
                  inline: 'nearest',
                });
            } else {
              figure.scrollIntoView({
                block: 'center',
                inline: 'nearest',
              });
            }
            requestAnimationFrame(() =>
              requestAnimationFrame(() => {
                root.style.scrollBehavior = previousScrollBehavior;
                done(true);
              }));
            """,
            index,
            deep_link,
        )
        return driver.execute_script(
            """
            const index = arguments[0];
            const deepLink = arguments[1];
            const figures = [
              ...document.querySelectorAll('figure.paper-figure')
            ];
            const figure = figures[index];
            if (!figure) {
              return {
                id: '',
                index,
                issues: ['figure disappeared during audit'],
              };
            }
            const issues = [];
            const round = (value) => Math.round(value * 10) / 10;
            const images = [...figure.querySelectorAll('img')];
            const directImages = [
              ...figure.querySelectorAll(
                ':scope > p > a > img, '
                + ':scope > .paper-figure__media > img'
              )
            ];
            const mediaLinks = [
              ...figure.querySelectorAll(
                ':scope > p > a, '
                + ':scope > a.paper-figure__media'
              )
            ];
            const captions = [
              ...figure.querySelectorAll(':scope > figcaption')
            ];
            const image = directImages[0] || images[0] || null;
            const mediaLink = mediaLinks[0] || null;
            const caption = captions[0] || null;
            if (!figure.id) {
              issues.push('missing stable id');
            }
            if (images.length !== 1 || directImages.length !== 1) {
              issues.push(
                `expected one direct image, found ${images.length}/`
                + `${directImages.length}`
              );
            }
            if (mediaLinks.length !== 1) {
              issues.push(`expected one media link, found ${mediaLinks.length}`);
            }
            if (captions.length !== 1) {
              issues.push(`expected one figcaption, found ${captions.length}`);
            }
            const sourceValue = figure.dataset.paperSource || '';
            const assetValue = figure.dataset.paperAsset || '';
            if (!!sourceValue !== !!assetValue) {
              issues.push('paper source and asset data attributes are incomplete');
            }
            const httpsLinks = caption
              ? [...caption.querySelectorAll(
                  '.paper-figure__source a[href^="https://"]'
                )]
              : [];
            if (httpsLinks.length < 2) {
              issues.push(
                `expected source and license links, found ${httpsLinks.length}`
              );
            }
            const focusable = (element) =>
              !!element
              && element.matches('a[href]')
              && element.tabIndex >= 0;
            if (!focusable(mediaLink)) {
              issues.push('media link is not keyboard focusable');
            }
            if (httpsLinks.some((link) => !focusable(link))) {
              issues.push('source or license link is not keyboard focusable');
            }
            let naturalWidth = 0;
            let naturalHeight = 0;
            let clientWidth = 0;
            let clientHeight = 0;
            let density = null;
            let imageBackground = null;
            let mediaBackground = null;
            if (!image) {
              issues.push('missing image');
            } else {
              naturalWidth = image.naturalWidth;
              naturalHeight = image.naturalHeight;
              clientWidth = image.clientWidth;
              clientHeight = image.clientHeight;
              density = clientWidth > 0
                ? round(naturalWidth / clientWidth)
                : null;
              if (!image.complete || naturalWidth === 0) {
                issues.push('image did not load');
              }
              const altLength = (image.alt || '').trim().length;
              if (altLength < 20 || altLength > 240) {
                issues.push(`alt length is ${altLength}, expected 20..240`);
              }
              const imageStyle = getComputedStyle(image);
              if (imageStyle.filter !== 'none') {
                issues.push(`image filter is ${imageStyle.filter}`);
              }
              if (clientWidth <= 0 || clientHeight <= 0) {
                issues.push('image has no rendered area');
              } else if (naturalWidth / clientWidth < 1.5) {
                issues.push(
                  `image density ${density}x is below the 1.5x floor`
                );
              }
              imageBackground = imageStyle.backgroundColor;
            }
            const white = (value) => {
              const match = value && value.match(
                /^rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/
              );
              return !!match
                && Number(match[1]) >= 250
                && Number(match[2]) >= 250
                && Number(match[3]) >= 250;
            };
            if (mediaLink) {
              mediaBackground = getComputedStyle(mediaLink).backgroundColor;
            }
            if (!white(imageBackground) || !white(mediaBackground)) {
              issues.push(
                `media background is not opaque white `
                + `(image=${imageBackground}, link=${mediaBackground})`
              );
            }
            if (deepLink && figure.id) {
              figure.scrollIntoView({
                block: 'start',
                inline: 'nearest',
              });
            }
            const figureRect = figure.getBoundingClientRect();
            const article = document.querySelector(
              'article.md-content__inner'
            );
            const articleRect = article ? article.getBoundingClientRect() : null;
            if (
              articleRect
              && (
                figureRect.left < articleRect.left - 1
                || figureRect.right > articleRect.right + 1
              )
            ) {
              issues.push(
                `figure leaves article bounds `
                + `(${round(figureRect.left)}..${round(figureRect.right)} vs `
                + `${round(articleRect.left)}..${round(articleRect.right)})`
              );
            }
            if (figure.scrollWidth > figure.clientWidth + 1) {
              issues.push(
                `figure overflows horizontally `
                + `(${figure.scrollWidth}>${figure.clientWidth})`
              );
            }
            const captionRect = caption
              ? caption.getBoundingClientRect()
              : null;
            if (!captionRect || captionRect.height <= 0) {
              issues.push('figcaption is not visible');
            }
            let deepLinkTop = null;
            let headerBottom = null;
            if (deepLink && figure.id) {
              const header = document.querySelector('.md-header');
              headerBottom = header
                ? round(header.getBoundingClientRect().bottom)
                : 0;
              deepLinkTop = round(figureRect.top);
              if (
                figureRect.top < headerBottom - 1
                || figureRect.top >= window.innerHeight
              ) {
                issues.push(
                  `deep link position ${deepLinkTop}px is outside visible `
                  + `content below header ${headerBottom}px`
                );
              }
            }
            return {
              id: figure.id || '',
              index,
              source: sourceValue || null,
              asset: assetValue || null,
              issues,
              naturalSize: [naturalWidth, naturalHeight],
              renderedSize: [clientWidth, clientHeight],
              density,
              figureWidth: round(figureRect.width),
              figureScrollWidth: figure.scrollWidth,
              deepLinkTop,
              headerBottom,
              imageBackground,
              mediaBackground,
              sourceLinks: httpsLinks.length,
            };
            """,
            index,
            deep_link,
        )

    def capture_figure_failure(
        route: str,
        width: int,
        scheme: str,
        index: int,
        figure_id: str,
    ) -> Optional[str]:
        if visual_artifacts_dir is None:
            return None
        stem = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            route.strip("/") or "index",
        ).strip("-")
        anchor = re.sub(
            r"[^a-zA-Z0-9._-]+",
            "-",
            figure_id or f"figure-{index}",
        ).strip("-")
        destination = (
            visual_artifacts_dir
            / "screenshots"
            / f"{stem}-{width}-{scheme}-{anchor}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            elements = driver.find_elements(
                "css selector",
                "figure.paper-figure",
            )
            if index < len(elements):
                elements[index].screenshot(str(destination))
            else:
                driver.save_screenshot(str(destination))
            return destination.relative_to(visual_artifacts_dir).as_posix()
        except Exception:
            return None

    def write_audit_report() -> None:
        if visual_artifacts_dir is None:
            return
        visual_artifacts_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "site_dir": str(site_dir),
            "records": audit_records,
        }
        (visual_artifacts_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    with serve_directory(site_dir) as base_url:
        try:
            for viewport_index, (width, height, mobile) in enumerate(
                (
                    (1440, 1000, False),
                    (390, 844, True),
                )
            ):
                if viewport_index:
                    with contextlib.suppress(Exception):
                        driver.quit()
                    driver, wait = start_driver()
                disclosure_checked = False
                set_color_scheme("light")
                driver.execute_cdp_cmd(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": width,
                        "height": height,
                        "deviceScaleFactor": 1,
                        "mobile": mobile,
                    },
                )
                for page in pages:
                    relative = page.relative_to(site_dir)
                    route = (
                        "/"
                        if relative == Path("index.html")
                        else "/" + relative.parent.as_posix() + "/"
                    )
                    visits += 1
                    route_error_start = len(errors)
                    route_record: dict[str, object] = {
                        "route": route,
                        "viewport": {
                            "width": width,
                            "height": height,
                            "mobile": mobile,
                        },
                        "paper_figure_count": 0,
                        "paper_figures": [],
                    }
                    try:
                        wrappers = 0
                        for attempt in range(3):
                            try:
                                driver.get_log("browser")
                                driver.get(base_url + route)
                                wait.until(
                                    lambda current: current.execute_script(
                                        """
                                        const route = arguments[0];
                                        const canonical = document.querySelector(
                                          'link[rel="canonical"]'
                                        );
                                        return document.readyState === 'complete'
                                          && location.pathname === route
                                          && !!canonical
                                          && new URL(canonical.href).pathname
                                            === '/llm' + route
                                          && !!document.querySelector(
                                            'article.md-content__inner'
                                          );
                                        """,
                                        route,
                                    )
                                )
                                wrappers = driver.execute_script(
                                    "return document.querySelectorAll("
                                    "'.arithmatex').length"
                                )
                                if wrappers:
                                    wait.until(
                                        lambda current: current.execute_script(
                                            "return !!(window.MathJax && "
                                            "window.MathJax.version)"
                                        )
                                    )
                                    ready = driver.execute_async_script(
                                        """
                                        const done = arguments[0];
                                        Promise.resolve(
                                          window.MathJax.startup.promise
                                        )
                                          .then(() => document.fonts
                                            ? document.fonts.ready
                                            : Promise.resolve())
                                          .then(() => new Promise((resolve) =>
                                            requestAnimationFrame(() =>
                                              requestAnimationFrame(resolve))))
                                          .then(() => done(true))
                                          .catch((error) =>
                                            done(String(error)));
                                        """
                                    )
                                    if ready is not True:
                                        raise RuntimeError(
                                            f"MathJax startup failed: {ready}"
                                        )
                                else:
                                    driver.execute_async_script(
                                        """
                                        const done = arguments[0];
                                        Promise.resolve(document.fonts
                                          ? document.fonts.ready
                                          : Promise.resolve())
                                          .then(() => requestAnimationFrame(() =>
                                            requestAnimationFrame(() =>
                                              done(true))));
                                        """
                                    )
                                break
                            except Exception:
                                driver.get_log("browser")
                                if attempt == 2:
                                    raise
                        paper_image_count = driver.execute_script(
                            """
                            const images = [
                              ...document.querySelectorAll(
                                'figure.paper-figure img'
                              )
                            ];
                            images.forEach((image) => {
                              image.loading = 'eager';
                            });
                            return images.length;
                            """
                        )
                        if paper_image_count:
                            wait.until(
                                lambda current: current.execute_script(
                                    """
                                    return [...document.querySelectorAll(
                                      'figure.paper-figure img'
                                    )].every((image) => image.complete);
                                    """
                                )
                            )
                        stats = driver.execute_script(
                            """
                            const wrappers = [
                              ...document.querySelectorAll('.arithmatex')
                            ];
                            const article = document.querySelector(
                              'article.md-content__inner'
                            );
                            const clone = article ? article.cloneNode(true) : null;
                            if (clone) {
                              clone.querySelectorAll(
                                'pre, code, script, style, .arithmatex'
                              ).forEach((node) => node.remove());
                            }
                            const raw = clone
                              ? /\\\\(?:[A-Za-z]+|[\\[\\]()])/.test(
                                  clone.textContent
                                )
                              : false;
                            const rawMarkdown = clone
                              ? /\\*\\*[^*\\n]+\\*\\*/.test(clone.textContent)
                              : false;
                            const invalid = wrappers.filter((element) =>
                              element.querySelectorAll(
                                ':scope > mjx-container'
                              ).length !== 1
                              || element.querySelectorAll(
                                'mjx-assistive-mml mjx-container'
                              ).length !== 0
                            ).length;
                            const mathErrors = [
                              ...document.querySelectorAll(
                                'mjx-merror, .MathJax_Error'
                              )
                            ].map((node) => node.textContent.trim());
                            const badOverflow = wrappers.filter((element) => {
                              if (element.tagName !== 'DIV') return false;
                              if (element.scrollWidth <= element.clientWidth + 1) {
                                return false;
                              }
                              const overflow = getComputedStyle(element).overflowX;
                              return overflow !== 'auto'
                                && overflow !== 'scroll';
                            }).length;
                            const inlineMathIssues = wrappers.filter((element) => {
                              if (element.tagName !== 'SPAN') return false;
                              const style = getComputedStyle(element);
                              const shift = Number.parseFloat(style.verticalAlign);
                              const fontSize = Number.parseFloat(style.fontSize);
                              const ratio = shift / fontSize;
                              return style.display !== 'inline-block'
                                || style.overflowX !== 'visible'
                                || style.overflowY !== 'visible'
                                || !Number.isFinite(ratio)
                                || ratio < -0.24
                                || ratio > -0.20;
                            }).length;
                            const displayMathIssues = wrappers.filter((element) => {
                              if (element.tagName !== 'DIV') return false;
                              const overflow = getComputedStyle(element).overflowX;
                              return overflow !== 'auto'
                                && overflow !== 'scroll';
                            }).length;
                            const brokenImages = [
                              ...document.querySelectorAll(
                                'article.md-content__inner img'
                              )
                            ].filter((image) =>
                              image.complete && image.naturalWidth === 0
                            ).length;
                            const paperFigures = [
                              ...document.querySelectorAll(
                                'figure.paper-figure'
                              )
                            ];
                            const documentOverflow =
                              document.documentElement.scrollWidth
                              > document.documentElement.clientWidth + 1;
                            const disclosures = [
                              ...document.querySelectorAll(
                                'details.code-disclosure'
                              )
                            ];
                            const invalidDisclosures = disclosures.filter(
                              (details) => {
                                const summary = details.firstElementChild;
                                return !summary
                                  || summary.tagName !== 'SUMMARY'
                                  || !summary.id
                                  || details.querySelectorAll(
                                    ':scope > summary'
                                  ).length !== 1
                                  || !details.querySelector('pre > code')
                                  || !!details.querySelector(
                                    'details.code-disclosure'
                                  );
                              }
                            ).length;
                            const openDisclosures = disclosures.filter(
                              (details) => details.open
                            ).length;
                            const missingDisclosureCopies = disclosures.filter(
                              (details) =>
                                details.querySelectorAll(
                                  '.md-code__button[data-md-type="copy"]'
                                ).length
                                < details.querySelectorAll('pre > code').length
                            ).length;
                            const overflowingElements = documentOverflow
                              ? [...document.querySelectorAll(
                                  'article.md-content__inner *'
                                )]
                                  .map((element) => {
                                    const rect =
                                      element.getBoundingClientRect();
                                    return {
                                      element,
                                      right: rect.right,
                                      width: rect.width,
                                    };
                                  })
                                  .filter(({ right }) =>
                                    right
                                    > document.documentElement.clientWidth + 1
                                  )
                                  .sort((left, right) =>
                                    right.right - left.right
                                  )
                                  .slice(0, 5)
                                  .map(({ element, right, width }) => ({
                                    tag: element.tagName.toLowerCase(),
                                    className:
                                      typeof element.className === 'string'
                                        ? element.className
                                        : '',
                                    right: Math.round(right * 10) / 10,
                                    width: Math.round(width * 10) / 10,
                                    text: (
                                      element.textContent || ''
                                    ).trim().replace(/\\s+/g, ' ').slice(0, 80),
                                  }))
                              : [];
                            return {
                              version: window.MathJax
                                && window.MathJax.version
                                ? window.MathJax.version
                                : null,
                              wrappers: wrappers.length,
                              invalid,
                              mathErrors,
                              badOverflow,
                              inlineMathIssues,
                              displayMathIssues,
                              brokenImages,
                              paperFigures: paperFigures.length,
                              raw,
                              rawMarkdown,
                              article: !!article,
                              documentOverflow,
                              overflowingElements,
                              disclosures: disclosures.length,
                              invalidDisclosures,
                              openDisclosures,
                              missingDisclosureCopies,
                            };
                            """
                        )
                        if stats["version"]:
                            version = stats["version"]
                        rendered += stats["wrappers"]
                        route_record["paper_figure_count"] = stats[
                            "paperFigures"
                        ]
                        paper_results: list[dict[str, object]] = []
                        if stats["paperFigures"]:
                            for scheme in ("light", "dark"):
                                set_color_scheme(scheme)
                                for figure_index in range(
                                    stats["paperFigures"]
                                ):
                                    result = audit_paper_figure(
                                        figure_index,
                                        deep_link=scheme == "light",
                                    )
                                    result["scheme"] = scheme
                                    issues = result.get("issues", [])
                                    if not isinstance(issues, list):
                                        issues = ["audit returned invalid issues"]
                                        result["issues"] = issues
                                    if issues:
                                        artifact = capture_figure_failure(
                                            route,
                                            width,
                                            scheme,
                                            figure_index,
                                            str(result.get("id", "")),
                                        )
                                        if artifact:
                                            result["screenshot"] = artifact
                                        metrics = (
                                            f"natural={result.get('naturalSize')}, "
                                            f"rendered={result.get('renderedSize')}, "
                                            f"density={result.get('density')}, "
                                            f"figure={result.get('figureWidth')}/"
                                            f"{result.get('figureScrollWidth')}, "
                                            f"deep-link={result.get('deepLinkTop')}/"
                                            f"{result.get('headerBottom')}"
                                        )
                                        errors.append(
                                            f"{route}#"
                                            f"{result.get('id') or figure_index}: "
                                            f"paper figure failed at {width}px/"
                                            f"{scheme}: "
                                            + "; ".join(
                                                str(issue) for issue in issues
                                            )
                                            + f" [{metrics}]"
                                        )
                                    paper_results.append(result)
                            set_color_scheme("light")
                        route_record["paper_figures"] = paper_results
                        if (
                            stats["wrappers"]
                            and stats["version"] != EXPECTED_MATHJAX_VERSION
                        ):
                            errors.append(
                                f"{route}: expected MathJax "
                                f"{EXPECTED_MATHJAX_VERSION}, loaded "
                                f"{stats['version']}"
                            )
                        if not stats["article"]:
                            errors.append(f"{route}: missing documentation article")
                        if stats["invalid"]:
                            errors.append(
                                f"{route}: {stats['invalid']} formulas were not "
                                "rendered exactly once"
                            )
                        if stats["mathErrors"]:
                            errors.append(
                                f"{route}: MathJax errors: "
                                + "; ".join(stats["mathErrors"])
                            )
                        if stats["raw"]:
                            errors.append(f"{route}: visible prose leaks raw TeX")
                        if stats["rawMarkdown"]:
                            raw_markdown = True
                            for _ in range(2):
                                wait_for_paint()
                                raw_markdown = driver.execute_script(
                                    """
                                    const article = document.querySelector(
                                      'article.md-content__inner'
                                    );
                                    const clone = article
                                      ? article.cloneNode(true)
                                      : null;
                                    if (!clone) return false;
                                    clone.querySelectorAll(
                                      'pre, code, script, style, .arithmatex'
                                    ).forEach((node) => node.remove());
                                    return /\\*\\*[^*\\n]+\\*\\*/.test(
                                      clone.textContent
                                    );
                                    """
                                )
                                if not raw_markdown:
                                    break
                            if raw_markdown:
                                errors.append(
                                    f"{route}: visible prose leaks raw Markdown "
                                    "emphasis"
                                )
                        if stats["badOverflow"]:
                            errors.append(
                                f"{route}: {stats['badOverflow']} formulas "
                                f"overflow at {width}px without scrolling"
                            )
                        if stats["inlineMathIssues"]:
                            errors.append(
                                f"{route}: {stats['inlineMathIssues']} inline "
                                "formulas violate the shared baseline or "
                                "clipping contract"
                            )
                        if stats["displayMathIssues"]:
                            errors.append(
                                f"{route}: {stats['displayMathIssues']} display "
                                "formulas lack horizontal overflow handling"
                            )
                        if stats["brokenImages"]:
                            errors.append(
                                f"{route}: {stats['brokenImages']} broken images"
                            )
                        if stats["invalidDisclosures"]:
                            errors.append(
                                f"{route}: {stats['invalidDisclosures']} "
                                "invalid code disclosures"
                            )
                        if stats["openDisclosures"]:
                            errors.append(
                                f"{route}: {stats['openDisclosures']} code "
                                "disclosures are initially open"
                            )
                        if stats["missingDisclosureCopies"]:
                            errors.append(
                                f"{route}: {stats['missingDisclosureCopies']} "
                                "code disclosures are missing copy controls"
                            )
                        if stats["documentOverflow"]:
                            details = "; ".join(
                                (
                                    f"{item['tag']}"
                                    f".{item['className'].replace(' ', '.')}"
                                    f" right={item['right']}"
                                    f" width={item['width']}"
                                    f" text={item['text']!r}"
                                )
                                for item in stats["overflowingElements"]
                            )
                            errors.append(
                                f"{route}: page has horizontal overflow at "
                                f"{width}px"
                                + (f": {details}" if details else "")
                            )
                        if stats["disclosures"] and not disclosure_checked:
                            summary_id = driver.execute_script(
                                "return document.querySelector("
                                "'details.code-disclosure > summary').id"
                            )
                            enter_handled = driver.execute_script(
                                """
                                const summary = document.getElementById(
                                  arguments[0]
                                );
                                summary.focus();
                                const event = new KeyboardEvent('keydown', {
                                  key: 'Enter',
                                  code: 'Enter',
                                  bubbles: true,
                                  cancelable: true,
                                });
                                summary.dispatchEvent(event);
                                return event.defaultPrevented;
                                """,
                                summary_id,
                            )
                            if not enter_handled:
                                raise RuntimeError(
                                    "Enter was not handled by the disclosure"
                                )
                            wait.until(
                                lambda current: current.execute_script(
                                    "return document.getElementById("
                                    "arguments[0]).closest('details').open",
                                    summary_id,
                                )
                            )
                            space_handled = driver.execute_script(
                                """
                                const summary = document.getElementById(
                                  arguments[0]
                                );
                                const event = new KeyboardEvent('keydown', {
                                  key: ' ',
                                  code: 'Space',
                                  bubbles: true,
                                  cancelable: true,
                                });
                                summary.dispatchEvent(event);
                                return event.defaultPrevented;
                                """,
                                summary_id,
                            )
                            if not space_handled:
                                raise RuntimeError(
                                    "Space was not handled by the disclosure"
                                )
                            wait.until(
                                lambda current: not current.execute_script(
                                    "return document.getElementById("
                                    "arguments[0]).closest('details').open",
                                    summary_id,
                                )
                            )
                            driver.execute_script(
                                "location.hash = arguments[0];",
                                summary_id,
                            )
                            wait.until(
                                lambda current: current.execute_script(
                                    "return document.getElementById("
                                    "arguments[0]).closest('details').open",
                                    summary_id,
                                )
                            )
                            driver.execute_script(
                                "document.getElementById(arguments[0])"
                                ".closest('details').open = false; "
                                "history.replaceState(null, '', "
                                "location.pathname + location.search);",
                                summary_id,
                            )
                            disclosure_checked = True
                            disclosure_interactions += 1
                        logs = driver.get_log("browser")
                        severe = [
                            item["message"]
                            for item in logs
                            if item["level"] == "SEVERE"
                            and re.search(
                                r"mathjax|tex-mml-chtml",
                                item["message"],
                                re.IGNORECASE,
                            )
                        ]
                        if severe and stats["wrappers"]:
                            errors.append(
                                f"{route}: browser reported MathJax load errors: "
                                + "; ".join(severe)
                            )
                    except Exception as exc:
                        message = (
                            f"{route}: browser audit failed at {width}px "
                            f"after three attempts: {type(exc).__name__}: {exc}"
                        )
                        errors.append(message)
                        route_record["exception"] = message
                        artifact = capture_figure_failure(
                            route,
                            width,
                            "exception",
                            0,
                            "browser-audit",
                        )
                        if artifact:
                            route_record["screenshot"] = artifact
                    finally:
                        route_errors = errors[route_error_start:]
                        route_record["status"] = (
                            "failed" if route_errors else "passed"
                        )
                        if route_errors:
                            route_record["errors"] = route_errors
                        audit_records.append(route_record)
            driver.execute_cdp_cmd("Emulation.clearDeviceMetricsOverride", {})
        finally:
            driver.quit()
            write_audit_report()
    return rendered, version, visits, disclosure_interactions


def source_paths(root: Path) -> list[Path]:
    paths = [root / "README.md"]
    paths.extend(sorted((root / "docs").rglob("*.md")))
    includes = root / "includes"
    if includes.is_dir():
        paths.extend(sorted(includes.rglob("*.md")))
    return [path for path in paths if path.is_file()]


def validate_fixtures(errors: list[str]) -> None:
    bad = {
        "legacy": r"Bad \(x\).",
        "unclosed": "Bad $x.",
        "brace": "$x_{i$",
        "plain": r"Bad \frac{1}{2}.",
    }
    for name, text in bad.items():
        _, fixture_errors = scan_markdown(Path(f"<fixture:{name}>"), text)
        if not fixture_errors:
            errors.append(f"checker fixture did not reject {name}")
    good = "$x_i$.\n\n$$\n\\frac{1}{2}\n$$\n\n`$code$`\n"
    _, fixture_errors = scan_markdown(Path("<fixture:good>"), good)
    if fixture_errors:
        errors.append(
            "checker rejected its valid fixture: " + "; ".join(fixture_errors)
        )


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-dir",
        type=Path,
        help="compare source mathematics with an existing MkDocs output",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="render every generated page at desktop and mobile widths",
    )
    parser.add_argument(
        "--chrome-binary",
        help="explicit Chrome or Chromium executable for --browser",
    )
    parser.add_argument(
        "--visual-artifacts-dir",
        type=Path,
        help="write browser audit JSON and failure screenshots to this directory",
    )
    parser.add_argument(
        "--fix-legacy-delimiters",
        action="store_true",
        help="replace legacy math delimiters outside code with canonical dollars",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    paths = source_paths(root)
    if args.fix_legacy_delimiters:
        changed = 0
        replacements = 0
        for path in paths:
            text = path.read_text(encoding="utf-8")
            updated, count = canonicalize_legacy_math(
                path.relative_to(root),
                text,
            )
            if updated != text:
                path.write_text(updated, encoding="utf-8")
                changed += 1
                replacements += count
        print(
            f"canonicalized {replacements} delimiters across {changed} files"
        )
    errors: list[str] = []
    validate_fixtures(errors)
    expression_total = 0
    math_files = 0
    for path in paths:
        expressions, path_errors = scan_markdown(
            path.relative_to(root),
            path.read_text(encoding="utf-8"),
        )
        expression_total += len(expressions)
        math_files += bool(expressions)
        errors.extend(path_errors)
    print(
        f"source: {expression_total} expressions in {math_files} "
        f"of {len(paths)} Markdown files"
    )
    site_dir: Optional[Path] = None
    if args.site_dir:
        site_dir = args.site_dir
        if not site_dir.is_absolute():
            site_dir = root / site_dir
        if not site_dir.is_dir():
            errors.append(f"generated site directory does not exist: {site_dir}")
        else:
            generated = compare_generated_site(root, site_dir, paths, errors)
            print(f"generated HTML: {generated} Arithmatex expressions")
    if args.browser:
        if site_dir is None:
            site_dir = root / "site"
        visual_artifacts_dir = args.visual_artifacts_dir
        if (
            visual_artifacts_dir is not None
            and not visual_artifacts_dir.is_absolute()
        ):
            visual_artifacts_dir = root / visual_artifacts_dir
        if not site_dir.is_dir():
            errors.append(f"browser site directory does not exist: {site_dir}")
        else:
            rendered, version, visits, disclosure_interactions = browser_audit(
                site_dir,
                args.chrome_binary,
                visual_artifacts_dir,
                errors,
            )
            print(
                f"browser: {rendered} rendered expressions across "
                f"{visits} page visits with MathJax {version}; "
                f"{disclosure_interactions} disclosure interaction checks"
            )
    if errors:
        print("\nRendering check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Rendering check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
