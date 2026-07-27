#!/usr/bin/env python3
"""Validate rendered internal links and fragments inside documentation articles."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import posixpath
import sys
from urllib.parse import unquote, urljoin, urlsplit


SITE_PREFIX = "/llm/"
REQUIRED_DEEP_LINKS = {
    "/reinforcement-learning/trust-region-ppo/": {"trpo"},
    "/reinforcement-learning/critic-free-baselines/": {"rloo"},
    "/landscape/works/sao-compactionrl/": {"sao", "compactionrl"},
    "/reinforcement-learning/grpo/": {"group-std", "dynamic-sampling"},
    "/landscape/works/dapo/": {
        "clip-higher",
        "dynamic-sampling",
        "token-loss",
        "overlong",
    },
    "/landscape/works/vapo/": {
        "value-pretraining",
        "decoupled-gae",
        "length-adaptive-gae",
    },
}


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.ids: set[str] = set()
        self.links: list[str] = []

    @staticmethod
    def classes(attrs: list[tuple[str, str | None]]) -> list[str]:
        return (dict(attrs).get("class") or "").split()

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if tag == "article" and "md-content__inner" in self.classes(attrs):
            self.article_depth += 1
        identifier = attributes.get("id")
        if identifier:
            self.ids.add(unquote(identifier))
        if self.article_depth and tag == "a":
            href = attributes.get("href")
            if href is not None:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self.article_depth:
            self.article_depth -= 1


def route_for(path: Path, site_dir: Path) -> str:
    relative = path.relative_to(site_dir)
    if relative == Path("index.html"):
        return "/"
    return "/" + relative.parent.as_posix().strip("/") + "/"


def normalize_route(source_route: str, href: str) -> tuple[str, str]:
    parsed = urlsplit(href)
    route = unquote(parsed.path)
    if route.startswith(SITE_PREFIX):
        route = "/" + route[len(SITE_PREFIX):]
    elif not route.startswith("/"):
        route = urlsplit(urljoin("https://local" + source_route, route)).path
    route = posixpath.normpath(route)
    if href.endswith("/") and route != "/":
        route += "/"
    if route.endswith("/index.html"):
        route = route[: -len("index.html")]
    elif route.endswith(".html"):
        route = route[: -len(".html")] + "/"
    if not route.endswith("/") and PurePosixPath(route).suffix == "":
        route += "/"
    return route or "/", unquote(parsed.fragment)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-dir", type=Path, default=Path("site"))
    args = parser.parse_args()
    site_dir = args.site_dir.resolve()
    errors: list[str] = []
    pages: dict[str, tuple[Path, PageParser]] = {}
    for path in sorted(site_dir.glob("**/index.html")):
        parsed = PageParser()
        parsed.feed(path.read_text(encoding="utf-8"))
        route = route_for(path, site_dir)
        pages[route] = (path, parsed)
    if not pages:
        print(f"{site_dir}: 没有生成页面", file=sys.stderr)
        return 1
    for route, fragments in REQUIRED_DEEP_LINKS.items():
        if route not in pages:
            errors.append(f"缺少稳定页面路由：{route}")
            continue
        missing = sorted(fragments - pages[route][1].ids)
        for fragment in missing:
            errors.append(f"{route}: 缺少稳定 fragment：#{fragment}")
    checked = 0
    for source_route, (source_path, parsed) in pages.items():
        for href in parsed.links:
            split = urlsplit(href)
            if split.scheme in {"http", "https", "mailto", "tel"} or split.netloc:
                continue
            if split.scheme or href.startswith(("javascript:", "data:")):
                errors.append(
                    f"{source_path.relative_to(site_dir)}: 不支持的内部链接：{href}"
                )
                continue
            target_route, fragment = normalize_route(source_route, href)
            if PurePosixPath(target_route).suffix:
                target_file = site_dir / target_route.lstrip("/")
                if not target_file.is_file():
                    errors.append(
                        f"{source_path.relative_to(site_dir)}: "
                        f"内部资源不存在：{href}"
                    )
                checked += 1
                continue
            if target_route not in pages:
                errors.append(
                    f"{source_path.relative_to(site_dir)}: "
                    f"内部页面不存在：{href} -> {target_route}"
                )
                continue
            if fragment and fragment not in pages[target_route][1].ids:
                errors.append(
                    f"{source_path.relative_to(site_dir)}: "
                    f"fragment 不存在：{href}"
                )
            checked += 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"渲染链接检查通过：{len(pages)} 个页面，{checked} 条内部链接")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
