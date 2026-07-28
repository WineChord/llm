#!/usr/bin/env python3
"""Validate cropped paper figures, provenance, and semantic page integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
PAPERS_ROOT = DOCS_ROOT / "assets" / "papers"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_ASSET_BYTES = 3 * 1024 * 1024
MAX_PAGE_AREA_RATIO = 0.65
FIGURE_BLOCK = re.compile(
    r"<figure\b(?P<attrs>[^>]*)>(?P<body>.*?)</figure>",
    re.DOTALL,
)
IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<target>[^)\s]+)[^)]*\)")
ATTRIBUTE = re.compile(r'\b(?P<name>[a-z]+)="(?P<value>[^"]+)"')
GENERIC_ALT = re.compile(r"^(?:图|图片|截图|figure|paper figure)\s*\d*$", re.IGNORECASE)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError("不是带有效 IHDR 的 PNG")
    return struct.unpack(">II", header[16:24])


def figure_blocks() -> tuple[dict[str, tuple[Path, str, str]], list[str]]:
    blocks: dict[str, tuple[Path, str, str]] = {}
    errors: list[str] = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in FIGURE_BLOCK.finditer(text):
            attrs = match.group("attrs")
            classes = re.search(r'\bclass="([^"]+)"', attrs)
            if not classes or "paper-figure" not in classes.group(1).split():
                continue
            identifier = re.search(r'\bid="([^"]+)"', attrs)
            if not identifier:
                errors.append(f"{path.relative_to(ROOT)}: paper-figure 缺少稳定 id")
                continue
            figure_id = identifier.group(1)
            if figure_id in blocks:
                previous = blocks[figure_id][0].relative_to(ROOT)
                errors.append(
                    f"{path.relative_to(ROOT)}: paper-figure id {figure_id} "
                    f"与 {previous} 重复"
                )
                continue
            blocks[figure_id] = (path, attrs, match.group("body"))
    return blocks, errors


def main() -> int:
    errors: list[str] = []
    blocks, block_errors = figure_blocks()
    errors.extend(block_errors)
    manifests = sorted(PAPERS_ROOT.glob("*/manifest.json"))
    registered_assets: set[Path] = set()
    registered_ids: set[str] = set()
    figure_count = 0

    if not manifests:
        errors.append("docs/assets/papers/: 缺少论文图表 manifest")

    for manifest_path in manifests:
        relative_manifest = manifest_path.relative_to(ROOT)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{relative_manifest}: manifest 无法读取：{exc}")
            continue

        source = manifest.get("source", {})
        render = manifest.get("render", {})
        figures = manifest.get("figures", [])
        page_size = source.get("page_size_points", [])
        pdf_url = source.get("pdf_url", "")
        license_url = source.get("license_url", "")
        copyright_notice = source.get("copyright", "")
        dpi = render.get("dpi")

        if manifest.get("schema_version") != 1:
            errors.append(f"{relative_manifest}: schema_version 必须为 1")
        if (
            not isinstance(pdf_url, str)
            or not pdf_url.startswith("https://")
            or "/main/" in pdf_url
        ):
            errors.append(f"{relative_manifest}: PDF 必须使用固定 revision 的 HTTPS URL")
        if not re.fullmatch(r"[0-9a-f]{64}", source.get("pdf_sha256", "")):
            errors.append(f"{relative_manifest}: PDF SHA-256 无效")
        if (
            not isinstance(page_size, list)
            or len(page_size) != 2
            or not all(isinstance(value, (int, float)) and value > 0 for value in page_size)
        ):
            errors.append(f"{relative_manifest}: page_size_points 无效")
            page_size = [1, 1]
        if not isinstance(dpi, int) or dpi < 144 or dpi > 600:
            errors.append(f"{relative_manifest}: render.dpi 应位于 144–600")
            dpi = 300
        if not isinstance(figures, list) or not figures:
            errors.append(f"{relative_manifest}: figures 不得为空")
            continue

        license_file = manifest_path.parent / source.get("license_file", "")
        if (
            not license_file.is_file()
            or not copyright_notice
            or copyright_notice not in license_file.read_text(encoding="utf-8")
        ):
            errors.append(f"{relative_manifest}: 缺少匹配版权声明的本地许可证")
        if not isinstance(license_url, str) or not license_url.startswith("https://"):
            errors.append(f"{relative_manifest}: license_url 必须是 HTTPS 链接")

        for record in figures:
            figure_count += 1
            figure_id = record.get("id", "")
            filename = record.get("file", "")
            used_by = ROOT / record.get("used_by", "")
            asset = manifest_path.parent / filename
            crop = record.get("crop_points", [])
            page = record.get("page")
            expected_width = record.get("pixel_width")
            expected_height = record.get("pixel_height")

            if not re.fullmatch(r"[a-z0-9][a-z0-9-]+", figure_id):
                errors.append(f"{relative_manifest}: figure id 无效：{figure_id!r}")
            if figure_id in registered_ids:
                errors.append(f"{relative_manifest}: figure id 重复登记：{figure_id}")
            registered_ids.add(figure_id)
            if not re.fullmatch(r"[a-z0-9][a-z0-9-]+\.png", filename):
                errors.append(f"{relative_manifest}: 图片文件名无效：{filename!r}")
            registered_assets.add(asset.resolve())
            if not asset.is_file():
                errors.append(f"{relative_manifest}: 图片不存在：{filename}")
                continue
            try:
                actual_width, actual_height = png_size(asset)
            except ValueError as exc:
                errors.append(f"{asset.relative_to(ROOT)}: {exc}")
                continue
            if (actual_width, actual_height) != (expected_width, expected_height):
                errors.append(
                    f"{asset.relative_to(ROOT)}: 像素尺寸 "
                    f"{actual_width}x{actual_height} 与 manifest 不符"
                )
            if actual_width < 900 or actual_height < 200:
                errors.append(f"{asset.relative_to(ROOT)}: 裁图分辨率过低")
            if asset.stat().st_size > MAX_ASSET_BYTES:
                errors.append(f"{asset.relative_to(ROOT)}: 图片超过 3 MiB")
            if digest(asset) != record.get("sha256"):
                errors.append(f"{asset.relative_to(ROOT)}: SHA-256 与 manifest 不符")

            if (
                not isinstance(crop, list)
                or len(crop) != 4
                or not all(isinstance(value, (int, float)) for value in crop)
            ):
                errors.append(f"{relative_manifest}: {figure_id} crop_points 无效")
            else:
                x0, y0, x1, y1 = crop
                page_width, page_height = page_size
                if not (0 <= x0 < x1 <= page_width and 0 <= y0 < y1 <= page_height):
                    errors.append(f"{relative_manifest}: {figure_id} crop 超出页面")
                else:
                    ratio = (x1 - x0) * (y1 - y0) / (page_width * page_height)
                    if ratio >= MAX_PAGE_AREA_RATIO:
                        errors.append(
                            f"{relative_manifest}: {figure_id} 裁取面积接近整页 "
                            f"({ratio:.1%})"
                        )
                    rendered_width = round((x1 - x0) * dpi / 72)
                    rendered_height = round((y1 - y0) * dpi / 72)
                    if (
                        abs(rendered_width - actual_width) > 1
                        or abs(rendered_height - actual_height) > 1
                    ):
                        errors.append(
                            f"{asset.relative_to(ROOT)}: 像素尺寸与 crop/DPI 不一致"
                        )

            if not isinstance(page, int) or page < 1:
                errors.append(f"{relative_manifest}: {figure_id} 页码无效")
            if not used_by.is_file() or used_by.suffix != ".md":
                errors.append(f"{relative_manifest}: {figure_id} used_by 无效")
            block = blocks.get(figure_id)
            if not block:
                errors.append(f"{relative_manifest}: {figure_id} 未嵌入正文")
                continue
            block_path, _, body = block
            if block_path.resolve() != used_by.resolve():
                errors.append(
                    f"{relative_manifest}: {figure_id} 实际位于 "
                    f"{block_path.relative_to(ROOT)}，与 used_by 不符"
                )
            if filename not in body or body.count(filename) < 2:
                errors.append(f"{block_path.relative_to(ROOT)}: {figure_id} 缺少高清图片链接")
            image_alts = [
                match.group("alt").strip()
                for match in IMAGE.finditer(body)
                if match.group("target").endswith(filename)
            ]
            if (
                len(image_alts) != 1
                or len(image_alts[0]) < 24
                or GENERIC_ALT.fullmatch(image_alts[0])
            ):
                errors.append(f"{block_path.relative_to(ROOT)}: {figure_id} alt 不够具体")
            attrs = dict(ATTRIBUTE.findall(body))
            for name, value in (
                ("width", str(expected_width)),
                ("height", str(expected_height)),
                ("loading", "lazy"),
                ("decoding", "async"),
            ):
                if attrs.get(name) != value:
                    errors.append(
                        f"{block_path.relative_to(ROOT)}: {figure_id} "
                        f"缺少 {name}={value}"
                    )
            caption = re.search(r"<figcaption>(.*?)</figcaption>", body, re.DOTALL)
            if not caption or len(re.sub(r"<[^>]+>", "", caption.group(1)).strip()) < 80:
                errors.append(f"{block_path.relative_to(ROOT)}: {figure_id} 图注过短")
            elif re.search(r"\$|\\[\[(]", caption.group(1)):
                errors.append(
                    f"{block_path.relative_to(ROOT)}: {figure_id} "
                    "图注不得泄漏未解析的数学分隔符"
                )
            source_link = f"{pdf_url}#page={page}"
            if source_link not in body or license_url not in body or "© 2026 Moonshot AI" not in body:
                errors.append(
                    f"{block_path.relative_to(ROOT)}: {figure_id} "
                    "缺少固定来源、版权或许可证"
                )

    unregistered = sorted(
        path.relative_to(ROOT)
        for path in PAPERS_ROOT.rglob("*.png")
        if path.resolve() not in registered_assets
    )
    if unregistered:
        errors.append(
            "docs/assets/papers/: 存在未登记图片："
            + "、".join(map(str, unregistered))
        )
    extra_blocks = sorted(set(blocks) - registered_ids)
    if extra_blocks:
        errors.append("docs/: 存在未登记 paper-figure：" + "、".join(extra_blocks))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"论文图表检查通过：{len(manifests)} 份 manifest，{figure_count} 幅裁图")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
