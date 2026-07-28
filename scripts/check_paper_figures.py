#!/usr/bin/env python3
"""Validate cropped paper figures, provenance, and semantic integration.

Schema v1 remains readable for compatibility. New and migrated sources use
schema v2, which separates immutable source artifacts, cropped assets, and page
placements so one asset can be reused without duplicating the binary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
from typing import Any, Optional, Sequence
from urllib.parse import unquote, urlsplit
import zlib


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
PAPERS_ROOT = DOCS_ROOT / "assets" / "papers"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40}")
ARXIV_VERSION = re.compile(r"\d{4}\.\d{4,5}v\d+")
SLUG = re.compile(r"[a-z0-9][a-z0-9-]*")
PNG_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\.png")
MAX_ASSET_BYTES = 3 * 1024 * 1024
V1_MAX_PAGE_AREA_RATIO = 0.65
V2_MAX_PAGE_AREA_RATIO = 0.85
ALLOWED_PNG_CHUNKS = {
    "IHDR",
    "PLTE",
    "IDAT",
    "IEND",
    "sRGB",
    "gAMA",
    "cHRM",
    "pHYs",
}
FORBIDDEN_METADATA_CHUNKS = {"tEXt", "zTXt", "iTXt", "eXIf"}
ALLOWED_KINDS = {"figure", "table", "algorithm", "listing", "diagram", "panel"}
ALLOWED_ROLES = {"normal", "compact", "portrait", "wide"}
ALLOWED_MEDIA_LINKS = {"asset", "source"}
ALLOWED_COORDINATE_SPACES = {
    "rendered-page-points-top-left",
    "pdf-points-bottom-left",
    "source-pixels-top-left",
}
ALLOWED_VERSION_KINDS = {"git_commit", "arxiv_version", "release_tag", "sha256_only"}
ALLOWED_ARTIFACT_KINDS = {"pdf", "standalone_raster"}
VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
GENERIC_ALT = re.compile(
    r"^(?:图|图片|截图|表|表格|figure|table|paper figure)\s*[\w.-]*$",
    re.IGNORECASE,
)
FIGURE_BLOCK = re.compile(
    r"<figure\b(?P<attrs>[^>]*)>(?P<body>.*?)</figure>",
    re.DOTALL,
)
HTML_ATTRIBUTE = re.compile(
    r"""(?P<name>[A-Za-z_:][A-Za-z0-9_.:-]*)
        (?:\s*=\s*(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'))?""",
    re.VERBOSE,
)
MARKDOWN_IMAGE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]"
    r"\((?P<target>[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)"
    r"(?:\s*\{(?P<attrs>[^}]*)\})?",
)
MARKDOWN_ATTRIBUTE = re.compile(
    r"""(?P<name>[A-Za-z_:][A-Za-z0-9_.:-]*)
        (?:=(?:"(?P<double>[^"]*)"|'(?P<single>[^']*)'|(?P<bare>[^\s]+)))?""",
    re.VERBOSE,
)


class DuplicateKeyError(ValueError):
    pass


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    bit_depth: int
    color_type: int
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class Placement:
    page_relative: str
    page_path: Path
    anchor: str
    role: str
    media_link: str


@dataclass
class SourceRecord:
    schema_version: int
    manifest_path: Path
    source_id: str
    title: str
    artifact_kind: str
    artifact_url: str
    artifact_sha256: str
    version_kind: str
    version_value: str
    page_count: Optional[int]
    license_name: str
    license_url: str
    license_file: Optional[Path]
    license_sha256: Optional[str]
    credit: str
    dpi: int
    coordinate_space: str
    colorspace: str
    image_format: str
    metadata_policy: str
    assets: list["AssetRecord"] = field(default_factory=list)


@dataclass
class AssetRecord:
    source: SourceRecord
    asset_id: str
    file_name: str
    file_path: Optional[Path]
    kind: str
    source_label: str
    page: int
    page_box: tuple[float, float]
    rotation: int
    crop_box: tuple[float, float, float, float]
    pixel_width: int
    pixel_height: int
    sha256: str
    placements: list[Placement]


@dataclass(frozen=True)
class FigureBlock:
    path: Path
    anchor: str
    attrs: dict[str, str]
    body: str


@dataclass
class HtmlFigure:
    attrs: dict[str, str]
    images: list[dict[str, str]] = field(default_factory=list)
    media_hrefs: list[str] = field(default_factory=list)
    caption_links: list[str] = field(default_factory=list)
    caption_parts: list[str] = field(default_factory=list)

    @property
    def caption_text(self) -> str:
        return re.sub(r"\s+", " ", unescape("".join(self.caption_parts))).strip()


def duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"重复 JSON key：{key}")
        result[key] = value
    return result


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def parse_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in HTML_ATTRIBUTE.finditer(raw):
        value = match.group("double")
        if value is None:
            value = match.group("single")
        attrs[match.group("name")] = value or ""
    return attrs


def parse_markdown_attrs(raw: Optional[str]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if not raw:
        return attrs
    for match in MARKDOWN_ATTRIBUTE.finditer(raw):
        value = match.group("double")
        if value is None:
            value = match.group("single")
        if value is None:
            value = match.group("bare")
        attrs[match.group("name")] = value or ""
    return attrs


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def expect_dict(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label}: 必须是 JSON object")
        return {}
    return value


def expect_list(value: Any, label: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{label}: 必须是 JSON array")
        return []
    return value


def expect_keys(
    value: dict[str, Any],
    label: str,
    errors: list[str],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        errors.append(f"{label}: 缺少字段：{', '.join(missing)}")
    if unknown:
        errors.append(f"{label}: 未知字段：{', '.join(unknown)}")


def expect_string(
    value: Any,
    label: str,
    errors: list[str],
    *,
    pattern: Optional[re.Pattern[str]] = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{label}: 必须是非空字符串")
        return ""
    result = value.strip()
    if pattern is not None and pattern.fullmatch(result) is None:
        errors.append(f"{label}: 格式无效：{result!r}")
    return result


def expect_int(
    value: Any,
    label: str,
    errors: list[str],
    *,
    minimum: Optional[int] = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{label}: 必须是整数")
        return 0
    if minimum is not None and value < minimum:
        errors.append(f"{label}: 必须大于等于 {minimum}")
    return value


def expect_pair(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(is_number(item) and item > 0 for item in value)
    ):
        errors.append(f"{label}: 必须是两个正数")
        return (1.0, 1.0)
    return (float(value[0]), float(value[1]))


def expect_int_pair(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
    ):
        errors.append(f"{label}: 必须是两个正整数")
        return (1, 1)
    return (value[0], value[1])


def expect_box(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(is_number(item) for item in value)
    ):
        errors.append(f"{label}: 必须是四个数值")
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def expect_int_box(
    value: Any,
    label: str,
    errors: list[str],
) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in value
        )
    ):
        errors.append(f"{label}: 必须是四个整数")
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def https_url(value: Any, label: str, errors: list[str]) -> str:
    result = expect_string(value, label, errors)
    if not result:
        return ""
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        errors.append(f"{label}: 必须是无凭据的 HTTPS URL")
    return result


def safe_path(
    base: Path,
    value: Any,
    label: str,
    errors: list[str],
    *,
    allowed_root: Optional[Path] = None,
    file_name_only: bool = False,
) -> Optional[Path]:
    raw = expect_string(value, label, errors)
    if not raw:
        return None
    pure = PurePosixPath(raw)
    if (
        pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or (file_name_only and len(pure.parts) != 1)
    ):
        errors.append(f"{label}: 路径必须是受限的 POSIX 相对路径")
        return None
    root = (allowed_root or base).resolve()
    candidate = (base / Path(*pure.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append(f"{label}: 路径逃逸允许目录")
        return None
    return candidate


def png_info(path: Path) -> PngInfo:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("不是 PNG")
    offset = len(PNG_SIGNATURE)
    chunks: list[str] = []
    width = height = bit_depth = color_type = 0
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("PNG chunk 被截断")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind_bytes = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("PNG chunk 长度越界")
        payload = data[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(kind_bytes)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise ValueError("PNG chunk CRC 无效")
        try:
            kind = kind_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("PNG chunk 名称无效") from exc
        chunks.append(kind)
        if kind == "IHDR":
            if length != 13:
                raise ValueError("PNG IHDR 长度无效")
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
        if kind == "IEND":
            saw_iend = True
            if end != len(data):
                raise ValueError("PNG IEND 后仍有数据")
            break
        offset = end
    if not chunks or chunks[0] != "IHDR" or not saw_iend:
        raise ValueError("PNG 缺少 IHDR 或 IEND")
    if "IDAT" not in chunks:
        raise ValueError("PNG 缺少 IDAT")
    return PngInfo(width, height, bit_depth, color_type, tuple(chunks))


def version_is_bound(
    kind: str,
    value: str,
    artifact_url: str,
    label: str,
    errors: list[str],
) -> None:
    decoded = unquote(artifact_url)
    if kind == "git_commit":
        if GIT_REVISION.fullmatch(value) is None:
            errors.append(f"{label}: git_commit 必须是完整 40 位小写 SHA")
        elif value not in decoded:
            errors.append(f"{label}: artifact URL 未绑定声明的 git commit")
    elif kind == "arxiv_version":
        if ARXIV_VERSION.fullmatch(value) is None:
            errors.append(f"{label}: arxiv_version 必须显式包含 vN")
        elif value not in decoded:
            errors.append(f"{label}: artifact URL 未绑定声明的 arXiv version")
    elif kind == "release_tag":
        if not value or value not in decoded:
            errors.append(f"{label}: artifact URL 未绑定声明的 release tag")
    elif kind == "sha256_only":
        return
    else:
        errors.append(f"{label}: 不支持的 version kind：{kind!r}")


def credit_variants(credit: str) -> set[str]:
    variants = {credit.strip()}
    subject = re.sub(
        r"^\s*(?:Copyright\s*)?(?:\(c\)|©)?\s*",
        "",
        credit,
        flags=re.IGNORECASE,
    ).strip()
    if subject:
        variants.update(
            {
                subject,
                f"© {subject}",
                f"Copyright (c) {subject}",
                f"Copyright © {subject}",
            }
        )
    return {item for item in variants if item}


def source_figure_blocks(errors: list[str]) -> dict[tuple[Path, str], FigureBlock]:
    blocks: dict[tuple[Path, str], FigureBlock] = {}
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in FIGURE_BLOCK.finditer(text):
            attrs = parse_attrs(match.group("attrs"))
            classes = attrs.get("class", "").split()
            if "paper-figure" not in classes:
                continue
            anchor = attrs.get("id", "")
            if not anchor:
                errors.append(f"{path.relative_to(ROOT)}: paper-figure 缺少稳定 id")
                continue
            key = (path.resolve(), anchor)
            if key in blocks:
                errors.append(
                    f"{path.relative_to(ROOT)}: paper-figure id {anchor} 在同页重复"
                )
                continue
            blocks[key] = FigureBlock(path, anchor, attrs, match.group("body"))
    return blocks


def validate_notice(
    source: SourceRecord,
    errors: list[str],
    *,
    require_sha: bool,
) -> None:
    label = source.manifest_path.relative_to(ROOT)
    notice = source.license_file
    if notice is None or not notice.is_file():
        errors.append(f"{label}: 缺少本地许可证或 notice 文件")
        return
    try:
        text = notice.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{notice.relative_to(ROOT)}: notice 必须是 UTF-8 文本")
        return
    if not any(variant in text for variant in credit_variants(source.credit)):
        errors.append(f"{label}: notice 未包含声明的版权或署名")
    actual_sha = digest(notice)
    if require_sha:
        if (
            source.license_sha256 is None
            or SHA256.fullmatch(source.license_sha256) is None
        ):
            errors.append(f"{label}: schema v2 必须提供 notice_sha256")
        elif actual_sha != source.license_sha256:
            errors.append(f"{notice.relative_to(ROOT)}: notice SHA-256 不匹配")


def parse_v1(
    manifest_path: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> SourceRecord:
    label = str(manifest_path.relative_to(ROOT))
    source = expect_dict(manifest.get("source"), f"{label}: source", errors)
    render = expect_dict(manifest.get("render"), f"{label}: render", errors)
    source_id = expect_string(
        manifest_path.parent.name, f"{label}: source id", errors, pattern=SLUG
    )
    title = expect_string(source.get("title"), f"{label}: source.title", errors)
    revision = expect_string(
        source.get("revision"), f"{label}: source.revision", errors
    )
    artifact_url = https_url(source.get("pdf_url"), f"{label}: source.pdf_url", errors)
    artifact_sha = expect_string(
        source.get("pdf_sha256"),
        f"{label}: source.pdf_sha256",
        errors,
        pattern=SHA256,
    )
    version_kind = "git_commit"
    if GIT_REVISION.fullmatch(revision) is None:
        errors.append(f"{label}: schema v1 source.revision 必须是完整 40 位小写 SHA")
    if artifact_url:
        version_is_bound(version_kind, revision, artifact_url, label, errors)
        if "/main/" in artifact_url:
            errors.append(f"{label}: schema v1 PDF URL 不得指向 main")
    page_box = expect_pair(
        source.get("page_size_points"),
        f"{label}: source.page_size_points",
        errors,
    )
    license_name = expect_string(
        source.get("license_name"), f"{label}: source.license_name", errors
    )
    license_url = https_url(
        source.get("license_url"), f"{label}: source.license_url", errors
    )
    license_file = safe_path(
        manifest_path.parent,
        source.get("license_file"),
        f"{label}: source.license_file",
        errors,
        allowed_root=manifest_path.parent,
        file_name_only=True,
    )
    credit = expect_string(
        source.get("copyright"), f"{label}: source.copyright", errors
    )
    dpi = expect_int(render.get("dpi"), f"{label}: render.dpi", errors, minimum=144)
    if dpi > 600:
        errors.append(f"{label}: render.dpi 不得超过 600")
    colorspace = expect_string(
        render.get("colorspace"), f"{label}: render.colorspace", errors
    )
    image_format = expect_string(
        render.get("format"), f"{label}: render.format", errors
    )
    if colorspace.casefold() != "srgb":
        errors.append(f"{label}: schema v1 仅支持 sRGB")
    if image_format.casefold() != "png":
        errors.append(f"{label}: schema v1 仅支持 PNG")
    record = SourceRecord(
        schema_version=1,
        manifest_path=manifest_path,
        source_id=source_id,
        title=title,
        artifact_kind="pdf",
        artifact_url=artifact_url,
        artifact_sha256=artifact_sha,
        version_kind=version_kind,
        version_value=revision,
        page_count=None,
        license_name=license_name,
        license_url=license_url,
        license_file=license_file,
        license_sha256=None,
        credit=credit,
        dpi=dpi or 300,
        coordinate_space="rendered-page-points-top-left",
        colorspace=colorspace,
        image_format=image_format,
        metadata_policy="strip",
    )
    validate_notice(record, errors, require_sha=False)
    figures = expect_list(manifest.get("figures"), f"{label}: figures", errors)
    if not figures:
        errors.append(f"{label}: figures 不得为空")
    for index, raw in enumerate(figures):
        item_label = f"{label}: figures[{index}]"
        item = expect_dict(raw, item_label, errors)
        asset_id = expect_string(
            item.get("id"), f"{item_label}.id", errors, pattern=SLUG
        )
        file_name = expect_string(
            item.get("file"), f"{item_label}.file", errors, pattern=PNG_NAME
        )
        file_path = safe_path(
            manifest_path.parent,
            file_name,
            f"{item_label}.file",
            errors,
            allowed_root=manifest_path.parent,
            file_name_only=True,
        )
        figure_number = expect_int(
            item.get("figure"), f"{item_label}.figure", errors, minimum=1
        )
        page = expect_int(item.get("page"), f"{item_label}.page", errors, minimum=1)
        crop = expect_box(item.get("crop_points"), f"{item_label}.crop_points", errors)
        width = expect_int(
            item.get("pixel_width"),
            f"{item_label}.pixel_width",
            errors,
            minimum=1,
        )
        height = expect_int(
            item.get("pixel_height"),
            f"{item_label}.pixel_height",
            errors,
            minimum=1,
        )
        asset_sha = expect_string(
            item.get("sha256"), f"{item_label}.sha256", errors, pattern=SHA256
        )
        page_path = safe_path(
            ROOT,
            item.get("used_by"),
            f"{item_label}.used_by",
            errors,
            allowed_root=DOCS_ROOT,
        )
        page_relative = (
            page_path.relative_to(ROOT).as_posix() if page_path is not None else ""
        )
        placement = Placement(
            page_relative=page_relative,
            page_path=page_path or DOCS_ROOT / "__invalid__.md",
            anchor=asset_id,
            role="normal",
            media_link="asset",
        )
        record.assets.append(
            AssetRecord(
                source=record,
                asset_id=asset_id,
                file_name=file_name,
                file_path=file_path,
                kind="figure",
                source_label=f"Figure {figure_number}",
                page=page,
                page_box=page_box,
                rotation=0,
                crop_box=crop,
                pixel_width=width,
                pixel_height=height,
                sha256=asset_sha,
                placements=[placement],
            )
        )
    return record


def parse_v2(
    manifest_path: Path,
    manifest: dict[str, Any],
    errors: list[str],
) -> SourceRecord:
    label = str(manifest_path.relative_to(ROOT))
    expect_keys(
        manifest,
        label,
        errors,
        required={"schema_version", "source", "render", "assets"},
    )
    source = expect_dict(manifest.get("source"), f"{label}: source", errors)
    render = expect_dict(manifest.get("render"), f"{label}: render", errors)
    version = expect_dict(source.get("version"), f"{label}: source.version", errors)
    license_data = expect_dict(
        source.get("license"), f"{label}: source.license", errors
    )
    expect_keys(
        source,
        f"{label}: source",
        errors,
        required={
            "id",
            "title",
            "artifact_url",
            "artifact_sha256",
            "version",
            "page_count",
            "license",
        },
        optional={"canonical_url", "artifact_kind"},
    )
    expect_keys(
        version,
        f"{label}: source.version",
        errors,
        required={"kind", "value"},
    )
    expect_keys(
        license_data,
        f"{label}: source.license",
        errors,
        required={"name", "url", "credit", "notice_file", "notice_sha256"},
    )
    expect_keys(
        render,
        f"{label}: render",
        errors,
        required={
            "renderer",
            "dpi",
            "coordinate_space",
            "colorspace",
            "format",
            "metadata_policy",
        },
        optional={"renderer_version"},
    )
    source_id = expect_string(
        source.get("id"), f"{label}: source.id", errors, pattern=SLUG
    )
    if source_id and source_id != manifest_path.parent.name:
        errors.append(f"{label}: source.id 必须与 manifest 所在目录名一致")
    title = expect_string(source.get("title"), f"{label}: source.title", errors)
    artifact_kind = expect_string(
        source.get("artifact_kind", "pdf"),
        f"{label}: source.artifact_kind",
        errors,
    )
    if artifact_kind not in ALLOWED_ARTIFACT_KINDS:
        errors.append(f"{label}: source.artifact_kind 不受支持")
    canonical_url = source.get("canonical_url")
    if canonical_url is not None:
        https_url(canonical_url, f"{label}: source.canonical_url", errors)
    artifact_url = https_url(
        source.get("artifact_url"), f"{label}: source.artifact_url", errors
    )
    artifact_sha = expect_string(
        source.get("artifact_sha256"),
        f"{label}: source.artifact_sha256",
        errors,
        pattern=SHA256,
    )
    version_kind = expect_string(
        version.get("kind"), f"{label}: source.version.kind", errors
    )
    version_value = expect_string(
        version.get("value"), f"{label}: source.version.value", errors
    )
    if version_kind not in ALLOWED_VERSION_KINDS:
        errors.append(f"{label}: source.version.kind 不受支持")
    elif artifact_url:
        version_is_bound(version_kind, version_value, artifact_url, label, errors)
    page_count = expect_int(
        source.get("page_count"),
        f"{label}: source.page_count",
        errors,
        minimum=1,
    )
    license_name = expect_string(
        license_data.get("name"), f"{label}: source.license.name", errors
    )
    license_url = https_url(
        license_data.get("url"), f"{label}: source.license.url", errors
    )
    credit = expect_string(
        license_data.get("credit"), f"{label}: source.license.credit", errors
    )
    license_file = safe_path(
        manifest_path.parent,
        license_data.get("notice_file"),
        f"{label}: source.license.notice_file",
        errors,
        allowed_root=manifest_path.parent,
        file_name_only=True,
    )
    license_sha = expect_string(
        license_data.get("notice_sha256"),
        f"{label}: source.license.notice_sha256",
        errors,
        pattern=SHA256,
    )
    minimum_dpi = 72 if artifact_kind == "standalone_raster" else 144
    dpi = expect_int(
        render.get("dpi"),
        f"{label}: render.dpi",
        errors,
        minimum=minimum_dpi,
    )
    if dpi > 600:
        errors.append(f"{label}: render.dpi 不得超过 600")
    coordinate_space = expect_string(
        render.get("coordinate_space"),
        f"{label}: render.coordinate_space",
        errors,
    )
    if coordinate_space not in ALLOWED_COORDINATE_SPACES:
        errors.append(f"{label}: render.coordinate_space 不受支持")
    colorspace = expect_string(
        render.get("colorspace"), f"{label}: render.colorspace", errors
    )
    image_format = expect_string(
        render.get("format"), f"{label}: render.format", errors
    )
    metadata_policy = expect_string(
        render.get("metadata_policy"),
        f"{label}: render.metadata_policy",
        errors,
    )
    renderer = expect_string(
        render.get("renderer"), f"{label}: render.renderer", errors
    )
    if artifact_kind == "standalone_raster":
        if page_count != 1:
            errors.append(f"{label}: standalone_raster 的 page_count 必须为 1")
        if renderer != "imagemagick":
            errors.append(f"{label}: standalone_raster 要求 renderer=imagemagick")
        if coordinate_space != "source-pixels-top-left":
            errors.append(
                f"{label}: standalone_raster 要求 "
                "coordinate_space=source-pixels-top-left"
            )
        if dpi != 72:
            errors.append(f"{label}: standalone_raster 要求 dpi=72")
    else:
        if renderer != "pdftocairo":
            errors.append(f"{label}: PDF 来源要求 renderer=pdftocairo")
        if coordinate_space == "source-pixels-top-left":
            errors.append(f"{label}: PDF 来源不得使用 source-pixels-top-left")
    if colorspace.casefold() != "srgb":
        errors.append(f"{label}: schema v2 当前仅支持 sRGB")
    if image_format.casefold() != "png":
        errors.append(f"{label}: schema v2 当前仅支持 PNG")
    if metadata_policy != "strip":
        errors.append(f"{label}: schema v2 要求 metadata_policy=strip")
    record = SourceRecord(
        schema_version=2,
        manifest_path=manifest_path,
        source_id=source_id,
        title=title,
        artifact_kind=artifact_kind,
        artifact_url=artifact_url,
        artifact_sha256=artifact_sha,
        version_kind=version_kind,
        version_value=version_value,
        page_count=page_count or None,
        license_name=license_name,
        license_url=license_url,
        license_file=license_file,
        license_sha256=license_sha,
        credit=credit,
        dpi=dpi or 300,
        coordinate_space=coordinate_space,
        colorspace=colorspace,
        image_format=image_format,
        metadata_policy=metadata_policy,
    )
    validate_notice(record, errors, require_sha=True)
    assets = expect_list(manifest.get("assets"), f"{label}: assets", errors)
    if not assets:
        errors.append(f"{label}: assets 不得为空")
    for index, raw in enumerate(assets):
        item_label = f"{label}: assets[{index}]"
        item = expect_dict(raw, item_label, errors)
        expect_keys(
            item,
            item_label,
            errors,
            required={
                "id",
                "file",
                "kind",
                "source_label",
                "page",
                "rotation",
                "pixel_size",
                "sha256",
                "placements",
            }
            | (
                {"source_size_pixels", "source_crop_box_pixels"}
                if artifact_kind == "standalone_raster"
                else {"page_box_points", "crop_box_points"}
            ),
        )
        asset_id = expect_string(
            item.get("id"), f"{item_label}.id", errors, pattern=SLUG
        )
        file_name = expect_string(
            item.get("file"), f"{item_label}.file", errors, pattern=PNG_NAME
        )
        file_path = safe_path(
            manifest_path.parent,
            file_name,
            f"{item_label}.file",
            errors,
            allowed_root=manifest_path.parent,
            file_name_only=True,
        )
        kind = expect_string(item.get("kind"), f"{item_label}.kind", errors)
        if kind not in ALLOWED_KINDS:
            errors.append(f"{item_label}.kind: 不受支持")
        source_label = expect_string(
            item.get("source_label"), f"{item_label}.source_label", errors
        )
        page = expect_int(item.get("page"), f"{item_label}.page", errors, minimum=1)
        if record.page_count is not None and page > record.page_count:
            errors.append(f"{item_label}.page: 超过 source.page_count")
        if artifact_kind == "standalone_raster":
            page_size = expect_int_pair(
                item.get("source_size_pixels"),
                f"{item_label}.source_size_pixels",
                errors,
            )
            page_box = (float(page_size[0]), float(page_size[1]))
        else:
            page_box = expect_pair(
                item.get("page_box_points"),
                f"{item_label}.page_box_points",
                errors,
            )
        rotation = expect_int(item.get("rotation", 0), f"{item_label}.rotation", errors)
        if rotation not in {0, 90, 180, 270}:
            errors.append(f"{item_label}.rotation: 必须是 0/90/180/270")
        if artifact_kind == "standalone_raster":
            crop = expect_int_box(
                item.get("source_crop_box_pixels"),
                f"{item_label}.source_crop_box_pixels",
                errors,
            )
        else:
            crop = expect_box(
                item.get("crop_box_points"),
                f"{item_label}.crop_box_points",
                errors,
            )
        pixel_size = expect_int_pair(
            item.get("pixel_size"), f"{item_label}.pixel_size", errors
        )
        width, height = pixel_size
        asset_sha = expect_string(
            item.get("sha256"), f"{item_label}.sha256", errors, pattern=SHA256
        )
        placements: list[Placement] = []
        placement_values = expect_list(
            item.get("placements"), f"{item_label}.placements", errors
        )
        if not placement_values:
            errors.append(f"{item_label}.placements: 不得为空")
        for placement_index, placement_raw in enumerate(placement_values):
            placement_label = f"{item_label}.placements[{placement_index}]"
            placement_data = expect_dict(placement_raw, placement_label, errors)
            expect_keys(
                placement_data,
                placement_label,
                errors,
                required={"page", "anchor"},
                optional={"role", "media_link"},
            )
            page_path = safe_path(
                ROOT,
                placement_data.get("page"),
                f"{placement_label}.page",
                errors,
                allowed_root=DOCS_ROOT,
            )
            page_relative = (
                page_path.relative_to(ROOT).as_posix() if page_path is not None else ""
            )
            anchor = expect_string(
                placement_data.get("anchor"),
                f"{placement_label}.anchor",
                errors,
                pattern=SLUG,
            )
            role = expect_string(
                placement_data.get("role", "normal"),
                f"{placement_label}.role",
                errors,
            )
            if role not in ALLOWED_ROLES:
                errors.append(f"{placement_label}.role: 不受支持")
            media_link = expect_string(
                placement_data.get("media_link", "asset"),
                f"{placement_label}.media_link",
                errors,
            )
            if media_link not in ALLOWED_MEDIA_LINKS:
                errors.append(f"{placement_label}.media_link: 不受支持")
            placements.append(
                Placement(
                    page_relative=page_relative,
                    page_path=page_path or DOCS_ROOT / "__invalid__.md",
                    anchor=anchor,
                    role=role,
                    media_link=media_link,
                )
            )
        record.assets.append(
            AssetRecord(
                source=record,
                asset_id=asset_id,
                file_name=file_name,
                file_path=file_path,
                kind=kind,
                source_label=source_label,
                page=page,
                page_box=page_box,
                rotation=rotation,
                crop_box=crop,
                pixel_width=width,
                pixel_height=height,
                sha256=asset_sha,
                placements=placements,
            )
        )
    return record


def parse_manifest(path: Path, errors: list[str]) -> Optional[SourceRecord]:
    label = str(path.relative_to(ROOT))
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=duplicate_safe_object,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        DuplicateKeyError,
    ) as exc:
        errors.append(f"{label}: manifest 无法读取：{exc}")
        return None
    if not isinstance(manifest, dict):
        errors.append(f"{label}: manifest 顶层必须是 JSON object")
        return None
    version = manifest.get("schema_version")
    if version == 1:
        return parse_v1(path, manifest, errors)
    if version == 2:
        return parse_v2(path, manifest, errors)
    errors.append(
        f"{label}: schema_version 必须为 1 或 2；"
        "新来源请使用 schemas/paper-figure-manifest-v2.schema.json"
    )
    return None


def validate_asset(asset: AssetRecord, errors: list[str]) -> None:
    source = asset.source
    label = str(source.manifest_path.relative_to(ROOT))
    item_label = f"{label}: {source.source_id}/{asset.asset_id}"
    if asset.file_path is None or not asset.file_path.is_file():
        errors.append(f"{item_label}: 图片不存在：{asset.file_name}")
        return
    try:
        info = png_info(asset.file_path)
    except (OSError, ValueError) as exc:
        errors.append(f"{asset.file_path.relative_to(ROOT)}: {exc}")
        return
    if (info.width, info.height) != (asset.pixel_width, asset.pixel_height):
        errors.append(
            f"{asset.file_path.relative_to(ROOT)}: 像素尺寸 "
            f"{info.width}x{info.height} 与 manifest 不符"
        )
    if info.bit_depth != 8 or info.color_type != 2:
        errors.append(
            f"{asset.file_path.relative_to(ROOT)}: 论文裁图必须是 8-bit opaque RGB PNG"
        )
    forbidden = sorted(set(info.chunks) & FORBIDDEN_METADATA_CHUNKS)
    unknown = sorted(set(info.chunks) - ALLOWED_PNG_CHUNKS)
    if forbidden or unknown:
        details = ", ".join(forbidden + unknown)
        errors.append(
            f"{asset.file_path.relative_to(ROOT)}: "
            f"PNG 含未允许的 metadata/chunk：{details}"
        )
    if max(info.width, info.height) < 900 or min(info.width, info.height) < 150:
        errors.append(f"{asset.file_path.relative_to(ROOT)}: 裁图分辨率过低")
    if asset.file_path.stat().st_size > MAX_ASSET_BYTES:
        errors.append(f"{asset.file_path.relative_to(ROOT)}: 图片超过 3 MiB")
    if digest(asset.file_path) != asset.sha256:
        errors.append(f"{asset.file_path.relative_to(ROOT)}: SHA-256 与 manifest 不符")
    x0, y0, x1, y1 = asset.crop_box
    page_width, page_height = asset.page_box
    if not (0 <= x0 < x1 <= page_width and 0 <= y0 < y1 <= page_height):
        errors.append(f"{item_label}: crop box 超出页面或为空")
        return
    ratio = (x1 - x0) * (y1 - y0) / (page_width * page_height)
    if source.artifact_kind != "standalone_raster":
        maximum = (
            V1_MAX_PAGE_AREA_RATIO
            if source.schema_version == 1
            else V2_MAX_PAGE_AREA_RATIO
        )
        if ratio >= maximum:
            errors.append(f"{item_label}: 裁取面积接近整页 ({ratio:.1%})")
    expected_width = round((x1 - x0) * source.dpi / 72)
    expected_height = round((y1 - y0) * source.dpi / 72)
    if abs(expected_width - info.width) > 1 or abs(expected_height - info.height) > 1:
        errors.append(
            f"{asset.file_path.relative_to(ROOT)}: 像素尺寸与 crop/DPI 不一致"
        )


def source_link_for(asset: AssetRecord) -> str:
    if asset.source.artifact_kind == "standalone_raster":
        return asset.source.artifact_url
    return f"{asset.source.artifact_url}#page={asset.page}"


def validate_source_block(
    asset: AssetRecord,
    placement: Placement,
    block: FigureBlock,
    errors: list[str],
) -> None:
    relative = block.path.relative_to(ROOT)
    source = asset.source
    if source.schema_version == 2:
        if block.attrs.get("data-paper-source") != source.source_id:
            errors.append(
                f"{relative}: #{placement.anchor} 缺少 "
                f'data-paper-source="{source.source_id}"'
            )
        if block.attrs.get("data-paper-asset") != asset.asset_id:
            errors.append(
                f"{relative}: #{placement.anchor} 缺少 "
                f'data-paper-asset="{asset.asset_id}"'
            )
    expected_role = (
        None if placement.role == "normal" else f"paper-figure--{placement.role}"
    )
    if expected_role and expected_role not in block.attrs.get("class", "").split():
        errors.append(f"{relative}: #{placement.anchor} 缺少 {expected_role}")
    images = [
        (
            match.group("alt").strip(),
            match.group("target"),
            parse_markdown_attrs(match.group("attrs")),
        )
        for match in MARKDOWN_IMAGE.finditer(block.body)
        if match.group("target").split("#", 1)[0].endswith(asset.file_name)
    ]
    if len(images) != 1:
        errors.append(f"{relative}: #{placement.anchor} 必须恰有一张登记图片")
        return
    alt, _, image_attrs = images[0]
    if not (20 <= len(alt) <= 240) or GENERIC_ALT.fullmatch(alt):
        errors.append(f"{relative}: #{placement.anchor} alt 必须具体且不过度冗长")
    expected_attrs = {
        "width": str(asset.pixel_width),
        "height": str(asset.pixel_height),
        "loading": "lazy",
        "decoding": "async",
    }
    for name, value in expected_attrs.items():
        if image_attrs.get(name) != value:
            errors.append(
                f"{relative}: #{placement.anchor} 缺少图片属性 {name}={value}"
            )
    source_link = source_link_for(asset)
    required_media = asset.file_name if placement.media_link == "asset" else source_link
    if block.body.count(required_media) < 2:
        errors.append(
            f"{relative}: #{placement.anchor} 媒体链接与 manifest.media_link 不一致"
        )
    caption = re.search(
        r"<figcaption(?:\s+[^>]*)?>(.*?)</figcaption>",
        block.body,
        re.DOTALL,
    )
    if not caption:
        errors.append(f"{relative}: #{placement.anchor} 缺少 figcaption")
        return
    caption_html = caption.group(1)
    caption_text = re.sub(r"<[^>]+>", "", caption_html)
    caption_text = re.sub(r"\s+", " ", unescape(caption_text)).strip()
    if not (40 <= len(caption_text) <= 1200):
        errors.append(f"{relative}: #{placement.anchor} 图注长度不合适")
    if re.search(r"\$|\\[\[(]", caption_html):
        errors.append(f"{relative}: #{placement.anchor} 图注含未解析数学分隔符")
    if source_link not in caption_html:
        errors.append(f"{relative}: #{placement.anchor} 缺少固定来源链接")
    if source.license_url not in caption_html:
        errors.append(f"{relative}: #{placement.anchor} 缺少许可证链接")
    if not any(variant in caption_text for variant in credit_variants(source.credit)):
        errors.append(f"{relative}: #{placement.anchor} 缺少版权或署名")
    if asset.source_label not in caption_text:
        errors.append(
            f"{relative}: #{placement.anchor} 图注未标明 {asset.source_label}"
        )


class PaperFigureHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.figures: list[HtmlFigure] = []
        self.current: Optional[HtmlFigure] = None
        self.figure_depth = 0
        self.in_caption = False
        self.anchor_stack: list[tuple[str, bool]] = []

    def handle_starttag(
        self,
        tag: str,
        attrs_list: list[tuple[str, Optional[str]]],
    ) -> None:
        attrs = {key: value or "" for key, value in attrs_list}
        if tag == "figure" and "paper-figure" in attrs.get("class", "").split():
            if self.current is not None:
                return
            self.current = HtmlFigure(attrs=attrs)
            self.figure_depth = 1
            return
        if self.current is None:
            return
        if tag not in VOID_HTML_TAGS:
            self.figure_depth += 1
        if tag == "figcaption":
            self.in_caption = True
        if tag == "a":
            href = attrs.get("href", "")
            self.anchor_stack.append((href, self.in_caption))
            if self.in_caption:
                self.current.caption_links.append(href)
        elif tag == "img":
            self.current.images.append(attrs)
            if self.anchor_stack:
                self.current.media_hrefs.append(self.anchor_stack[-1][0])

    def handle_endtag(self, tag: str) -> None:
        if self.current is None:
            return
        if tag in VOID_HTML_TAGS:
            return
        if tag == "a" and self.anchor_stack:
            self.anchor_stack.pop()
        if tag == "figcaption":
            self.in_caption = False
        self.figure_depth -= 1
        if tag == "figure" and self.figure_depth == 0:
            self.figures.append(self.current)
            self.current = None

    def handle_data(self, data: str) -> None:
        if self.current is not None and self.in_caption:
            self.current.caption_parts.append(data)


def site_path_for_doc(site_dir: Path, page_relative: str) -> Path:
    relative = PurePosixPath(page_relative)
    parts = relative.parts
    if not parts or parts[0] != "docs":
        return site_dir / "__invalid__" / "index.html"
    markdown_path = PurePosixPath(*parts[1:])
    if markdown_path.name == "index.md":
        parent = Path(*markdown_path.parent.parts)
        return site_dir / parent / "index.html"
    return site_dir / Path(*markdown_path.with_suffix("").parts) / "index.html"


def validate_generated_site(
    site_dir: Path,
    assets: list[AssetRecord],
    errors: list[str],
) -> None:
    cache: dict[Path, list[HtmlFigure]] = {}
    for asset in assets:
        source_link = source_link_for(asset)
        for placement in asset.placements:
            html_path = site_path_for_doc(site_dir, placement.page_relative)
            if not html_path.is_file():
                errors.append(f"{placement.page_relative}: 生成站点缺少对应 index.html")
                continue
            if html_path not in cache:
                figures: list[HtmlFigure] = []
                html = html_path.read_text(encoding="utf-8")
                for match in FIGURE_BLOCK.finditer(html):
                    parser = PaperFigureHtmlParser()
                    parser.feed(match.group(0))
                    figures.extend(parser.figures)
                cache[html_path] = figures
            matches = [
                figure
                for figure in cache[html_path]
                if figure.attrs.get("id") == placement.anchor
            ]
            route = html_path.relative_to(site_dir)
            if len(matches) != 1:
                errors.append(
                    f"{route}: #{placement.anchor} 必须恰有一个生成后的 paper-figure"
                )
                continue
            figure = matches[0]
            if asset.source.schema_version == 2:
                if figure.attrs.get("data-paper-source") != asset.source.source_id:
                    errors.append(f"{route}: #{placement.anchor} source data attr 丢失")
                if figure.attrs.get("data-paper-asset") != asset.asset_id:
                    errors.append(f"{route}: #{placement.anchor} asset data attr 丢失")
            if len(figure.images) != 1 or len(figure.media_hrefs) != 1:
                errors.append(f"{route}: #{placement.anchor} 生成后媒体结构无效")
                continue
            image = figure.images[0]
            source_url = urlsplit(source_link)
            media_href = figure.media_hrefs[0]
            media_url = urlsplit(media_href)
            if placement.media_link == "asset":
                if not (
                    image.get("src", "").split("?", 1)[0].endswith(asset.file_name)
                    and media_url.path.endswith(asset.file_name)
                ):
                    errors.append(
                        f"{route}: #{placement.anchor} 生成后媒体未链接登记图片"
                    )
            elif (
                media_url.scheme != source_url.scheme
                or media_url.netloc != source_url.netloc
                or media_url.path != source_url.path
                or media_url.fragment != source_url.fragment
            ):
                errors.append(
                    f"{route}: #{placement.anchor} 生成后媒体未链接固定来源页"
                )
            for name, value in {
                "width": str(asset.pixel_width),
                "height": str(asset.pixel_height),
                "loading": "lazy",
                "decoding": "async",
            }.items():
                if image.get(name) != value:
                    errors.append(
                        f"{route}: #{placement.anchor} 生成后缺少 {name}={value}"
                    )
            if not (20 <= len(image.get("alt", "").strip()) <= 240):
                errors.append(f"{route}: #{placement.anchor} 生成后 alt 无效")
            if source_link not in figure.caption_links:
                errors.append(f"{route}: #{placement.anchor} 生成后缺少固定来源链接")
            if asset.source.license_url not in figure.caption_links:
                errors.append(f"{route}: #{placement.anchor} 生成后缺少许可链接")
            if not figure.caption_text:
                errors.append(f"{route}: #{placement.anchor} 生成后图注不可见")


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site-dir",
        type=Path,
        help="also validate semantic figures in an existing MkDocs output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    errors: list[str] = []
    blocks = source_figure_blocks(errors)
    manifests = sorted(PAPERS_ROOT.rglob("manifest.json"))
    if not manifests:
        errors.append("docs/assets/papers/: 缺少论文图表 manifest")
    sources: list[SourceRecord] = []
    for manifest_path in manifests:
        record = parse_manifest(manifest_path, errors)
        if record is not None:
            sources.append(record)
    source_ids: set[str] = set()
    asset_keys: set[tuple[str, str]] = set()
    placement_keys: set[tuple[Path, str]] = set()
    registered_assets: set[Path] = set()
    used_blocks: set[tuple[Path, str]] = set()
    assets: list[AssetRecord] = []
    for source in sources:
        if source.source_id in source_ids:
            errors.append(f"paper source id 重复：{source.source_id}")
        source_ids.add(source.source_id)
        for asset in source.assets:
            assets.append(asset)
            key = (source.source_id, asset.asset_id)
            if key in asset_keys:
                errors.append(
                    f"paper asset id 重复：{source.source_id}/{asset.asset_id}"
                )
            asset_keys.add(key)
            if asset.file_path is not None:
                registered_assets.add(asset.file_path.resolve())
            validate_asset(asset, errors)
            for placement in asset.placements:
                placement_key = (placement.page_path.resolve(), placement.anchor)
                if placement_key in placement_keys:
                    errors.append(
                        f"{placement.page_relative}: paper-figure id "
                        f"{placement.anchor} 在同页重复登记"
                    )
                placement_keys.add(placement_key)
                if (
                    not placement.page_path.is_file()
                    or placement.page_path.suffix != ".md"
                ):
                    errors.append(
                        f"{source.manifest_path.relative_to(ROOT)}: "
                        f"{source.source_id}/{asset.asset_id} placement 页面无效"
                    )
                    continue
                block = blocks.get(placement_key)
                if block is None:
                    errors.append(
                        f"{source.manifest_path.relative_to(ROOT)}: "
                        f"{source.source_id}/{asset.asset_id} 未嵌入 "
                        f"{placement.page_relative}#{placement.anchor}"
                    )
                    continue
                used_blocks.add(placement_key)
                validate_source_block(asset, placement, block, errors)
    unregistered = sorted(
        path.relative_to(ROOT)
        for path in PAPERS_ROOT.rglob("*.png")
        if path.resolve() not in registered_assets
    )
    if unregistered:
        errors.append(
            "docs/assets/papers/: 存在未登记图片：" + "、".join(map(str, unregistered))
        )
    extra_blocks = sorted(
        (
            f"{path.relative_to(ROOT)}#{anchor}"
            for path, anchor in set(blocks) - used_blocks
        )
    )
    if extra_blocks:
        errors.append("docs/: 存在未登记 paper-figure：" + "、".join(extra_blocks))
    if args.site_dir:
        site_dir = args.site_dir
        if not site_dir.is_absolute():
            site_dir = ROOT / site_dir
        if not site_dir.is_dir():
            errors.append(f"生成站点目录不存在：{site_dir}")
        else:
            validate_generated_site(site_dir, assets, errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    v1_count = sum(source.schema_version == 1 for source in sources)
    v2_count = sum(source.schema_version == 2 for source in sources)
    suffix = f"，生成站点 {args.site_dir}" if args.site_dir else ""
    print(
        f"论文图表检查通过：{len(sources)} 份 manifest "
        f"(v1={v1_count}, v2={v2_count})，{len(assets)} 幅裁图{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
