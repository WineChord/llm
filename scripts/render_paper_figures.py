#!/usr/bin/env python3
"""Render registered paper crops from an explicitly supplied local PDF.

The command never downloads a source document. It verifies the local PDF
against the manifest before invoking Poppler, strips ancillary PNG metadata,
and either checks the committed assets or writes deterministic crop outputs.
"""

from __future__ import annotations

import argparse
import hashlib
from html import unescape
import json
import os
from pathlib import Path, PurePath
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Optional, Sequence
import zlib


ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\.png")
PDF_WORD = re.compile(
    r'<word\s+xMin="(?P<x0>[0-9.]+)"\s+yMin="(?P<y0>[0-9.]+)"\s+'
    r'xMax="(?P<x1>[0-9.]+)"\s+yMax="(?P<y1>[0-9.]+)">'
    r"(?P<text>.*?)</word>",
    re.DOTALL,
)
KEPT_PNG_CHUNKS = {"IHDR", "PLTE", "IDAT", "IEND", "sRGB", "gAMA", "cHRM"}


class RenderError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RenderError(f"manifest 包含重复 key：{key}")
        result[key] = value
    return result


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RenderError(f"{label} 必须是 object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RenderError(f"{label} 必须是 array")
    return value


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderError(f"{label} 必须是非空字符串")
    return value.strip()


def require_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RenderError(f"{label} 必须是数值")
    return float(value)


def safe_manifest_path(value: Any, label: str) -> Path:
    raw = require_string(value, label)
    path = PurePath(raw)
    if path.is_absolute() or len(path.parts) != 1 or raw in {".", ".."}:
        raise RenderError(f"{label} 必须是单个安全文件名")
    if SAFE_NAME.fullmatch(raw) is None:
        raise RenderError(f"{label} 必须是小写 kebab-case PNG 文件名")
    return Path(raw)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_without_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"无法读取 manifest：{exc}") from exc
    return require_dict(value, "manifest")


def command_path(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RenderError(f"缺少必需命令：{name}")
    return path


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RenderError(
            f"{Path(command[0]).name} 退出码 {result.returncode}：{detail}"
        )
    return result


def pdf_metadata(pdf: Path, page: int) -> tuple[int, float, float, int]:
    result = run_command(
        [
            command_path("pdfinfo"),
            "-f",
            str(page),
            "-l",
            str(page),
            str(pdf),
        ]
    )
    pages_match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
    size_match = re.search(
        rf"^Page\s+{page}\s+size:\s+"
        r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
        result.stdout,
        re.MULTILINE,
    )
    rotation_match = re.search(
        rf"^Page\s+{page}\s+rot:\s+(\d+)\s*$",
        result.stdout,
        re.MULTILINE,
    )
    if pages_match is None or size_match is None:
        raise RenderError(f"pdfinfo 未返回第 {page} 页的尺寸信息")
    rotation = int(rotation_match.group(1)) if rotation_match else 0
    return (
        int(pages_match.group(1)),
        float(size_match.group(1)),
        float(size_match.group(2)),
        rotation,
    )


def validate_crop_does_not_split_pdf_text(
    pdf: Path,
    page: int,
    crop: tuple[float, float, float, float],
    page_box: tuple[float, float],
    coordinate_space: str,
) -> None:
    x0, y0, x1, y1 = crop
    if coordinate_space == "pdf-points-bottom-left":
        y0, y1 = page_box[1] - y1, page_box[1] - y0
    result = run_command(
        [
            command_path("pdftotext"),
            "-f",
            str(page),
            "-l",
            str(page),
            "-bbox",
            str(pdf),
            "-",
        ]
    )
    tolerance = 0.25
    split_words: list[str] = []
    for match in PDF_WORD.finditer(result.stdout):
        word_box = tuple(
            float(match.group(name)) for name in ("x0", "y0", "x1", "y1")
        )
        wx0, wy0, wx1, wy1 = word_box
        overlaps = (
            wx1 > x0 + tolerance
            and wx0 < x1 - tolerance
            and wy1 > y0 + tolerance
            and wy0 < y1 - tolerance
        )
        contained = (
            wx0 >= x0 - tolerance
            and wx1 <= x1 + tolerance
            and wy0 >= y0 - tolerance
            and wy1 <= y1 + tolerance
        )
        if overlaps and not contained:
            text = unescape(re.sub(r"<[^>]+>", "", match.group("text"))).strip()
            split_words.append(
                f"{text or '<empty>'} "
                f"[{wx0:.2f},{wy0:.2f},{wx1:.2f},{wy1:.2f}]"
            )
    if split_words:
        sample = "；".join(split_words[:5])
        suffix = "；…" if len(split_words) > 5 else ""
        raise RenderError(
            f"第 {page} 页 crop 会截断 PDF 文字：{sample}{suffix}"
        )


def rewrite_png_without_metadata(source: Path, destination: Path) -> None:
    data = source.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise RenderError(f"{source}: pdftocairo 未生成有效 PNG")
    output = bytearray(PNG_SIGNATURE)
    offset = len(PNG_SIGNATURE)
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise RenderError(f"{source}: PNG chunk 被截断")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            raise RenderError(f"{source}: PNG chunk 越界")
        kind_bytes = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        actual_crc = zlib.crc32(kind_bytes)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != stored_crc:
            raise RenderError(f"{source}: PNG CRC 无效")
        try:
            kind = kind_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise RenderError(f"{source}: PNG chunk 名称无效") from exc
        if kind in KEPT_PNG_CHUNKS:
            output.extend(data[offset:end])
        if kind == "IEND":
            saw_iend = True
            break
        offset = end
    if not saw_iend:
        raise RenderError(f"{source}: PNG 缺少 IEND")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(output)
    temporary.replace(destination)


def png_dimensions(path: Path) -> tuple[int, int, int, int]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE) or len(data) < 33:
        raise RenderError(f"{path}: PNG 无效")
    if data[12:16] != b"IHDR":
        raise RenderError(f"{path}: PNG 缺少首个 IHDR")
    width, height, bit_depth, color_type = struct.unpack(">IIBB", data[16:26])
    return width, height, bit_depth, color_type


def parse_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)):
        raise RenderError(f"{label} 必须是 array")
    items = value
    if len(items) != 2:
        raise RenderError(f"{label} 必须包含两个数值")
    pair = (require_number(items[0], label), require_number(items[1], label))
    if min(pair) <= 0:
        raise RenderError(f"{label} 必须为正数")
    return pair


def parse_box(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)):
        raise RenderError(f"{label} 必须是 array")
    items = value
    if len(items) != 4:
        raise RenderError(f"{label} 必须包含四个数值")
    return tuple(require_number(item, label) for item in items)  # type: ignore[return-value]


def manifest_model(
    manifest: dict[str, Any],
) -> tuple[str, int, str, Optional[int], list[dict[str, Any]]]:
    version = manifest.get("schema_version")
    source = require_dict(manifest.get("source"), "source")
    render = require_dict(manifest.get("render"), "render")
    dpi = render.get("dpi")
    if not isinstance(dpi, int) or isinstance(dpi, bool) or not 144 <= dpi <= 600:
        raise RenderError("render.dpi 必须是 144 到 600 的整数")
    if version == 1:
        sha = require_string(source.get("pdf_sha256"), "source.pdf_sha256")
        page_box = parse_pair(source.get("page_size_points"), "source.page_size_points")
        raw_assets = require_list(manifest.get("figures"), "figures")
        assets = []
        for index, raw in enumerate(raw_assets):
            item = require_dict(raw, f"figures[{index}]")
            assets.append(
                {
                    "id": require_string(item.get("id"), f"figures[{index}].id"),
                    "file": safe_manifest_path(
                        item.get("file"), f"figures[{index}].file"
                    ),
                    "page": item.get("page"),
                    "page_box": page_box,
                    "rotation": 0,
                    "crop": item.get("crop_points"),
                    "pixel_size": [
                        item.get("pixel_width"),
                        item.get("pixel_height"),
                    ],
                    "sha256": item.get("sha256"),
                }
            )
        return sha, dpi, "rendered-page-points-top-left", None, assets
    if version == 2:
        sha = require_string(source.get("artifact_sha256"), "source.artifact_sha256")
        page_count = source.get("page_count")
        if not isinstance(page_count, int) or isinstance(page_count, bool):
            raise RenderError("source.page_count 必须是整数")
        coordinate_space = require_string(
            render.get("coordinate_space"), "render.coordinate_space"
        )
        if coordinate_space not in {
            "rendered-page-points-top-left",
            "pdf-points-bottom-left",
        }:
            raise RenderError("render.coordinate_space 不受支持")
        raw_assets = require_list(manifest.get("assets"), "assets")
        assets = []
        for index, raw in enumerate(raw_assets):
            item = require_dict(raw, f"assets[{index}]")
            assets.append(
                {
                    "id": require_string(item.get("id"), f"assets[{index}].id"),
                    "file": safe_manifest_path(
                        item.get("file"), f"assets[{index}].file"
                    ),
                    "page": item.get("page"),
                    "page_box": item.get("page_box_points"),
                    "rotation": item.get("rotation", 0),
                    "crop": item.get("crop_box_points"),
                    "pixel_size": item.get("pixel_size"),
                    "sha256": item.get("sha256"),
                }
            )
        return sha, dpi, coordinate_space, page_count, assets
    raise RenderError("schema_version 必须为 1 或 2")


def validate_asset_model(
    asset: dict[str, Any],
    label: str,
) -> tuple[int, tuple[float, float], int, tuple[float, float, float, float], tuple[int, int], str]:
    page = asset["page"]
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise RenderError(f"{label}.page 必须是正整数")
    page_box = parse_pair(asset["page_box"], f"{label}.page_box")
    rotation = asset["rotation"]
    if rotation not in {0, 90, 180, 270}:
        raise RenderError(f"{label}.rotation 必须是 0/90/180/270")
    crop = parse_box(asset["crop"], f"{label}.crop")
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= page_box[0] and 0 <= y0 < y1 <= page_box[1]):
        raise RenderError(f"{label}.crop 超出页面或为空")
    size = parse_pair(asset["pixel_size"], f"{label}.pixel_size")
    pixel_size = (int(size[0]), int(size[1]))
    if size != pixel_size:
        raise RenderError(f"{label}.pixel_size 必须是整数")
    sha = require_string(asset["sha256"], f"{label}.sha256")
    if SHA256.fullmatch(sha) is None:
        raise RenderError(f"{label}.sha256 必须是 64 位小写十六进制")
    return page, page_box, rotation, crop, pixel_size, sha


def crop_pixels(
    crop: tuple[float, float, float, float],
    page_box: tuple[float, float],
    dpi: int,
    coordinate_space: str,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    if coordinate_space == "pdf-points-bottom-left":
        y0, y1 = page_box[1] - y1, page_box[1] - y0
    scale = dpi / 72
    x = round(x0 * scale)
    y = round(y0 * scale)
    width = round((x1 - x0) * scale)
    height = round((y1 - y0) * scale)
    if min(width, height) <= 0:
        raise RenderError("crop 转为像素后为空")
    return x, y, width, height


def render_crop(
    pdf: Path,
    page: int,
    crop: tuple[int, int, int, int],
    dpi: int,
    destination: Path,
    temporary_root: Path,
) -> None:
    x, y, width, height = crop
    prefix = temporary_root / f"render-{page}-{destination.stem}"
    run_command(
        [
            command_path("pdftocairo"),
            "-png",
            "-singlefile",
            "-f",
            str(page),
            "-l",
            str(page),
            "-r",
            str(dpi),
            "-x",
            str(x),
            "-y",
            str(y),
            "-W",
            str(width),
            "-H",
            str(height),
            str(pdf),
            str(prefix),
        ]
    )
    generated = prefix.with_suffix(".png")
    if not generated.is_file():
        raise RenderError(f"pdftocairo 未生成 {generated.name}")
    rewrite_png_without_metadata(generated, destination)


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render registered paper figures from a verified local PDF."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="render only this asset id; repeat for multiple assets",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output-dir",
        type=Path,
        help="write verified crops to a separate directory",
    )
    destination.add_argument(
        "--write-assets",
        action="store_true",
        help="replace exactly the registered asset files after all checks pass",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="report generated size and SHA without requiring manifest equality",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = args.manifest.expanduser().resolve()
        pdf_path = args.pdf.expanduser().resolve()
        if not manifest_path.is_file():
            raise RenderError(f"manifest 不存在：{manifest_path}")
        if not pdf_path.is_file():
            raise RenderError(f"本地 PDF 不存在：{pdf_path}")
        manifest = load_manifest(manifest_path)
        source_sha, dpi, coordinate_space, page_count, assets = manifest_model(
            manifest
        )
        if SHA256.fullmatch(source_sha) is None:
            raise RenderError("源 PDF SHA-256 格式无效")
        actual_source_sha = digest(pdf_path)
        if actual_source_sha != source_sha:
            raise RenderError(
                "本地 PDF 与 manifest 的源文件 SHA-256 不一致；"
                "拒绝从错误版本生成裁图"
            )
        selected = set(args.asset)
        known = {asset["id"] for asset in assets}
        unknown = sorted(selected - known)
        if unknown:
            raise RenderError("未知 asset id：" + "、".join(unknown))
        work = [asset for asset in assets if not selected or asset["id"] in selected]
        if not work:
            raise RenderError("没有可生成的 asset")
        if args.output_dir:
            output_root = args.output_dir.expanduser().resolve()
        elif args.write_assets:
            output_root = manifest_path.parent
        else:
            output_root = None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        reports: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="paper-figures-") as raw_tmp:
            temporary_root = Path(raw_tmp)
            for index, asset in enumerate(work):
                label = f"asset[{index}]/{asset['id']}"
                page, page_box, rotation, crop, expected_size, expected_sha = (
                    validate_asset_model(asset, label)
                )
                if page_count is not None and page > page_count:
                    raise RenderError(f"{label}.page 超过 source.page_count")
                actual_count, width, height, actual_rotation = pdf_metadata(
                    pdf_path, page
                )
                if page_count is not None and actual_count != page_count:
                    raise RenderError(
                        f"PDF 页数 {actual_count} 与 manifest {page_count} 不一致"
                    )
                if abs(width - page_box[0]) > 0.5 or abs(height - page_box[1]) > 0.5:
                    raise RenderError(
                        f"{label}: PDF 页面 {width:g}x{height:g} pt "
                        f"与 manifest {page_box[0]:g}x{page_box[1]:g} pt 不一致"
                    )
                if actual_rotation != rotation:
                    raise RenderError(
                        f"{label}: PDF 页面旋转 {actual_rotation} "
                        f"与 manifest {rotation} 不一致"
                    )
                if rotation == 0:
                    validate_crop_does_not_split_pdf_text(
                        pdf_path,
                        page,
                        crop,
                        page_box,
                        coordinate_space,
                    )
                pixel_crop = crop_pixels(crop, page_box, dpi, coordinate_space)
                generated = temporary_root / "normalized" / asset["file"]
                render_crop(
                    pdf_path,
                    page,
                    pixel_crop,
                    dpi,
                    generated,
                    temporary_root,
                )
                png_width, png_height, bit_depth, color_type = png_dimensions(
                    generated
                )
                generated_size = (png_width, png_height)
                generated_sha = digest(generated)
                if bit_depth != 8 or color_type != 2:
                    raise RenderError(
                        f"{label}: 生成结果不是 8-bit opaque RGB PNG"
                    )
                mismatches = []
                if generated_size != expected_size:
                    mismatches.append(
                        f"尺寸 {generated_size[0]}x{generated_size[1]} "
                        f"!= {expected_size[0]}x{expected_size[1]}"
                    )
                if generated_sha != expected_sha:
                    mismatches.append(
                        f"SHA-256 {generated_sha} != {expected_sha}"
                    )
                if mismatches and not args.allow_drift:
                    raise RenderError(f"{label}: " + "；".join(mismatches))
                destination = (
                    output_root / asset["file"] if output_root is not None else None
                )
                if destination is not None:
                    staged_path = temporary_root / "staged" / asset["file"]
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(generated, staged_path)
                    staged.append((staged_path, destination))
                reports.append(
                    {
                        "id": asset["id"],
                        "file": asset["file"].as_posix(),
                        "page": page,
                        "pixel_size": [png_width, png_height],
                        "sha256": generated_sha,
                        "matches_manifest": not mismatches,
                    }
                )
            for staged_path, destination in staged:
                temporary = destination.with_name(f".{destination.name}.tmp")
                shutil.copyfile(staged_path, temporary)
                temporary.replace(destination)
        print(json.dumps({"assets": reports}, ensure_ascii=False, indent=2))
        return 0
    except (RenderError, OSError) as exc:
        print(f"论文裁图生成失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
