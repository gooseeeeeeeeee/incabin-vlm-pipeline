#!/usr/bin/env python3
"""Materialize SVIRO use-case bucket JSONL files into pure-image folders.

The JSONL buckets remain the traceable index. This script creates a
benchmark-friendly image view where each use case is a folder containing only
PNG candidates. Hardlinks are used when possible to avoid duplicating data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path


BUCKET_FILES = OrderedDict(
    [
        ("01_empty_seat", "01_empty_seat.jsonl"),
        ("02_adult_occupant", "02_adult_occupant.jsonl"),
        (
            "03_child_in_forward_facing_child_seat_candidates",
            "03_child_in_forward_facing_child_seat_candidates.jsonl",
        ),
        ("04_rear_facing_child_seat_candidates", "04_rear_facing_child_seat_candidates.jsonl"),
        ("05_infant_recognition", "05_infant_recognition.jsonl"),
        (
            "06_isofix_child_seat_orientation_candidates",
            "06_isofix_child_seat_orientation_candidates.jsonl",
        ),
        ("07_pet_presence_unsupported", "07_pet_presence_unsupported.jsonl"),
        ("08_seatbelt_worn_unsupported", "08_seatbelt_worn_unsupported.jsonl"),
        ("09_seatbelt_not_worn_unsupported", "09_seatbelt_not_worn_unsupported.jsonl"),
        ("10_occupant_sleeping_unsupported", "10_occupant_sleeping_unsupported.jsonl"),
        ("11_out_of_position_candidates", "11_out_of_position_candidates.jsonl"),
        ("12_left_objects", "12_left_objects.jsonl"),
        ("13_vehicle_door_status_unsupported", "13_vehicle_door_status_unsupported.jsonl"),
    ]
)


def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
    return records


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "skipped_existing"
    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def write_readme(output_dir: Path, source_bucket_dir: Path) -> None:
    lines = [
        "# SVIRO Use Case Candidate Images / SVIRO Use Case 候选图片集",
        "",
        "This directory is a pure-image view of the current SVIRO use-case buckets.",
        "",
        "本目录是当前 SVIRO use case 分桶的纯图片视图。",
        "",
        "## Purpose / 用途",
        "",
        "- Each subfolder corresponds to one of the 13 in-cabin use cases.",
        "- Use-case subfolders contain PNG images only, so they can be used directly for VLM inspection, benchmark sampling, or manual review.",
        "- The source of truth for why an image entered a bucket remains `gt_facts/use_case_buckets/*.jsonl`.",
        "- Images are hardlinked to the extracted RGB files when possible; hardlinked files behave like normal PNG files but avoid extra disk usage.",
        "- Unsupported use cases are kept as empty folders to make current SVIRO coverage explicit.",
        "",
        "- 每个子文件夹对应 13 个车内 use case 之一。",
        "- use case 子文件夹只放 PNG 图片，方便直接用于 VLM 检查、benchmark 抽样或人工复核。",
        "- 图片为什么进入某个 bucket，以 `gt_facts/use_case_buckets/*.jsonl` 为准。",
        "- 图片优先 hardlink 到已解压 RGB；hardlink 看起来是普通 PNG，但不会重复占用一份图片空间。",
        "- unsupported use case 保留为空文件夹，用来明确当前 SVIRO 不支持这些标签。",
        "",
        "## Important Notes / 注意",
        "",
        "- This is candidate bucketing, not final confirmed labels.",
        "- One image may appear in multiple use-case folders.",
        "- Forward-facing, rear-facing, ISOFIX, and OOP folders are recall-oriented candidates for later VLM/manual filtering.",
        "",
        "- 这是候选召回，不是最终确认标签。",
        "- 同一张图片可以出现在多个 use case 文件夹中。",
        "- forward-facing、rear-facing、ISOFIX、OOP 当前都是偏召回的候选集，后续需要 VLM/人工筛选。",
        "",
        "## Source / 来源",
        "",
        f"- Source bucket directory: `{source_bucket_dir}`",
        "- Source RGB images: `0-datasets/sviro/raw/bmw_x5_random/extracted/rgb/`",
        "",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(output_dir: Path, source_bucket_dir: Path, stats: OrderedDict[str, dict]) -> None:
    lines = [
        "# SVIRO Use Case Candidate Images Report / SVIRO Use Case 候选图片报告",
        "",
        "Generated from `gt_facts/use_case_buckets/*.jsonl`.",
        "",
        "由 `gt_facts/use_case_buckets/*.jsonl` 生成。",
        "",
        "## Summary / 汇总",
        "",
        "| Folder | Images | Hardlinked | Copied | Existing skipped | Missing source | Source bucket |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for folder, stat in stats.items():
        lines.append(
            "| {folder} | {images} | {hardlinked} | {copied} | {skipped_existing} | {missing_source} | `{bucket}` |".format(
                folder=folder,
                images=stat["images"],
                hardlinked=stat["hardlinked"],
                copied=stat["copied"],
                skipped_existing=stat["skipped_existing"],
                missing_source=stat["missing_source"],
                bucket=source_bucket_dir.joinpath(BUCKET_FILES[folder]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation / 解读",
            "",
            "- Folder counts should match the corresponding JSONL bucket row counts when no source image is missing.",
            "- Empty unsupported folders are intentional.",
            "- Use-case folders contain images only; metadata and evidence stay in the JSONL bucket files.",
            "",
            "- 如果没有源图片缺失，文件夹图片数应与对应 JSONL 行数一致。",
            "- unsupported 空文件夹是有意保留的。",
            "- use case 文件夹只放图片；元数据和证据保留在 JSONL bucket 文件中。",
            "",
        ]
    )
    output_dir.joinpath("use_case_candidates_report.md").write_text("\n".join(lines), encoding="utf-8")


def materialize(project_root: Path, source_bucket_dir: Path, output_dir: Path, clean: bool) -> OrderedDict[str, dict]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: OrderedDict[str, dict] = OrderedDict()
    for folder, bucket_file in BUCKET_FILES.items():
        folder_dir = output_dir / folder
        folder_dir.mkdir(parents=True, exist_ok=True)
        records = read_jsonl(source_bucket_dir / bucket_file)
        stat = {
            "images": 0,
            "hardlinked": 0,
            "copied": 0,
            "skipped_existing": 0,
            "missing_source": 0,
        }
        for record in records:
            image_rel = record.get("image")
            if not image_rel:
                stat["missing_source"] += 1
                continue
            src = project_root / image_rel
            if not src.exists():
                stat["missing_source"] += 1
                continue
            dst = folder_dir / src.name
            result = link_or_copy(src, dst)
            stat[result] += 1
            if result in {"hardlinked", "copied", "skipped_existing"}:
                stat["images"] += 1
        stats[folder] = stat

    write_readme(output_dir, source_bucket_dir)
    write_report(output_dir, source_bucket_dir, stats)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".", help="Incabin project root.")
    parser.add_argument(
        "--source-bucket-dir",
        default="1-gt_facts/use_case_buckets",
        help="Directory containing use-case bucket JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        default="0-datasets/sviro/use_case_candidates",
        help="Output directory for pure-image use-case folders.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not remove the output directory before rebuilding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    source_bucket_dir = (project_root / args.source_bucket_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    stats = materialize(project_root, source_bucket_dir, output_dir, clean=not args.no_clean)
    for folder, stat in stats.items():
        print(
            f"{folder}: images={stat['images']} hardlinked={stat['hardlinked']} "
            f"copied={stat['copied']} skipped_existing={stat['skipped_existing']} missing={stat['missing_source']}"
        )


if __name__ == "__main__":
    main()
