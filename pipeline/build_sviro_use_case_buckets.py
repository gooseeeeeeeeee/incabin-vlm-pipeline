#!/usr/bin/env python3
"""Export SVIRO use-case bucket views from GT Facts suited_cases.

The source of truth is `suited_cases` inside each GT Facts record. Bucket JSONL
files are lightweight, traceable views for review, sampling, and prompt-planning.
They are not final benchmark labels.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path


GT_FACTS_REF = "1-gt_facts/sviro_bmw_x5_random_train_gt_facts.jsonl"

BUCKETS = OrderedDict(
    [
        ("01_empty_seat", {"file": "01_empty_seat.jsonl", "bucket_type": "gt_supported"}),
        ("02_adult_occupant", {"file": "02_adult_occupant.jsonl", "bucket_type": "gt_supported"}),
        (
            "03_child_in_forward_facing_child_seat_candidates",
            {"file": "03_child_in_forward_facing_child_seat_candidates.jsonl", "bucket_type": "candidate"},
        ),
        (
            "04_rear_facing_child_seat_candidates",
            {"file": "04_rear_facing_child_seat_candidates.jsonl", "bucket_type": "candidate"},
        ),
        ("05_infant_recognition", {"file": "05_infant_recognition.jsonl", "bucket_type": "gt_supported"}),
        (
            "06_isofix_child_seat_orientation_candidates",
            {"file": "06_isofix_child_seat_orientation_candidates.jsonl", "bucket_type": "candidate"},
        ),
        ("07_pet_presence_unsupported", {"file": "07_pet_presence_unsupported.jsonl", "bucket_type": "unsupported"}),
        ("08_seatbelt_worn_unsupported", {"file": "08_seatbelt_worn_unsupported.jsonl", "bucket_type": "unsupported"}),
        (
            "09_seatbelt_not_worn_unsupported",
            {"file": "09_seatbelt_not_worn_unsupported.jsonl", "bucket_type": "unsupported"},
        ),
        (
            "10_occupant_sleeping_unsupported",
            {"file": "10_occupant_sleeping_unsupported.jsonl", "bucket_type": "unsupported"},
        ),
        ("11_out_of_position_candidates", {"file": "11_out_of_position_candidates.jsonl", "bucket_type": "candidate"}),
        ("12_left_objects", {"file": "12_left_objects.jsonl", "bucket_type": "gt_supported"}),
        (
            "13_vehicle_door_status_unsupported",
            {"file": "13_vehicle_door_status_unsupported.jsonl", "bucket_type": "unsupported"},
        ),
    ]
)

UNSUPPORTED_NOTES = {
    "07_pet_presence_unsupported": "Current SVIRO GT has no pet labels.",
    "08_seatbelt_worn_unsupported": "Current SVIRO GT has no seatbelt-worn labels.",
    "09_seatbelt_not_worn_unsupported": "Current SVIRO GT has no seatbelt misuse labels.",
    "10_occupant_sleeping_unsupported": "Current SVIRO GT has no sleeping or microsleep labels.",
    "13_vehicle_door_status_unsupported": "Current SVIRO GT has no door open/ajar/closed labels.",
}


def make_entry(record: dict, suited_case: dict) -> dict:
    return {
        "sample_id": record["sample_id"],
        "image": record["image"]["path"],
        "use_case": suited_case["case_id"],
        "case_name": suited_case["case_name"],
        "support_level": suited_case["support_level"],
        "purpose": suited_case["purpose"],
        "evidence": suited_case["evidence"],
        "limitations": suited_case["limitations"],
        "gt_facts_ref": GT_FACTS_REF,
    }


def write_readme(out_dir: Path) -> None:
    content = """# SVIRO Use Case Buckets / SVIRO Use Case 分桶

This folder contains lightweight JSONL views derived from `suited_cases` in GT Facts.

本目录保存从 GT Facts 的 `suited_cases` 字段派生出来的轻量 JSONL 视图。

## Role / 作用

- Source of truth: `1-gt_facts/sviro_bmw_x5_random_train_gt_facts.jsonl`.
- This folder is useful for checking which samples are eligible for each prompt/use-case direction.
- It is not the final benchmark dataset and not final confirmed classification.
- Unsupported buckets are intentionally empty to make current SVIRO coverage limits explicit.

- 真正源头是 `1-gt_facts/sviro_bmw_x5_random_train_gt_facts.jsonl`。
- 本目录用于检查每个 prompt/use-case 方向有哪些 eligible samples。
- 它不是最终 benchmark 数据集，也不是最终确认分类。
- unsupported bucket 故意保留为空，用来明确当前 SVIRO 覆盖边界。

## Chain / 链路

```text
1-gt_facts GT Facts.suited_cases
↓
1-gt_facts/use_case_buckets JSONL views
↓
prompt routing / pilot CoT + QA generation
↓
QC / frozen split / ShareGPT export
```

See `../use_case_buckets_report.md` for counts and examples.
"""
    (out_dir / "README.md").write_text(content, encoding="utf-8")


def write_report(report_path: Path, counts: dict[str, int], examples: dict[str, list[dict]]) -> None:
    lines = [
        "# SVIRO Use Case Buckets Report / SVIRO Use Case 分桶报告",
        "",
        "These bucket files are derived from `suited_cases` in GT Facts.",
        "",
        "这些 bucket 文件由 GT Facts 中的 `suited_cases` 字段派生。",
        "",
        "## Important Notes / 重要说明",
        "",
        "- `suited_cases` is prompt-routing eligibility, not final benchmark labeling.",
        "- One sample may appear in multiple buckets.",
        "- Candidate buckets such as orientation, ISOFIX, and OOP require later VLM/manual/QC confirmation before they become benchmark labels.",
        "- Unsupported buckets are empty because current SVIRO GT does not support those tasks.",
        "",
        "- `suited_cases` 是 prompt routing 资格，不是最终 benchmark 标签。",
        "- 同一个样本可以出现在多个 bucket。",
        "- orientation、ISOFIX、OOP 等 candidate bucket 后续需要 VLM/人工/QC 确认，才能成为 benchmark 标签。",
        "- unsupported bucket 为空，因为当前 SVIRO GT 不支持这些任务。",
        "",
        "## Counts / 数量",
        "",
        "| Bucket | Support | Count |",
        "|---|---|---:|",
    ]
    for bucket, meta in BUCKETS.items():
        lines.append(f"| `{meta['file']}` | {meta['bucket_type']} | {counts.get(bucket, 0)} |")

    lines.extend(["", "## First Examples / 前 5 条样例", ""])
    for bucket, meta in BUCKETS.items():
        lines.extend([f"### {bucket}", "", f"File: `{meta['file']}`", ""])
        if bucket in UNSUPPORTED_NOTES:
            lines.append(f"Unsupported note: {UNSUPPORTED_NOTES[bucket]}")
            lines.append("")
        if not examples.get(bucket):
            lines.extend(["No current SVIRO samples. / 当前 SVIRO 无样本。", ""])
            continue
        lines.append("```json")
        for entry in examples[bucket][:5]:
            lines.append(json.dumps(entry, ensure_ascii=False))
        lines.extend(["```", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-facts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {bucket: [] for bucket in BUCKETS}

    with args.gt_facts.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            for suited_case in record.get("suited_cases", []):
                case_id = suited_case["case_id"]
                if case_id not in buckets:
                    raise ValueError(f"Unknown suited_case case_id: {case_id}")
                buckets[case_id].append(make_entry(record, suited_case))

    counts = {}
    examples = {}
    for bucket, meta in BUCKETS.items():
        rows = buckets[bucket]
        counts[bucket] = len(rows)
        examples[bucket] = rows[:5]
        with (args.out_dir / meta["file"]).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_readme(args.out_dir)
    write_report(args.report, counts, examples)
    print(json.dumps({"out_dir": str(args.out_dir), "report": str(args.report), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
