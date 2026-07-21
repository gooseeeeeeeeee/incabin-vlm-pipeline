#!/usr/bin/env python3
"""
build_splits.py — deterministic train / val_dev / frozen_test / candidate_pool splits.

Local, no API. Follows 3-splits/PLAN_SPLITS.md:
  - key = sample_id; splits are disjoint (train / val_dev / frozen_test).
  - frozen_test: GT-supported only, balanced by use case, never enters training.
  - candidate-only rows (no gt_supported case) never go to frozen_test.
  - candidate_pool is a routing VIEW (may overlap train/val), not a benchmark set.
  - writes per-split index.jsonl, frozen_test/by_use_case/, an assignment map, and a manifest.

Usage:  python scripts/build_splits.py [--frozen-per 40] [--val 150] [--seed 0]
"""
import argparse, json, os, random
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "1-gt_facts", "sviro_bmw_x5_random_train_gt_facts.jsonl")
SPLIT_DIR = os.path.join(ROOT, "3-splits")
FROZEN_CASES = ["01_empty_seat", "02_adult_occupant", "05_infant_recognition", "12_left_objects"]


def load():
    return [json.loads(l) for l in open(GT, encoding="utf-8") if l.strip()]


def summarize(row):
    sc = {c["case_id"]: c["support_level"] for c in row["suited_cases"]}
    gt_sup = sorted(k for k, v in sc.items() if v == "gt_supported")
    cand = sorted(k for k, v in sc.items() if v == "candidate")
    has_flag = any(p.get("pose_flags") for p in row.get("persons_summary", []))
    return gt_sup, cand, has_flag


def index_row(row, gt_sup, cand, has_flag):
    return {"sample_id": row["sample_id"], "image": row["image"]["path"],
            "seat_states": row["seat_states"], "gt_supported": gt_sup,
            "candidate": cand, "has_pose_flag": has_flag}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-per", type=int, default=40, help="target frozen_test samples per gt_supported use case")
    ap.add_argument("--val", type=int, default=150, help="val_dev size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load()
    meta = {r["sample_id"]: summarize(r) for r in rows}
    by_id = {r["sample_id"]: r for r in rows}
    all_ids = [r["sample_id"] for r in rows]

    # frozen_test: balanced greedy over GT-supported cases; candidate-only rows excluded
    eligible = [sid for sid in all_ids if meta[sid][0]]  # has >=1 gt_supported case
    frozen = set()
    for case in FROZEN_CASES:
        pool = [sid for sid in eligible if case in meta[sid][0] and sid not in frozen]
        rng.shuffle(pool)
        frozen.update(pool[:args.frozen_per])

    remaining = [sid for sid in all_ids if sid not in frozen]
    rng.shuffle(remaining)
    val = set(remaining[:args.val])
    train = set(remaining[args.val:])
    candidate_pool = [sid for sid in all_ids if meta[sid][1] and sid not in frozen]  # routing view

    assign = {}
    for sid in frozen: assign[sid] = "frozen_test"
    for sid in val: assign[sid] = "val_dev"
    for sid in train: assign[sid] = "train"

    # write assignment map + per-split indexes
    write_jsonl(os.path.join(SPLIT_DIR, "split_assignment.jsonl"),
                [{"sample_id": sid, "split": assign[sid]} for sid in all_ids])
    for name, ids in (("train", train), ("val_dev", val), ("frozen_test", frozen)):
        rows_out = [index_row(by_id[sid], *meta[sid]) for sid in all_ids if sid in ids]
        write_jsonl(os.path.join(SPLIT_DIR, name, "index.jsonl"), rows_out)
    write_jsonl(os.path.join(SPLIT_DIR, "candidate_pool", "index.jsonl"),
                [index_row(by_id[sid], *meta[sid]) for sid in candidate_pool])

    # frozen_test by_use_case
    fbuckets = defaultdict(list)
    for sid in frozen:
        for c in meta[sid][0]:
            fbuckets[c].append(index_row(by_id[sid], *meta[sid]))
    for c, rws in fbuckets.items():
        write_jsonl(os.path.join(SPLIT_DIR, "frozen_test", "by_use_case", f"{c}.jsonl"), rws)

    # manifest
    fcov = Counter(c for sid in frozen for c in meta[sid][0])
    man = os.path.join(SPLIT_DIR, "splits_manifest.md")
    with open(man, "w", encoding="utf-8") as fh:
        fh.write("# SVIRO Splits Manifest / 切分清单\n\n")
        fh.write(f"- seed={args.seed}, frozen_per={args.frozen_per}, val={args.val}\n")
        fh.write(f"- total={len(all_ids)}  train={len(train)}  val_dev={len(val)}  frozen_test={len(frozen)}\n")
        fh.write(f"- candidate_pool (view, overlaps train/val)={len(candidate_pool)}\n\n")
        fh.write("## Rules applied\n")
        fh.write("- Splits keyed by sample_id; train/val_dev/frozen_test are disjoint.\n")
        fh.write("- frozen_test drawn ONLY from rows with a gt_supported case; balanced across "
                 f"{FROZEN_CASES}; never enters training.\n")
        fh.write("- candidate-only rows (no gt_supported case) are excluded from frozen_test; they live in "
                 "train/val and candidate_pool, phrased as candidates.\n")
        fh.write("- candidate_pool is a routing view for generation/review, NOT a benchmark set.\n\n")
        fh.write("## frozen_test coverage by use case (multi-membership counted)\n\n")
        for c in FROZEN_CASES:
            fh.write(f"- {c}: {fcov.get(c, 0)}\n")
        fh.write(f"\n## frozen_test rows with a pose_flag (difficulty): "
                 f"{sum(1 for sid in frozen if meta[sid][2])}\n")
    print(f"train={len(train)} val_dev={len(val)} frozen_test={len(frozen)} candidate_pool={len(candidate_pool)}")
    print("frozen coverage:", dict(fcov))
    print("wrote ->", SPLIT_DIR)


if __name__ == "__main__":
    main()
