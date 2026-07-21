#!/usr/bin/env python3
"""
explore_driveact.py — discover what's inside the raw Drive&Act dataset.

Run from the project root ON THE INSTANCE (where the raw data lives):
    python scripts/explore_driveact.py
Prints: directory structure, modalities/views, annotation files + schemas, and label vocabularies.
"""
import collections
import csv
import json
import os

ROOT = "0-datasets/driveact"


def tree(root, max_depth=3, max_children=25):
    print(f"\n=== DIRECTORY TREE (depth<= {max_depth}) ===")
    for dp, dn, fn in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth > max_depth:
            dn[:] = []
            continue
        # don't descend into per-participant frame dirs (thousands of images)
        if len(fn) > max_children:
            dn[:] = [d for d in dn if not any(x in d.lower() for x in ("vp", "run", "frame"))]
        indent = "  " * depth
        exts = collections.Counter(os.path.splitext(f)[1].lower() for f in fn)
        ext_s = " ".join(f"{k or '<noext>'}:{v}" for k, v in exts.most_common(6)) if fn else ""
        print(f"{indent}{os.path.basename(dp) or root}/  [{len(fn)} files {ext_s}] [{len(dn)} subdirs]")


def modalities():
    print("\n=== MODALITIES / VIEWS (top-level under raw/) ===")
    raw = os.path.join(ROOT, "raw")
    if os.path.isdir(raw):
        for d in sorted(os.listdir(raw)):
            p = os.path.join(raw, d)
            if os.path.isdir(p):
                subs = sorted(os.listdir(p))[:30]
                print(f"  raw/{d}/: {subs}")


def annotation_files():
    print("\n=== ANNOTATION FILES (csv/txt/json under raw) ===")
    ann = []
    for dp, _dn, fn in os.walk(os.path.join(ROOT, "raw")):
        for f in fn:
            if os.path.splitext(f)[1].lower() in (".csv", ".txt", ".json", ".tsv"):
                ann.append(os.path.join(dp, f))
    print(f"total annotation-like files: {len(ann)}")
    # group by parent folder (annotation type / view)
    by_dir = collections.Counter(os.path.dirname(p) for p in ann)
    for d, c in by_dir.most_common(20):
        print(f"  {c:5d}  {d}")
    return ann


def peek_and_vocab(ann):
    print("\n=== SAMPLE SCHEMAS + LABEL VOCABULARIES ===")
    # take one representative csv/txt per distinct parent dir
    seen_dirs = set()
    vocab = collections.defaultdict(collections.Counter)
    for p in ann:
        d = os.path.dirname(p)
        ext = os.path.splitext(p)[1].lower()
        if ext in (".csv", ".tsv", ".txt") and d not in seen_dirs:
            seen_dirs.add(d)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    lines = [next(fh) for _ in range(4)]
            except Exception:
                continue
            print(f"\n--- {p}")
            for ln in lines:
                print("   ", ln.rstrip()[:160])
    # aggregate categorical vocab across ALL csv/tsv label files
    for p in ann:
        if os.path.splitext(p)[1].lower() not in (".csv", ".tsv"):
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as fh:
                rdr = csv.DictReader(fh)
                for row in rdr:
                    for k, v in row.items():
                        if k and v and not v.replace(".", "").replace("-", "").isdigit() and len(v) < 40:
                            vocab[k][v] += 1
        except Exception:
            continue
    print("\n=== LABEL VOCABULARIES (categorical columns, top values) ===")
    for col, ctr in vocab.items():
        if 1 < len(ctr) <= 400:
            print(f"\n[{col}] {len(ctr)} distinct:")
            for v, c in ctr.most_common(25):
                print(f"   {c:6d}  {v}")


def processed_recap():
    print("\n=== PROCESSED GT FACTS (already built) ===")
    gt = os.path.join("1-gt_facts", "driveact", "driveact_a_column_co_driver_rgb_gt_facts.jsonl")
    if os.path.exists(gt):
        rows = [json.loads(l) for l in open(gt, encoding="utf-8")]
        mid = collections.Counter(r["activity_facts"]["midlevel"]["activity"] for r in rows)
        obj = collections.Counter((r["activity_facts"].get("objectlevel") or {}).get("activity") for r in rows)
        loc = collections.Counter((r["activity_facts"].get("objectlevel") or {}).get("location") for r in rows)
        things = collections.Counter((r["activity_facts"].get("objectlevel") or {}).get("object") for r in rows)
        print(f"processed rows: {len(rows)}  (subset a_column_co_driver_rgb only)")
        print("midlevel activities:", len(mid), "->", dict(mid.most_common(12)))
        print("objectlevel actions:", dict(obj.most_common(12)))
        print("objectlevel objects:", dict(things.most_common(12)))
        print("objectlevel locations:", dict(loc.most_common(12)))


if __name__ == "__main__":
    print("ROOT:", os.path.abspath(ROOT))
    tree(ROOT)
    modalities()
    ann = annotation_files()
    peek_and_vocab(ann)
    processed_recap()
