#!/usr/bin/env python3
"""
spotcheck_occupancy.py — dump occupancy MISSES (gt vs model text) for manual real-error vs scorer-artifact review.

Usage:  python 5-evaluation/spotcheck_occupancy.py 5-evaluation/results_v2.jsonl [N]
"""
import json, random, sys

path = sys.argv[1] if len(sys.argv) > 1 else "5-evaluation/results_v2.jsonl"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

misses = []
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    for p in r.get("preds", []):
        if p.get("type") == "occupancy" and not p.get("ok"):
            misses.append((r["sample_id"], p.get("id"), p.get("gt"), (p.get("pred") or "").replace("\n", " ")))

print(f"file: {path}  |  total occupancy misses: {len(misses)}")
random.seed(0); random.shuffle(misses)
for sid, qid, gt, pred in misses[:n]:
    print(f"\n[{sid} {qid}]  GT = {gt}")
    print(f"   PRED: {pred[:220]}")
