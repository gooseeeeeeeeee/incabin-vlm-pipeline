#!/usr/bin/env python3
"""
clean_and_export.py — content-clean the full raw JSONL and export training ShareGPT.

Local, no API. Re-validates each record at the CONTENT level (ignores gt_evidence, which is
metadata that never enters ShareGPT) and splits into:
  - clean   -> raw_outputs/sviro_full_clean.jsonl + sharegpt/sviro_train_val_sharegpt.json
  - flagged -> quality/sviro_full_flagged.jsonl  (real content issues, for regeneration/review)
Rows with an `error` field (e.g. API 504s) are reported separately for a --resume backfill.

Content rules (same as the generator's validator): exactly 1 Reject; Recognition/Reasoning/Decision
present; no forbidden claim (seatbelt/pet/door/sleep/gender) outside Reject; no appearance/state
inference (race/alertness/emotion); orientation/ISOFIX only inside a refusal.

Usage:  python scripts/clean_and_export.py
"""
import argparse, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "2-generation", "raw_outputs", "sviro_full_raw.jsonl")
CLEAN = os.path.join(ROOT, "2-generation", "raw_outputs", "sviro_full_clean.jsonl")
SG = os.path.join(ROOT, "2-generation", "sharegpt", "sviro_train_val_sharegpt.json")
FLAG = os.path.join(ROOT, "2-generation", "quality", "sviro_full_flagged.jsonl")
CAP_PROMPT = "<image>\nDescribe the rear-seat cabin state, safety-relevant risk, and decision."

HARD = re.compile(r"\b(seatbelt|seat belt|belt|pet|dog|cat|door open|door closed|asleep|sleeping|"
                  r"unconscious|gender)\b", re.I)
ORI = re.compile(r"forward-facing|rear-facing|isofix|latched", re.I)
SAFE = re.compile(r"candidate|cannot|not confirm|not directly label|not labeled|does not|do not|"
                  r"no .*label|manual review|manual or vlm|vlm review|needs? review|review is|"
                  r"before driving|verify|manually|unable|not provided|not specify|is not", re.I)
APP = re.compile(r"\b(caucasian|asian|african|hispanic|latino|ethnicit\w*)\b|"
                 r"appears?\s+(alert|awake|calm|tired|drowsy|asleep)|looks?\s+(alert|awake|tired|calm)|"
                 r"\b(emotion|happy|sad|angry|anxious)\b", re.I)


def content_issues(r):
    iss = []
    caps = [q.get("capability") for q in r.get("qa", [])]
    if caps.count("Reject") != 1:
        iss.append("reject_count")
    for n in ("Recognition", "Reasoning", "Decision"):
        if n not in caps:
            iss.append("missing_" + n)
    for q in r.get("qa", []):
        a = q.get("answer") or ""
        low = ((q.get("question") or "") + " " + a).lower()
        if q.get("capability") != "Reject":
            if HARD.search(low):
                iss.append("forbidden")
            if APP.search(low):
                iss.append("appearance")
        if ORI.search(low) and not SAFE.search(a):
            iss.append("asserted_orientation")
    return sorted(set(iss))


def sharegpt(r, image_root=""):
    conv = [{"from": "human", "value": CAP_PROMPT},
            {"from": "gpt", "value": (r.get("caption_prose") or "").strip()}]
    for q in r.get("qa", []):
        qq, aa = (q.get("question") or "").strip(), (q.get("answer") or "").strip()
        if qq and aa:
            conv += [{"from": "human", "value": qq}, {"from": "gpt", "value": aa}]
    img = r["image"]
    if image_root:
        img = os.path.join(image_root, img)
    return {"conversations": conv, "images": [img]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=SG, help="output ShareGPT json path")
    ap.add_argument("--image-root", default="", help="prefix for image paths (e.g. instance project root)")
    ap.add_argument("--split-file", default=None, help="3-splits/split_assignment.jsonl")
    ap.add_argument("--splits", default="train,val_dev", help="comma splits to include in ShareGPT")
    ap.add_argument("--trust-qc", action="store_true",
                    help="filter on each record's own qc.passed instead of SVIRO content rules (use for DriveAct)")
    ap.add_argument("--soft-issues", default="candidate_not_flagged,bad_evidence",
                    help="qc issue types treated as non-blocking under --trust-qc (benign/metadata)")
    args = ap.parse_args()
    soft = {s.strip() for s in args.soft_issues.split(",") if s.strip()}
    recs = [json.loads(l) for l in open(args.raw, encoding="utf-8") if l.strip()]
    errors = [r for r in recs if r.get("error")]
    ok = [r for r in recs if not r.get("error")]
    if args.split_file:
        inc = {s.strip() for s in args.splits.split(",")}
        amap = {}
        for line in open(args.split_file, encoding="utf-8"):
            try:
                o = json.loads(line); amap[o["sample_id"]] = o["split"]
            except Exception:  # noqa: BLE001
                pass
        ok = [r for r in ok if amap.get(r["sample_id"]) in inc]
    clean, flagged = [], []
    for r in ok:
        if args.trust_qc:
            raw_iss = [] if r.get("qc", {}).get("passed") else (r.get("qc", {}).get("issues") or ["qc_fail"])
            iss = [i for i in raw_iss if i.split("=")[0] not in soft]  # drop only blocking issues
        else:
            iss = content_issues(r)
        if iss or not (r.get("caption_prose") and r.get("qa")):
            r["_flag"] = iss or ["missing_content"]
            flagged.append(r)
        else:
            clean.append(r)

    for path, rows in ((CLEAN, clean), (FLAG, flagged)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump([sharegpt(r, args.image_root) for r in clean], fh, ensure_ascii=False, indent=1)

    from collections import Counter
    fb = Counter(i for r in flagged for i in r["_flag"])
    turns = sum(len(sharegpt(r)["conversations"]) for r in clean)
    print(f"total={len(recs)}  ok={len(ok)}  error(504,need backfill)={len(errors)}")
    print(f"clean={len(clean)}  flagged={len(flagged)}  flag types={dict(fb)}")
    print(f"ShareGPT: {len(clean)} records, {turns} turns -> {args.out}")
    print(f"clean raw -> {CLEAN}\nflagged -> {FLAG}")


if __name__ == "__main__":
    main()
