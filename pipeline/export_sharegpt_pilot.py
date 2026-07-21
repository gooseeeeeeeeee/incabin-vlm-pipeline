#!/usr/bin/env python3
"""
export_sharegpt_pilot.py — convert pilot raw JSONL -> ShareGPT training file.

Local, no API. Keeps ONLY `conversations` + `images` (per specs/SHAREGPT_FORMAT_SPEC of the
exterior pipeline and 2-generation/PLAN_COT_QA.md). All metadata (sample_id, suited_cases,
gt_evidence, qc, _draft) stays in the raw JSONL for QC/provenance — it must not leak into training.

Turn layout:
  human: "<image>\nDescribe the rear-seat cabin state, safety-relevant risk, and decision."
  gpt  : <caption_prose>          # label-free prose (G1)
  human: <q1> / gpt: <a1> / ...   # QA turns, in order

Usage:  python scripts/export_sharegpt_pilot.py
"""
import argparse, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "2-generation", "raw_outputs", "sviro_pilot_llm_raw.jsonl")
OUT = os.path.join(ROOT, "2-generation", "sharegpt", "sviro_pilot_sharegpt.json")
CAP_PROMPT = "<image>\nDescribe the rear-seat cabin state, safety-relevant risk, and decision."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW, help="input raw jsonl")
    ap.add_argument("--out", default=OUT, help="output sharegpt json")
    args = ap.parse_args()
    out_path = args.out
    recs = [json.loads(l) for l in open(args.raw, encoding="utf-8") if l.strip()]
    out, skipped = [], []
    for r in recs:
        if r.get("error") or not r.get("caption_prose") or not r.get("qa"):
            skipped.append(r.get("sample_id")); continue
        conv = [{"from": "human", "value": CAP_PROMPT},
                {"from": "gpt", "value": r["caption_prose"].strip()}]
        for q in r["qa"]:
            question, answer = (q.get("question") or "").strip(), (q.get("answer") or "").strip()
            if not question or not answer:
                continue
            conv += [{"from": "human", "value": question},
                     {"from": "gpt", "value": answer}]
        out.append({"conversations": conv, "images": [r["image"]]})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    turns = sum(len(o["conversations"]) for o in out)
    print(f"wrote {len(out)} ShareGPT records ({turns} turns) -> {out_path}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
