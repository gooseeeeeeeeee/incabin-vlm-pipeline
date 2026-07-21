#!/usr/bin/env python3
"""
check_adversarial.py — prove the cross-check gate actually strips planted errors.

RUN ON THE AUTODL INSTANCE (needs API + .env). Builds a few DRAFTS that each contain ONE
deliberately false claim, sends them through the same cross-check used in generation, and
reports whether the correction removed/fixed the planted error.

Planted error types:
  A occupancy contradiction  — an 'empty' seat claimed as occupied by an adult
  B asserted orientation     — child seat stated as confirmed rear-facing + ISOFIX-latched
  C seatbelt claim           — an occupant stated to be wearing a seatbelt (not in GT)
  D person miscount          — person count inflated beyond GT object_counts

Usage:  python scripts/check_adversarial.py
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sviro_cot_qa_llm as B  # reuse CHECK_SYSTEM/CHECK_USER, parse_json, SAFE_NEG, llm_client

GT = os.path.join(B.ROOT, "1-gt_facts", "sviro_bmw_x5_random_train_gt_facts.jsonl")


def find(rows, pred):
    return next(r for r in rows if pred(r))


def draft(planted_answer, filler_answer="All three rear seats are described per GT."):
    return {"caption": {"state_scene": "", "risk": "", "decision": ""},
            "caption_prose": planted_answer,
            "qa": [{"question": "Describe the scene.", "answer": planted_answer,
                    "reason": "planted", "capability": "Recognition", "use_case": "01_empty_seat",
                    "gt_evidence": ["seat_states"]},
                   {"question": "Anything else?", "answer": filler_answer, "reason": "filler",
                    "capability": "Reasoning", "use_case": "01_empty_seat", "gt_evidence": ["seat_states"]}],
            "limitations_used": []}


def crosscheck_answer(gt_row, d):
    raw = B.llm_client.crosscheck(json.dumps(gt_row, ensure_ascii=False),
                                  json.dumps(d, ensure_ascii=False), B.CHECK_SYSTEM, B.CHECK_USER)
    corr = B.parse_json(raw)
    return (corr.get("qa") or [{}])[0].get("answer", ""), corr


def main():
    rows = [json.loads(l) for l in open(GT, encoding="utf-8") if l.strip()]
    r_occ = find(rows, lambda r: "empty" in r["seat_states"].values() and "adult" in r["seat_states"].values())
    r_child = find(rows, lambda r: any(("child" in v or "infant" in v) for v in r["seat_states"].values()))
    empty_seat = next(k for k, v in r_occ["seat_states"].items() if v == "empty")
    gt_person = r_occ.get("object_counts", {}).get("person", 0)

    cases = [
        ("A occupancy contradiction", r_occ,
         f"The {empty_seat} rear seat is occupied by an adult.",
         lambda a: "empty" in a.lower() and "occupied by an adult" not in a.lower()),
        ("B asserted orientation", r_child,
         "The child seat is installed rear-facing and is ISOFIX-latched securely.",
         lambda a: bool(B.SAFE_NEG.search(a)) or ("rear-facing" not in a.lower())),
        ("C seatbelt claim", r_occ,
         "The adult occupant is properly wearing a seatbelt.",
         lambda a: "wearing a seatbelt" not in a.lower()),
        ("D person miscount", r_occ,
         f"There are {gt_person + 3} people visible in the rear seats.",
         lambda a: str(gt_person + 3) not in a),
    ]

    caught = 0
    for name, row, planted, detect in cases:
        try:
            corrected, _ = crosscheck_answer(row, draft(planted))
        except Exception as e:  # noqa: BLE001
            print(f"== {name}: ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        ok = detect(corrected)
        caught += ok
        print(f"== {name}: {'CAUGHT' if ok else 'MISSED'}")
        print(f"   planted  : {planted}")
        print(f"   corrected: {corrected[:220]}")
    print(f"\ncross-check caught {caught}/{len(cases)} planted errors")


if __name__ == "__main__":
    main()
