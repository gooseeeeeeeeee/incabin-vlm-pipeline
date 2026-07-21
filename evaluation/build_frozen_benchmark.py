#!/usr/bin/env python3
"""
build_frozen_benchmark.py — closed-form objective benchmark for frozen_test.

Local, no API. For each frozen_test sample (GT-supported), emits questions whose ground-truth
answer is derived deterministically from GT Facts, so scoring is objective:
  - occupancy (per seat): category of left/middle/right
  - count       : number of persons in the rear seats
  - left_object : is an everyday object present (yes/no)
  - reject      : seatbelt status -> must refuse ("cannot determine")

Output: 5-evaluation/frozen_benchmark.jsonl  (image paths are relative; eval prepends --image-root)

Usage:  python scripts/build_frozen_benchmark.py
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "1-gt_facts", "sviro_bmw_x5_random_train_gt_facts.jsonl")
ASSIGN = os.path.join(ROOT, "3-splits", "split_assignment.jsonl")
OUT = os.path.join(ROOT, "5-evaluation", "frozen_benchmark.jsonl")

OCC_CHOICES = "empty, adult, child in child seat, infant in infant seat, empty child seat, empty infant seat, everyday object"
CAT = {"empty": "empty", "adult": "adult", "child_in_child_seat": "child_in_child_seat",
       "infant_in_infant_seat": "infant_in_infant_seat", "empty_child_seat": "empty_child_seat",
       "empty_infant_seat": "empty_infant_seat", "everyday_object": "everyday_object"}


def main():
    frozen = {json.loads(l)["sample_id"] for l in open(ASSIGN, encoding="utf-8")
              if json.loads(l)["split"] == "frozen_test"}
    rows = [json.loads(l) for l in open(GT, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            if r["sample_id"] not in frozen:
                continue
            seats = r["seat_states"]
            oc = r.get("object_counts", {})
            qs = []
            for seat in ("left", "middle", "right"):
                qs.append({"id": f"occ_{seat}", "type": "occupancy", "use_case": "occupancy",
                           "q": f"Look at the {seat.upper()} rear seat. What is on it? "
                                f"Answer with exactly one of: {OCC_CHOICES}.",
                           "gt": CAT[seats[seat]]})
            qs.append({"id": "count_person", "type": "count", "use_case": "02_adult_occupant",
                       "q": "How many people are in the rear seats? Reply with a single integer.",
                       "gt": int(oc.get("person", 0))})
            has_obj = any(v == "everyday_object" for v in seats.values()) or bool(oc.get("everyday_object"))
            qs.append({"id": "left_object", "type": "yesno", "use_case": "12_left_objects",
                       "q": "Is there an everyday object left on any rear seat? Answer yes or no.",
                       "gt": "yes" if has_obj else "no"})
            qs.append({"id": "reject_seatbelt", "type": "reject", "use_case": "08_seatbelt",
                       "q": "Are the rear occupants wearing seatbelts? If the image and annotations "
                            "cannot determine this, answer 'cannot determine'.",
                       "gt": "cannot determine"})
            fh.write(json.dumps({"sample_id": r["sample_id"], "image": r["image"]["path"],
                                 "questions": qs}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} frozen samples, {n*6} questions -> {OUT}")


if __name__ == "__main__":
    main()
