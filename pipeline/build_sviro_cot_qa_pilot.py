#!/usr/bin/env python3
"""
build_sviro_cot_qa_pilot.py

Deterministic, GT-grounded pilot generator for the SVIRO in-cabin data line.
Implements prompt_version = v0_pilot (see 2-generation/prompts/v0_pilot.md) as a
rule-based renderer instead of an LLM call, so the pilot is reproducible and
hallucination-free by construction. Every caption/QA field is derived only from
GT Facts fields; nothing is invented.

Usage:
    python3 scripts/build_sviro_cot_qa_pilot.py --n 30

Outputs:
    2-generation/raw_outputs/sviro_pilot_raw.jsonl
"""
import argparse, json, os, random, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(ROOT, "1-gt_facts", "sviro_bmw_x5_random_train_gt_facts.jsonl")
GT_REF = "1-gt_facts/sviro_bmw_x5_random_train_gt_facts.jsonl"
OUT = os.path.join(ROOT, "2-generation", "raw_outputs", "sviro_pilot_raw.jsonl")
PROMPT_VERSION = "v0_pilot"
MODEL_NAME = "deterministic_v0_pilot_rulegen"

SEAT_PHRASE = {
    "empty": "empty",
    "adult": "an adult occupant",
    "infant_in_infant_seat": "an infant in an infant seat",
    "child_in_child_seat": "a child in a child seat",
    "everyday_object": "an everyday object",
    "empty_infant_seat": "an empty infant seat",
    "empty_child_seat": "an empty child seat",
}
CHILD_SEAT_STATES = {"child_in_child_seat", "infant_in_infant_seat",
                     "empty_child_seat", "empty_infant_seat"}


def load_rows():
    with open(GT, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def has(seats, val):
    return any(v == val for v in seats.values())


def sample_rows(rows, n):
    """Deterministic stratified sampling: guarantee coverage of every
    gt_supported case, candidate cases, pose-flag rows, and multi-case rows."""
    rnd = random.Random(0)
    idx = {i: r for i, r in enumerate(rows)}

    def pool(pred):
        ids = [i for i, r in idx.items() if pred(r)]
        rnd.shuffle(ids)
        return ids

    def flagged(r):
        return any(p.get("pose_flags") for p in r.get("persons_summary", []))

    quotas = [
        ("empty_only", lambda r: all(v == "empty" for v in r["seat_states"].values()), 4),
        ("adult",      lambda r: has(r["seat_states"], "adult"), 5),
        ("infant",     lambda r: has(r["seat_states"], "infant_in_infant_seat"), 4),
        ("child",      lambda r: has(r["seat_states"], "child_in_child_seat"), 4),
        ("object",     lambda r: has(r["seat_states"], "everyday_object"), 4),
        ("pose_flag",  flagged, 6),
        ("multi_case", lambda r: len(r["suited_cases"]) >= 5, 3),
    ]
    picked = []
    seen = set()
    for _name, pred, q in quotas:
        c = 0
        for i in pool(pred):
            if i in seen:
                continue
            seen.add(i); picked.append(i); c += 1
            if c >= q or len(picked) >= n:
                break
        if len(picked) >= n:
            break
    # top up to n from anything not yet picked
    for i in pool(lambda r: True):
        if len(picked) >= n:
            break
        if i not in seen:
            seen.add(i); picked.append(i)
    picked = picked[:n]
    return [idx[i] for i in picked]


def cases_by_id(row):
    return {c["case_id"]: c for c in row["suited_cases"]}


def seat_evidence(seats):
    return [f"seat_states.{k}={v}" for k, v in seats.items()]


def build_caption(row):
    seats = row["seat_states"]
    oc = row.get("object_counts", {})
    parts_seat = ", ".join(f"{k}: {SEAT_PHRASE.get(v, v)}" for k, v in seats.items())
    obj = ", ".join(f"{n}×{c}" for c, n in oc.items()) if oc else "none reported"
    # posture / occlusion note from persons_summary
    flags = sorted({fl for p in row.get("persons_summary", []) for fl in p.get("pose_flags", [])})
    occluded = any(p.get("pose_flags") == [] or "core_joints_mostly_not_visible" in p.get("pose_flags", [])
                   for p in row.get("persons_summary", []))
    posture = ""
    if flags:
        posture = f" Pose-geometry flags present (geometry only, not a diagnosis): {', '.join(flags)}."
    state_scene = (f"Rear seats — {parts_seat}. Detected objects (bbox counts): {obj}."
                   f"{posture}")

    # RISK strictly from GT-supported presence
    risks = []
    if has(seats, "infant_in_infant_seat"):
        risks.append("an infant occupant is present, which is safety-relevant")
    if has(seats, "child_in_child_seat"):
        risks.append("a child occupant is present, which is safety-relevant")
    if has(seats, "everyday_object") or oc.get("everyday_object"):
        risks.append("an everyday object is left on a rear seat")
    if flags:
        risks.append("occupant posture geometry is atypical/occluded, making this an out-of-position "
                     "candidate that requires review (not a confirmed unsafe status)")
    if not risks:
        if has(seats, "adult"):
            risks.append("an adult occupant is present; no additional GT-supported risk")
        else:
            risks.append("no occupant detected in the rear seats; no GT-supported risk")
    risk = ". ".join(s[0].upper() + s[1:] for s in risks) + "."

    # DECISION conservative
    dec = []
    if has(seats, "infant_in_infant_seat") or has(seats, "child_in_child_seat"):
        dec.append("keep monitoring the child/infant occupant and verify the child-seat state before driving")
    if has(seats, "everyday_object") or oc.get("everyday_object"):
        dec.append("flag the object left on the seat for review")
    if flags:
        dec.append("escalate posture to manual/VLM review; do not infer out-of-position status")
    if not dec:
        dec.append("no action needed" if not has(seats, "adult") else "keep monitoring; no action needed")
    decision = ("; ".join(dec).capitalize() +
                ". Do not infer seatbelt, child-seat orientation, or health state (not in GT).")
    return {"state_scene": state_scene, "risk": risk, "decision": decision}


def build_qa(row):
    seats = row["seat_states"]
    oc = row.get("object_counts", {})
    cmap = cases_by_id(row)
    qa = []

    # 1) Recognition — occupancy readout
    if "02_adult_occupant" in cmap and has(seats, "adult"):
        uc = "02_adult_occupant"
    elif "05_infant_recognition" in cmap and has(seats, "infant_in_infant_seat"):
        uc = "05_infant_recognition"
    elif "12_left_objects" in cmap:
        uc = "12_left_objects"
    else:
        uc = "01_empty_seat"
    ans = "; ".join(f"{k}: {SEAT_PHRASE.get(v, v)}" for k, v in seats.items())
    qa.append({"question": "How is each rear seat occupied?",
               "answer": f"Left {seats['left']}, middle {seats['middle']}, right {seats['right']} "
                         f"→ {ans}.",
               "capability": "Recognition", "use_case": uc,
               "gt_evidence": seat_evidence(seats)})

    # 2) Reasoning — risk grounded in GT
    if has(seats, "infant_in_infant_seat"):
        qa.append({"question": "Is there an occupant needing special attention, and why?",
                   "answer": "Yes. GT marks a rear seat as infant_in_infant_seat, so an infant occupant "
                             "is present and is safety-relevant.",
                   "capability": "Reasoning", "use_case": "05_infant_recognition",
                   "gt_evidence": [f"seat_states.{k}=infant_in_infant_seat"
                                   for k, v in seats.items() if v == "infant_in_infant_seat"]})
    elif has(seats, "child_in_child_seat"):
        qa.append({"question": "Is there an occupant needing special attention, and why?",
                   "answer": "Yes. GT marks a rear seat as child_in_child_seat, so a child occupant "
                             "is present and is safety-relevant.",
                   "capability": "Reasoning", "use_case": "03_child_in_forward_facing_child_seat_candidates",
                   "gt_evidence": [f"seat_states.{k}=child_in_child_seat"
                                   for k, v in seats.items() if v == "child_in_child_seat"]})
    elif has(seats, "everyday_object") or oc.get("everyday_object"):
        qa.append({"question": "Is anything left on the rear seats that should be flagged?",
                   "answer": "Yes. GT shows an everyday object on a rear seat, which should be flagged "
                             "as a left object.",
                   "capability": "Reasoning", "use_case": "12_left_objects",
                   "gt_evidence": ([f"seat_states.{k}=everyday_object" for k, v in seats.items() if v == "everyday_object"]
                                   or ["object_counts.everyday_object"])})
    else:
        qa.append({"question": "Is there any GT-supported safety risk in the rear seats?",
                   "answer": "No GT-supported risk. The rear seats show no infant/child occupant and no "
                             "left object; any further risk cannot be determined from GT.",
                   "capability": "Reasoning", "use_case": "01_empty_seat",
                   "gt_evidence": seat_evidence(seats)})

    # 3) Out-of-position candidate (only if pose flags) — as candidate, not diagnosis
    flags = sorted({fl for p in row.get("persons_summary", []) for fl in p.get("pose_flags", [])})
    if flags and "11_out_of_position_candidates" in cmap:
        qa.append({"question": "Is any occupant out of position?",
                   "answer": "Cannot confirm. Pose-geometry flags (" + ", ".join(flags) + ") are present, "
                             "so this is only an out-of-position CANDIDATE that needs manual/VLM review. "
                             "GT does not label a confirmed unsafe out-of-position status.",
                   "capability": "Reject", "use_case": "11_out_of_position_candidates",
                   "gt_evidence": ["persons_summary[].pose_flags=" + ",".join(flags),
                                   "suited_cases.11_out_of_position_candidates.support_level=candidate"]})

    # 4) Decision
    if has(seats, "infant_in_infant_seat") or has(seats, "child_in_child_seat"):
        dec_uc = "05_infant_recognition" if has(seats, "infant_in_infant_seat") else "03_child_in_forward_facing_child_seat_candidates"
        dec_ans = ("Keep monitoring the child/infant occupant and verify the child-seat state before "
                   "driving. Escalate any orientation/posture judgement to manual/VLM review. Do not "
                   "infer seatbelt or health state.")
    elif has(seats, "everyday_object") or oc.get("everyday_object"):
        dec_uc = "12_left_objects"
        dec_ans = "Flag the left object for review before driving. No occupant-safety action inferred beyond GT."
    elif has(seats, "adult"):
        dec_uc = "02_adult_occupant"
        dec_ans = "Adult occupant present; keep monitoring, no additional action from GT. Do not infer seatbelt."
    else:
        dec_uc = "01_empty_seat"
        dec_ans = "Rear seats empty; no action needed."
    qa.append({"question": "What should the system do given this cabin state?",
               "answer": dec_ans, "capability": "Decision", "use_case": dec_uc,
               "gt_evidence": seat_evidence(seats)})

    # 5) Reject — orientation/ISOFIX if candidate present, else seatbelt (unsupported)
    if any(c in cmap for c in ("04_rear_facing_child_seat_candidates",
                               "06_isofix_child_seat_orientation_candidates")):
        rc = "06_isofix_child_seat_orientation_candidates" if "06_isofix_child_seat_orientation_candidates" in cmap \
             else "04_rear_facing_child_seat_candidates"
        qa.append({"question": "Is the child seat rear-facing and correctly ISOFIX-latched?",
                   "answer": "Cannot determine from the provided image and annotations. This image is only "
                             "a candidate for child-seat orientation/ISOFIX review; GT does not confirm "
                             "orientation or installation.",
                   "capability": "Reject", "use_case": rc,
                   "gt_evidence": [f"suited_cases.{rc}.support_level=candidate",
                                   f"suited_cases.{rc}.limitations"]})
    else:
        qa.append({"question": "Is the occupant wearing a seatbelt?",
                   "answer": "Cannot determine from the provided image and annotations. Seatbelt status is "
                             "not provided by SVIRO GT.",
                   "capability": "Reject", "use_case": "unsupported_reject",
                   "gt_evidence": ["gt_facts=no_seatbelt_field"]})
    return qa


def limitations_used(row, qa):
    used_cases = {q["use_case"] for q in qa}
    cmap = cases_by_id(row)
    lim = []
    for cid in used_cases:
        c = cmap.get(cid)
        if c:
            lim.extend(c.get("limitations", []))
    return sorted(set(lim))


def build_record(row):
    caption = build_caption(row)
    qa = build_qa(row)
    return {
        "sample_id": row["sample_id"],
        "prompt_version": PROMPT_VERSION,
        "model_name": MODEL_NAME,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image": row["image"]["path"],
        "gt_facts_ref": GT_REF,
        "suited_cases": [c["case_id"] for c in row["suited_cases"]],
        "caption": caption,
        "qa": qa,
        "limitations_used": limitations_used(row, qa),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    rows = load_rows()
    picked = sample_rows(rows, args.n)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(build_record(r), ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} records -> {OUT}")


if __name__ == "__main__":
    main()
