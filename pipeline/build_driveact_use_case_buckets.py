#!/usr/bin/env python3
"""build_driveact_use_case_buckets.py

Map Drive&Act (A-Column co-driver RGB, activity-only) GT facts onto the 13
in-cabin use cases, using the SAME 01_..13_ naming/format as the SVIRO buckets
(`1-gt_facts/use_case_buckets/`).

Why this is a separate mapper (not just `suited_cases`):
  The SVIRO GT facts already embed rich `suited_cases` with `case_id`. Drive&Act
  facts only carry 4 coarse suited_cases (general_activity / seat_belt /
  door_status / out_of_position_candidate). So here we derive the 13-use-case
  view from the Drive&Act ACTIVITY taxonomy (midlevel.activity + objectlevel
  {activity, object, location}).

Honesty rules (match `6-docs/target_spec_coverage.md`):
  - Drive&Act is a single ADULT co-driver performing an activity in RGB. It has
    NO occupancy / child-seat / infant / pet labels, and NO eye/sleep/3D state.
  - Cases with no matching Drive&Act evidence are written as EMPTY files with an
    explicit `unsupported` reason -- we do not fabricate coverage.
  - Action != state. Seatbelt (fasten/unfasten action), door (open/close action)
    and left-object (active interaction, not "left unattended") are marked
    `candidate` / `partial`, never `gt_supported` for the exact safety claim.

Usage (run from the Incabin root, or pass explicit paths):
  python scripts/build_driveact_use_case_buckets.py \
    --gt-facts 1-gt_facts/driveact/driveact_a_column_co_driver_rgb_gt_facts.jsonl \
    --out-dir  1-gt_facts/driveact/use_case_13 \
    --report   1-gt_facts/driveact/use_case_13/coverage_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

GT_FACTS_REF = "1-gt_facts/driveact/driveact_a_column_co_driver_rgb_gt_facts.jsonl"

# ---------------------------------------------------------------------------
# Drive&Act activity taxonomy groupings used by the matchers below.
# ---------------------------------------------------------------------------

# Portable items an occupant could leave behind (drives use case 12).
# Excludes fixed cabin controls and the seatbelt itself.
LEFT_OBJECT_ITEMS = {
    "phone", "laptop", "newspaper", "magazine", "bottle", "food",
    "jacket", "writing_pad", "glasses", "backpack", "pen", "glasses_case",
}
FIXED_OR_NON_LEFT = {"no_object", "seatbelt", "automation_button",
                     "multimedia_display", "gearstick"}

FASTEN_ACTS = {"fastening_seat_belt"}
UNFASTEN_ACTS = {"unfastening_seat_belt"}


# ---------------------------------------------------------------------------
# Each matcher takes (mid, obj, suited) and returns (matched: bool, reason: str)
#   mid    = midlevel.activity string or None
#   obj    = objectlevel dict {activity, object, location, ...} or None
#   suited = set of coarse suited_cases on the fact
# ---------------------------------------------------------------------------

def _m_empty_seat(mid, obj, suited):
    return False, ""  # every Drive&Act frame shows an occupied co-driver seat


def _m_adult(mid, obj, suited):
    # Drive&Act subjects are all adult co-drivers -> presence is always true.
    return (mid is not None), "adult co-driver present (activity-labeled frame)"


def _m_child_fwd(mid, obj, suited):
    return False, ""


def _m_child_rear(mid, obj, suited):
    return False, ""


def _m_infant(mid, obj, suited):
    return False, ""


def _m_isofix(mid, obj, suited):
    return False, ""


def _m_pet(mid, obj, suited):
    return False, ""


def _m_belt_worn(mid, obj, suited):
    if mid in FASTEN_ACTS:
        return True, f"midlevel={mid} (belt-fastening action)"
    # objectlevel fallback: touching the belt while NOT explicitly unfastening it.
    if (mid not in UNFASTEN_ACTS and obj and obj.get("object") == "seatbelt"
            and obj.get("activity") in {"reaching_for", "interacting"}):
        return True, f"objectlevel: {obj.get('activity')} seatbelt @ {obj.get('location')} (mid={mid})"
    return False, ""


def _m_belt_not_worn(mid, obj, suited):
    if mid in UNFASTEN_ACTS:
        return True, f"midlevel={mid} (belt-unfastening action)"
    # objectlevel fallback: pulling away from the belt while NOT explicitly fastening it.
    if (mid not in FASTEN_ACTS and obj and obj.get("object") == "seatbelt"
            and obj.get("activity") == "retracting_from"):
        return True, f"objectlevel: retracting_from seatbelt @ {obj.get('location')} (mid={mid})"
    return False, ""


def _m_sleeping(mid, obj, suited):
    # No eye/drowsiness state in Drive&Act. `sitting_still` is NOT sleeping.
    return False, ""


def _m_oop(mid, obj, suited):
    if "out_of_position_candidate" in suited:
        return True, f"suited_cases=out_of_position_candidate (midlevel={mid})"
    return False, ""


def _m_left_objects(mid, obj, suited):
    if obj and obj.get("object") in LEFT_OBJECT_ITEMS:
        return True, f"objectlevel object={obj.get('object')} @ {obj.get('location')} ({obj.get('activity')})"
    return False, ""


def _m_door(mid, obj, suited):
    if "door_status" in suited:
        return True, f"suited_cases=door_status (midlevel={mid})"
    return False, ""


# ---------------------------------------------------------------------------
# Bucket registry: 13 use cases, in spec order.
# ---------------------------------------------------------------------------

BUCKETS = OrderedDict([
    ("01_empty_seat", dict(
        file="01_empty_seat_unsupported.jsonl", case_id="01", support="unsupported",
        name="Empty Seat", match=_m_empty_seat,
        purpose="Detect unoccupied seats for occupancy assessment.",
        limitation="Drive&Act co-driver frames are always occupied and activity-labeled; no empty-seat evidence. Occupancy is covered by SVIRO seat-states, not Drive&Act.")),

    ("02_adult_occupant", dict(
        file="02_adult_occupant_presence.jsonl", case_id="02", support="partial",
        name="Adult Occupant", match=_m_adult,
        purpose="Identify adult passengers; spec also wants 5th/50th/95th stature percentile.",
        limitation="Presence only. Drive&Act has NO anthropometric/stature-percentile labels, so the percentile requirement is unsupported.")),

    ("03_child_in_forward_facing_child_seat", dict(
        file="03_child_forward_facing_seat_unsupported.jsonl", case_id="03", support="unsupported",
        name="Child in Forward-Facing Child Seat", match=_m_child_fwd,
        purpose="Recognize children in forward-installed safety seats for airbag logic.",
        limitation="Drive&Act contains only adult co-drivers; no child or child-seat labels.")),

    ("04_rear_facing_child_seat", dict(
        file="04_rear_facing_child_seat_unsupported.jsonl", case_id="04", support="unsupported",
        name="Rear-Facing Child Seat", match=_m_child_rear,
        purpose="Detect rear-facing seats to command passenger airbag OFF (safety-critical).",
        limitation="No child-seat or orientation labels in Drive&Act. Do NOT claim -- lethal if wrong.")),

    ("05_infant_recognition", dict(
        file="05_infant_recognition_unsupported.jsonl", case_id="05", support="unsupported",
        name="Infant Recognition", match=_m_infant,
        purpose="Identify infants incl. sleeping / playing / unrestrained.",
        limitation="No infant labels in Drive&Act.")),

    ("06_isofix_child_seat_orientation", dict(
        file="06_isofix_orientation_unsupported.jsonl", case_id="06", support="unsupported",
        name="ISOFIX & Child Seat Orientation", match=_m_isofix,
        purpose="Verify ISOFIX install + orientation via mechanical tension/flush placement.",
        limitation="No child-seat/ISOFIX/mechanical labels; likely needs sensing beyond RGB.")),

    ("07_pet_presence", dict(
        file="07_pet_presence_unsupported.jsonl", case_id="07", support="unsupported",
        name="Pet Presence", match=_m_pet,
        purpose="Detect dogs/cats vs humans.",
        limitation="No pet labels in Drive&Act.")),

    ("08_seatbelt_worn", dict(
        file="08_seatbelt_worn_candidate.jsonl", case_id="08", support="candidate",
        name="Seatbelt Worn", match=_m_belt_worn,
        purpose="Confirm belt routing over clavicle/sternum.",
        limitation="Drive&Act gives the fastening ACTION only, not the worn-state / diagonal-routing geometry the spec requires. Recall-only candidate.")),

    ("09_seatbelt_not_worn", dict(
        file="09_seatbelt_not_worn_candidate.jsonl", case_id="09", support="candidate",
        name="Seatbelt Not Worn", match=_m_belt_not_worn,
        purpose="Identify unfastened/misused belts with misuse categories + 30s latency.",
        limitation="Only the unfastening ACTION is labeled; no misuse categories (buckle-only/behind-back/lap-only) and no temporal-latency labels. Candidate.")),

    ("10_occupant_sleeping", dict(
        file="10_occupant_sleeping_unsupported.jsonl", case_id="10", support="unsupported",
        name="Occupant Sleeping", match=_m_sleeping,
        purpose="Distinguish microsleep vs full sleep via eye-closure timing.",
        limitation="No eye-state/drowsiness/temporal labels. `sitting_still` is not sleeping -- deliberately not mapped.")),

    ("11_out_of_position", dict(
        file="11_out_of_position_candidate.jsonl", case_id="11", support="candidate",
        name="Out of Position (OOP)", match=_m_oop,
        purpose="Detect unsafe postures (head <20cm to airbag; feet on dash).",
        limitation="Reach/fetch/lean postures flagged as candidates. No 3D distance or precise-geometry GT, so exact <20cm / feet-position claims cannot be confirmed.")),

    ("12_left_objects", dict(
        file="12_left_objects_candidate.jsonl", case_id="12", support="candidate",
        name="Left Objects", match=_m_left_objects,
        purpose="Detect leftover items (phones/bags/parcels/keys) to suppress false chimes.",
        limitation="Drive&Act labels objects being ACTIVELY handled, not left/unattended. Good for object-presence recall; the 'left/unattended' state itself is not labeled.")),

    ("13_vehicle_door_status", dict(
        file="13_vehicle_door_status_candidate.jsonl", case_id="13", support="candidate",
        name="Vehicle Door Status", match=_m_door,
        purpose="Monitor open/ajar/closed door status.",
        limitation="Drive&Act gives door open/close ACTIONS, not a persistent open/ajar/closed STATE. Candidate.")),
])


def make_entry(record: dict, meta: dict, reason: str) -> dict:
    af = record.get("activity_facts", {}) or {}
    mid = (af.get("midlevel") or {}) or {}
    return {
        "sample_id": record.get("sample_id"),
        "image": (record.get("image") or {}).get("path"),
        "video_alias": record.get("video_alias"),
        "use_case": meta["case_id"],
        "case_name": meta["name"],
        "support_level": meta["support"],
        "matched_by": reason,
        "midlevel_activity": mid.get("activity"),
        "objectlevel": af.get("objectlevel"),
        "purpose": meta["purpose"],
        "limitation": meta["limitation"],
        "source_dataset": record.get("source_dataset", "DriveAct"),
        "gt_facts_ref": GT_FACTS_REF,
    }


def write_report(report_path: Path, counts: dict, examples: dict, total: int) -> None:
    tier_order = {"gt_supported": 0, "partial": 1, "candidate": 2, "unsupported": 3}
    lines = [
        "# Drive&Act -> 13 In-Cabin Use Cases: Coverage",
        "",
        f"Source: `{GT_FACTS_REF}`  ·  Total facts scanned: **{total}**",
        "",
        "Derived from the Drive&Act activity taxonomy (midlevel.activity + objectlevel).",
        "Same 01_..13_ scheme as the SVIRO buckets. A fact may land in multiple buckets.",
        "`unsupported` buckets are intentionally empty -- Drive&Act has no evidence for that claim.",
        "",
        "| # | Use case | Tier (Drive&Act) | Count |",
        "|---|---|---|---:|",
    ]
    for key, meta in BUCKETS.items():
        lines.append(f"| {meta['case_id']} | {meta['name']} | {meta['support']} | {counts.get(key, 0)} |")
    lines += [
        "",
        "## Tier meaning",
        "",
        "- **partial** — presence/recall supported, but the spec's exact requirement (e.g. stature percentile) is not labeled.",
        "- **candidate** — Drive&Act labels the ACTION, not the safety STATE the spec demands. Usable for recall; must not be asserted as a confirmed safety call.",
        "- **unsupported** — no Drive&Act evidence at all; needs another dataset (SVIRO or a new source).",
        "",
        "## Per-bucket notes & first examples",
        "",
    ]
    for key, meta in sorted(BUCKETS.items(), key=lambda kv: (tier_order[kv[1]["support"]], kv[1]["case_id"])):
        lines += [f"### {meta['case_id']} {meta['name']} — {meta['support']} ({counts.get(key,0)})",
                  "", f"File: `{meta['file']}`", "", meta["limitation"], ""]
        ex = examples.get(key) or []
        if not ex:
            lines += ["_No Drive&Act samples (empty by design)._", ""]
            continue
        lines += ["```json"] + [json.dumps(e, ensure_ascii=False) for e in ex[:3]] + ["```", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-facts", type=Path,
                   default=Path("1-gt_facts/driveact/driveact_a_column_co_driver_rgb_gt_facts.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("1-gt_facts/driveact/use_case_13"))
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args()
    report = args.report or (args.out_dir / "coverage_report.md")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    buckets = {k: [] for k in BUCKETS}
    total = 0

    with args.gt_facts.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            af = rec.get("activity_facts", {}) or {}
            mid = (af.get("midlevel") or {}).get("activity")
            obj = af.get("objectlevel")
            suited = set(rec.get("suited_cases", []))
            for key, meta in BUCKETS.items():
                matched, reason = meta["match"](mid, obj, suited)
                if matched:
                    buckets[key].append(make_entry(rec, meta, reason))

    counts, examples = {}, {}
    for key, meta in BUCKETS.items():
        rows = buckets[key]
        counts[key] = len(rows)
        examples[key] = rows[:3]
        with (args.out_dir / meta["file"]).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_report(report, counts, examples, total)

    covered = sum(1 for k, m in BUCKETS.items() if counts[k] > 0)
    print(json.dumps({
        "total_facts": total,
        "use_cases_with_driveact_data": covered,
        "use_cases_empty": 13 - covered,
        "counts": {BUCKETS[k]["case_id"] + " " + BUCKETS[k]["name"]: counts[k] for k in BUCKETS},
        "out_dir": str(args.out_dir),
        "report": str(report),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
