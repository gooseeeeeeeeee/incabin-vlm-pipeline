#!/usr/bin/env python3
"""Build SVIRO GT Facts.

Principles:
- keep semantic labels instead of raw numeric label ids;
- keep full keypoint coordinates because they are factual geometry;
- map keypoint visibility ids to text;
- add lightweight person summaries and conservative rule-based pose flags;
- add suited_cases for prompt routing / data-generation eligibility;
- keep raw_refs for traceability.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


IMAGE_W = 960
IMAGE_H = 640

FILENAME_RE = re.compile(
    r"(?P<vehicle>x5_random)_(?P<split>train)_imageID_(?P<image_id>\d+)_GT_"
    r"(?P<left>\d+)_(?P<middle>\d+)_(?P<right>\d+)"
)

SEAT_LABELS = {
    0: "empty",
    1: "infant_in_infant_seat",
    2: "child_in_child_seat",
    3: "adult",
    4: "everyday_object",
    5: "empty_infant_seat",
    6: "empty_child_seat",
}

BBOX_LABELS = {
    0: "background",
    1: "infant_seat",
    2: "child_seat",
    3: "person",
    4: "everyday_object",
}

VISIBILITY_LABELS = {
    0: "outside_image",
    1: "occluded",
    2: "visible_or_locatable",
}

CORE_JOINTS = ["head", "neck_01", "spine_03", "pelvis"]

SUITED_CASE_META = {
    "01_empty_seat": {
        "case_name": "Empty Seat",
        "support_level": "gt_supported",
        "limitations": ["SVIRO seat-state GT covers rear seats only in the current subset."],
    },
    "02_adult_occupant": {
        "case_name": "Adult Occupant",
        "support_level": "gt_supported",
        "limitations": ["SVIRO does not provide adult stature percentile labels."],
    },
    "03_child_in_forward_facing_child_seat_candidates": {
        "case_name": "Child in Forward-Facing Child Seat Candidates",
        "support_level": "candidate",
        "limitations": ["SVIRO indicates child/infant seat state but does not directly label forward-facing orientation."],
    },
    "04_rear_facing_child_seat_candidates": {
        "case_name": "Rear-Facing Child Seat Candidates",
        "support_level": "candidate",
        "limitations": ["SVIRO indicates child/infant seat state but does not directly label rear-facing orientation."],
    },
    "05_infant_recognition": {
        "case_name": "Infant Recognition",
        "support_level": "gt_supported",
        "limitations": ["SVIRO infant recognition uses occupied infant seats only; empty_infant_seat is not an infant occupant."],
    },
    "06_isofix_child_seat_orientation_candidates": {
        "case_name": "ISOFIX & Child Seat Orientation Candidates",
        "support_level": "candidate",
        "limitations": ["SVIRO has no ISOFIX, tension, or flush-placement labels; this is child-seat presence/orientation eligibility only."],
    },
    "11_out_of_position_candidates": {
        "case_name": "Out of Position Candidates",
        "support_level": "candidate",
        "limitations": ["SVIRO does not directly label unsafe OOP status; pose flags are supporting geometry only."],
    },
    "12_left_objects": {
        "case_name": "Left Objects",
        "support_level": "gt_supported",
        "limitations": ["SVIRO everyday_object does not distinguish phone, bag, parcel, or key subtypes."],
    },
}

CHILD_SEAT_STATES = {"child_in_child_seat", "empty_child_seat"}
INFANT_SEAT_STATES = {"infant_in_infant_seat", "empty_infant_seat"}
SEAT_STATES_WITH_CHILD_OR_INFANT_SEAT = CHILD_SEAT_STATES | INFANT_SEAT_STATES
HUMAN_SEAT_STATES = {"adult", "child_in_child_seat", "infant_in_infant_seat"}


def parse_name(path_or_name: str) -> dict:
    name = Path(path_or_name).name
    match = FILENAME_RE.search(name)
    if not match:
        raise ValueError(f"Cannot parse SVIRO filename: {name}")
    groups = match.groupdict()
    left, middle, right = (int(groups["left"]), int(groups["middle"]), int(groups["right"]))
    return {
        "split": groups["split"],
        "image_id": int(groups["image_id"]),
        "seat_states": {
            "left": SEAT_LABELS[left],
            "middle": SEAT_LABELS[middle],
            "right": SEAT_LABELS[right],
        },
        "base": f"x5_random_train_imageID_{groups['image_id']}_GT_{left}_{middle}_{right}",
    }


def rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def parse_bbox_file(path: Path) -> list[dict]:
    objects = []
    if not path.exists():
        return objects
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 5:
            raise ValueError(f"Bad bbox line {path}:{line_no}: {raw}")
        class_id, x1, y1, x2, y2 = map(int, parts)
        objects.append(
            {
                "class_name": BBOX_LABELS.get(class_id, f"unknown_{class_id}"),
                "bbox_xyxy": [x1, y1, x2, y2],
            }
        )
    return objects


def convert_joint(point: list[int]) -> dict:
    x, y, visibility = point
    return {
        "x": x,
        "y": y,
        "visibility": VISIBILITY_LABELS.get(visibility, f"unknown_{visibility}"),
    }


def visibleish(joint: dict | None) -> bool:
    return bool(joint and joint.get("visibility") == "visible_or_locatable")


def pose_flags(joints: dict) -> list[dict]:
    flags = []
    head = joints.get("head")
    neck = joints.get("neck_01")
    spine = joints.get("spine_03")
    pelvis = joints.get("pelvis")

    if visibleish(head) and visibleish(spine) and head["y"] > spine["y"] + 10:
        flags.append(
            {
                "flag": "head_below_spine_03",
                "meaning": "Head keypoint is lower than upper-spine keypoint in image coordinates.",
                "evidence": {"head_y": head["y"], "spine_03_y": spine["y"]},
            }
        )

    if visibleish(head) and visibleish(pelvis) and head["y"] > pelvis["y"] + 20:
        flags.append(
            {
                "flag": "head_below_pelvis",
                "meaning": "Head keypoint is lower than pelvis keypoint in image coordinates.",
                "evidence": {"head_y": head["y"], "pelvis_y": pelvis["y"]},
            }
        )

    if all(visibleish(j) for j in [head, neck, spine, pelvis]):
        if not (head["y"] < neck["y"] < spine["y"] < pelvis["y"]):
            flags.append(
                {
                    "flag": "core_order_not_upright",
                    "meaning": "Head, neck, spine, and pelvis do not follow a typical upright top-to-bottom order.",
                    "evidence": {
                        "head_y": head["y"],
                        "neck_01_y": neck["y"],
                        "spine_03_y": spine["y"],
                        "pelvis_y": pelvis["y"],
                    },
                }
            )

    outside_feet = [
        name
        for name in ["foot_l", "foot_r", "ball_l", "ball_r"]
        if joints.get(name, {}).get("visibility") == "outside_image"
    ]
    if len(outside_feet) >= 2:
        flags.append(
            {
                "flag": "lower_extremities_outside_image",
                "meaning": "Multiple foot/ball keypoints are outside the image; lower body geometry is incomplete.",
                "evidence": {"outside_joints": outside_feet},
            }
        )

    visible_core = sum(1 for name in CORE_JOINTS if visibleish(joints.get(name)))
    if visible_core <= 1:
        flags.append(
            {
                "flag": "core_joints_mostly_not_visible",
                "meaning": "Most core body joints are occluded or outside the image, so pose reasoning is less reliable.",
                "evidence": {"visible_or_locatable_core_joints": visible_core, "core_joint_count": len(CORE_JOINTS)},
            }
        )

    return flags


def matching_seat_evidence(seat_states: dict, states: set[str]) -> list[str]:
    return [
        f"seat_states.{seat}={state}"
        for seat, state in seat_states.items()
        if state in states
    ]


def has_person_fact(record: dict) -> bool:
    return (
        any(state in HUMAN_SEAT_STATES for state in record["seat_states"].values())
        or record.get("object_counts", {}).get("person", 0) > 0
        or bool(record.get("persons_summary"))
    )


def add_suited_case(cases: list[dict], case_id: str, evidence: list[str], extra_limitations: list[str] | None = None) -> None:
    meta = SUITED_CASE_META[case_id]
    limitations = list(meta["limitations"])
    if extra_limitations:
        limitations.extend(extra_limitations)
    cases.append(
        {
            "case_id": case_id,
            "case_name": meta["case_name"],
            "support_level": meta["support_level"],
            "purpose": "prompt_routing",
            "evidence": evidence,
            "limitations": limitations,
        }
    )


def build_suited_cases(record: dict) -> list[dict]:
    seat_states = record["seat_states"]
    object_counts = record.get("object_counts", {})
    persons_summary = record.get("persons_summary", [])
    cases: list[dict] = []

    empty_evidence = matching_seat_evidence(seat_states, {"empty"})
    if empty_evidence:
        add_suited_case(cases, "01_empty_seat", empty_evidence)

    adult_evidence = matching_seat_evidence(seat_states, {"adult"})
    if adult_evidence:
        evidence = adult_evidence[:]
        if object_counts.get("person", 0) > 0:
            evidence.append(f"object_counts.person={object_counts['person']}")
        add_suited_case(cases, "02_adult_occupant", evidence)

    child_infant_seat_evidence = matching_seat_evidence(seat_states, SEAT_STATES_WITH_CHILD_OR_INFANT_SEAT)
    if child_infant_seat_evidence:
        add_suited_case(cases, "03_child_in_forward_facing_child_seat_candidates", child_infant_seat_evidence)
        add_suited_case(cases, "04_rear_facing_child_seat_candidates", child_infant_seat_evidence)
        add_suited_case(cases, "06_isofix_child_seat_orientation_candidates", child_infant_seat_evidence)

    infant_evidence = matching_seat_evidence(seat_states, {"infant_in_infant_seat"})
    if infant_evidence:
        add_suited_case(cases, "05_infant_recognition", infant_evidence)

    if has_person_fact(record):
        evidence = []
        human_seats = matching_seat_evidence(seat_states, HUMAN_SEAT_STATES)
        evidence.extend(human_seats)
        if object_counts.get("person", 0) > 0:
            evidence.append(f"object_counts.person={object_counts['person']}")
        if persons_summary:
            person_refs = ", ".join(
                f"{person.get('person_id')}@{person.get('position')}"
                for person in persons_summary
            )
            evidence.append(f"persons_summary={person_refs}")
        add_suited_case(cases, "11_out_of_position_candidates", evidence)

    object_evidence = matching_seat_evidence(seat_states, {"everyday_object"})
    if object_counts.get("everyday_object", 0) > 0:
        object_evidence.append(f"object_counts.everyday_object={object_counts['everyday_object']}")
    if object_evidence:
        add_suited_case(cases, "12_left_objects", object_evidence)

    return cases


def parse_pose_file(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    data = json.loads(path.read_text(encoding="utf-8"))
    keypoints = []
    summaries = []
    for person_id, person in data.items():
        raw_bones = person.get("bones", {})
        joints = {name: convert_joint(value) for name, value in raw_bones.items()}
        visibility_counts = Counter(j["visibility"] for j in joints.values())
        counts = {
            "outside_image": visibility_counts.get("outside_image", 0),
            "occluded": visibility_counts.get("occluded", 0),
            "visible_or_locatable": visibility_counts.get("visible_or_locatable", 0),
        }
        flags = pose_flags(joints)
        keypoints.append(
            {
                "person_id": person_id,
                "position": person.get("position"),
                "joints": joints,
                "visibility_counts": counts,
                "pose_flags": flags,
            }
        )
        summaries.append(
            {
                "person_id": person_id,
                "position": person.get("position"),
                "num_joints": len(joints),
                "visibility_counts": counts,
                "core_joints": {
                    name: joints[name]["visibility"]
                    for name in CORE_JOINTS
                    if name in joints
                },
                "pose_flags": [flag["flag"] for flag in flags],
            }
        )
    return keypoints, summaries


def build_record(image_path: Path, project_root: Path, bbox_dir: Path, pose_dir: Path) -> dict:
    parsed = parse_name(image_path.name)
    base = parsed["base"]
    bbox_path = bbox_dir / f"{base}.txt"
    pose_path = pose_dir / f"{base}_pose.json"
    objects = parse_bbox_file(bbox_path)
    object_counts = dict(sorted(Counter(obj["class_name"] for obj in objects).items()))
    keypoints, persons_summary = parse_pose_file(pose_path)
    record = {
        "sample_id": f"sviro_bmw_x5_random_train_{parsed['image_id']:06d}",
        "source_dataset": "SVIRO",
        "subset": "bmw_x5_random",
        "split": parsed["split"],
        "image_id": parsed["image_id"],
        "image": {
            "path": rel(image_path, project_root),
            "width": IMAGE_W,
            "height": IMAGE_H,
        },
        "seat_states": parsed["seat_states"],
        "objects": objects,
        "object_counts": object_counts,
        "keypoints": keypoints,
        "persons_summary": persons_summary,
        "raw_refs": {
            "rgb": rel(image_path, project_root),
            "bbox": rel(bbox_path, project_root),
            "keypoints": rel(pose_path, project_root),
        },
    }
    record["suited_cases"] = build_suited_cases(record)
    return record


def special_score(record: dict) -> tuple[int, int, int]:
    flags = [flag for person in record["persons_summary"] for flag in person["pose_flags"]]
    core_unusual = sum(
        1 for flag in flags if flag in {"head_below_spine_03", "head_below_pelvis", "core_order_not_upright"}
    )
    all_flags = len(flags)
    occupants = sum(1 for state in record["seat_states"].values() if state not in {"empty"})
    return core_unusual, all_flags, occupants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--rgb-dir", type=Path, required=True)
    parser.add_argument("--bbox-dir", type=Path, required=True)
    parser.add_argument("--pose-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    args = parser.parse_args()

    images = sorted(args.rgb_dir.glob("*.png"), key=lambda p: parse_name(p.name)["image_id"])
    records = [
        build_record(image, args.project_root, args.bbox_dir, args.pose_dir)
        for image in images
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    special = [record for record in records if special_score(record)[0] > 0]
    if len(special) < args.sample_size:
        special = [record for record in records if special_score(record)[1] > 0]
    selected = sorted(special or records, key=special_score, reverse=True)[: args.sample_size]
    with args.sample_out.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    all_flags = Counter(
        flag
        for record in records
        for person in record["persons_summary"]
        for flag in person["pose_flags"]
    )
    suited_case_counts = Counter(
        case["case_id"]
        for record in records
        for case in record["suited_cases"]
    )
    suited_support_counts = Counter(
        case["support_level"]
        for record in records
        for case in record["suited_cases"]
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "sample_records": len(selected),
                "records_with_pose_flags": sum(1 for record in records if special_score(record)[1] > 0),
                "records_with_core_unusual_flags": sum(1 for record in records if special_score(record)[0] > 0),
                "pose_flag_counts": dict(sorted(all_flags.items())),
                "suited_case_counts": dict(sorted(suited_case_counts.items())),
                "suited_support_counts": dict(sorted(suited_support_counts.items())),
                "out": str(args.out),
                "sample_out": str(args.sample_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
