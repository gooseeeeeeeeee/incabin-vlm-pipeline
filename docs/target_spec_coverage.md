# Target-Spec Coverage: 13 In-Cabin Use Cases vs Current Data

Date: 2026-07-13 · Sources today: **SVIRO** (synthetic rear-seat occupancy / seat-states / objects / 2D keypoints)
and **DriveAct** (co-driver activity: coarse + fine actions).

Bottom line: the pipeline (grounded generation → cross-check → validate → ShareGPT → train → eval) can serve
**any** dataset. The gap to this spec is **data/labels**, not method. Below, each use case is tiered:
**Supported** (GT exists) · **Candidate** (recall only, GT can't confirm the exact claim) · **Unsupported** (needs a new dataset).

| # | Use case | Spec's hard requirement | Current GT source | Tier | What's missing (data needed) |
|---|---|---|---|---|---|
| 1 | Empty Seat | no cargo/pet/child-seat; monitor front row | SVIRO `seat=empty` (rear only) | **Supported** (rear) | front-row view; pet/cargo disambiguation |
| 2 | Adult Occupant | classify stature 5th/50th/95th percentile | SVIRO adult *presence* | Presence **Supported**; percentile **Unsupported** | anthropometric / stature-labeled occupants |
| 3 | Child in Forward-Facing Seat | confirm forward orientation for airbag logic | SVIRO `child_in_child_seat`, no orientation | **Candidate** | child-seat orientation labels |
| 4 | Rear-Facing Child Seat | confirm rear-facing → command airbag OFF | SVIRO, no orientation | **Candidate** (safety-critical) | orientation GT (critical — lethal if wrong) |
| 5 | Infant Recognition | incl. sleeping / playing / unrestrained | SVIRO `infant_in_infant_seat` only | **Partial** | unrestrained-infant + infant-state data |
| 6 | ISOFIX & Orientation | mechanical tension + flush placement | none | **Unsupported** | installation/mechanical labels or sensing |
| 7 | Pet Presence | dogs/cats vs humans | none | **Unsupported** | pet-in-cabin dataset |
| 8 | Seatbelt Worn | diagonal-belt routing over clavicle/sternum | DriveAct fasten *action* only | **Unsupported** (state/routing) | belt-routing-labeled data |
| 9 | Seatbelt Not Worn | misuse types + 30 s latency warning | none | **Unsupported** | belt-misuse + temporal data |
| 10 | Occupant Sleeping | microsleep vs full sleep, eye-closure timing | DriveAct activities, no eye state | **Unsupported** | face/eye + temporal drowsiness data |
| 11 | Out of Position | head <20 cm to airbag; feet-on-dash positions | SVIRO 2D keypoints; DriveAct OOP-candidate | **Candidate** (no 3D distance) | depth/3D + precise-geometry labels |
| 12 | Left Objects | phones/bags/parcels/keys → suppress chime | SVIRO `everyday_object` | **Supported** (presence); type-detail partial | object-type labels |
| 13 | Vehicle Door Status | open / ajar / closed in diverse lighting | DriveAct door activities | **Partial** (action, not state) | door-state-labeled data |

## Where we stand

- **Supported / mostly supported (4):** 1 empty, 2 adult-presence, 12 left-objects, and 13 door (as action).
- **Candidate only (3–4):** 3, 4, 6 (child-seat orientation / ISOFIX), 11 (OOP) — the model can *recall* these but GT can't *confirm* the safety-critical claim, so we phrase them as candidates and REJECT confident assertions.
- **Unsupported — needs new data (6+):** 2-stature-percentile, 4-airbag-critical-orientation, 6-ISOFIX-mechanical, 7-pet, 8-seatbelt-routing, 9-seatbelt-misuse, 10-sleep-timing, 11-precise-OOP.

## Why the reject discipline matters here

This spec is safety-critical (e.g. #4: wrong rear-facing call → "lethal hazard"; #8/#9 seatbelt; #11 airbag distance). Our pipeline **refuses** to claim these until GT supports them, instead of hallucinating — which is exactly the correct behavior for a safety system. The frozen-test "reject" metric (base 33% → tuned model ~100%) is evidence this discipline is learned, not just imposed.

## Roadmap to the full spec

The method scales; the work is **adding one dataset per gap**, each run through the same pipeline as a new GT-facts adapter (like DriveAct):

1. Per unsupported use case, source a dataset with **matching GT** (e.g. seatbelt-routing data for #8/#9; pet-in-cabin for #7; depth/3D for #11; drowsiness/eye for #10; stature-labeled for #2).
2. Build its GT-facts → generate grounded caption/QA (action/attribute-specific prompt) → cross-check → ShareGPT.
3. Joint-train and evaluate on a **matching frozen-test slice** — only claim a capability that is `gt_supported` AND has a test slice.
4. Prioritize by (safety-criticality × data feasibility): child-seat orientation (#3/#4) and seatbelt (#8/#9) are high-value; ISOFIX-tension (#6) and precise-OOP-distance (#11) likely need sensors beyond RGB.

Until then, the current model covers occupancy + object + driver-activity honestly, and rejects the rest — a sound, non-hallucinating foundation to extend one dataset at a time.
