# Public Datasets for Uncovered Use Cases / 未覆盖用例的公开数据集查漏补缺

Date: 2026-07-19 · Companion to `target_spec_coverage.md` and
`1-gt_facts/driveact/use_case_13/coverage_report.md`.

Context: Drive&Act (co-driver RGB activity) covers **6/13** use cases at
`partial`/`candidate` level (02 presence, 08/09 belt *action*, 11 OOP, 12 objects,
13 door *action*), and leaves 7 with **no evidence**. This doc lists concrete,
mostly-public datasets to close each gap, plus an honest note where **no public
dataset exists** and the signal really needs a different sensor (CAN bus / anchor
switch / radar), not more vision data.

> Recommendation legend: **[strong]** direct labels for the exact claim ·
> **[partial]** related labels, needs adaptation · **[synth]** synthetic /
> render-your-own · **[none]** no public dataset — sensor or custom-labeling problem.

---

## #2 Adult Occupant — stature 5th/50th/95th percentile

Drive&Act gives adult *presence* only. No public **image** dataset ships
5th/50th/95th stature-percentile labels; this is usually solved with synthetic
rendering over a controlled anthropometry distribution.

- **[partial] TICaM** — real + synthetic front-cabin, `person` class with 2D/3D
  boxes and pose; body size is derivable from 3D pose, not a percentile label.
  https://vizta-tof.kl.dfki.de/cabin-dataset/
- **[synth] Synthetic render pipelines** (SkyEngine, Anyverse) parametrize
  occupant height/weight directly — the practical way to get percentile labels.
  https://www.skyengine.ai/use-cases/synthetic-data-for-next-gen-driver-monitoring-systems ·
  https://anyverse.ai/in-cabin-monitoring-overview/
- **[reference] Label definition, not images**: FMVSS §571.208 percentile classes
  (5th female / 50th male / 95th male) and NHANES / UMTRI anthropometry give the
  ground-truth size bins to render/label against.
  https://mreed.umtri.umich.edu/SAE-GI-2024/UMTRI%20Occupant%20Anthro%20SAE%20GI%20Presentation.4.pdf ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10448517/

**Verdict:** best path = SMPL-body synthetic render conditioned on NHANES/UMTRI
percentiles. No off-the-shelf public image set with the percentile label.

## #3 / #4 Child in Forward-Facing & Rear-Facing Child Seat (orientation)

This is the highest-value + most-attainable gap — a real public dataset covers it.

- **[strong] TICaM** — real driving-simulator images that explicitly include
  **child seats with `ff` (forward-facing) and `rf` (rear-facing)** object classes,
  plus `child` and `infant`, in depth/RGB/IR. This directly supplies the
  orientation label the airbag logic (#4) needs.
  https://ar5iv.labs.arxiv.org/html/2103.11719 · https://vizta-tof.kl.dfki.de/cabin-dataset/
- **[synth] SVIRO (already in pipeline)** — the *generator* randomly rotates infant
  seats 180° and offsets child-seat yaw (handle up/down), so orientation exists in
  the raw synthetic scene even though the released classification labels don't
  expose it. Re-rendering with orientation exported would upgrade 03/04 from
  `candidate` to `gt_supported`. https://sviro.kl.dfki.de/data/
- **[synth] TICaM synthetic** — same taxonomy, unlimited variations.
  https://datasetninja.com/ticam-synthetic-images

**Verdict:** adopt **TICaM** as the orientation GT source; optionally re-export
SVIRO orientation. This is safety-critical (#4 = "lethal if wrong"), so keep the
reject-until-GT discipline until a matching frozen-test slice exists.

## #5 Infant Recognition (incl. unrestrained / sleeping / playing)

- **[strong] TICaM** — has `infant` and `child` classes (recorded with dolls),
  including out-of-seat scenes. https://vizta-tof.kl.dfki.de/cabin-dataset/
- **[synth] SVIRO** — `infant_in_infant_seat` (already used) but only in-seat.
  Unrestrained-infant states need TICaM or new renders.

**Verdict:** TICaM covers infant presence + some unrestrained cases; infant
*state* (sleeping/playing) still thin.

## #6 ISOFIX & Child-Seat Orientation (mechanical tension / flush placement)

- **[none]** No public dataset — and largely **not an RGB problem**. ISOFIX
  install correctness (latch engagement, anchor tension, flush seatback contact)
  is defined by the ISO 13216 mechanical standard and is best sensed by anchor /
  tension sensors, not a camera. https://www.iso.org/isofix-child-seats-iso-13216.html
- Vision can at most infer *visible* base flushness (borrow orientation frames from
  TICaM/SVIRO); the "mechanical tension" requirement is out of scope for vision GT.

**Verdict:** genuine sensor gap. Don't promise a vision label for tension.

## #7 Pet Presence (dogs / cats)

- **[none] in-cabin pets**: no public dataset places real dogs/cats inside a car
  cabin. In-cabin pet detection appears only in patents, not open data.
- **[partial] general animal detection** to composite/fine-tune from:
  Oxford-IIIT Pet (37 breeds, boxes) https://public.roboflow.com/object-detection/oxford-pets ·
  Cat/Dog YOLO https://www.kaggle.com/datasets/hemanthganeshvilluri/cats-dogs-yolo ·
  Animal-Pose (dogs/cats keypoints) for pose.

**Verdict:** best path = composite Oxford-Pets animals into cabin backgrounds, or
render synthetically. No ready-made in-cabin pet set.

## #8 / #9 Seatbelt Worn / Not-Worn — routing & misuse

Drive&Act has the fasten/unfasten *action* only. For the **worn-state + diagonal
routing over clavicle/sternum** (#8) and **misuse categories** (#9):

- **[strong] Neuromorphic Seatbelt State Detection** — 108,691 synthetic
  neuromorphic (event-camera) frames from an NIR base, labeled
  fastened/unfastened, F1 0.99 sim / 0.94 real. Direct worn/not-worn state.
  https://arxiv.org/pdf/2308.07802
- **[partial] "Robust Seatbelt Detection and Usage Recognition for DMS"** (Hu et
  al.) — a fully-annotated in-cabin seatbelt-usage dataset (availability = ask
  authors). https://www.semanticscholar.org/paper/5352941fb9983032baf4e91a3135a4ceba17069a
- **[partial] TICaM** — front-cabin persons with belts visible; usable for
  routing-segmentation labeling.

**Gap that remains:** the specific **misuse taxonomy** (buckle-only / behind-back /
lap-only) and the **30-second latency** temporal label — no public set has these;
would need custom annotation on top of the above.

## #10 Occupant Sleeping — microsleep vs full sleep (eye-closure timing)

Well-served publicly; pick by whether you need physiological ground truth.

- **[strong] NTHU-DDD** — 36 subjects, day/night, glasses/sunglasses; **per-frame
  eye status (normal/sleepy)**, head pose, mouth (yawn) — ideal for microsleep vs
  sleep timing. http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- **[strong] ULg DROZY** — 14 subjects, KSS + PVT physiological drowsiness ground
  truth alongside IR video. (search: "ULg DROZY database")
- **[partial] DMD** (Vicomtech) — 41 h, includes a drowsiness subset with
  eye/blink. https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- **[partial] UTA-RLDD** — 60-subject real-life drowsiness videos.

**Verdict:** NTHU-DDD (+DROZY for physiological GT) closes #10 cleanly. Note both
are driver-facing; co-driver sleep would need re-view/render.

## #11 Out of Position — precise head-to-airbag distance / feet-on-dash

Drive&Act OOP is a coarse posture candidate with no depth. For the **<20 cm to
airbag** metric you need 3D:

- **[strong] TICaM** — ToF **depth** + 2D/3D detection + pose → real metric
  distance. https://vizta-tof.kl.dfki.de/cabin-dataset/
- **[strong] DMD** — RGB+**depth**+IR, body pose from 3 views.
  https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- **[partial] SVIRO (already in pipeline)** — synthetic depth + 2D keypoints;
  metric distance derivable in synthetic space.

**Verdict:** move #11 to depth-based GT (TICaM / DMD) to confirm the geometric
claim instead of recall-only.

## #13 Vehicle Door Status — open / ajar / closed

- **[none] as a vision dataset**: door **state** is normally a vehicle signal
  (door-ajar switch / CAN bus), not a camera-labeled dataset. Drive&Act only labels
  the door open/close *action* (209 frames), which is what we have.
- If a vision approach is required, the exterior-door datasets don't map to cabin
  view; this would need custom in-cabin labeling of door-gap state.

**Verdict:** treat door open/ajar/closed as a CAN-bus/sensor signal; vision GT for
it is a custom-labeling task, not an available public set.

---

## Priority (safety-criticality × data feasibility)

1. **#3/#4 child-seat orientation** — **TICaM has direct `ff`/`rf` labels**;
   highest safety value, immediately actionable. → adopt TICaM.
2. **#10 sleeping** — **NTHU-DDD / DROZY** are ready and public. → adopt.
3. **#11 precise OOP** — **TICaM / DMD depth** upgrades recall→geometric. → adopt.
4. **#8/#9 seatbelt state** — neuromorphic-seatbelt set for worn/not-worn; misuse
   taxonomy + 30 s latency still need custom labels.
5. **#2 stature percentile** — synthetic render on NHANES/UMTRI bins (no public
   image set).
6. **#7 pet** — composite Oxford-Pets/Animal-Pose into cabin (no in-cabin set).
7. **#6 ISOFIX tension** & **#13 door state** — **not vision problems**; sensor /
   CAN-bus signals. Don't promise a vision label.

**Single best acquisition:** **TICaM** alone touches #3, #4, #5, #11 (and helps
#2/#8) in one dataset with matching modalities to the current pipeline — run it
through the same GT-facts→generate→cross-check→ShareGPT adapter as Drive&Act.

## Sources

- TICaM — https://ar5iv.labs.arxiv.org/html/2103.11719 · https://vizta-tof.kl.dfki.de/cabin-dataset/
- SVIRO — https://sviro.kl.dfki.de/data/ · https://openaccess.thecvf.com/content_WACV_2020/papers/Da_Cruz_SVIRO_Synthetic_Vehicle_Interior_Rear_Seat_Occupancy_Dataset_and_Benchmark_WACV_2020_paper.pdf
- NTHU-DDD — http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/
- DMD — https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset
- Neuromorphic Seatbelt State Detection — https://arxiv.org/pdf/2308.07802
- Robust Seatbelt Detection (Hu et al.) — https://www.semanticscholar.org/paper/5352941fb9983032baf4e91a3135a4ceba17069a
- Occupant anthropometry (UMTRI / NHANES / adaptive restraint) — https://mreed.umtri.umich.edu/SAE-GI-2024/UMTRI%20Occupant%20Anthro%20SAE%20GI%20Presentation.4.pdf · https://pmc.ncbi.nlm.nih.gov/articles/PMC10448517/
- Oxford-IIIT Pet — https://public.roboflow.com/object-detection/oxford-pets
- ISO 13216 (ISOFIX) — https://www.iso.org/isofix-child-seats-iso-13216.html
- Synthetic pipelines — https://www.skyengine.ai/use-cases/synthetic-data-for-next-gen-driver-monitoring-systems · https://anyverse.ai/in-cabin-monitoring-overview/
