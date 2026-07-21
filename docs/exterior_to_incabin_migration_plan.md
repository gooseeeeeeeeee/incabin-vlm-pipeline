# Exterior → In-Cabin Pipeline Migration Plan / 车外流水线迁移到车内的方案

- Date / 日期: 2026-07-03
- Reference / 参考: `Justin0504/Visteon-CabinVLMBenchmark` (`caption_pipeline/PROMPTS.md`, `README.md`), `6-docs/exterior_reuse_for_incabin.md`, `2-generation/PLAN_COT_QA.md`, `2-generation/quality/sviro_pilot_review.md`
- Method / 方法: for each exterior stage — **thesis** (reuse as-is), **antithesis** (why a naive transplant breaks in-cabin), **synthesis** (the adapted design). 对每一环做正—反—合。

---

## 0. The two differences that drive every decision / 决定一切的两点差异

1. **GT nature.** Exterior GT is 3D/metric (class, count, distance, bbox, camera role) → spatial-metric QA and a temporal `prediction` are well-founded. In-cabin GT is **categorical + 2D** (seat occupancy states, 2D bbox, 2D keypoints, pose_flags), single rear-seat view, near-static scene.
2. **Claim surface.** Exterior "reject = not visible" is binary. In-cabin adds a **three-tier support model** (`gt_supported` / `candidate` / `unsupported`) and a set of **visually tempting but forbidden** claims (seatbelt, age, gender, sleeping, seat orientation, ISOFIX).

Three non-obvious consequences (logic, not taste):

- **C1 — Invert the hallucination-risk profile.** Exterior must *ask a vision model to count objects' context* because counts come from sensors the model can't see; the risky part is fusion. In-cabin, **almost every fact (occupancy, counts, object class, seat state, posture geometry) is already categorical GT** → we should lean *more* on deterministic generation for facts and use the LLM only for natural language + QA diversity. Net hallucination surface is *smaller* than exterior — if we exploit it.
- **C2 — `prediction` is the weakest transplant.** A rear-seat frame does not evolve in 1–3 s the way traffic does; SVIRO is a static synthetic frame. A temporal intent link is mostly ungrounded → drop it, optionally replace with a non-temporal `attention` (which occupant to keep monitoring).
- **C3 — The VISION stage is higher-risk and lower-value in-cabin.** A vision model looking at a cabin is *tempted* to assert exactly the forbidden labels (belt, age, asleep), and SVIRO's synthetic single-cabin gives it little unique context to add. So the VISION stage must be **guarded and optional for SVIRO**, mandatory only for future real, varied cabin data.

---

## 1. Stage-by-stage migration / 逐环迁移

### Stage 0 — GT extraction & GT Facts  → **REUSE (already built)**
- Thesis: exterior grounds on nuScenes-devkit; in-cabin grounds on SVIRO parsers. Same principle: authoritative facts first.
- Synthesis: **done** — `1-gt_facts/*.jsonl` with `seat_states`, `objects`, `object_counts`, `keypoints`, `persons_summary`, `suited_cases` (3-tier). One fix pending: **P1 taxonomy** — add a `gt_supported` "child occupant present" concept so a supported fact isn't forced onto candidate case ids (`03/04/06`). Update `GT_FACTS_SCHEMA.md`.

### Stage A — VISION context (image → context only)  → **ADAPT + GUARD (optional for SVIRO)**
- Thesis: reuse the exterior split — VISION supplies context labels lack; counts/classes stay GT.
- Antithesis: in-cabin VISION is tempted into forbidden claims (belt/age/asleep/orientation) and adds little on a synthetic single cabin (C3).
- Synthesis: keep the VISION/GT split but (a) **hard negative list** in the system prompt (never mention belt, age, gender, emotion, identity, pet, door, sleep, orientation, ISOFIX; never count persons — counts are GT); (b) restrict output to *cabin appearance only* (lighting, general layout, occlusion appearance, upholstery/scene); (c) **make the stage a switch** — off for the SVIRO pilot, on for future real cabin datasets.

### Stage B1 — Caption fusion (GT [+VISION] → causal caption)  → **ADAPT**
- Thesis: reuse chain-of-causation.
- Antithesis: (i) exterior's `scene→risk→decision→prediction` has a temporal `prediction` that in-cabin can't ground (C2); (ii) exterior A/B proved **labeled captions score *lower* on the judge (6.45 vs 7.65)** than prose — but a safety system may want to *parse* state/risk/decision.
- Synthesis: internal schema `{state_scene, risk, decision, (optional) attention}` (drop `prediction`); **the model thinks in JSON, but the ingested training caption is emitted as label-free prose** (resolves review item **G1**). Keep the JSON in `raw_outputs` for QC/provenance and any downstream parser. Candidate cases must read as candidates. `risk`/`decision` drawn only from GT-supported facts; conservative decision vocabulary (`keep monitoring`, `verify before driving`, `manual/VLM review needed`, `no action needed`).

### Stage B2 — QA generation  → **ADAPT (adopt exterior rigor, in-cabin capabilities)**
- Thesis: reuse exterior QA rules — diverse perspectives, ≥N capabilities, exactly one reject, every answer carries a `reason`, driver/system voice, no "in the image".
- Antithesis: exterior capabilities (counting/distance/traffic-intent) are exterior-specific; in-cabin needs occupancy/posture/candidate handling; and much of it is deterministic.
- Synthesis: capability set = `occupancy_recognition`, `object_recognition`, `spatial_seat` (left/middle/right), `posture_geometry` (from keypoints/pose_flags, **non-diagnostic**), `risk_reasoning`, `decision`, `reject`, `candidate_review`. Rules kept from exterior: exactly **one forced reject** per record; **each answer carries `reason`**; vary Why/What/Where/Which/How-many/Is-Can. **Split of labor (C1):** facts (occupancy/counts/object/seat) generated **deterministically**; the LLM only rephrases for naturalness + adds diversity; then cross-check. Fixes review items **P2–P6** (context-aware reject, reject-evidence convention, real case-id mapping, diversity, `reason`).

### Stage C — Cross-check gate  → **REUSE (critical) + EXTEND**
- Thesis: reuse the second-model fact-checker (exterior lifted faithfulness **53%→80%**).
- Synthesis: two layers. (1) **Deterministic QC first** (already have it in the pilot review script): seat-state evidence resolves, `suited_cases` match, no forbidden label outside reject, candidate phrased as candidate. (2) **LLM cross-check second** (only when an LLM generator is used): strip any claim contradicting GT occupancy/counts/classes, **any forbidden-label claim**, and **any candidate stated as confirmed**. This fuses exterior's faithfulness gate with in-cabin's anti-hallucination + candidate-guard.

### Stage D — Deterministic-from-GT extraction  → **REUSE pattern, larger role**
- Exterior did this only for VRU separation. In-cabin, occupancy/counts/object/seat-state/pose-flags are *all* deterministic → this pattern carries a **much larger share** of in-cabin generation (C1). Never ask a model to re-count or re-classify what GT already states.

### Stage E — ShareGPT export  → **REUSE 1:1**
- `conversations` + `images` only; caption prose first, then QA turns; all metadata (seat labels, bbox, keypoints, suited_cases, prompt_version) stays in `raw_outputs`/manifest. Matches `specs/SHAREGPT_FORMAT_SPEC.md`.

### Stage F — Splits + Frozen test  → **REUSE, restratify**
- Replace exterior's per-camera/per-category stratification with in-cabin strata: **seat-state combination, occupant presence, object presence, occlusion/pose-flag difficulty**. Frozen test never trained. Objective accuracy for `gt_supported` questions (occupancy/object/reject); judge score for open captions. **Candidate cases are NOT scored as objective labels** — only recall/needs-review (this is the in-cabin-specific eval rule).

### Stage G — Training + Evaluation  → **REUSE**
- Same base family (Qwen2.5-VL-7B, LoRA) and multi-version frozen eval. Carry exterior's two lessons: **caption quality ↑ ⇒ model ↑**, and **more data ≠ better unless the frozen test has a matching slice** → only claim a capability if it is `gt_supported` *and* has a matching test slice.

### Not migrated (exterior-specific)  → **DROP**
- POI RAG + OSM/Wikipedia verify, traffic-light extraction, 3D distance / six-camera mapping, weather/lane/road-surface prompting, Stanford-Cars/GTSRB/TextVQA/SUN397 build logic. Camera-role mapping → **replaced** by seat-zone (left/middle/right) mapping. (Consistent with `exterior_reuse_for_incabin.md §3.)

---

## 2. Target architecture (one line) / 目标架构

```
SVIRO raw
  → [0] GT Facts (deterministic, 3-tier suited_cases)
  → [A] VISION context (guarded, OPTIONAL for SVIRO)
  → [B1] caption: think in {state_scene,risk,decision,attention?} JSON, EMIT prose
  → [B2] QA: deterministic facts + LLM diversity, 1 forced reject, reason on every answer
  → [C] cross-check: deterministic QC + (LLM) GT/forbidden/candidate guard
  → [E] ShareGPT (conversations+images)
  → [F] splits + frozen test (restratified; candidates = recall only)
  → [G] LoRA fine-tune + multi-version frozen eval (matching slices only)
```

Difference from exterior in one sentence: **facts move from "vision-guessed, GT-corrected" to "GT-first, LLM only for language"; the temporal `prediction` link is dropped; a `candidate` tier and a strict forbidden-label guard are added end to end.**

---

## 3. Sequencing / 实施顺序（按风险与依赖排序）

**Phase 1 — no LLM/GPU needed, do now.** Fix P1 taxonomy; upgrade the deterministic generator to: emit prose caption + keep internal JSON, diverse QA with `reason`, context-aware reject, real case-id mapping (fixes P2–P6). Re-run 30, pass the deterministic QC. → a clean, fully-grounded, model-independent pilot.

**Phase 2 — needs an LLM/VLM served on the AutoDL GPU.** Add guarded VISION (optional), LLM caption/QA rendering for naturalness + diversity, and the LLM cross-check gate. A/B the LLM output vs the Phase-1 deterministic baseline on a small judged set (adopt exterior's judge protocol).

**Phase 3 — scale + train.** Export ShareGPT, build frozen test (restratified), LoRA fine-tune, multi-version eval per use case. Only after Phase-2 faithfulness is stable. Do **not** generate all ~2000 before this.

---

## 4. Decisions I recommend (state, don't ask) / 我的推荐判断

1. Caption: **prose external, JSON internal.** (Follow exterior's tested result; keep JSON because a safety monitor may parse it.)
2. **Drop `prediction`;** optionally add non-temporal `attention`.
3. **VISION stage off for SVIRO pilot,** on for future real cabin data.
4. **Lean deterministic for facts,** LLM for language/diversity + cross-check (inverts exterior's emphasis).
5. **`candidate` is first-class** in prompt, cross-check, and eval; candidates are never scored as objective labels.
6. Only claim a capability that is **`gt_supported` and has a matching frozen-test slice.**

---

## 5. 中文小结

把车外流水线整体搬过来，骨架不变（GT→caption→QA→交叉核查→ShareGPT→冻结测试→训练评测），但针对车内做三处关键改动：

1. **事实生成方式反过来**：车外是"视觉猜、GT 纠"，车内的占用/计数/座位/姿态几乎全是类别型 GT，应该"GT 优先、确定性生成事实，LLM 只负责自然语言和多样性"，幻觉面反而更小。
2. **砍掉 `prediction`**：车内是近静态单帧，1-3s 意图预测缺乏依据；如需要可换成非时序的 `attention`（该盯哪个乘员）。
3. **补两样车外没有的东西**：三级支持里的 **candidate（待复核）**要在 prompt/核查/评测里都当一等公民（candidate 永远不当客观标签打分）；以及对安全带/年龄/性别/睡眠/朝向等**视觉可猜但禁止断言**标签的端到端硬性拦截（VISION 阶段因此更危险、对 SVIRO 价值又低，所以设为可选、默认关）。

Caption 格式采纳车外实测结论：**内部按 JSON schema 思考、最终输出散文**，但保留 JSON 供安全系统解析。VISION 阶段对 SVIRO 先关、对未来真实车内数据再开。先做 Phase 1（不依赖模型、纯确定性、修完 P1–P6 重跑 30 条），再上 LLM 生成和交叉核查。
