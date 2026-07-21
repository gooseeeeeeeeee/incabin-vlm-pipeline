# Model v2 Summary — Data, Evaluation Method, Results

Date: 2026-07-13 · Model: `Qwen3-VL-8B-Instruct` + LoRA (`4-training/sviro_driveact_lora_v2`)

## 1. Training data

| Source | Domain | GT rows available | Generated (raw) | Clean | **Used in v2 train** | Category coverage |
|---|---|---:|---:|---:|---:|---|
| **SVIRO** (`bmw_x5_random`, synthetic rear-seat) | occupancy / seats / objects | 2 000 | 1 840 (train+val) → 1 777 ok | 1 761 | **1 620** (train split only) | 7 seat states; 4 gt_supported use cases (01 empty, 02 adult, 05 infant, 12 left-objects) |
| **DriveAct** (`a_column_co_driver`, RGB) | driver activity / object interaction | 12 743 | 3 181 (balanced, cap 120/activity) → 3 177 ok | 2 908 | **2 908** | **100 % of categories**: 39 mid-level activities, 7 object-actions, 18 objects, 15 locations, 4 suited_cases |
| **Total** | | | | | **4 528 records / 54 336 turns** | |

Held out (never trained): SVIRO `frozen_test` **160** samples + `val_dev` **150**.
DriveAct currently has **no held-out benchmark** — its activity capability is not yet measured.

## 2. Model / training

Qwen3-VL-8B-Instruct, LoRA **rank 8 / α 16 / dropout 0.05 / target=all**, vision tower frozen;
2 epochs (~1 130 steps), effective batch 8, lr 1e-4 cosine, bf16 + gradient checkpointing,
image cap 512²; RTX 4090, **1 h 07 min**; train_loss 0.5608 (final steps ≈ 0.38, converged).

## 3. Evaluation method

- **Set**: SVIRO `frozen_test` — 160 samples never seen in training.
- **Questions**: 6 per sample = **960 objective questions**, ground truth derived *deterministically* from SVIRO GT facts (no LLM judging):
  - `occupancy` ×3 (one per seat: left/middle/right, 6-way category)
  - `count` ×1 (number of persons — integer match)
  - `yesno` ×1 (is an everyday object left on a seat)
  - `reject` ×1 (seatbelt status → the model **must refuse**, since SVIRO has no seatbelt GT)
- **Scoring**: greedy decoding, identical prompts for base and LoRA. Occupancy is scored at **three levels**:
  - **FINE** — exact 6-way category;
  - **COARSE** — safety grouping {unoccupied / child / adult / object}, i.e. "empty vs empty-child-seat" and "infant vs child" count as correct;
  - **SAFETY** — *missed occupant* (GT has a person, model says unoccupied) and *hallucinated occupant*.

## 4. Results — base vs v2

| Metric | base Qwen3-VL-8B | **v2 (LoRA)** | Δ |
|---|---:|---:|---:|
| occupancy **FINE** (exact 6-way) | 15.6 % | **45.8 %** | **+30.2** |
| occupancy **COARSE** (safety grouping) | 47.3 % | **75.0 %** | **+27.7** |
| **missed occupant** (lower = better) | 36.7 % | **9.0 %** | **−27.7** |
| hallucinated occupant (lower = better) | 0.0 % | 0.4 % | +0.4 |
| count (persons) | _pending_ | _pending_ | — |
| yesno (left object) | _pending_ | _pending_ | — |
| reject (must refuse seatbelt) | _pending_ | _pending_ | — |

> Fill the pending rows with:
> `python 5-evaluation/eval_qwen_vl.py --compare 5-evaluation/results_base_qwen3.jsonl 5-evaluation/results_v2.jsonl`

## 5. Reading the numbers

- **The headline is the safety metric, not accuracy.** *Missed occupant* fell **36.7 % → 9.0 %** — the base model calls an occupied seat "empty" roughly one time in three; v2 reduces that ~4×. For occupant sensing this is the error that matters.
- **Coarse +27.7 pts** is the fair "did it understand the cabin" number. **Fine 6-way (45.8 %) understates the model**: a manual spot-check of 20 misses showed ~70 % were safety-irrelevant sub-type confusions (*empty* vs *empty child seat*, *infant* vs *child*), not real failures. That is why all three levels are reported.
- **Remaining weakness**: objects are still often missed (`object → unoccupied`, the largest single confusion) and 9 % of real occupants are still missed.

## 6. Limitations (do not overclaim)

1. SVIRO is **synthetic, single vehicle** (`bmw_x5_random`) → this measures in-distribution fit, not real-cabin generalization.
2. Evaluation GT comes from the **same GT facts** used to ground training data → tests "did it learn the mapping", not transfer.
3. **No DriveAct benchmark yet** → the activity/object-interaction capability added in v2 is unmeasured.
4. Keyword-based scoring of free text may **favour the LoRA's learned phrasing**; closed-choice questions would remove this confound.
5. `reject` uses a single unsupported type and does **not** measure over-refusal (a model that refuses everything would score 100 %).
6. Captions (State/Scene → Risk → Decision) are **not evaluated** — no faithfulness judge yet.
