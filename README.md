# In-Cabin VLM Pipeline

Closed-loop curation of a **ground-truth-grounded** vision-language dataset and model for **in-cabin
occupant monitoring** and **driver activity understanding**.

Raw dataset labels → GT Facts → LLM generation → independent cross-check → deterministic validation →
cleaning → ShareGPT → splits → LoRA fine-tune → frozen-test evaluation.

The design principle throughout: **the model may only state what the ground truth supports.** Anything the
dataset cannot confirm (seatbelt state, child-seat orientation, ISOFIX, sleep) is either phrased as a
*candidate for review* or explicitly **refused** — never guessed. This matters because the target use cases
are safety-critical.

The repo hosts **two GT-grounded data lines** that share this principle:
- **In-cabin** (primary) — SVIRO / Drive&Act occupant & activity understanding.
- **Exterior / nuScene** (Task 2) — out-of-vehicle recaptioning + finetune, objectively evaluated on
  held-out driving datasets. Files live under `*/exterior/`. See [`docs/exterior_finetune_results.md`](docs/exterior_finetune_results.md).

---

## Repository layout

```
pipeline/            in-cabin data construction: GT facts → grounded QA → QC → ShareGPT
pipeline/exterior/   exterior data-gen: sharegpt flatten, JAAD crossing, YOLO presence/count QA
training/            LoRA configs+launcher (in-cabin, LLaMA-Factory) + train_lora_qwen3vl.py (exterior)
evaluation/          in-cabin frozen-test benchmark + scoring
evaluation/exterior/ exterior objective eval (YOLO-GT in-domain nuScenes + cross-dataset nuImages)
docs/                method, coverage, results (both lines)
.env.example         API configuration template (never commit a real .env)
```

### `pipeline/`

| File | Purpose |
|---|---|
| `build_sviro_gt_facts.py` | Parse SVIRO raw labels (seat states, bboxes, keypoints) into unified **GT Facts** JSONL |
| `build_sviro_use_case_buckets.py` | Derive per-use-case buckets from `suited_cases` |
| `materialize_sviro_use_case_candidate_images.py` | Materialise image views per use case for inspection |
| `explore_driveact.py` | Discover the Drive&Act raw structure, modalities and label vocabularies |
| `build_driveact_use_case_buckets.py` | Drive&Act use-case bucketing |
| `llm_client.py` | OpenAI-compatible client (generation / cross-check roles, global throttle, retries) |
| `build_sviro_cot_qa_pilot.py` | Deterministic rule-based pilot generator (zero-hallucination baseline) |
| `build_sviro_cot_qa_llm.py` | **SVIRO** 3-stage generator: generate → cross-check → validate |
| `build_driveact_cot_qa_llm.py` | **Drive&Act** 3-stage generator (activity-grounded prompt) |
| `check_adversarial.py` | Adversarial test proving the cross-check gate strips planted errors |
| `clean_and_export.py` | Content cleaning + ShareGPT export (`conversations` + `images` only) |
| `export_sharegpt_pilot.py` | Minimal pilot ShareGPT exporter |
| `build_splits.py` | Deterministic train / val_dev / frozen_test splits (frozen never trained) |

### `training/`

`run_lora.sh` installs LLaMA-Factory, builds the train-only ShareGPT and launches training.
`sviro_lora_qwen2vl.yaml` (SVIRO only) and `sviro_driveact_lora_v2.yaml` (joint SVIRO + Drive&Act)
fine-tune **Qwen3-VL-8B-Instruct** with LoRA (rank 8, frozen vision tower, 2 epochs).

### `evaluation/`

`build_frozen_benchmark.py` builds closed-form objective questions whose answers are derived
deterministically from GT Facts. `eval_qwen_vl.py` runs base vs LoRA inference, scores, compares, and
re-scores occupancy at three granularities. `spotcheck_occupancy.py` dumps misses for manual review.

---

## Quick start

```bash
pip install openai python-dotenv                 # generation
cp .env.example .env                             # then fill in your API key (never commit .env)

# 1. GT facts  →  2. grounded generation  →  3. clean + export
python pipeline/build_sviro_gt_facts.py
python pipeline/build_sviro_cot_qa_llm.py --all --workers 5 --resume \
       --split-file 3-splits/split_assignment.jsonl --include-splits train,val_dev \
       --out 2-generation/raw_outputs/sviro_full_raw.jsonl
python pipeline/clean_and_export.py --raw 2-generation/raw_outputs/sviro_full_raw.jsonl \
       --split-file 3-splits/split_assignment.jsonl --splits train \
       --out 2-generation/sharegpt/sviro_train_sharegpt.json

# 4. splits (before full generation, so frozen_test is never trained)
python pipeline/build_splits.py --frozen-per 40 --val 150

# 5. train
bash training/run_lora.sh

# 6. evaluate
python evaluation/build_frozen_benchmark.py
python evaluation/eval_qwen_vl.py --tag base
python evaluation/eval_qwen_vl.py --adapter <adapter_dir> --tag v2
python evaluation/eval_qwen_vl.py --compare evaluation/results_base.jsonl evaluation/results_v2.jsonl
python evaluation/eval_qwen_vl.py --rescore evaluation/results_v2.jsonl   # fine / coarse / safety
```

Paths above assume the data directories (`0-datasets/`, `1-gt_facts/`, `2-generation/`, `3-splits/`)
from the working project. **Datasets, generated data and model weights are not included in this repo.**

---

## Data sources

| Source | Domain | Licence note |
|---|---|---|
| **SVIRO** | synthetic rear-seat occupancy, seat states, objects, keypoints | non-commercial research use; not redistributed |
| **Drive&Act** | driver activity, object-level interaction (action + object + location) | non-commercial research use; not redistributed |

## Results (v2, Qwen3-VL-8B + LoRA)

Frozen test: 160 unseen SVIRO samples, 960 deterministic objective questions.

| Metric | base | v2 (LoRA) | Δ |
|---|---:|---:|---:|
| occupancy, exact 6-way | 15.6 % | 45.8 % | +30.2 |
| occupancy, safety grouping | 47.3 % | 75.0 % | +27.7 |
| **missed occupant** (lower better) | 36.7 % | **9.0 %** | **−27.7** |

The headline is the safety metric: the base model calls an occupied seat "empty" about one time in three;
the fine-tuned model reduces that roughly four-fold. See [`docs/MODEL_V2_SUMMARY.md`](docs/MODEL_V2_SUMMARY.md)
for the full table, method and **limitations** (synthetic single-vehicle data, in-distribution evaluation,
Drive&Act capability not yet benchmarked).

## Documentation

- [`docs/MODEL_V2_SUMMARY.md`](docs/MODEL_V2_SUMMARY.md) — data counts, evaluation method, results, limitations
- [`docs/WORK_SUMMARY.md`](docs/WORK_SUMMARY.md) — what was built, step by step
- [`docs/exterior_to_incabin_migration_plan.md`](docs/exterior_to_incabin_migration_plan.md) — method design and reuse decisions
- [`docs/target_spec_coverage.md`](docs/target_spec_coverage.md) — 13 target use cases vs current data coverage
- [`docs/gap_datasets_public_sources.md`](docs/gap_datasets_public_sources.md) — candidate datasets for uncovered use cases
- [`docs/exterior_finetune_results.md`](docs/exterior_finetune_results.md) — **exterior/nuScene line**: ablation + v26 results (+10.2 in-domain / +4.2 cross-dataset)
- [`docs/exterior_dataset.md`](docs/exterior_dataset.md) — exterior dataset manifest (sources / GT / held-out / leakage audit)

## Notes

- Generation uses an OpenAI-compatible endpoint; generation and cross-check must be **non-thinking** models
  (reasoning models emit chain-of-thought instead of JSON and break parsing).
- Long runs are resumable (`--resume`) and stream to disk, so an interrupted job never loses completed work.
