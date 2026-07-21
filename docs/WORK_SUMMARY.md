# In-Cabin Data Line — Work Summary

Date: 2026-07-06 · Dataset: SVIRO `bmw_x5_random` (2000 GT rows) · Base model: Qwen2.5-VL-7B-Instruct

## What was built

An end-to-end in-cabin VLM data pipeline, migrated from the exterior pipeline (`Justin0504/Visteon-CabinVLMBenchmark`) and adapted to the cabin domain, ending in a LoRA fine-tune.

**Pipeline:** GT Facts → LLM generation → independent cross-check → deterministic validation → cleaning → ShareGPT → splits → LoRA training (→ evaluation, next).

## Steps completed

1. **Connected** local machine to the AutoDL data disk over SSH/rsync; worked directly on the synced project.
2. **Prompt design** (`2-generation/prompts/v0_pilot.md`): GT-grounded caption (State/Scene→Risk→Decision) + Recognition/Reasoning/Decision/Reject QA, with a strict anti-hallucination contract (no seatbelt/pet/door/sleep/age/orientation claims; candidate cases phrased as candidates).
3. **Migration plan** (`6-docs/exterior_to_incabin_migration_plan.md`): mapped each exterior stage → in-cabin (reuse / adapt / drop), dropped the temporal `prediction`, added the `candidate` support tier, kept caption as prose but structured internally.
4. **Generator** (`scripts/build_sviro_cot_qa_llm.py`): 3-stage — generate → cross-check → validate. Models via Vultr Serverless Inference (one endpoint/key): gen + check = `DeepSeek-V4-Flash` (non-thinking, clean JSON), safety folded into deterministic regex checks. Parallel, throttled, resumable, streams to disk.
5. **Cross-check verified** (`scripts/check_adversarial.py`): planted 4 error types (occupancy contradiction, asserted orientation, seatbelt, miscount) → **4/4 caught**.
6. **Splits** (`scripts/build_splits.py`, `3-splits/`): train 1690 / val_dev 150 / frozen_test 160 (disjoint; frozen is GT-supported only, balanced by use case, never trained).
7. **Full generation**: 1840 train+val rows → 1777 ok. Cleaned (`scripts/clean_and_export.py`) to **1761 content-clean** records (16 flagged: appearance/forbidden; 63 API-504 rows pending backfill).
8. **ShareGPT export**: `2-generation/sharegpt/sviro_train_sharegpt.json` — 1620 clean train records (val_dev/frozen_test held out).
9. **LoRA fine-tune** (`4-training/`, LLaMA-Factory): Qwen2.5-VL-7B, LoRA rank 8, frozen ViT, 2 epochs (406 steps) on RTX 4090 (24 min). Final loss ~0.35, converged. Adapter → `4-training/sviro_lora_v1/`.

![Training loss — SVIRO LoRA v1](sviro_lora_v1_loss.png)

*Training loss (Qwen2.5-VL-7B LoRA, 2 epochs, 406 steps): drops from ~2.6, converges by ~step 150, stable at ~0.35.*

## Key decisions

- **Generate facts deterministically-first, LLM for language** — in-cabin GT is categorical, so the hallucination surface is smaller than exterior.
- **No vision stage for SVIRO** (synthetic single cabin, low visual value; text generation from GT is cleaner).
- **Independent cross-check gate** kept and verified (mirrors exterior's faithfulness gate).
- **Frozen_test held out** before full generation to avoid train/test contamination.

## Open / next

- Backfill the 63 API-504 rows (`--resume`); optionally regenerate the 16 flagged.
- **Evaluation** (`5-evaluation/`): run base vs LoRA on `frozen_test` — per-use-case accuracy + caption judge.
- **Joint vs separate with exterior**: train in-cabin adapter first, score on in-cabin frozen_test, then run a balanced joint experiment and compare on both frozen tests before adopting one model.
- Bigger lever for quality: more in-cabin data variety (other vehicles / real cabins) — dataset expansion, currently deferred.
