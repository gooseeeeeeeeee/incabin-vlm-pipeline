# Exterior (nuScene) Line — Finetune Results

Date: 2026-07 · Model: `Qwen3-VL-4B-Instruct` + LoRA · Task 2 (Nuscene recaptioning + finetune)

The exterior/out-of-vehicle line, complementary to the in-cabin line. Same principle:
**GT-grounded, no hallucination.** Objective evaluation on held-out driving datasets.

## Data
Exterior training = 8 sharegpt capability blocks (nuScenes VLA CoT + Stanford Cars + GTSRB/TT100K signs +
TextVQA OCR + SUN397 landscape + traffic-light + JAAD VRU) → 31,631 QA, plus balanced negatives
(JAAD crossing, YOLO presence/count) to fix answer biases. Full manifest: [`exterior_dataset.md`](exterior_dataset.md).

## Ablation (leakage-audited held-out, YOLOv8x objective GT)
| ver | key change | note |
|---|---|---|
| v21 | narrow slice | mode-collapse (crossing yes-bias) |
| v24 | + diverse JAAD crossing negatives | fixed crossing collapse |
| v25 | + full exterior data | signs/crossing up, but objective perception regressed (pedestrian yes-bias) |
| **v26** | + balanced YOLO presence/count negatives | **both axes positive (recommended)** |
| v27 | + BDD100K diverse | (training) strengthen cross-dataset generalization |

## Deliverable result (v26 vs base, objective YOLO GT)
| Axis | base | v26 | Δ |
|---|---:|---:|---:|
| **In-domain nuScenes** (sweeps, N=500, 0 leak) | 83.9 % | 94.1 % | **+10.2** |
| **Cross-dataset nuImages** (same-country, N=600) | 90.7 % | 94.9 % | **+4.2** |

Sub-metrics all positive (vehicle count +16.0 / +10.2, pedestrian, traffic-light). Matches the task-table
deliverable: finetuned Qwen3-VL improves on (1) same dataset (nuScenes) and (2) cross-dataset same-country.

## Method notes
- Objective GT = YOLOv8x (vehicle count ±1 / pedestrian presence / traffic-light presence). Training-image
  labels and eval GT come from the same detector; **train and eval images are disjoint** (leakage-audited),
  so this is genuine perception generalization, not same-image fitting.
- Held-out data all direct-download (no login): nuScenes-mini / nuImages-mini (nuscenes.org/data), JAAD
  (YorkU), BDD100K (HF mirror `dgural/bdd100k`).
- Training needs `max_pixels` cap + gradient checkpointing or full-res exterior images OOM 40 GB.

See [`../evaluation/exterior/`](../evaluation/exterior/) (eval) and [`../pipeline/exterior/`](../pipeline/exterior/) (data-gen), [`../training/train_lora_qwen3vl.py`](../training/train_lora_qwen3vl.py) (trainer).
