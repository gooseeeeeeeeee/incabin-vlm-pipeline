#!/usr/bin/env bash
# SVIRO in-cabin LoRA fine-tune (Qwen2.5-VL-7B) on an AutoDL GPU instance (RTX 4090 24GB).
# Run FROM the project root on the instance:  bash 4-training/run_lora.sh
set -e
PROJ=/root/autodl-tmp/.autodl/Incabin
cd "$PROJ"

# 0) GPU check
nvidia-smi || { echo "No GPU — power on the instance with a 4090 first."; exit 1; }

# 0b) HF mirror for the base-model download (fast in CN; no GitHub/accel needed)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true  # accel slows pip
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 1) Install LLaMA-Factory + latest transformers (Qwen3-VL needs recent versions)
pip install -U "llamafactory[torch,metrics]" --break-system-packages
pip install -U transformers qwen-vl-utils accelerate --break-system-packages
# NOTE: if training reports "Cannot find valid samples" (Qwen3-VL <image> vs <|image_pad|> bug,
#       LLaMA-Factory issue #9605), upgrade LLaMA-Factory to the newest version (or install from source).

# 2) Build the TRAIN-only ShareGPT with absolute image paths (val_dev/frozen_test excluded)
python "$PROJ/scripts/clean_and_export.py" \
  --raw "$PROJ/2-generation/raw_outputs/sviro_full_raw.jsonl" \
  --split-file "$PROJ/3-splits/split_assignment.jsonl" --splits train \
  --image-root "$PROJ" \
  --out "$PROJ/2-generation/sharegpt/sviro_train_sharegpt.json"

# 3) Register the dataset for LLaMA-Factory (dataset_info.json lives in dataset_dir)
cat > "$PROJ/2-generation/sharegpt/dataset_info.json" <<'JSON'
{
  "sviro_incabin": {
    "file_name": "sviro_train_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations", "images": "images"},
    "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"}
  }
}
JSON

# 4) Train (LoRA). First run downloads the ~16GB base model.
llamafactory-cli train "$PROJ/4-training/sviro_lora_qwen2vl.yaml"

echo "Done. Adapter -> $PROJ/4-training/sviro_lora_v1"
echo "Merge for inference:  llamafactory-cli export (see 4-training/README.md)"
