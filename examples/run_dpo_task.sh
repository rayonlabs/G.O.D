#!/bin/bash

# Example configuration for DPO (Direct Preference Optimization) Task

# Unique task identifier
TASK_ID="7719761b-73c5-4100-98fb-cbe5a6847737"

# Model to fine-tune (from HuggingFace)
MODEL="Qwen/Qwen1.5-7B-Chat"

# Dataset location - can be:
# - S3 URL: "s3://bucket/path/to/dataset.json"
# - Local file: "/path/to/dataset.json"
# - HuggingFace dataset: "username/dataset-name"
DATASET="s3://your-bucket/path/to/dpo_dataset.json"

# Dataset type mapping for DPO:
# - field_prompt: column containing the prompt
# - field_chosen: column containing the preferred response
# - field_rejected: column containing the rejected response
# - Optional format templates (use {prompt}, {chosen}, {rejected} as placeholders)
DATASET_TYPE='{
  "field_prompt":"prompt",
  "field_chosen":"chosen",
  "field_rejected":"rejected",
  "prompt_format":"{prompt}",
  "chosen_format":"{chosen}",
  "rejected_format":"{rejected}"
}'

# File format: "csv", "json", "hf" (HuggingFace), or "s3"
FILE_FORMAT="s3"

# Optional: Repository name for the trained model (just the model name, not username/model-name)
EXPECTED_REPO_NAME="my-dpo-finetuned-model"

# Create secure data directory
DATA_DIR="$(pwd)/secure_data"
mkdir -p "$DATA_DIR"
chmod 700 "$DATA_DIR"

# Build the downloader image
docker build --no-cache -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build --no-cache -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$DATA_DIR:/cache:rw" \
  --name downloader-dpo \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT"

# Run the training
echo "Starting DPO training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --volume "$DATA_DIR:/cache:rw" \
  --name dpo-trainer \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME"
