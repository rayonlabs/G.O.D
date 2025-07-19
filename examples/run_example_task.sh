#!/bin/bash

# Another DPO Task example with different model and dataset

# Unique task identifier
TASK_ID="f7d81131-3b6f-4532-a86b-14d62ebb615a"

# Model to fine-tune (from HuggingFace)
MODEL="Qwen/Qwen1.5-14B-Chat"

# Dataset location
DATASET="s3://your-bucket/path/to/dpo_dataset.json"

# Dataset type mapping for DPO
DATASET_TYPE='{
  "field_prompt":"instruction",
  "field_chosen":"chosen",
  "field_rejected":"rejected"
}'

# File format
FILE_FORMAT="s3"

# Optional: Repository name for the trained model
EXPECTED_REPO_NAME="my-qwen-dpo-finetuned"

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
  --name downloader-qwen \
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
  --name dpo-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "DpoTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --huggingface-token "$HUGGINGFACE_TOKEN" \
  --wandb-token "$WANDB_TOKEN" \
  --huggingface-username "$HUGGINGFACE_USERNAME"