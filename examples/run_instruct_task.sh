#!/bin/bash

# Example configuration for InstructTextTask training

# Unique task identifier
TASK_ID="0ace46bc-8f88-4e70-95b9-9502b5a4d1dc"

# Model to fine-tune (from HuggingFace)
MODEL="TinyLlama/TinyLlama_v1.1"

# Dataset location - can be:
# - S3 URL: "s3://bucket/path/to/dataset.json"
# - Local file: "/path/to/dataset.json"
# - HuggingFace dataset: "username/dataset-name"
DATASET="s3://your-bucket/path/to/instruct_dataset.json"

# Dataset type mapping - maps your dataset columns to expected format
# For InstructTextTask:
# - field_instruction: column containing the instruction/question
# - field_output: column containing the expected output/answer
DATASET_TYPE='{
  "field_instruction":"instruct",
  "field_output":"output"
}'

# File format: "csv", "json", "hf" (HuggingFace), or "s3"
FILE_FORMAT="s3"

# Optional: Repository name for the trained model (just the model name, not username/model-name)
EXPECTED_REPO_NAME="my-finetuned-model"

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
  --name downloader-example \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT"

# Run the training
echo "Starting training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --volume "$DATA_DIR:/cache:rw" \
  --name instruct-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT" \
  --expected-repo-name "$EXPECTED_REPO_NAME"
