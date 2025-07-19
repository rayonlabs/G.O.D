#!/bin/bash

# Example configuration for Image Model Training

# Unique task identifier
TASK_ID="9a877904-5fe9-402a-8c75-be5eb1b51f7e"

# Base model to fine-tune (from HuggingFace)
MODEL="zenless-lab/sdxl-anima-pencil-xl-v5"

# Dataset ZIP file location (must be a ZIP file with images)
DATASET_ZIP="s3://your-bucket/path/to/image_dataset.zip"

# Model type: "sdxl" or "flux"
MODEL_TYPE="sdxl"

# Optional: Repository name for the trained model
EXPECTED_REPO_NAME="my-sdxl-finetuned"

CHECKPOINTS_DIR="$(pwd)/secure_checkpoints"
mkdir -p "$CHECKPOINTS_DIR"
chmod 700 "$CHECKPOINTS_DIR"

# Build the downloader image
docker build --no-cache -t trainer-downloader -f dockerfiles/trainer-downloader.dockerfile .

# Build the trainer image
docker build --no-cache -t standalone-image-trainer -f dockerfiles/standalone-image-trainer.dockerfile .

# Download model and dataset
echo "Downloading model and dataset..."
docker run --rm \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name downloader-image \
  trainer-downloader \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET_ZIP" \
  --task-type "ImageTask"

# Run the training
echo "Starting image training..."
docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=32g \
  --cpus=8 \
  --volume "$CHECKPOINTS_DIR:/cache:rw" \
  --name image-trainer-example \
  standalone-image-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset-zip "$DATASET_ZIP" \
  --model-type "$MODEL_TYPE" \
  --expected-repo-name "$EXPECTED_REPO_NAME"
