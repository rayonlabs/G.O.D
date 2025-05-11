#!/bin/bash
# GRPO Evaluation Shell Script
#
# This script provides a simplified way to run GRPO evaluations for testing purposes.
# It downloads a dataset from S3, prepares the environment, and runs the evaluation
# in a Docker container similar to how it's done in production.
#
# Usage:
#   ./run_grpo_evaluation.sh <s3_dataset_url> <original_model> <model1,model2,...>
#
# Example:
#   ./run_grpo_evaluation.sh https://bucket.s3.amazonaws.com/dataset.jsonl \
#       mistralai/Mistral-7B-v0.1 repo1/adapter,repo2/adapter

set -e

if [ $# -lt 3 ]; then
  echo "Usage: $0 <s3_dataset_url> <original_model> <model1,model2,...>"
  exit 1
fi

S3_DATASET_URL="$1"
ORIGINAL_MODEL="$2"
MODELS="$3"

# Set up temporary workspace
TEMP_DIR=$(mktemp -d)
trap 'rm -rf -- "$TEMP_DIR"' EXIT

echo "=== GRPO Evaluation Test Runner ==="
echo "Dataset URL: $S3_DATASET_URL"
echo "Original Model: $ORIGINAL_MODEL"
echo "Models to evaluate: $MODELS"
echo "Using temp directory: $TEMP_DIR"

# Download the dataset from S3
echo "Downloading dataset..."
DATASET_FILENAME=$(basename "$S3_DATASET_URL")
LOCAL_DATASET_PATH="$TEMP_DIR/$DATASET_FILENAME"
curl -s -o "$LOCAL_DATASET_PATH" "$S3_DATASET_URL"
echo "Downloaded dataset to $LOCAL_DATASET_PATH"

# Run evaluation in Docker container
echo "Starting Docker container for GRPO evaluation..."
docker run --rm \
  -e DATASET="/workspace/input_data/$DATASET_FILENAME" \
  -e MODELS="$MODELS" \
  -e ORIGINAL_MODEL="$ORIGINAL_MODEL" \
  -e DATASET_TYPE='{"field_prompt":"prompt"}' \
  -e FILE_FORMAT="jsonl" \
  -v "$TEMP_DIR:/workspace/input_data:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw" \
  --runtime nvidia \
  --gpus all \
  -it \
  validator \
  python -m validator.evaluation.eval_grpo

echo "GRPO evaluation completed"