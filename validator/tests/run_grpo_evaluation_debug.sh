#!/bin/bash
# GRPO Evaluation Shell Script with Debug Logging
#
# This script provides a simplified way to run GRPO evaluations for testing purposes
# with enhanced logging and debugging capabilities.
#
# Usage:
#   ./run_grpo_evaluation_debug.sh <s3_dataset_url> <original_model> <model1,model2,...>

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
LOG_FILE="$TEMP_DIR/grpo_eval_$(date +%s).log"
trap 'rm -rf -- "$TEMP_DIR"' EXIT

echo "=== GRPO Evaluation Test Runner (Debug Mode) ==="
echo "Dataset URL: $S3_DATASET_URL"
echo "Original Model: $ORIGINAL_MODEL"
echo "Models to evaluate: $MODELS"
echo "Using temp directory: $TEMP_DIR"
echo "Logging to: $LOG_FILE"

# Download the dataset from S3
echo "Downloading dataset..."
DATASET_FILENAME=$(basename "$S3_DATASET_URL" | cut -d'?' -f1)  # Strip query parameters
LOCAL_DATASET_PATH="$TEMP_DIR/$DATASET_FILENAME"
curl -L -v -o "$LOCAL_DATASET_PATH" "$S3_DATASET_URL" 2>&1 | tee -a "$LOG_FILE"
echo "Downloaded dataset to $LOCAL_DATASET_PATH"

# Examine the dataset content
echo "Dataset content preview:" | tee -a "$LOG_FILE"
head -n 10 "$LOCAL_DATASET_PATH" | tee -a "$LOG_FILE"
echo "Dataset file size: $(wc -c < "$LOCAL_DATASET_PATH") bytes" | tee -a "$LOG_FILE"
echo "Dataset has $(grep -c . "$LOCAL_DATASET_PATH") lines" | tee -a "$LOG_FILE"

# Prepare Docker command
echo "Setting up Docker evaluation command..." | tee -a "$LOG_FILE"
DOCKER_CMD=(
  docker run --rm
  -e "DATASET=/workspace/input_data/$DATASET_FILENAME"
  -e "MODELS=$MODELS"
  -e "ORIGINAL_MODEL=$ORIGINAL_MODEL"
  -e "DATASET_TYPE={\"field_prompt\":\"prompt\"}"
  -e "FILE_FORMAT=jsonl"
  -v "$TEMP_DIR:/workspace/input_data:rw"
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw"
  --runtime nvidia
  --gpus all
  -it
  validator
  python -m validator.evaluation.eval_grpo
)

# Print the Docker command
echo "Docker command:" | tee -a "$LOG_FILE"
echo "${DOCKER_CMD[@]}" | tee -a "$LOG_FILE"

# Run evaluation in Docker container
echo "Starting Docker container for GRPO evaluation..." | tee -a "$LOG_FILE"
"${DOCKER_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

# Check for evaluation results in the mounted volume
echo "Checking for evaluation results..." | tee -a "$LOG_FILE"
find "$TEMP_DIR" -type f -name "*evaluation_results.json" | tee -a "$LOG_FILE"

# If results exist, display them
RESULTS_FILE=$(find "$TEMP_DIR" -type f -name "*evaluation_results.json" | head -n 1)
if [ -n "$RESULTS_FILE" ]; then
  echo "Evaluation results found: $RESULTS_FILE" | tee -a "$LOG_FILE"
  echo "Results content:" | tee -a "$LOG_FILE"
  cat "$RESULTS_FILE" | tee -a "$LOG_FILE"
else
  echo "No evaluation results found in the temp directory" | tee -a "$LOG_FILE"
fi

echo "GRPO evaluation completed. Log file is at $LOG_FILE"
# Make a copy of the log file in the current directory
cp "$LOG_FILE" "./grpo_eval_log_$(date +%s).log"
echo "Log file copied to ./$(basename "$LOG_FILE")"