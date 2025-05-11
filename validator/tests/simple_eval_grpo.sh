#!/bin/bash
# Simple GRPO evaluation script
# Usage: ./simple_eval_grpo.sh

TEMP_DIR=$(mktemp -d)
DATASET_FILE="$TEMP_DIR/dataset.json"

# Download the dataset
echo "Downloading dataset..."
curl -L -o "$DATASET_FILE" "https://gradients.s3.eu-north-1.amazonaws.com/13abfd7d95984f0d_test_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250510%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250510T034142Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9bcc25185db0e3d411f46b5063134bb66154734c21401cef6ec5427380ef4841"

echo "Dataset downloaded to $DATASET_FILE"
echo "Dataset size: $(wc -c < "$DATASET_FILE") bytes"

# Print first few lines of dataset
echo "Dataset preview:"
head -n 5 "$DATASET_FILE"

# Run Docker container
echo "Starting GRPO evaluation..."
docker run --rm \
  -e DATASET="/workspace/input_data/dataset.json" \
  -e MODELS="robiual-awal/c621c6f1-40be-4a54-add1-38585b4e002f,Alphatao/3e05bf5e-0a8a-4c96-bf01-7a2d82bd333c" \
  -e ORIGINAL_MODEL="EleutherAI/pythia-70m" \
  -e DATASET_TYPE='{"field_prompt":"prompt"}' \
  -e FILE_FORMAT="jsonl" \
  -v "$TEMP_DIR:/workspace/input_data:rw" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:rw" \
  --runtime nvidia \
  --gpus all \
  weightswandering/tuning_vali:latest \
  python -m validator.evaluation.eval_grpo

# Check for results
RESULTS_FILE=$(find "$TEMP_DIR" -type f -name "*evaluation_results.json" | head -n 1)
if [ -n "$RESULTS_FILE" ]; then
  echo "Evaluation results found: $RESULTS_FILE"
  echo "Results content:"
  cat "$RESULTS_FILE" 
  # Copy results to current directory
  cp "$RESULTS_FILE" "./grpo_eval_results_$(date +%s).json"
  echo "Results copied to current directory"
else
  echo "No evaluation results found in the temp directory"
fi

echo "Evaluation complete"
# Cleanup
rm -rf "$TEMP_DIR"