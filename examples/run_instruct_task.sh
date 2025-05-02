#!/bin/bash

TASK_ID="0ace46bc-8f88-4e70-95b9-9502b5a4d1dc"
MODEL="TinyLlama/TinyLlama_v1.1"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/e1230b33949f9bdf_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250430%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250430T023943Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=847c416b0015e19470188979543b5778ae51d1acff922d19f08487a4e6c37db9"
DATASET_TYPE='{
  "field_instruction":"question",
  "field_output":"chosen"
}'
FILE_FORMAT="s3"
HOURS_TO_COMPLETE=6
# Use your actual HuggingFace credentials for model upload
HUGGINGFACE_TOKEN="hf_your_token_here"
WANDB_TOKEN="your_wandb_token_here"
HUGGINGFACE_USERNAME="your_hf_username"

DATA_DIR="$(pwd)/secure_data"
mkdir -p "$DATA_DIR"
chmod 700 "$DATA_DIR"

docker build --no-cache -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

docker run --rm --gpus all \
  --security-opt=no-new-privileges \
  --cap-drop=ALL \
  --memory=64g \
  --cpus=8 \
  --volume "$DATA_DIR:/workspace/input_data:rw" \
  --name instruct-text-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --task-type "InstructTextTask" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --huggingface-token "$HUGGINGFACE_TOKEN" \
  --wandb-token "$WANDB_TOKEN" \
  --huggingface-username "$HUGGINGFACE_USERNAME"