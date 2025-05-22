#!/bin/bash

TASK_ID="7719761b-73c5-4100-98fb-cbe5a6847737"
MODEL="Qwen/Qwen1.5-7B-Chat"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/755b05591666e560_synth_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250502%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250502T001419Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=8eb5998d93e0434e2d7a516981dbd1f9eeff829901aa79b83a77d19adc9f1402"
DATASET_TYPE='{
  "field_prompt":"prompt",
  "field_chosen":"chosen",
  "field_rejected":"rejected",
  "prompt_format":"{prompt}",
  "chosen_format":"{chosen}",
  "rejected_format":"{rejected}"
}'


FILE_FORMAT="s3"
HOURS_TO_COMPLETE=8
HUGGINGFACE_TOKEN="your_huggingface_token_here"
WANDB_TOKEN="your_wandb_token_here"
HUGGINGFACE_USERNAME="your_hf_username_here"

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
  --name dpo-trainer \
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
