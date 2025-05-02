#!/bin/bash
# Example script to run the standalone DPO trainer based on real task data with S3 link

# Task details from the database
TASK_ID="f7d81131-3b6f-4532-a86b-14d62ebb615a"
MODEL="Qwen/Qwen1.5-14B-Chat"  # Example model, replace with actual model
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/6644544492ff8929_synth_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250415%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250415T143435Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=545e2fe88f9219d9c2b427c181979308d43ab0a12992258f37807cf9e7dd57eb"
DATASET_TYPE='{
  "field_prompt":"instruction",
  "field_chosen":"chosen",
  "field_rejected":"rejected"
}'
FILE_FORMAT="s3"  # S3 signed URL
HOURS_TO_COMPLETE=8  # Example hours, replace with actual hours
HUGGINGFACE_TOKEN="hf_your_token_here"
WANDB_TOKEN="your_wandb_token_here"
HUGGINGFACE_USERNAME="your_hf_username_here"

# Build the container first
docker build -t standalone-text-trainer -f dockerfiles/standalone-text-trainer.dockerfile .

# Run the container with parameters for the DPO task using S3 link
docker run --gpus all \
  -v "$(pwd)/data:/workspace/input_data" \
  --name dpo-trainer-example \
  standalone-text-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --dataset-type "$DATASET_TYPE" \
  --file-format "$FILE_FORMAT" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --huggingface-token "$HUGGINGFACE_TOKEN" \
  --wandb-token "$WANDB_TOKEN" \
  --huggingface-username "$HUGGINGFACE_USERNAME"