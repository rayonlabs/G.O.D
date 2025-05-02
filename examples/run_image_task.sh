#!/bin/bash
# Example script to run the standalone image trainer based on real task data

# Task details from the database
TASK_ID="9a877904-5fe9-402a-8c75-be5eb1b51f7e"
MODEL="zenless-lab/sdxl-anima-pencil-xl-v5"
DATASET_ZIP="https://gradients.s3.eu-north-1.amazonaws.com/889c0bada81315f0_train_data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20250303%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20250303T123641Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=178cf09ca66f745a51ade601af84a079051778f1b46a8ad60128ca1cd47a739d"
MODEL_TYPE="sdxl"
HOURS_TO_COMPLETE=2
HUGGINGFACE_TOKEN="hf_your_token_here"
WANDB_TOKEN="your_wandb_token_here"
HUGGINGFACE_USERNAME="your_hf_username_here"

# Build the container first
docker build -t standalone-image-trainer -f dockerfiles/standalone-image-trainer.dockerfile .

# Run the container with parameters from the real image task
docker run --gpus all \
  --name image-trainer-example \
  standalone-image-trainer \
  --task-id "$TASK_ID" \
  --model "$MODEL" \
  --dataset-zip "$DATASET_ZIP" \
  --model-type "$MODEL_TYPE" \
  --hours-to-complete "$HOURS_TO_COMPLETE" \
  --huggingface-token "$HUGGINGFACE_TOKEN" \
  --wandb-token "$WANDB_TOKEN" \
  --huggingface-username "$HUGGINGFACE_USERNAME"