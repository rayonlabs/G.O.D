FROM axolotlai/axolotl:main-20241128-py3.11-cu124-2.5.1

# Install required dependencies
RUN pip install mlflow huggingface_hub wandb aiohttp pydantic requests

# Create required directories
WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

# Set environment variables for Axolotl
ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"

# Copy project code and scripts
COPY core /workspace/core
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml

# Make scripts executable
RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

# Entrypoint script accepts TrainRequestText fields:
# --task-id: Task ID
# --model: Name or path of the model to be trained
# --dataset: Path to the dataset file or HF dataset name
# --dataset-type: JSON string of dataset type config
# --file-format: Format of the dataset (csv, json, hf, s3)
# --expected-repo-name: Expected repository name (optional)
# --huggingface-token: Hugging Face token (optional)
# --wandb-token: Weights & Biases token (optional)
ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]