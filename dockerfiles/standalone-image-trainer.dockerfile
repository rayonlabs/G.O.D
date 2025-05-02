FROM diagonalge/kohya_latest:latest

# Install required dependencies
RUN pip install aiohttp pydantic requests

# Create required directories
RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/scripts \
    /workspace/core

# Set environment variables
ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"

# Copy project code and scripts
COPY core /workspace/core
COPY scripts /workspace/scripts

# Make scripts executable
RUN chmod +x /workspace/scripts/run_image_trainer.sh
RUN chmod +x /workspace/scripts/image_trainer.py

# Entrypoint script accepts TrainRequestImage fields:
# --task-id: Task ID
# --model: Name or path of the model to be trained
# --dataset-zip: Link to dataset zip file
# --model-type: Model type (sdxl or flux)
# --expected-repo-name: Expected repository name (optional)
# --huggingface-token: Hugging Face token (optional)
# --wandb-token: Weights & Biases token (optional)
ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]