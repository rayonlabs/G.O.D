FROM axolotlai/axolotl:main-20241128-py3.11-cu124-2.5.1

# Install core dependencies from pyproject.toml
RUN pip install mlflow huggingface_hub wandb aiohttp pydantic requests toml \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0" \
    fastapi uvicorn httpx loguru python-dotenv \
    scipy numpy datasets tenacity minio \
    transformers==4.46.2 pandas==2.2.3 tiktoken==0.8.0 sentencepiece==0.2.0 peft Pillow==11.1.0 PyYAML \
    requests huggingface_hub

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"

COPY core /workspace/core
COPY miner /workspace/miner
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml

RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]
