FROM diagonalge/ai-toolkit:latest

RUN pip install mlflow huggingface_hub aiohttp pydantic requests toml \
    "fiber @ git+https://github.com/rayonlabs/fiber.git@2.4.0" \
    fastapi uvicorn httpx loguru python-dotenv \
    scipy numpy datasets tenacity minio \
    transformers pandas==2.2.3 tiktoken==0.8.0 sentencepiece==0.2.0 peft Pillow==11.1.0 PyYAML \
    requests huggingface_hub textstat==0.7.7 langcheck detoxify

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/scripts \
    /workspace/core

COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_image_trainer.sh

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]

