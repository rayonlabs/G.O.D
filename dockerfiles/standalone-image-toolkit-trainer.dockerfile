FROM diagonalge/ai-toolkit:latest

RUN pip install --no-cache-dir PyYAML

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

