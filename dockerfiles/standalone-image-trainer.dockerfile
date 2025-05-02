FROM diagonalge/kohya_latest:latest

RUN pip install aiohttp pydantic requests

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /dataset/images \
    /workspace/scripts \
    /workspace/core

ENV CONFIG_DIR="/dataset/configs"
ENV OUTPUT_DIR="/dataset/outputs"
ENV DATASET_DIR="/dataset/images"

COPY core /workspace/core
COPY scripts /workspace/scripts

RUN chmod +x /workspace/scripts/run_image_trainer.sh
RUN chmod +x /workspace/scripts/image_trainer.py

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]