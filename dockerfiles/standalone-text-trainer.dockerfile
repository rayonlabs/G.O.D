FROM axolotlai/axolotl:main-20241128-py3.11-cu124-2.5.1

RUN pip install mlflow huggingface_hub wandb aiohttp pydantic requests

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data

ENV CONFIG_DIR="/workspace/axolotl/configs"
ENV OUTPUT_DIR="/workspace/axolotl/outputs"

COPY core /workspace/core
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml

RUN chmod +x /workspace/scripts/run_text_trainer.sh
RUN chmod +x /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]