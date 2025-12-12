FROM diagonalge/ai-toolkit:latest

RUN pip install --no-cache-dir PyYAML

RUN mkdir -p /dataset/configs \
    /dataset/outputs \
    /workspace/scripts

COPY scripts/image_trainer.py /workspace/scripts/image_trainer.py
COPY scripts/run_image_trainer.sh /workspace/scripts/run_image_trainer.sh

RUN chmod +x /workspace/scripts/run_image_trainer.sh

ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"]

