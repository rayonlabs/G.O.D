FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y git curl git-lfs && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub

COPY trainer/utils/hf_upload.py /app/hf_upload.py

ENTRYPOINT ["python", "/app/hf_upload.py"]