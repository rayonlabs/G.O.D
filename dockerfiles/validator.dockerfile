FROM winglian/axolotl:main-20250429

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force upgrade PyTorch to fix CVE-2025-32434 vulnerability
# Uninstall existing torch first to ensure clean upgrade
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Reinstall textstat after torch upgrade in case it was affected
RUN pip install --no-cache-dir textstat==0.7.7

COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"

RUN mkdir /aplp
