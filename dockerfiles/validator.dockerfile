FROM winglian/axolotl:main-20250429

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade PyTorch to fix CVE-2025-32434 vulnerability
RUN pip install --no-cache-dir torch>=2.6.0

COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"

RUN mkdir /aplp
