FROM winglian/axolotl:main-20250429

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force upgrade PyTorch to fix CVE-2025-32434 vulnerability
# Uninstall existing torch first to ensure clean upgrade
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Reinstall textstat and its dependencies after torch upgrade
# Uninstall any existing version first, then install the correct version
# Note: textstat <0.7.8 has issues with words not in CMU dictionary (e.g., "Gradients.io" -> KeyError: 'gradientsio')
# Using version 0.7.8 which handles unknown words gracefully
RUN pip uninstall -y textstat pyphen && \
    pip install --no-cache-dir --force-reinstall textstat==0.7.8 && \
    python -c "import textstat; print(f'textstat version: {textstat.__version__}')"

COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""
ENV TRANSFORMERS_ALLOW_TORCH_LOAD="true"

RUN mkdir /aplp
