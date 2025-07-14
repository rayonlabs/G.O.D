FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget && rm -rf /var/lib/apt/lists/*

RUN mkdir /aplp

WORKDIR /app/validator/evaluation
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI && \
    cd ComfyUI && \
    git fetch --depth 1 origin a220d11e6b8dd0dbbf50f81ab9398ec202a96fe6 && \
    git checkout a220d11e6b8dd0dbbf50f81ab9398ec202a96fe6 && \
    cd ..

RUN pip install -r ComfyUI/requirements.txt
RUN cd ComfyUI/custom_nodes && \
    git clone --depth 1 https://github.com/Acly/comfyui-tooling-nodes && \
    cd ..

RUN wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/clip_l.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors \
    https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors && \
    wget -O /app/validator/evaluation/ComfyUI/models/vae/ae.safetensors \
    https://huggingface.co/Albert-zp/flux-vaesft/resolve/main/fluxVaeSft_aeSft.sft

RUN pip install docker diffusers

ENV TEST_DATASET_PATH=""
ENV TRAINED_LORA_MODEL_REPOS=""
ENV BASE_MODEL_REPO=""
ENV BASE_MODEL_FILENAME=""
ENV LORA_MODEL_FILENAMES=""

WORKDIR /app

COPY validator/requirements.txt validator/requirements.txt
RUN pip install -r validator/requirements.txt

COPY . .

RUN echo '#!/bin/bash\n\
python /app/validator/evaluation/ComfyUI/main.py &\n\
python -m validator.evaluation.eval_diffusion' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
