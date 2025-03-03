FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

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

RUN pip install docker diffusers

ENV TEST_DATASET_PATH=""
ENV TRAINED_LORA_MODEL_REPOS=""
ENV BASE_MODEL_REPO=""
ENV BASE_MODEL_FILENAME=""
ENV LORA_MODEL_FILENAMES=""

WORKDIR /app
COPY . .

RUN pip install -r validator/requirements.txt

RUN echo '#!/bin/bash\n\
python /app/validator/evaluation/ComfyUI/main.py &\n\
python -m validator.evaluation.eval_diffusion' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
