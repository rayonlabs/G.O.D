FROM winglian/axolotl:main-20250401

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install docker toml

# Copy the rest of the application
COPY . .

RUN mkdir /aplp

ENV JOB_ID=""
ENV DATASET=""
ENV MODELS=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""

CMD ["python", "-m", "validator.evaluation.eval_dpo"]
