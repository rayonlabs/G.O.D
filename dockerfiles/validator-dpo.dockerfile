FROM winglian/axolotl:main-20250401

WORKDIR /app

RUN pip install --no-cache-dir trl==1.3.0

# Copy requirements and install dependencies
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

CMD ["python", "-m", "validator.evaluation.dpo_eval"]
