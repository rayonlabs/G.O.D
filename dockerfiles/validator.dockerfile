FROM winglian/axolotl:main-20241101-py3.11-cu124-2.5.0

WORKDIR /app

COPY validator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV JOB_ID=""
ENV DATASET=""
ENV MODEL=""
ENV ORIGINAL_MODEL=""
ENV DATASET_TYPE=""
ENV FILE_FORMAT=""

RUN mkdir /aplp

CMD ["python", "-m", "validator.evaluation.eval"]
