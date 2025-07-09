FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1

WORKDIR /app

RUN pip install --no-cache-dir huggingface_hub aiohttp pydantic transformers textstat==0.7.7 langcheck detoxify

COPY trainer/ trainer/
COPY core/ core/

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "trainer/utils/trainer_downloader.py"]
