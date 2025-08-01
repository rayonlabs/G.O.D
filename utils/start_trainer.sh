#!/bin/bash

# Read port from .trainer.env
TRAINER_PORT=$(grep TRAINER_PORT .trainer.env | cut -d '=' -f2)

# Delete old trainer services
pm2 delete trainer_api || true

# Load variables from .trainer.env
set -a # Automatically export all variables
. .trainer.env
set +a # Stop automatic export

# Start the trainer service using opentelemetry-instrument with combined env vars
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name trainer \
    uvicorn \
    --factory trainer.asgi:factory \
    --host 0.0.0.0 \
    --port ${TRAINER_PORT} \
    --env-file .trainer.env" \
    --name trainer_api

echo "Trainer service started with PM2"
echo "Use 'pm2 logs trainer_api' to view logs"
echo "Use 'pm2 status' to check service status" 