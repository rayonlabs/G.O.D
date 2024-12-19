#!/bin/bash

# Read port from .vali.env
VALIDATOR_PORT=$(grep VALIDATOR_PORT .vali.env | cut -d '=' -f2)

# Delete old validator services
pm2 delete validator || true
pm2 delete validator_cycle || true

# Load variables from .vali.env
set -a # Automatically export all variables
source .vali.env
set +a # Stop automatic export

# Start the validator service using opentelemetry-instrument with combined env vars

OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name validator \
    uvicorn \
    --factory validator.asgi:factory \
    --host 0.0.0.0 \
    --port ${VALIDATOR_PORT} \
    --env-file .vali.env" \
    --name validator

# Start the validator_cycle service using opentelemetry-instrument
OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf" \
OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:4317" \
OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED="true" \
OTEL_PYTHON_LOG_CORRELATION="true" \
pm2 start \
    "opentelemetry-instrument \
    --logs_exporter otlp \
    --traces_exporter none \
    --metrics_exporter otlp \
    --service_name validator_cycle \
    python -u -m validator.cycle.main" \
    --name validator_cycle
