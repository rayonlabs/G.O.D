#!/bin/bash

# Read port from env file
VALIDATOR_PORT=$(grep VALIDATOR_PORT .vali.env | cut -d '=' -f2)


# Start the validator service
pm2 start \
    "ENV_FILE=.vali.env uvicorn \
    --factory validator.asgi:factory \
    --host 0.0.0.0 \
    --port ${VALIDATOR_PORT} \
    --env-file .vali.env" \
    --name validator
