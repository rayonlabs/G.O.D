#!/bin/bash
# Simple wrapper script for the image_trainer.py script
# Handles both SDXL and Flux training

set -e

# Run the Python script with all arguments passed to this shell script
python3 /workspace/scripts/image_trainer.py "$@"