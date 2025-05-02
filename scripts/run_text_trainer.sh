#!/bin/bash
# Simple wrapper script for the text_trainer.py script
# Handles both InstructText and DPO training

set -e

# Run the Python script with all arguments passed to this shell script
python3 /workspace/scripts/text_trainer.py "$@"