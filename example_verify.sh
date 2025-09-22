#!/bin/bash

# Example script to verify image pairs using your trained model
# Make sure to update the paths according to your setup

# IMPORTANT: Run this script from the CAL_REID project root directory
# cd /home/mossbee/Work/twin_verification/adaptation/CAL_REID

# Path to your trained model
MODEL_PATH="path/to/your/resnet50_latest.pth"

# Path to your pairs file (create this file with your image pairs)
PAIRS_FILE="/kaggle/input/nd-twin-256/test_pairs.txt"

# Configuration file
CONFIG_FILE="configs/softmax_triplet.yml"

# Run verification (make sure you're in the project root directory)
python tools/verify.py \
    --config_file $CONFIG_FILE \
    --pairs_file $PAIRS_FILE \
    --weights $MODEL_PATH \
    --batch_size 64 \
    --device cuda

echo "Verification completed!"
