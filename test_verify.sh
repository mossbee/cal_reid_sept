#!/bin/bash

# Test script for verification
# Make sure you have actual image paths in your pairs file

echo "Testing verification script..."

# Check if files exist
if [ ! -f "configs/softmax_triplet.yml" ]; then
    echo "Error: config file not found"
    exit 1
fi

if [ ! -f "/kaggle/input/nd-twin-256/resnet50_latest.pth" ]; then
    echo "Error: model file not found"
    exit 1
fi

if [ ! -f "simple_test_pairs.txt" ]; then
    echo "Error: pairs file not found"
    exit 1
fi

echo "All files found. Running verification..."

python tools/verify.py \
    --config_file configs/softmax_triplet.yml \
    --pairs_file simple_test_pairs.txt \
    --weights /kaggle/input/nd-twin-256/resnet50_latest.pth \
    --batch_size 64 \
    --device cuda

echo "Verification completed!"
