#!/bin/bash
# Training script for joint RNN-T + LAS model

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training configuration
EXPERIMENT="joint_ssi"
OUTPUT_DIR="exp/${EXPERIMENT}"

echo "Starting joint RNN-T + LAS training..."
echo "Experiment: ${EXPERIMENT}"
echo "Output directory: ${OUTPUT_DIR}"

# Run training
python -m src.train_joint \
    experiment=${EXPERIMENT} \
    training.output_dir=${OUTPUT_DIR} \
    "$@"

echo "Training completed!"
