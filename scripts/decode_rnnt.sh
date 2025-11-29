#!/bin/bash
# Decode test set using RNN-T only

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
CHECKPOINT="exp/joint_ssi/best.ckpt"
OUTPUT_PATH="exp/joint_ssi/decode_rnnt.jsonl"

echo "Decoding with RNN-T (streaming, first-pass)..."
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_PATH}"

python -m src.decode_rnnt \
    model.checkpoint=${CHECKPOINT} \
    decoding.output_path=${OUTPUT_PATH} \
    "$@"

echo "Decoding completed!"
