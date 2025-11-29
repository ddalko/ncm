#!/bin/bash
# Decode test set using two-pass ASR (RNN-T + LAS)

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
CHECKPOINT="exp/joint_ssi/best.ckpt"
OUTPUT_PATH="exp/joint_ssi/decode_2pass.jsonl"

echo "Decoding with Two-Pass ASR (RNN-T + LAS rescoring)..."
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_PATH}"

python -m src.decode_two_pass \
    model.checkpoint=${CHECKPOINT} \
    decoding.output_path=${OUTPUT_PATH} \
    "$@"

echo "Decoding completed!"
