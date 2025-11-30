#!/bin/bash
# Train Neural Confidence Model (NCM)

set -e

echo "=================================================="
echo "Training NCM"
echo "=================================================="

# Configuration
EXPERIMENT="ncm_ssi"

# Run NCM training
python -m src.train_ncm \
    experiment=${EXPERIMENT} \
    "$@"

echo "=================================================="
echo "NCM training completed!"
echo "=================================================="
