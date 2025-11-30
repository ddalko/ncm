#!/bin/bash
# Extract NCM features from trained joint model

set -e

echo "=================================================="
echo "Extracting NCM Features"
echo "=================================================="

# Configuration
EXPERIMENT="ncm_ssi"
CONFIG_PATH="configs/experiment/${EXPERIMENT}.yaml"

# Run feature extraction
python -m src.extract_features \
    experiment=${EXPERIMENT} \
    "$@"

echo "=================================================="
echo "Feature extraction completed!"
echo "=================================================="
