#!/bin/bash
# Complete NCM Pipeline: Extract features and train NCM

set -e

echo "=================================================="
echo "NCM Complete Pipeline"
echo "=================================================="

EXPERIMENT="ncm_ssi"

# Check if joint model checkpoint exists
CHECKPOINT="exp/joint_ssi/best.ckpt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Joint model checkpoint not found at $CHECKPOINT"
    echo "Please run joint training first: bash scripts/run_joint.sh"
    exit 1
fi

echo ""
echo "Step 1: Extracting NCM features..."
echo "=================================================="
python -m src.extract_features \
    experiment=${EXPERIMENT} \
    "$@"

# Check if features were created
FEATURES_PATH="data/ncm_features/features.pkl"
if [ ! -f "$FEATURES_PATH" ]; then
    echo "ERROR: Features not created at $FEATURES_PATH"
    exit 1
fi

echo ""
echo "Step 2: Training NCM model..."
echo "=================================================="
python -m src.train_ncm \
    experiment=${EXPERIMENT} \
    "$@"

echo ""
echo "=================================================="
echo "NCM pipeline completed!"
echo "=================================================="
