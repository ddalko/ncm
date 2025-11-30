#!/bin/bash

echo "=================================================="
echo "Training Lightweight NCM"
echo "=================================================="

cd /workspace

python -m src.train_ncm experiment=ncm_ssi_light

echo "=================================================="
echo "Lightweight NCM training completed!"
echo "=================================================="
