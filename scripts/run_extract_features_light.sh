#!/bin/bash

echo "=================================================="
echo "Extracting Lightweight NCM Features"
echo "=================================================="

cd /workspace

python -m src.extract_features_light experiment=ncm_ssi_light

echo "=================================================="
echo "Lightweight feature extraction completed!"
echo "=================================================="
