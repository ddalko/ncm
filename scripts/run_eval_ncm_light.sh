#!/bin/bash

echo "=================================================="
echo "Evaluating Lightweight NCM"
echo "=================================================="

cd /workspace

python -m src.eval_ncm experiment=ncm_ssi_light

echo "=================================================="
echo "Lightweight NCM evaluation completed!"
echo "=================================================="
