#!/bin/bash

echo "=================================================="
echo "NCM Evaluation with CS@x% RIER metrics"
echo "=================================================="

cd /workspace

python -m src.eval_ncm experiment=ncm_ssi

echo "=================================================="
echo "NCM evaluation completed!"
echo "=================================================="
