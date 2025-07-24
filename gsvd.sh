#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python GSVD.py \
    --ratio 0.2 \
    --calib-samples 256 \
    --calib-dataset "wikitext2" \
    --calib-batch-size 10 \
    --gradient-epochs 50 \
    --batch-size 8 \
    --output-dir-base "results/gsvd_hmsecos" \
    --grads-path "grads/llama7b_gradsppl_out.pt" \

# set -euo pipefail

# # list of ratios to try, one by one
# for ratio in 0.2 0.4 0.8; do
#   echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running WGSVD with --ratio $ratio"
#   python WGSVD.py \
#     --gradient-iters 500 \
#     --matrix-iters 1 \
#     --ratio "$ratio" \
#     --calib-samples 256
#   echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finished ratio $ratio"
#   echo "---------------------------------------------------------------"
# done

# echo "All jobs complete at $(date +'%Y-%m-%d %H:%M:%S')"