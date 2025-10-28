#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python GSVD.py \
    --model-name "meta-llama/Llama-3.1-8B" \
    --ratio 0.4 \
    --calib-samples 256 \
    --calib-dataset "wikitext2" \
    --calib-batch-size 10 \
    --gradient-epochs 50 \
    --batch-size 4 \
    --output-dir-base "results/llama3" \
    --grads-path "grads/Llama-3.1-8B_gradsppl_out.pt" \
    #--superweight "layers.2.mlp.down_proj" \

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