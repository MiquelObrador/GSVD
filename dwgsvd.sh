#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python DWGSVD.py \
    --gradient-iters 500 \
    --matrix-iters 1 \
    --ratio 0.6 \
    --calib-samples 512 \
    --dynamic-delta 0.10 \

# set -euo pipefail

# # list of ratios to try, one by one
# for ratio in 0.2 0.4 0.8; do
#   echo "[$(date +'%Y-%m-%d %H:%M:%S')] Running DWGSVD with --ratio $ratio"
#   python DWGSVD.py \
#     --gradient-iters 500 \
#     --matrix-iters 1 \
#     --ratio "$ratio" \
#     --calib-samples 256 \
#     --dynamic-delta 0.10
#   echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finished ratio $ratio"
#   echo "---------------------------------------------------------------"
# done

# echo "All DWGSVD jobs complete at $(date +'%Y-%m-%d %H:%M:%S')"