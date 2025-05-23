#!/bin/bash

python evaluate.py \
    --model_base "huggyllama/llama-7b" \
    --model_path "results/llama-7b/gsvd_llama-7b_r0.6_g500_c512_m1_d0.1_w.pt" \
    --ratios_path "results/llama-7b/gsvd_llama-7b_r0.6_g500_c512_m1_d0.1_ratios_dict_w.csv" \
    --eval_metric "ppl" \
    --eval_dataset "ptb" \

# #!/usr/bin/env bash
# set -euo pipefail

# MODEL_BASE="huggyllama/llama-7b"
# MODEL_PATH="results/llama-7b/gsvd_llama-7b_r0.6_g500_c256_m1_d0.1_w.pt"
# RATIOS_PATH="results/llama-7b/gsvd_llama-7b_r0.6_g500_c256_m1_d0.1_ratios_dict_w.csv"
# EVAL_METRIC="accuracy"

# DATASETS=(
#   "arc_easy"
#   "arc_challenge"
#   "openbookqa"
#   "winogrande"
#   "hellaswag"
#   "piqa"
#   "mathqa"
# )

# for DS in "${DATASETS[@]}"; do
#   echo "=== Evaluating on ${DS} ==="
#   python evaluate.py \
#     --model_base "$MODEL_BASE" \
#     --model_path "$MODEL_PATH" \
#     --ratios_path "$RATIOS_PATH" \
#     --eval_metric "$EVAL_METRIC" \
#     --eval_dataset "$DS"
#   echo
# done

# echo "All evaluations complete!"
