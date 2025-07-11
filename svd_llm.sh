#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python SVD-LLM.py \
    --ratio 0.6 \
    --calib-samples 256 \
    --calib-dataset "wikitext2" \
    --calib-batch-size 4 \
    --batch-size 8 \
    --seed 42 \
    --output-dir-base "results/wsvd_llm_pplg" \
    --grads-path "grads/llama7b_gradsppl_out.pt" \
    --dynamic-rank False \
    --dynamic-rank-alpha 0.0 \