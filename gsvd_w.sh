#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python GSVD_W.py \
    --gradient-iters 500 \
    --matrix-iters 1 \
    --ratio 0.6 \
    --calib-samples 256 \