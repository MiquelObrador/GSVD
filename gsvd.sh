#!/bin/bash

# This script runs the GSVD algorithm on a given dataset.

python GSVD.py \
    --gradient-iters 500 \
    --matrix-iters 1 \
    --ratio 0.2 \
    --calib-samples 256 \