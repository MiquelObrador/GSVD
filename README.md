# GSVD Compression for Large Language Models

This repository contains code for compressing large language models using a **gradient-optimized SVD** (GSVD) method. The approach replaces `nn.Linear` layers in Transformer-based models with low-rank approximations optimized via gradient descent, significantly reducing model size with minimal loss in performance.

## Overview

This project enables:

- **Low-rank compression** of Hugging Face-compatible models using SVD.
- **Layer-wise gradient-based optimization** of decomposed matrices to minimize reconstruction error.
- **Per-layer performance tracking** (loss deltas, final reconstruction loss).
- Evaluation using **perplexity** on WikiText for compressed model quality.

## Features

- Gradient-based fine-tuning of compressed layers  
- Supports any Hugging Face causal language model  
- Handles GPU memory efficiently during compression  
- Saves compressed model, loss metrics, and perplexity  

## Compression Pipeline

1. Load a pretrained causal language model (e.g., LLaMA 7B)  
2. Collect activations using a small calibration set (WikiText)  
3. Apply SVD with a configurable compression ratio  
4. Optimize the decomposed matrices with gradient descent  
5. Replace original layers with optimized low-rank modules  
6. Evaluate perplexity of the final model  

## Usage

### Requirements

- PyTorch  
- Hugging Face Transformers  
- tqdm  

Install requirements with:

```bash
pip install torch transformers tqdm
```

## Run Compression

```bash
python main.py \
  --model-name huggyllama/llama-7b \
  --gradient-iters 500 \
  --ratio 0.6 \
  --calib-samples 256 \
  --batch-size 4 \
  --output-dir-base results

```

## Arguments

| Argument          | Description                                   | Default                |
|-------------------|-----------------------------------------------|------------------------|
| `--model-name`    | Hugging Face model ID                        | `huggyllama/llama-7b` |
| `--gradient-iters`| Number of optimization steps per layer        | `500`                 |
| `--ratio`         | Target SVD compression ratio                 | `0.6`                 |
| `--calib-samples` | Calibration dataset samples (WikiText)        | `256`                 |
| `--batch-size`    | Batch size for calibration                   | `4`                   |
| `--output-dir-base`| Base folder to save results                 | `results`             |

## Output

The script saves:

- *.pt: compressed model weights

- *_losses.txt: loss deltas for each module

- *_ppl.txt: model perplexity after compression

## License

This project is licensed under the MIT License.