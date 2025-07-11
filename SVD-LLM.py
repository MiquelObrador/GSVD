import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (replace_module_by_name, get_calib_train_data,
                   get_truncate, get_unique_path, load_eval_tokenized_dataset,
                   find_layers, calculate_truncation_ranks)
from modules import SVDLinearLayer
from tqdm import tqdm
import os
import argparse
import gc # --- CHANGE: Import garbage collector
from collections import OrderedDict

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Compress a Hugging Face model using GSVD.")
parser.add_argument('--model-name', type=str, default="huggyllama/llama-7b",
                    help='Hugging Face model identifier (default: huggyllama/llama-7b)')
parser.add_argument('--ratio', type=float, default=0.6,
                    help='Compression ratio target for SVD (default: 0.6)')
parser.add_argument('--calib-samples', type=int, default=256,
                    help='Number of calibration samples from WikiText (default: 256)')
parser.add_argument('--calib-dataset', type=str, default="wikitext2",
                    help='Calibration dataset to use (default: wikitext2)')
parser.add_argument('--calib-batch-size', type=int, default=10,
                    help='Batch size for calibration gradient descent (default: 10)')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size for calibration data loading (default: 4)')
parser.add_argument('--output-dir-base', type=str, default="results",
                    help='Base directory to save results (default: results)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--grads-path', type=str, default=None,
                    help='Path to the precomputed gradients importance weights')
parser.add_argument('--dynamic-rank', type=bool, default=False,
                    help='Enable dynamic rank allocation based on layer importance, you must provide --grads-path')
parser.add_argument('--dynamic-rank-alpha', type=float, default=0.5,
                    help='Alpha smoothing value for dynamic rank allocation (default: 0.5)')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Use arguments parsed from CLI
model_name = args.model_name
RATIO = args.ratio
CALIB_SAMPLES = args.calib_samples
CALIB_DATASET = args.calib_dataset
CALIB_BATCH_SIZE = args.calib_batch_size
BATCH_SIZE = args.batch_size
OUTPUT_DIR_BASE = args.output_dir_base
SEED = args.seed
GRADS_PATH = args.grads_path
DYNAMIC_RANK = args.dynamic_rank
DR_ALPHA = args.dynamic_rank_alpha
# Create output directory if it doesn't exist
if not os.path.exists(f"{OUTPUT_DIR_BASE}"):
    os.makedirs(f"{OUTPUT_DIR_BASE}")
    
# Set seed for reproducibility and deterministic results
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For multi-GPU setups
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"--- Configurations ---")
# ... (rest of config printing is unchanged)
print(f"Model Name: {model_name}")
print(f"Compression Ratio: {RATIO}")
print(f"Calibration Samples: {CALIB_SAMPLES}")
print(f"Calibration Dataset: {CALIB_DATASET}")
print(f"Calibration Batch Size: {CALIB_BATCH_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Output Directory Base: {OUTPUT_DIR_BASE}")
print(f"Seed: {SEED}")
print(f"Grads Path: {GRADS_PATH}")
print(f"Dynamic Rank: {DYNAMIC_RANK}")

print(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

# --- CHANGE: Load gradient importance weights ---
grad_weights = None
if GRADS_PATH:
    if os.path.exists(GRADS_PATH):
        print(f"Loading gradient importance weights from {GRADS_PATH}...")
        with open(GRADS_PATH, 'rb') as f:
            grad_weights = torch.load(f)
        print("Gradient weights loaded.")
    else:
        print(f"Warning: Gradients path specified but not found: {GRADS_PATH}")
        DYNAMIC_RANK = False  # Disable dynamic rank if gradients are not available
        
if DYNAMIC_RANK and grad_weights is not None:
    print("Calculating dynamic ranks based on gradient importance weights...")
    importance_avg = OrderedDict()
    for layer_name, importance in grad_weights.items():
        importance_avg[layer_name] = torch.mean(importance).item()
    
    final_ranks, _ = calculate_truncation_ranks(model=model,
                                                importance_dict=importance_avg,
                                                compression_ratio=RATIO,
                                                smoothing_alpha=DR_ALPHA)
    
    print("Dynamic ranks calculated.")
    
else:
    final_ranks = None
    print("Dynamic rank allocation is disabled or gradients are not provided.")

og_num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters before GSVD compression:", og_num_params)

SEQLEN = model.config.max_position_embeddings
print(f"Sequence Length: {SEQLEN}")

print("Loading calibration data...")
calib_data = get_calib_train_data(name=CALIB_DATASET,
                                  tokenizer=tokenizer,
                                  seqlen=SEQLEN,
                                  batch_size=CALIB_BATCH_SIZE,
                                  nsamples=CALIB_SAMPLES,
                                  seed=SEED)
print("Calibration data loaded.")

model.to(DEVICE)

if "llama" in model_name or "mistral" in model_name or "vicuna" in model_name:
    layers = model.model.layers
elif "opt" in model_name:
    layers = model.model.decoder.layers

print("Start obtaining the whitening matrix...")

def hook(module, input, output):
    inp = input[0].detach().float()
    if inp.dim() == 2:
        inp = inp.unsqueeze(0)
    module.raw_scaling_diag_matrix += torch.sum(torch.matmul(inp.transpose(1, 2), inp), dim=0).to("cpu")
    del inp
    torch.cuda.empty_cache()  # --- CHANGE: Clear cache after each hook call
    
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and "lm_head" not in name:
        module.raw_scaling_diag_matrix = 0
        module.register_forward_hook(hook)
        
with torch.no_grad():
    for batch in tqdm(calib_data, desc="Calculating whitening matrix"):
        inputs = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = model(inputs, attention_mask=attention_mask, use_cache=False)
        
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module._forward_hooks.clear()  # Clear hooks to free memory
        
torch.cuda.empty_cache()  # --- CHANGE: Clear cache after calculating whitening matrix
model.to("cpu")  # Move model back to CPU to save memory

for i in range(len(layers)):
    subset = find_layers(layers[i])
    for name in subset:
        subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
        
profiling_mat = {}
print("Start Cholesky Decomposition...")

for name, module in tqdm(model.named_modules(), desc="Cholesky Decomposition", total=len(list(model.named_modules()))):
    if isinstance(module, nn.Linear) and hasattr(module, 'raw_scaling_diag_matrix'):
        raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(DEVICE)
        try:
            scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
            raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(DEVICE)
            scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            eigenvalues = None
            del eigenvalues
        scaling_diag_matrix = scaling_diag_matrix.cpu()
        profiling_mat[name] = scaling_diag_matrix
        scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
        del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
        torch.cuda.empty_cache()
    
print("Start SVD decomposition after whitening...")

for name, module in tqdm(model.named_modules(), desc="SVD Decomposition", total=len(list(model.named_modules()))):
    if isinstance(module, nn.Linear) and name in profiling_mat:
        W = module.weight.data.float().to(DEVICE)
        b = module.bias.data.float().to(DEVICE) if module.bias is not None else None
        dtype = W.dtype
        scaling_diag_matrix = profiling_mat[name].to(DEVICE)
        
        if grad_weights and name in grad_weights:
            
            importance_scores = grad_weights[name].to(DEVICE).float()
            
            importance_scores = torch.clamp(importance_scores, min=0.01, max=1.0)
            
        try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
        except Exception as e:
            print("Warning: scaling_diag_matrix is not full rank!")
            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(DEVICE)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            
        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
        
        W_scale = torch.matmul(torch.matmul(torch.diag(importance_scores), W), scaling_diag_matrix)
        
        U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
        
        if DYNAMIC_RANK and name in final_ranks:
            
            num_s_after_trunc = final_ranks[name]
            
        else:
            num_s_after_trunc = get_truncate(in_features=W.shape[0],
                                             out_features=W.shape[1],
                                             ratio=RATIO)
        
        truc_s = S[:num_s_after_trunc]
        truc_u = torch.matmul(torch.linalg.inv(torch.diag(importance_scores)), U[:, :num_s_after_trunc])
        truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
        
        truc_sigma = torch.diag(truc_s)

        sqrtSigma = torch.sqrt(truc_sigma)
        svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
        svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)
        
        svd_layer = SVDLinearLayer(
            vt_parameter=svd_v,
            u_parameter=svd_u,
            bias=b,
        )
        
        replace_module_by_name(model, name, svd_layer)
        W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
        del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        if grad_weights and name in grad_weights:
            del importance_scores  # Clear importance scores to free memory
        
        profiling_mat[name] = None  # Clear profiling matrix entry
        del profiling_mat[name]  # Remove entry from profiling matrix
    torch.cuda.empty_cache()  # --- CHANGE: Clear cache after processing each layer
    gc.collect()  # --- CHANGE: Collect garbage to free memory
    
print("SVD decomposition completed.")

model.to("cpu")  # Move model back to GPU for further processing
torch.cuda.empty_cache()  # Clear cache after SVD decomposition

filename = f"SVD_LLM_{model_name.split('/')[-1]}_r{RATIO}_cs{CALIB_SAMPLES}_s{SEED}"
save_path = get_unique_path(f"{OUTPUT_DIR_BASE}/{filename}.pt")
ppl_path = get_unique_path(f"{OUTPUT_DIR_BASE}/{filename}_ppl.txt")

torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Evaluate the model on the calibration dataset
print("Evaluating model on calibration dataset...")
with torch.no_grad():
    model.eval()
    model.half()
    model.to(DEVICE)
    
    BATCH_SIZE = 8
    SEQ_LEN = model.config.max_position_embeddings
    
    loader = load_eval_tokenized_dataset(tokenizer,
                            seq_len=SEQ_LEN,
                            batch_size=BATCH_SIZE)
    nlls = []
    for batch in tqdm(loader, desc="Calculating PPL", total=len(loader)):
        batch = batch.to(DEVICE)
        logits = model(input_ids=batch, use_cache=False).logits
        if torch.isfinite(logits).all():
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.cpu())
        else:
            print("Bad logits detected, skipping batch.")
            continue
        
    mean_loss = torch.cat(nlls).mean()
    ppl = torch.exp(mean_loss).item()
    if ppl > 1000:
        ppl = int(ppl)
        
final_params = sum(p.numel() for p in model.parameters())
print("Number of parameters after GSVD compression:", final_params)
model_size_gb = final_params * 4 / (1024 ** 3)
print(f"Model size after GSVD compression: {model_size_gb:.2f} GB")
print(f"Compression ratio: {final_params / og_num_params:.2f}")
print(f"Perplexity obtained after GSVD compression: {ppl}")

with open(ppl_path, "w") as f:
    f.write(str(ppl))
