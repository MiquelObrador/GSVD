import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (replace_module_by_name, get_calib_train_data,
                   get_truncate, get_unique_path, load_eval_tokenized_dataset)
from modules import SVDLinearLayer, WeightedMSELoss, HybridLoss
from tqdm import tqdm
import os
import argparse
import csv
import gc # --- CHANGE: Import garbage collector
from transformers.activations import ACT2FN

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
parser.add_argument('--gradient-epochs', type=int, default=50,
                    help='Number of epochs for gradient descent calibration (default: 500)')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size for calibration data loading (default: 4)')
parser.add_argument('--output-dir-base', type=str, default="results",
                    help='Base directory to save results (default: results)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--grads-path', type=str, default=None,
                    help='Path to the precomputed gradients importance weights (default: grads/llama7b_grads.pkl)')

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Use arguments parsed from CLI
model_name = args.model_name
RATIO = args.ratio
CALIB_SAMPLES = args.calib_samples
CALIB_DATASET = args.calib_dataset
CALIB_BATCH_SIZE = args.calib_batch_size
GRADIENT_EPOCHS = args.gradient_epochs
BATCH_SIZE = args.batch_size
OUTPUT_DIR_BASE = args.output_dir_base
SEED = args.seed
GRADS_PATH = args.grads_path
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
print(f"Gradient Epochs: {GRADIENT_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Output Directory Base: {OUTPUT_DIR_BASE}")
print(f"Seed: {SEED}")
print(f"Grads Path: {GRADS_PATH}")

print(f"Loading model: {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

og_num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters before GSVD compression:", og_num_params)

# Securely get activation function from config, fallback to nn.Identity if not present
activation_fn = ACT2FN[model.config.hidden_act] if hasattr(model.config, "hidden_act") and model.config.hidden_act in ACT2FN else nn.Identity()
print(f"Activation Function: {getattr(activation_fn, '__name__', activation_fn.__class__.__name__)}")

# Securely get sequence length from config, fallback to 2048 if not present
SEQLEN = getattr(model.config, "max_position_embeddings", 2048)
print(f"Sequence Length: {SEQLEN}")

print("Loading calibration data...")
calib_data = get_calib_train_data(name=CALIB_DATASET,
                                  tokenizer=tokenizer,
                                  seqlen=SEQLEN,
                                  batch_size=BATCH_SIZE,
                                  nsamples=CALIB_SAMPLES,
                                  seed=SEED)
print("Calibration data loaded.")

losses_per_module = {}
completed_layers = set() # --- CHANGE: Keep track of completed layers

# --- CHANGE: Checkpoint logic added ---
checkpoint_path = f"{OUTPUT_DIR_BASE}/gsvd_checkpoint.pth"
if os.path.exists(checkpoint_path):
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    completed_layers = checkpoint['completed_layers']
    losses_per_module = checkpoint['losses_per_module']
    
    # Replace completed layers with empty SVDLinearLayer instances
    for name in tqdm(completed_layers, desc="Replacing completed layers", total=len(completed_layers)):
        original_module = model.get_submodule(name)
        if not isinstance(original_module, nn.Linear):
            continue
        
        u_key = f"{name}.u_linear.weight"
        vt_key = f"{name}.vt_linear.weight"
        rank = state_dict[u_key].shape[1]
        
        # SVD needs a vt parameter and a u_parameter, we can create a dummy ones to initialize the SVD layer and the load the weights later
        # Create dummy parameters for SVD
        vt_parameter = torch.zeros(rank, original_module.in_features)
        u_parameter = torch.zeros(original_module.out_features, rank)
        
        svd_layer = SVDLinearLayer(
            vt_parameter=vt_parameter,
            u_parameter=u_parameter,
            bias=original_module.bias.clone().detach() if original_module.bias is not None else None,
        )
        # Replace the original layer with the SVD layer
        replace_module_by_name(model, name, svd_layer)
        
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")
    print(f"Resuming GSVD compression with {len(completed_layers)} completed layers.")
# --- End of Checkpoint logic ---
print(completed_layers)
# Load grads if provided
if GRADS_PATH is not None:
    print(f"Using precomputed gradients from {GRADS_PATH}")
    with open(GRADS_PATH, 'rb') as f:
        grads = torch.load(f)
else:
    print("No precomputed gradients provided, using default behavior.")

print(" Starting GSVD compression...")

# 1. Get only the names of the linear layers first.
linear_layer_names = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and name != "lm_head":
        linear_layer_names.append(name)

# 2. Iterate through the names and fetch the module inside the loop.
# Reverse the list to process from the last layer to the first
for name in tqdm(reversed(linear_layer_names), desc="Modules", total=len(linear_layer_names)):
    # To check if the layer has already been processed use the name till the last dot
    if any(name.startswith(completed_name) for completed_name in completed_layers):
        print(f"Skipping already processed layer: {name}")
        continue
    
    # Fetch the specific module from the model *right before* you use it.
    module = model.get_submodule(name)
    
    # --- The rest of your loop logic remains the same ---
    torch.cuda.empty_cache()

    # Get the activations of the module
    activations = []
    def forward_hook(module, input, output):
        activations.append(input[0].detach().cpu())

    # The hook is registered on the specific module we just fetched
    hook = module.register_forward_hook(forward_hook)
    model.eval()
    model.half()
    model.to(DEVICE)
    with torch.no_grad():
        for batch in tqdm(calib_data, desc=f"Getting activations for {name}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    hook.remove()

    # --- CHANGE: Aggressively clear VRAM after activation gathering ---
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    # --- End of VRAM clearing ---
    
    activations = torch.cat(activations, dim=0).float()

    module.to(DEVICE).float()

    W = module.weight.data.clone().detach().to(DEVICE)
    b = module.bias.data.clone().detach().to(DEVICE) if module.bias is not None else None

    out_features, in_features = W.shape
    low_rank = get_truncate(in_features, out_features, RATIO)
    print(f"Applying GSVD to {name} with low rank: {low_rank}")

    U, S, VT = torch.linalg.svd(W, full_matrices=False)
    s_sqrt = torch.diag(torch.sqrt(S))
    u_parameter = torch.matmul(U[:, :low_rank], s_sqrt[:low_rank, :low_rank])
    vt_parameter = torch.matmul(s_sqrt[:low_rank, :low_rank], VT[:low_rank, :])

    svd_layer = SVDLinearLayer(
        vt_parameter=vt_parameter,
        u_parameter=u_parameter,
        bias=b,
    )
    svd_layer.to(DEVICE).train()
    
    # If the layer is a mlp.gate_proj let's calibrate it with the activation function to ensure a more accurate representation
    if 'mlp.gate_proj' in name:
        activation = activation_fn
    else:
        activation = nn.Identity()

    del W, U, S, VT, s_sqrt, u_parameter, vt_parameter, b
    torch.cuda.empty_cache()
    
    importance_weights = grads[name] if GRADS_PATH is not None else None
    #Ensure no 0 importance weights are passed, if they are, use a threshold of 0.01
    if importance_weights is not None:
        importance_weights = torch.where(importance_weights < 0.01, torch.tensor(0.01, device=importance_weights.device), importance_weights)

    loss_fn = HybridLoss(weights=importance_weights,
                        alpha=0.5,  # Adjust alpha as needed
                        reduction="mean").to(DEVICE)
    
    optimizer = torch.optim.AdamW(svd_layer.parameters(), lr=1e-4, weight_decay=1e-3)

    print(f"Training GSVD layer for {name}...")
    activation_dataset = torch.utils.data.TensorDataset(activations)
    activation_loader = torch.utils.data.DataLoader(
        activation_dataset,
        batch_size=CALIB_BATCH_SIZE,
        shuffle=True
    )
    
    loss_log = []
    for epoch in range(GRADIENT_EPOCHS):
        epoch_loss_log = []
        pbar = tqdm(activation_loader, desc=f"Epoch {epoch+1}/{GRADIENT_EPOCHS}", leave=False)

        for batch_tuple in pbar:
            batch = batch_tuple[0].to(DEVICE)
            optimizer.zero_grad()
            output = activation(svd_layer(batch))
            # The target needs to be computed on the GPU with the original module
            with torch.no_grad():
                target = activation(module(batch))
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss_log.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        loss_log.append(sum(epoch_loss_log) / len(epoch_loss_log))

    print(f"Finished training GSVD layer for {name}. Loss: {loss_log[-1]}")
    losses_per_module[name] = loss_log[-1]
    
    svd_layer.to('cpu').eval().half()
    replace_module_by_name(model, name, svd_layer)
    
    # --- CHANGE: Clean up and save checkpoint ---
    del svd_layer, activations, loss_fn, optimizer, module
    completed_layers.add(name)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Saving checkpoint after completing layer {name}...")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'completed_layers': completed_layers,
        'losses_per_module': losses_per_module,
    }
    torch.save(checkpoint, checkpoint_path)
    # --- End of cleanup and checkpointing ---

print("GSVD compression completed.")

# ... (The rest of the script for final saving and evaluation is unchanged)

model.to('cpu')
torch.cuda.empty_cache()

filename = f"gsvd_{model_name.split('/')[-1]}_r{RATIO}_g{GRADIENT_EPOCHS}_cb{CALIB_BATCH_SIZE}"
save_path = get_unique_path(f"{OUTPUT_DIR_BASE}/{filename}.pt")
losses_path = get_unique_path(f"{OUTPUT_DIR_BASE}/{filename}_losses.csv")
ppl_path = get_unique_path(f"{OUTPUT_DIR_BASE}/{filename}_ppl.txt")

torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# If we reached this point, we can safely delete the model checkpoint
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)
    print(f"Checkpoint removed: {checkpoint_path}")

with open(losses_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Module', 'Loss'])
    for name, loss in losses_per_module.items():
        writer.writerow([name, loss])
print(f"Losses saved to {losses_path}")

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
print("\nGSVD Compression complete.")

final_params = sum(p.numel() for p in model.parameters())
print("Number of parameters after GSVD compression:", final_params)
model_size_gb = final_params * 4 / (1024 ** 3)
print(f"Model size after GSVD compression: {model_size_gb:.2f} GB")
print(f"Compression ratio: {final_params / og_num_params:.2f}")
print(f"Perplexity obtained after GSVD compression: {ppl}")

with open(ppl_path, "w") as f:
    f.write(str(ppl))