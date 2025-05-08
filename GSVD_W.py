import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import replace_module_by_name, load_wikitext, get_ratios, get_truncate, get_unique_path
from modules import SVDLinearLayer
from tqdm import tqdm
import os
import argparse
import pickle

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Compress a Hugging Face model using GSVD.")
parser.add_argument('--model-name', type=str, default="huggyllama/llama-7b",
                    help='Hugging Face model identifier (default: huggyllama/llama-7b)')
parser.add_argument('--gradient-iters', type=int, default=500,
                    help='Number of gradient descent iterations per layer (default: 500)')
parser.add_argument('--matrix-iters', type=int, default=1,
                    help='Number of matrix iterations (used in filename, default: 1)') # Note: This variable isn't used in the core logic beyond filename
parser.add_argument('--ratio', type=float, default=0.6,
                    help='Compression ratio target for SVD (default: 0.6)')
parser.add_argument('--calib-samples', type=int, default=256,
                    help='Number of calibration samples from WikiText (default: 256)')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Batch size for calibration data loading (default: 4)')
parser.add_argument('--output-dir-base', type=str, default="results",
                    help='Base directory to save results (default: results)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')

args = parser.parse_args()

# Set seed for reproducibility and deterministic results
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Use arguments parsed from CLI
model_name = args.model_name
GRADIENT_ITERS = args.gradient_iters
MATRIX_ITERS = args.matrix_iters
RATIO = args.ratio
CALIB_SAMPLES = args.calib_samples
BATCH_SIZE = args.batch_size

output_dir = f"{args.output_dir_base}/{model_name.split('/')[-1]}"

print(f"--- Configuration ---")
print(f"Model: {model_name}")
print(f"Gradient Iterations: {GRADIENT_ITERS}")
print(f"Matrix Iterations (filename): {MATRIX_ITERS}")
print(f"Compression Ratio: {RATIO}")
print(f"Calibration Samples: {CALIB_SAMPLES}")
print(f"Calibration Batch Size: {BATCH_SIZE}")
print(f"Output Directory: {output_dir}")
print(f"Seed: {args.seed}")
print(f"--------------------")


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    

print(f"Loading model: {model_name}...")
# Consider adding error handling for model/tokenizer loading
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

og_num_params = sum(p.numel() for p in model.parameters())
print("Number of parameters before GSVD compression:", og_num_params)

print("Loading calibration data...")
calib_data = load_wikitext(tokenizer,
                            seq_len=2048,
                            batch_size=BATCH_SIZE,
                            samples=CALIB_SAMPLES)

print("Loaded calibration data.")

model.eval()
model.to(DEVICE)

def hook(module, input, output):
    """
    Hook function to store the pre-activation and post-activation outputs of a module.
    """
    # Move to CPU immediately to free GPU memory, clone to avoid modifying original tensor downstream
    current_pre_act = input[0].detach().cpu()
    current_post_act = output.detach().cpu()

    # Ensure tensors are concatenated correctly, even if initial tensor is empty
    module.pre_act = torch.cat((module.pre_act, current_pre_act), dim=0) if module.pre_act.numel() != 0 else current_pre_act
    module.post_act = torch.cat((module.post_act, current_post_act), dim=0) if module.post_act.numel() != 0 else current_post_act
    
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.pre_act = torch.empty(0).to(DEVICE)
        module.post_act = torch.empty(0).to(DEVICE)
        module.register_forward_hook(hook)
        
with torch.no_grad():
    print("Running calibration...")
    for batch in tqdm(calib_data, desc="Calibrating", total=len(calib_data)):
        batch = batch.to(DEVICE)
        model(input_ids=batch, use_cache=False)
        
print("Calibration complete.")

for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module._forward_hooks.clear()  # Clear hooks to avoid memory leaks
        
model.to("cpu")
torch.cuda.empty_cache()

with open("grads/llama7b_weights.pkl", "rb") as f:
    grads = pickle.load(f)

class WeightedFrobeniusLoss(nn.Module):
    def __init__(self, feature_weights: torch.Tensor):
        """
        feature_weights: 1D tensor of shape (num_features,), with values in [0,1]
        """
        super().__init__()
        # make sure it's the right shape and on the right device
        self.register_buffer('w', feature_weights.view(1, -1))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # inputs, targets: (batch_size, num_features)
        diff = inputs - targets                     # (B, F)
        weighted_diff = diff * self.w                # broadcast w over batch
        # flatten and take Frobenius (i.e. ℓ₂) norm
        return torch.norm(weighted_diff.view(-1), p='fro')

losses_per_module = {}

print("Starting GSVD Compression...")

for name, module in tqdm(list(model.named_modules()), desc="Modules"):
    if isinstance(module, nn.Linear):
        if name == "lm_head":
            continue
        torch.cuda.empty_cache()
        
        ratios = get_ratios(RATIO, MATRIX_ITERS)
        
        importance = grads[name]
        importance = importance.to(DEVICE).float()
        
        W = module.weight.data.clone().to(DEVICE).float()
        b = module.bias.data.clone().to(DEVICE).float() if module.bias is not None else None
        
        pre_act = module.pre_act.to(DEVICE).float()
        post_act = module.post_act.to(DEVICE).float()
        
        module.pre_act = torch.empty(0)
        module.post_act = torch.empty(0)
        
        out_features, in_features = W.shape
        
        for i, ratio in enumerate(ratios):
            
            loss_fn = WeightedFrobeniusLoss(importance)
            
            print(f"Optimizing {name} with ratio {ratio:.2f}")
            
            low_rank = get_truncate(in_features, out_features, ratio)
                
            new_module = SVDLinearLayer(weights=W,
                                        bias=b,
                                        truncate=low_rank,
                                        data=pre_act,
                                        from_savepoint=False)
            
            new_module.train()
            new_module.to(DEVICE)
            
            with torch.no_grad():
                starting_loss = loss_fn(new_module(pre_act), post_act).item()
            
            if i == 0:
                initial_loss = starting_loss
                
            # Calculate inital lr based on the initial loss
            
            if starting_loss > 30:
                initial_lr = 0.0001
            else:
                initial_lr = 0.00001
            
            optimizer = torch.optim.Adam(new_module.parameters(), lr=initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GRADIENT_ITERS, eta_min=0.00001)
            pbar = tqdm(range(GRADIENT_ITERS), desc=f"Optimizing {name}", leave=False)
            for _ in pbar:
                optimizer.zero_grad()
                output_pred = new_module(pre_act)
                loss = loss_fn(output_pred, post_act)
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({"loss": loss.item()})
                
            optimizer.zero_grad()
            
            new_module.eval()
            
            if len(ratios) > 1:
                with torch.no_grad():
                    W = new_module.reconstruct_weights().to(DEVICE)
            
        new_module.eval()
        with torch.no_grad():
            final_loss = loss_fn(new_module(pre_act), post_act).detach().cpu().item()
        new_module.to("cpu")
        replace_module_by_name(model, name, new_module)
        loss_delta = final_loss - initial_loss
        losses_per_module[name] = (loss_delta, initial_loss, final_loss)
        del W, b, pre_act, post_act, optimizer, loss_fn, new_module, initial_loss, final_loss, loss, loss_delta, output_pred, starting_loss, importance
        torch.cuda.empty_cache()
        
model.to("cpu")
torch.cuda.empty_cache()
        
# Save the modified model, the name should contain the model name, the compression ratio, the number of gradient iterations, teh calibration samples, and the matrix iteration

filename = f"gsvd_{model_name.split('/')[-1]}_r{RATIO}_g{GRADIENT_ITERS}_c{CALIB_SAMPLES}_m{MATRIX_ITERS}"
save_path = get_unique_path(os.path.join(output_dir, filename + "_w.pt"))
losses_path = get_unique_path(os.path.join(output_dir, filename + "_losses_w.txt"))
ppl_path = get_unique_path(os.path.join(output_dir, filename + "_ppl_w.txt"))

torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

with open(losses_path, "w") as f:
    for module_name, loss_delta in losses_per_module.items():
        f.write(f"{module_name}: {loss_delta}\n")
print(f"Loss deltas saved to {losses_path}")

# Calculate the ppl of the model

with torch.no_grad():
    model.eval()
    model.half()
    model.to(DEVICE)
    
    BATCH_SIZE = 8
    SEQ_LEN = model.config.max_position_embeddings
    
    loader = load_wikitext(tokenizer,
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
model_size_gb = final_params * 4 / (1024 ** 3)  # Assuming float32 (4 bytes)
print(f"Model size after GSVD compression: {model_size_gb:.2f} GB")

# Compare the initial and final model number of parameters

print(f"Compression ratio: {final_params / og_num_params:.2f}")

print(f"Perplexity obtained after GSVD compression: {ppl}")
# Save perplexity to a file with the ratio and iterations in the filename
with open(ppl_path, "w") as f:
    f.write(str(ppl))