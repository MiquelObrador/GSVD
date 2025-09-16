import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import replace_module_by_name, get_truncate, get_calib_train_data
from modules import SVDLinearLayerDynamicSigma
from tqdm import tqdm
import gc
import matplotlib.pyplot as plt

RATIO = 0.6
GAMMA = 2.0
LAMBDA = 1e-6
CLIP_VALUE = 10.0
CALIB_DATASET = "wikitext2"
model_name = "huggyllama/llama-7b"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

SEQLEN = model.config.max_position_embeddings
BATCH_SIZE = 4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 30

model.to(DEVICE)
model.eval()

# Freeze all model parameters
for param in model.parameters():
    param.requires_grad = False

for name, module in tqdm(model.named_modules(), desc="Replacing layers", total=len(list(model.named_modules()))):
    if isinstance(module, nn.Linear) and "lm_head" not in name:
        W = module.weight.data.clone().float()
        b = module.bias.data.clone().float() if module.bias is not None else None

        out_features, in_features = W.shape
        low_rank = get_truncate(in_features, out_features, 1)

        new_module = SVDLinearLayerDynamicSigma(W, low_rank, b, from_savepoint=True)
        
        new_module.to("cpu")
        new_module.half()
        new_module.train()
        # Freeze U and VT, and S, only train DRA
        
        replace_module_by_name(model, name, new_module)

        del W, b, new_module
        gc.collect()
        torch.cuda.empty_cache()
        
# Load saved model
model.load_state_dict(torch.load("model_with_dra_layers.pt"))

train_dataloader = get_calib_train_data("wikitext2", tokenizer, nsamples=256, seqlen=SEQLEN, seed=SEED, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

model.train()
model.half()
model.to(DEVICE)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

svd_modules = [module for module in model.modules() if isinstance(module, SVDLinearLayerDynamicSigma)]

training_losses = []
compression_losses = []
dra_regularization_losses = []
crossentropy_losses = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    epoch_loss = 0.0
    epoch_compression_loss = 0.0
    epoch_dra_loss = 0.0
    crossentropy_losses_epoch = 0.0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        inputs_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        outputs = model(input_ids=inputs_ids, attention_mask=attention_mask, labels=inputs_ids)
        L_task = outputs.loss

        crossentropy_losses_epoch += L_task.item()

        total_sigmoid_sum = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
        #L_DRA = torch.tensor(0.0, device=DEVICE)  # Initialize L_DRA as a tensor on the device
        total_elements = 0

        for module in svd_modules:
            dra_param = module.DRA
            total_sigmoid_sum += torch.sum(torch.sigmoid(dra_param.float()))
            total_elements += dra_param.numel()

            #L_DRA -= torch.sum(torch.clamp(torch.abs(dra_param), max=CLIP_VALUE))
            
        print(svd_modules[0].DRA)

        R_now = total_sigmoid_sum / total_elements
    
        L_comp = GAMMA * torch.abs(R_now - RATIO)

        L_total = L_task + L_comp

        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()

        epoch_loss += L_total.item()
        epoch_compression_loss += L_comp.item()
        #epoch_dra_loss += (LAMBDA * L_DRA).item()

    avg_loss = epoch_loss / len(train_dataloader)
    avg_comp_loss = epoch_compression_loss / len(train_dataloader)
    avg_dra_loss = epoch_dra_loss / len(train_dataloader)
    avg_crossentropy_loss = crossentropy_losses_epoch / len(train_dataloader)
    print(f"Epoch {epoch+1} Summary:")
    print(f"  Avg Total Loss: {avg_loss:.4f}")
    print(f"  Avg Cross-Entropy Loss: {avg_crossentropy_loss:.4f}")
    print(f"  Ratio of used singular values (R_now): {R_now.item():.4f}")
    print(f"  Avg Compression Loss: {avg_comp_loss:.4f}")
    # print(f"  Avg DRA Regularization Loss: {avg_dra_loss:.4f}")
    training_losses.append(avg_loss)
    compression_losses.append(avg_comp_loss)
    # dra_regularization_losses.append(avg_dra_loss)
    # Save the DRA parameters after each epoch, save it as a dict with the name of the module as key
    dra_state_dict = {f"{name}": module.DRA.detach().cpu() for name, module in model.named_modules() if isinstance(module, SVDLinearLayerDynamicSigma)}
    torch.save(dra_state_dict, f"dra_epoch_{epoch+1}.pt")