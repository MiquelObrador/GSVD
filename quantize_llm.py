# quantize_llm.py

import argparse
import torch
import torch.nn as nn
import re
import os
import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from utils import replace_module_by_name
from tqdm import tqdm

def quantize_linear_layer(layer: nn.Linear, bits: int, threshold: float,
                          from_savepoint: bool = False) -> nn.Module:
    """
    Wrap a torch.nn.Linear into a bitsandbytes quantized Linear8bitLt or Linear4bit.
    """
    in_f, out_f = layer.in_features, layer.out_features
    bias = layer.bias is not None

    if bits == 8:
        # LLM.int8() style 8-bit
        qlayer = Linear8bitLt(
            in_f, out_f,
            bias=bias,
            has_fp16_weights=False,
            threshold=threshold,
            memory_efficient_backward=True,
        )
    elif bits == 4:
        # QLoRA-style 4-bit
        qlayer = Linear4bit(
            in_f, out_f,
            bias=bias,
            quant_type="nf4",
            compress_statistics=True,
        )
    else:
        raise ValueError(f"Unsupported bit width: {bits}")

    # copy over fp16 weights & bias
    if from_savepoint == False:
        sd = layer.state_dict()
        qlayer.load_state_dict(sd)
        # move to same device (and trigger quantization for int8)
        qlayer.to(layer.weight.device)
        
    return qlayer

def main():
    parser = argparse.ArgumentParser(
        description="Quantize all Linear/SVDLinearLayer weights in an LLM to 8‑bit or 4‑bit using bitsandbytes"
    )
    parser.add_argument("--base-model",   type=str, default="huggyllama/llama-7b",
                        help="HuggingFace model identifier for the original (uncompressed) model")
    parser.add_argument("--model-path",   type=str, required=True,
                        help="Path to fp16/bf16 model .pt state_dict")
    parser.add_argument("--output-path",  type=str, required=False,
                    help="Where to save the quantized model .pt (default: model_path with '_4bit.pt' or '_8bit.pt' suffix)")
    parser.add_argument("--bits",         type=int, default=8, choices=[4,8],
                        help="Quantization bit‑width (4 or 8)")
    parser.add_argument("--threshold",    type=float, default=6.0,
                        help="Threshold for LLM.int8(); ignored for 4‑bit")
    parser.add_argument("--device",       type=str, default="cuda",
                        help="Device to perform quantization on")
    args = parser.parse_args()
    
    # Extract ratio from model_path (e.g., '_r0.8')
    match = re.search(r"_r([0-9]+(?:\.[0-9]+)?)", args.model_path)
    if not match:
        parser.error(
            "Could not infer ratio from model_path. Ensure it contains '_r<ratio>' (e.g., '_r0.8')."
        )
    args.ratio = float(match.group(1))
    
    if args.output_path is None:
        args.output_path = f"{os.path.splitext(args.model_path)[0]}_{args.bits}bit.pt"

    # 1) load your model
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float16
    )
    
    SEQ_LEN = model.config.max_position_embeddings
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load SVD-compressed weights
    print(f"Applying SVD compression with ratio {args.ratio} from {args.model_path}")
    model = SVDModel.load_model(model, ratio=args.ratio, model_path=args.model_path)

    # Prepare for evaluation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.half().to(DEVICE).eval()

    for name, module in tqdm(model.named_modules(), desc="Quantizing layers", total=len(list(model.named_modules()))):
        if isinstance(module, nn.Linear):
            if name == "lm_head":
                print("Skipping lm_head")
                continue
            
            # Create the quantized layer
            qlayer = quantize_linear_layer(module, 
                                           bits=args.bits, 
                                           threshold=args.threshold,
                                           from_savepoint=False)
            
            # Replace the original layer with the quantized layer
            replace_module_by_name(model, name, qlayer)
            
    print(model)

    # 3) save only the quantized weights
    torch.save(model.state_dict(), args.output_path)
    print(f"✅ Saved quantized ({args.bits}-bit) model to {args.output_path}")
    
    loader = load_wikitext(
        tokenizer, seq_len=SEQ_LEN, batch_size=8)
    
    nlls = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
            batch = batch.to(DEVICE)
            logits = model(input_ids=batch, use_cache=False).logits
            if torch.isfinite(logits).all():
                shifted_logits = logits[:, :-1, :].contiguous()
                shifted_labels = batch[:, 1:].contiguous()
                loss_fnc = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fnc(
                    shifted_logits.view(-1, logits.size(-1)),
                    shifted_labels.view(-1)
                )
                nlls.append(loss.cpu())
            else:
                print("Non-finite logits detected, skipping batch.")
                
    mean_loss = torch.cat(nlls).mean()
    ppl = torch.exp(mean_loss).item()
    ppl = int(ppl) if ppl > 1000 else ppl
    print(f"Perplexity: {ppl}")
    
    ppl_path = re.sub(r"\.pt$", "", args.output_path) + "_pplEvaluation.txt"
    with open(ppl_path, "w") as f:
        f.write(str(ppl))
    print(f"Saved perplexity to {ppl_path}")

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from svdmodels import SVDModel
    from utils import load_wikitext
    main()