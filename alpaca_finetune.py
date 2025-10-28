import argparse
import json
import os
import re
import torch
import warnings
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.utils import cached_file

# Assuming svdmodels.py and Prompter.py are in the same directory or accessible
from Prompter import Prompter, ZeroPrompter
from svdmodels import SVDModel


def main():
    # Suppress deprecation warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    parser = argparse.ArgumentParser(description="Fine-tune a SVD-compressed model on the Alpaca dataset.")
    
    # --- Model and Path Arguments ---
    parser.add_argument("--base-model", type=str, default="huggyllama/llama-7b", help="Base Hugging Face model identifier.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the SVD-compressed model .pt state_dict.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the fine-tuned model and checkpoints.")
    
    # --- Dataset and Prompting Arguments ---
    parser.add_argument("--dataset", type=str, default="yahma/alpaca-cleaned", help="Dataset to use for fine-tuning.")
    parser.add_argument("--val-set-size", type=float, default=0.1, help="Proportion of the training set to use for validation.")
    parser.add_argument("--prompt-template-name", type=str, default="alpaca", help="The name of the prompt template to use from Prompter.")
    parser.add_argument("--no-instruction", action="store_true", help="Use ZeroPrompter for datasets without instructions.")
    parser.add_argument("--train-on-inputs", action="store_true", help="If False, masks out inputs in loss. Necessary for Alpaca-style.")
    parser.add_argument("--add-eos-token", action="store_true", default=True, help="Add EOS token to the end of prompts.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--micro-batch-size", type=int, default=4, help="Batch size per device for training.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--cutoff-len", type=int, default=256, help="Maximum sequence length for tokenization.")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--group-by-length", action="store_true", help="Group sequences of similar length for training efficiency.")
    
    # --- LoRA Hyperparameters ---
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling parameter.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    
    # --- System and Logging Arguments ---
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda', 'cpu').")
    parser.add_argument("--save-steps", type=int, default=200, help="Save a checkpoint every N optimization steps.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluate the model every N optimization steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- 1. Setup ---
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Infer compression ratio from model_path (e.g., '_r0.8')
    match = re.search(r"_r([0-9]+(?:\.[0-9]+)?)", args.model_path)
    if not match:
        raise ValueError("Could not infer ratio from model_path. Ensure it contains '_r<ratio>' (e.g., '_r0.8').")
    ratio = float(match.group(1))
    
    ### FIX: Removed redundant gradient_accumulation_steps calculation.
    # The Trainer's TrainingArguments handles this directly.
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Model and Tokenizer ---
    print(f"Loading base model: {args.base_model}")
    # Note: For INT8 training, you would typically load with `load_in_8bit=True`.
    # `prepare_model_for_int8_training` is still useful for type casting.
    
    def load_config_with_rope_fix(model_name):
        """Load model config and fix LLaMA 3 RoPE scaling if needed."""
        try:
            # Try loading normally first
            return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except ValueError as e:
            if "rope_scaling" in str(e):
                print(f"RoPE scaling error detected: {e}")
                print("Loading config file directly to bypass validation...")
                
                # Load config.json directly from HF Hub
                config_file = cached_file(model_name, "config.json", cache_dir=None)
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                
                # Fix rope_scaling if it's in LLaMA 3 format
                if "rope_scaling" in config_dict and config_dict["rope_scaling"] is not None:
                    rope_scaling = config_dict["rope_scaling"]
                    if isinstance(rope_scaling, dict) and "rope_type" in rope_scaling:
                        if rope_scaling.get("rope_type") == "llama3":
                            print("Converting LLaMA 3 RoPE scaling to compatible format...")
                            config_dict["rope_scaling"] = {
                                "type": "linear",
                                "factor": rope_scaling.get("factor", 8.0)
                            }
                
                # Create config from modified dict
                from transformers.models.llama.configuration_llama import LlamaConfig
                return LlamaConfig(**config_dict)
            else:
                raise e

    try:
        # Try loading with original configuration
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except ValueError as e:
        if "rope_scaling" in str(e):
            print(f"Model loading failed due to RoPE scaling: {e}")
            print("Loading with fixed configuration...")
            
            # Load config with RoPE fix
            fixed_config = load_config_with_rope_fix(args.base_model)
            
            # Load model with fixed config
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                config=fixed_config,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        else:
            raise e
    
    print(f"Applying SVD compression with ratio {ratio} from {args.model_path}")
    model = SVDModel.load_model(
        model=base_model,
        ratio=ratio,
        model_path=args.model_path,
        quantization_bits=None,  # Assuming no quantization for fine-tuning
        low_rank_dict=None,
        SEQLEN=args.cutoff_len  ### FIX: Use cutoff_len for consistency
    )
    
    model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # --- 3. PEFT/LoRA Setup ---
    model = prepare_model_for_kbit_training(model)
        
    # LoRA configuration
    target_modules = ["vt_linear", "u_linear"]  # Specific to your SVDModel structure

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # --- 4. Data Preparation ---
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point.get("instruction"),
            data_point.get("input"),
            data_point.get("output"),
        )
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point.get("instruction"), data_point.get("input")
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    ### FIX: Removed the unused `split_and_tokenizer` function.

    print("Loading and processing dataset...")
    data = load_dataset(args.dataset)
    
    # Handle datasets that may not have a validation split
    if "test" in data.keys() and args.val_set_size > 0:
         train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=args.seed)
         train_data = train_val["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
         val_data = train_val["test"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
        val_data = None

    # --- 5. Trainer Setup and Execution ---
    training_args = TrainingArguments(
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        logging_first_step=True,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_data else "no",
        save_strategy="steps",
        eval_steps=args.eval_steps if val_data else None,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        save_safetensors=False,
        save_total_limit=5,
        load_best_model_at_end=True if val_data else False,
        metric_for_best_model="eval_loss" if val_data else None,
        ddp_find_unused_parameters=None,
        group_by_length=args.group_by_length,
        report_to="none", # can be set to "wandb", "tensorboard" etc.
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    
    model.config.use_cache = False  # silence warnings
    
    print("Starting training...")
    trainer.train()

    # --- 6. Save Final Model ---
    ### FIX: Corrected the model saving path logic.
    basename = os.path.basename(args.model_path)
    filename, ext = os.path.splitext(basename)
    new_filename = f"{filename}_ft{ext}"
    model_save_path = os.path.join(args.output_dir, new_filename)

    # It's better to save the PEFT-trained adapters, not the full state_dict
    model.save_pretrained(args.output_dir)
    print(f"Fine-tuned LoRA adapters saved to {args.output_dir}")

    # If you still want to save the full state dict for some reason:
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Full model state_dict saved to {model_save_path}")


if __name__ == "__main__":
    main()