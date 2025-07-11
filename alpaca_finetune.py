import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import re

# Assuming svdmodels.py is in the same directory or accessible in PYTHONPATH
from svdmodels import SVDModel

def format_prompt(example):
    """Formats a single example from the Alpaca dataset into a prompt."""
    if example.get("input", ""):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a SVD-compressed model on the Alpaca dataset.")
    parser.add_argument("--base-model", type=str, default="huggyllama/llama-7b", help="Base Hugging Face model identifier.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the SVD-compressed model .pt state_dict.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="Dataset to use for fine-tuning.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Number of steps to accumulate gradients before an optimization step.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Number of warmup steps for the learning rate scheduler.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda', 'cpu').")
    parser.add_argument("--checkpoint-steps", type=int, default=100, help="Save a checkpoint every N optimization steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Infer ratio from model_path (e.g., '_r0.8')
    match = re.search(r"_r([0-9]+(?:\.[0-9]+)?)", args.model_path)
    if not match:
        parser.error("Could not infer ratio from model_path. Ensure it contains '_r<ratio>' (e.g., '_r0.8').")
    ratio = float(match.group(1))

    # --- 1. Load Tokenizer and Dataset ---
    print(f"Loading tokenizer for {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading and processing dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")

    def tokenize_batched(batch):
        # The input 'batch' is a dictionary of lists (e.g., {'instruction': [...], 'input': [...], ...})
        prompts = []
        # Iterate over each example in the batch
        for i in range(len(batch['instruction'])):
            # Reconstruct the example dict for format_prompt
            example = {key: batch[key][i] for key in batch}
            prompts.append(format_prompt(example) + tokenizer.eos_token)
        
        tokenized = tokenizer(
            prompts,
            truncation=True,
            padding="max_length", # Pad to max_seq_len
            max_length=args.max_seq_len,
        )
        # For Causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_batched,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True
    )

    # --- 2. Load Model ---
    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16)
    
    print(f"Applying SVD compression with ratio {ratio} from {args.model_path}")
    model = SVDModel.load_model(
        model=base_model,
        ratio=ratio,
        model_path=args.model_path,
        quantization_bits=None, # Assuming no quantization for fine-tuning
        ratios_dict=None
    )

    # --- 3. Prepare for Training ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.half()
    model.to(device)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # --- 4. Optimizer and Scheduler ---
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    # Adjust the number of training steps for the scheduler to account for gradient accumulation
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = num_update_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    # --- 5. Training Loop ---
    print("Starting fine-tuning...")
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Normalize loss to account for accumulation
            loss = loss / args.gradient_accumulation_steps
            
            loss.backward()
            
            # Update weights every `gradient_accumulation_steps`
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # --- Checkpointing ---
                if global_step % args.checkpoint_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    # Save the fine-tuned state_dict
                    output_model_path = os.path.join(checkpoint_dir, "finetuned_model.pt")
                    torch.save(model.state_dict(), output_model_path)
                    print(f"\n✅ Checkpoint saved to {output_model_path}")
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

    # --- 6. Save Model ---
    print("Training complete. Saving model...")
    os.makedirs(args.output_dir, exist_ok=True)
        
    # Save the fine-tuned state_dict
    output_model_path = os.path.join(args.output_dir, "finetuned_model.pt")
    torch.save(model.state_dict(), output_model_path)
    print(f"✅ Fine-tuned model saved to {output_model_path}")

if __name__ == "__main__":
    main()
