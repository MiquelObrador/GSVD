import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from svdmodels import SVDModel
from utils import load_wikitext

parser = argparse.ArgumentParser(
    description="Evaluate a (G)SVD-compressed LLM on Wikitext and compute perplexity"
)
parser.add_argument(
    "--base-model",
    type=str,
    default="huggyllama/llama-7b",
    help="HuggingFace model identifier for the original (uncompressed) model",
)
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help=(
        "Path to the SVD-compressed model weights."
        " The compression ratio will be inferred from the first occurrence of '_r' followed by a number in the filename."
    ),
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Batch size for evaluation on Wikitext",
)
args = parser.parse_args()

# Extract ratio from model_path (e.g., '_r0.8')
match = re.search(r"_r([0-9]+(?:\.[0-9]+)?)", args.model_path)
if not match:
    parser.error(
        "Could not infer ratio from model_path. Ensure it contains '_r<ratio>' (e.g., '_r0.8')."
    )
args.ratio = float(match.group(1))

# Set seed for reproducibility and deterministic results
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load original model and tokenizer
print(f"Loading base model: {args.base_model}")
model = AutoModelForCausalLM.from_pretrained(
    args.base_model, torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)

SEQ_LEN = model.config.max_position_embeddings

# Load SVD-compressed weights
print(f"Applying SVD compression with ratio {args.ratio} from {args.model_path}")
model = SVDModel.load_model(model, ratio=args.ratio, model_path=args.model_path)

# Prepare for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.half().to(DEVICE).eval()

# Load dataset
loader = load_wikitext(
    tokenizer, seq_len=SEQ_LEN, batch_size=args.batch_size)

# Evaluate perplexity
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

# Save the perplexity
ppl_path = re.sub(r"\.pt$", "", args.model_path) + "_pplEvaluation.txt"
with open(ppl_path, "w") as f:
    f.write(str(ppl))
print(f"Saved perplexity to {ppl_path}")