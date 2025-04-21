import random
import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset
import math
import os

def replace_module_by_name(root_module, module_full_name, new_module):
    """
    Replaces a module inside root_module given its full name (e.g. "layer1.conv").
    """
    name_parts = module_full_name.split('.')
    parent = root_module
    # Traverse to the parent of the target module.
    for part in name_parts[:-1]:
        parent = getattr(parent, part)
    # Replace the target module.
    setattr(parent, name_parts[-1], new_module)
    
    
def load_wikitext(tokenizer, seq_len=2048, batch_size = 4, samples=None):
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)
    ####
    def process_data(samples, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    if samples is not None:
        test_data = test_data.select(list(random.sample(range(len(test_data)), samples)))
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_ratios(final_ratio, matrix_iters):
    ratios = []
    for i in range(1, matrix_iters + 1):
        r = 1 - i * (1 - final_ratio) / matrix_iters
        ratios.append(r)
    return ratios

def get_truncate(in_features, out_features, ratio):
    if in_features == out_features:
        return math.ceil(ratio * in_features / 2)
    else:
        return math.ceil((ratio * in_features * out_features) / (in_features + out_features))
    
def get_unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path