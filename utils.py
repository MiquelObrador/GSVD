import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import random
import torch.nn as nn


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
    
class CustomDictDataset(Dataset):
    """
    A custom PyTorch Dataset that returns a dictionary with specified keys.

    Args:
        data_dict (dict): A dictionary where keys are strings (e.g., 'input_ids')
                          and values are lists or tensors of the data.
    """
    def __init__(self, data_dict):
        self.data = {key: torch.as_tensor(value) for key, value in data_dict.items()}
        self.keys = list(data_dict.keys())
        self._len = len(self.data[self.keys[0]])

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return {key: self.data[key][idx] for key in self.keys}

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    """
    Returns a DataLoader for calibration training data where each item is a dictionary.

    Args:
        name (str): The name of the dataset to use ('c4', 'ptb', 'wikitext2').
        tokenizer: The tokenizer to use for encoding the text.
        nsamples (int): The number of samples to generate.
        seqlen (int, optional): The sequence length of each sample. Defaults to 2048.
        seed (int, optional): The random seed for reproducibility. Defaults to 3.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 1.
        dataset_cache_dir (str, optional): The directory to cache the dataset. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the calibration data as dictionaries.
    """
    random.seed(seed)
    
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_file = os.path.join(cache_dir, f"{name}_{nsamples}_{seqlen}_{seed}_dict.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        data_dict = torch.load(cache_file)
    else:
        print("Generating and caching dataset...")
        if name == "c4":
            traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
            text_data = "\n\n".join(traindata["text"])
        elif name == "ptb":
            traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
            text_data = "\n\n".join(traindata["sentence"])
        elif name == "wikitext2":
            traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", cache_dir=dataset_cache_dir)
            text_data = "\n\n".join(traindata["text"])
        else:
            raise NotImplementedError

        all_input_ids = []
        all_attention_masks = []
        
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(text_data) - seqlen * 10 - 1)
                j = i + seqlen * 10
                encoded_text = tokenizer(text_data[i:j], return_tensors="pt")
                
                if encoded_text.input_ids.shape[1] >= seqlen:
                    inp = encoded_text.input_ids[:, :seqlen]
                    attention_mask = torch.ones_like(inp)
                    all_input_ids.append(inp)
                    all_attention_masks.append(attention_mask)
                    break
        
        data_dict = {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0)
        }
        
        torch.save(data_dict, cache_file)

    dataset = CustomDictDataset(data_dict)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_eval_tokenized_dataset(tokenizer,
                           dataset_name: str = 'wikitext-2-raw-v1',
                           seq_len: int = 2048,
                           batch_size: int = 4):
    """
    Load and tokenize a variety of NLP datasets with reproducible sampling.

    Args:
        tokenizer: a HuggingFace tokenizer instance
        dataset_name: config name or subset identifier (e.g. 'wikitext-2-raw-v1', 'penn_treebank')
        seq_len: sequence length for each example
        batch_size: batch size for DataLoader
        samples: optional limit on number of examples to sample
        seed: optional random seed for reproducible sampling

    Returns:
        DataLoader yielding batches of token IDs of shape [batch_size, seq_len]
    """
    class IndexDataset(Dataset):
        def __init__(self, tensors):
            self.tensors = tensors

        def __getitem__(self, index):
            return self.tensors[index]

        def __len__(self):
            return len(self.tensors)

    def process_data(samples, tokenizer, seq_len, field_name):
        # concatenate texts, tokenize, and split into fixed subsequences
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_ids_batch.append(batch)
        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    # Load the requested dataset split
    if "wikitext" in dataset_name.lower():
        test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    elif "ptb" in dataset_name.lower():
        test_data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')
    elif "c4" in dataset_name.lower():
        test_data = load_dataset('json', data_files="datasets/c4-validation.00000-of-00008.json.gz")["train"]
        test_dataset = process_data(test_data[0:2000], tokenizer, seq_len, 'text')
        
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def get_truncate(in_features, out_features, ratio):
    return int(in_features * out_features * ratio / (in_features + out_features))

def get_unique_path(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base}_{counter}{ext}"
    return new_path

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res