import torch.nn as nn
import torch
from modules import SVDLinearLayer
from utils import replace_module_by_name, get_truncate
from tqdm import tqdm

class SVDModel(nn.Module):
    def __init__(self, model, ratio=0.6):
        super(SVDModel, self).__init__()
        self.model = model
        self.ratio = ratio

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @classmethod
    def load_model(cls, model, ratio=0.6, model_path=None):
        instance = cls(model, ratio)
        # Replace original modules with SVD modules
        for name, module in tqdm(model.named_modules(), desc="Replacing modules", total=len(list(model.named_modules()))):
            if isinstance(module, nn.Linear):
                if name == "lm_head":
                    print("Skipping lm_head")
                    continue
                # Get the original weights and bias
                W = module.weight.data
                b = module.bias.data if module.bias is not None else None
                
                out_features, in_features = W.shape
                
                low_rank = get_truncate(in_features, out_features, ratio)
                    
                # Create the SVD layer
                svd_layer = SVDLinearLayer(W, low_rank, b, from_savepoint=True)
                
                # Replace the original layer with the SVD layer
                replace_module_by_name(instance.model, name, svd_layer)
                
        # Load the model state dict
        state_dict = torch.load(model_path)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        
        return instance