# SVD MODELS

import torch.nn as nn
import torch
from modules import SVDLinearLayer
from utils import replace_module_by_name, get_truncate
from tqdm import tqdm
from quantize_llm import quantize_linear_layer

class SVDModel(nn.Module):
    def __init__(self, model, ratio=0.6):
        super(SVDModel, self).__init__()
        self.model = model
        self.ratio = ratio

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @classmethod
    def load_model(cls, model, ratio=0.6, 
                   model_path=None, 
                   quantization_bits=None, 
                   ratios_dict=None):
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
                
                if ratios_dict is not None:
                    # Get the ratio for the current layer from the dictionary
                    ratio = ratios_dict.get(name, ratio)
                else:
                    # Use the default ratio
                    ratio = ratio
                
                low_rank = get_truncate(in_features, out_features, ratio)
                # SVD needs a vt parameter and a u_parameter, we can create a dummy ones to initialize the SVD layer and the load the weights later
                # Create dummy parameters for SVD
                vt_parameter = torch.zeros(low_rank, in_features)
                u_parameter = torch.zeros(out_features, low_rank)
                # Create the SVD layer
                svd_layer = SVDLinearLayer(
                    vt_parameter=vt_parameter,
                    u_parameter=u_parameter,
                    bias=b,
                )
            
                # Replace the original layer with the SVD layer
                replace_module_by_name(model, name, svd_layer)
                
        # Load the model state dict
        
        if quantization_bits is not None:
            for name, module in tqdm(model.named_modules(), desc="Quantizing layers", total=len(list(model.named_modules()))):
                if isinstance(module, nn.Linear):
                    if name == "lm_head":
                        print("Skipping lm_head")
                        continue
                    # Create the quantized layer
                    qlayer = quantize_linear_layer(module, 
                                                   bits=quantization_bits, 
                                                   threshold=6.0,
                                                   from_savepoint=True)
                    
                    # Replace the original layer with the quantized layer
                    replace_module_by_name(model, name, qlayer)
        
        state_dict = torch.load(model_path)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        
        return instance.model