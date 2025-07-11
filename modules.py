import torch
import torch.nn as nn
import torch.nn.functional as F

# class ChannelScaling(nn.Module):
#     """
#     Batch Normalization without mean and beta.
#     This is a custom implementation that normalizes the input by dividing by the standard deviation
#     """
#     def __init__(self, num_features, eps=1e-5, momentum=0.1):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum
#         self.gamma = nn.Parameter(torch.ones(num_features))
#         self.register_buffer("running_var", torch.ones(num_features))

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor of shape (batch_size, seq_len, num_features)
#         Returns:
#             Tensor of shape (batch_size, seq_len, num_features)
#         """
        
#         if self.training:
#             var = x.var(dim=(0, 1), unbiased=False)# Calculate variance over batch and sequence dimensions
#             self.running_var.data.copy_ = self.momentum * var + (1 - self.momentum) * self.running_var # Update running variance

#         # Ensure eps has the same dtype as var to avoid upcasting
#         current_eps = torch.tensor(self.eps, dtype=self.running_var.dtype, device=self.running_var.device)

#         x_norm = x / torch.sqrt(self.running_var.view(1, 1, -1) + current_eps) # Normalize the input by dividing by the standard deviation
#         return self.gamma.view(1, 1, -1) * x_norm  # Scale the normalized input by gamma
        
# class SVDLinearLayer(nn.Module):
#     def __init__(self, weights, truncate, bias=None, data=None, from_savepoint=False):
#         super(SVDLinearLayer, self).__init__()
        
#         torch.cuda.empty_cache()
        
#         device = weights.device
        
#         out_features, in_features = weights.shape
        
#         # If from_savepoint is True, initialize with random weights
#         if from_savepoint:
#             self.normalization1 = ChannelScaling(in_features).to(device)
#             self.normalization2 = ChannelScaling(truncate).to(device)
            
#             self.vt_linear = nn.Linear(in_features, truncate, bias=False).to(device)
#             self.u_linear = nn.Linear(truncate, out_features, bias=True if bias is not None else False).to(device)
                
#             return
        
#         if data.dim() == 2: # Add batch dimension if missing.
#             data = data.unsqueeze(0).to(device)
            
#         self.normalization1 = ChannelScaling(weights.shape[1]).to(device)
#         data_var = data.var(dim=(0, 1), unbiased=False).to(device)
#         self.normalization1.running_var.data.copy_(data_var)
        
#         # Compute normalization factors for the weights
#         diag_norm1 = torch.diag(self.normalization1.gamma / torch.sqrt(self.normalization1.running_var + self.normalization1.eps)).to(device)
        
#         weights = torch.matmul(weights, torch.inverse(diag_norm1))
        
#         #Perform SVD on the weights
#         U, S, Vt = torch.linalg.svd(weights, full_matrices=False)
#         U = U[:, :truncate]
#         S = S[:truncate]
#         Vt = Vt[:truncate, :]
        
#         diag_s = torch.diag(torch.sqrt(S))
        
#         vt_parameter = torch.matmul(diag_s, Vt)
        
#         self.vt_linear = nn.Linear(in_features, truncate, bias=False)
#         self.vt_linear.weight.data.copy_(vt_parameter)
        
#         self.normalization2 = ChannelScaling(U.shape[1]).to(device)
        
#         self.normalization1.eval()
#         data_var = F.linear(self.normalization1(data), vt_parameter, bias=None).var(dim=(0, 1), unbiased=False).to(device)
#         self.normalization1.train()
#         self.normalization2.running_var.data.copy_(data_var)
#         diag_norm2 = torch.diag(self.normalization2.gamma / torch.sqrt(self.normalization2.running_var + self.normalization2.eps)).to(device)
        
#         u_parameter = torch.matmul(U, torch.matmul(diag_s, torch.inverse(diag_norm2)))
            
#         self.bias = bias
        
#         self.u_linear = nn.Linear(truncate, out_features, bias=True if bias is not None else False)
#         if bias is not None:
#             self.u_linear.bias.data.copy_(bias)
#         self.u_linear.weight.data.copy_(u_parameter)
        
#         del weights, U, S, Vt, diag_s, diag_norm1, diag_norm2, data_var, data, device, u_parameter, vt_parameter
        
#         torch.cuda.empty_cache()
        
#     def forward(self, x):
        
#         """
#         Args:
#             x: Tensor of shape (batch_size, seq_len, in_features) or (seq_len, in_features)
#         Returns:
#             Tensor of shape (batch_size, seq_len, out_features)
#         """
        
#         if x.dim() == 2: # Add batch dimension if missing.
#             x = x.unsqueeze(0)
            
#         x = self.normalization1(x)
        
#         x = self.vt_linear(x)
        
#         x = self.normalization2(x)
        
#         x = self.u_linear(x)
        
#         return x
        
    
#     def reconstruct_weights(self):
#         """
#         Reconstruct the effective weight matrix, taking into account the normalization layers.
#         """
#         device = self.vt_linear.weight.device
        
#         # Incorporate normalization factors from normalization1.
#         if isinstance(self.normalization1, ChannelScaling):
#             norm1 = self.normalization1
#             diag_norm1 = torch.diag(norm1.gamma / torch.sqrt(norm1.running_var + norm1.eps))
#         else:
#             diag_norm1 = torch.eye(self.vt_linear.weight.shape[1], device=device)
            
#         # Incorporate normalization factors from normalization2.
#         if isinstance(self.normalization2, ChannelScaling):
#             norm2 = self.normalization2
#             diag_norm2 = torch.diag(norm2.gamma / torch.sqrt(norm2.running_var + norm2.eps))
#         else:
#             diag_norm2 = torch.eye(self.u_linear.weight.shape[1], device=device)
        
#         # Reconstruct the weight matrix.
#         return self.u_linear.weight @ diag_norm2 @ self.vt_linear.weight @ diag_norm1
    
class SVDLinearLayer(nn.Module):
    def __init__(self, vt_parameter, u_parameter, bias=None,
                 from_savepoint=False):
        """
        Args:
            vt_parameter: Tensor of shape (low_rank, in_features) for Vt matrix.
            u_parameter: Tensor of shape (out_features, low_rank) for U matrix.
            bias: Optional bias tensor of shape (out_features,).
            from_savepoint: If True, initializes the layer with random weights.
        """
        
        super(SVDLinearLayer, self).__init__()
        
        self.vt_linear = nn.Linear(vt_parameter.shape[1], 
                                   vt_parameter.shape[0], 
                                   bias=False)
        
        self.u_linear = nn.Linear(u_parameter.shape[1], 
                                   u_parameter.shape[0], 
                                   bias=True if bias is not None else False)
        
        if from_savepoint:
            # If from_savepoint is True, initialize with random weights
            return
        
        self.vt_linear.weight.data.copy_(vt_parameter)
        if bias is not None:
            self.u_linear.bias.data.copy_(bias)
        self.u_linear.weight.data.copy_(u_parameter)
        
    def forward(self, x):
        """ 
        Args:
            x: Tensor of shape (batch_size, seq_len, in_features) or (seq_len, in_features)
        Returns:
            Tensor of shape (batch_size, seq_len, out_features)
        """
        if x.dim() == 2:  # Add batch dimension if missing.
            x = x.unsqueeze(0)
        return self.u_linear(self.vt_linear(x))
    
class WeightedMSELoss(nn.Module):
    """
    Mean Squared Error loss with optional per-feature weighting.

    Args:
        weights (torch.Tensor, optional): 1D tensor of shape (features,) containing
            weight for each feature. If None, behaves like standard MSELoss.
        reduction (str, optional): 'mean', 'sum', or 'none'. Defaults to 'mean'.
    """
    def __init__(self, weights: torch.Tensor = None, reduction: str = 'mean'):
        super(WeightedMSELoss, self).__init__()
        self.reduction = reduction

        if weights is not None:
            # register as buffer so it gets moved with .to(device)
            # reshape to broadcast over batch and seqlen: (1, 1, features)
            self.register_buffer('weights', weights.view(1, 1, -1))
        else:
            self.weights = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # squared error
        diff = (input - target) ** 2  # shape: (batch, seqlen, features)

        if self.weights is not None:
            # apply per-feature weights
            diff = diff * self.weights

        if self.reduction == 'mean':
            return diff.mean()
        elif self.reduction == 'sum':
            return diff.sum()
        elif self.reduction == 'none':
            return diff
        else:
            raise ValueError(f"Unsupported reduction '{self.reduction}'")