import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelScaling(nn.Module):
    """
    Batch Normalization without mean and beta.
    This is a custom implementation that normalizes the input by dividing by the standard deviation
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, num_features)
        Returns:
            Tensor of shape (batch_size, seq_len, num_features)
        """
        
        if self.training:
            var = x.var(dim=(0, 1), unbiased=False)# Calculate variance over batch and sequence dimensions
            self.running_var.data.copy_ = self.momentum * var + (1 - self.momentum) * self.running_var # Update running variance
        else:
            var = self.running_var # Use running variance during inference

        x_norm = x / torch.sqrt(var.view(1, 1, -1) + self.eps) # Normalize the input by dividing by the standard deviation
        return self.gamma.view(1, 1, -1) * x_norm  # Scale the normalized input by gamma
        

class SVDLinearLayer(nn.Module):
    def __init__(self, weights, truncate, bias=None, data=None, from_savepoint=False):
        super(SVDLinearLayer, self).__init__()
        
        torch.cuda.empty_cache()
        
        device = weights.device
        
        out_features, in_features = weights.shape
        
        # If from_savepoint is True, initialize with random weights
        if from_savepoint:
            self.normalization1 = ChannelScaling(in_features).to(device)
            self.normalization2 = ChannelScaling(truncate).to(device)
            
            self.vt_linear = nn.Linear(in_features, truncate, bias=False)
            self.u_linear = nn.Linear(truncate, out_features, bias=True if bias is not None else False)
            
            if bias is not None:
                self.u_linear.bias.data.copy_(bias)
                
            return
        
        if data.dim() == 2: # Add batch dimension if missing.
            data = data.unsqueeze(0).to(device)
            
        self.normalization1 = ChannelScaling(weights.shape[1]).to(device)
        data_var = data.var(dim=(0, 1), unbiased=False).to(device)
        self.normalization1.running_var.data.copy_(data_var)
        
        # Compute normalization factors for the weights
        diag_norm1 = torch.diag(self.normalization1.gamma / torch.sqrt(self.normalization1.running_var + self.normalization1.eps)).to(device)
        
        weights = torch.matmul(weights, torch.inverse(diag_norm1))
        
        #Perform SVD on the weights
        U, S, Vt = torch.linalg.svd(weights, full_matrices=False)
        U = U[:, :truncate]
        S = S[:truncate]
        Vt = Vt[:truncate, :]
        
        diag_s = torch.diag(torch.sqrt(S))
        
        vt_parameter = torch.matmul(diag_s, Vt)
        
        self.vt_linear = nn.Linear(in_features, truncate, bias=False)
        self.vt_linear.weight.data.copy_(vt_parameter)
        
        self.normalization2 = ChannelScaling(U.shape[1]).to(device)
        
        self.normalization1.eval()
        data_var = F.linear(self.normalization1(data), vt_parameter, bias=None).var(dim=(0, 1), unbiased=False).to(device)
        self.normalization1.train()
        self.normalization2.running_var.data.copy_(data_var)
        diag_norm2 = torch.diag(self.normalization2.gamma / torch.sqrt(self.normalization2.running_var + self.normalization2.eps)).to(device)
        
        u_parameter = torch.matmul(U, torch.matmul(diag_s, torch.inverse(diag_norm2)))
            
        self.bias = bias
        
        self.u_linear = nn.Linear(truncate, out_features, bias=True if bias is not None else False)
        if bias is not None:
            self.u_linear.bias.data.copy_(bias)
        self.u_linear.weight.data.copy_(u_parameter)
        
        del weights, U, S, Vt, diag_s, diag_norm1, diag_norm2, data_var, data, device, u_parameter, vt_parameter
        
        torch.cuda.empty_cache()
        
    def forward(self, x):
        
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, in_features) or (seq_len, in_features)
        Returns:
            Tensor of shape (batch_size, seq_len, out_features)
        """
        
        if x.dim() == 2: # Add batch dimension if missing.
            x = x.unsqueeze(0)
            
        x = self.normalization1(x)
        
        x = self.vt_linear(x)
        
        x = self.normalization2(x) 
        
        x = self.u_linear(x)
        
        return x
        
    
    def reconstruct_weights(self):
        """
        Reconstruct the effective weight matrix, taking into account the normalization layers.
        """
        device = self.vt_linear.weight.device
        
        # Incorporate normalization factors from normalization1.
        if isinstance(self.normalization1, ChannelScaling):
            norm1 = self.normalization1
            diag_norm1 = torch.diag(norm1.gamma / torch.sqrt(norm1.running_var + norm1.eps))
        else:
            diag_norm1 = torch.eye(self.vt_linear.weight.shape[1], device=device)
            
        # Incorporate normalization factors from normalization2.
        if isinstance(self.normalization2, ChannelScaling):
            norm2 = self.normalization2
            diag_norm2 = torch.diag(norm2.gamma / torch.sqrt(norm2.running_var + norm2.eps))
        else:
            diag_norm2 = torch.eye(self.u_linear.weight.shape[1], device=device)
        
        # Reconstruct the weight matrix.
        return self.u_linear.weight @ diag_norm2 @ self.vt_linear.weight @ diag_norm1