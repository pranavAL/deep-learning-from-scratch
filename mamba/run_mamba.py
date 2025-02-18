"""
Adapted from:
- https://github.com/johnma2006/mamba-minimal
- https://github.com/PeaBrane/mamba-tiny/tree/master
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
         
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight         
        
class LocalConv(nn.Module):
    def __init__(self, d_model, kernel_size=4, conv_bias=True): 
        super().__init__()
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=conv_bias, padding=kernel_size - 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # Change to (batch, channels, length) for Conv1D
        x = self.conv1d(x)
        x = x[:, :, :-self.conv1d.kernel_size[0] + 1]  # Adjust shape after padding
        return x.transpose(1, 2)  # Restore original shape (batch, length, channels) 

class SelectiveScan(nn.Module):
    """
    Selective Scan module for state-space computation.
    Args:
        u: Input sequence (batch, length, dim)
        dt: Time step scaling (batch, length, dim)
        A: State transition matrix (dim, state_dim)
        B: Input projection matrix (batch, length, state_dim)
        C: Output projection matrix (batch, length, state_dim)
        D: Skip connection (dim)
    Returns:
        Output sequence (batch, length, dim)
    """
    def __init__(self):
        super().__init__()

    def forward(self, u, dt, A, B, C, D):
        # Discretize A: A_Δ = exp(dt * A)
        A_delta = torch.exp(torch.einsum('bld,dn->bldn', dt, A)).clamp(min=-20)
        
        # Input-dependent state update: B_Δ * x_t = dt * u * B
        B_delta_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
        
        # Cumulative state evolution: cumsum(exp(A_Δ))
        A_delta_cumsum = torch.exp(F.pad(A_delta[:, 1:], (0, 0, 0, 0, 1, 0)).cumsum(1))
        
        # Normalized state: h_t = (B_Δ * x_t) / (cumsum(exp(A_Δ)) + eps)
        h_t = B_delta_u / (A_delta_cumsum + 1e-12)
        
        # Output computation: y_t = C * cumsum(h_t * cumsum(exp(A_Δ)))
        y_t = torch.einsum('bldn,bln->bld', h_t.cumsum(1) * A_delta_cumsum, C)
        
        # Add skip connection: y_t = y_t + D * x_t
        return y_t + u * D            
        
class MambaModel(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, **kwargs):
        super().__init__()
        vocab_size = vocab_size + (8 - vocab_size % 8) % 8
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ResidualBlock(d_model, **kwargs) for _ in range(n_layers)])  
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))
    
class ResidualBlock(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, **kwargs)
    
    def forward(self, x):
        return x + self.mixer(self.norm(x))
    
class MambaBlock(nn.Module):
    """
    Mamba Block: Combines input projection, convolution, and selective state-space computation.
    """
    def __init__(self, d_model, d_state=16, expand=2, dt_rank='auto', d_conv=4, bias=False):
        super().__init__()
        d_inner = expand * d_model
        dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.conv = LocalConv(d_inner, d_conv)
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).repeat(d_inner, 1)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.scan = SelectiveScan()
        
    def forward(self, x):
        # Input projection and split into x and residual
        x, res = self.in_proj(x).chunk(2, dim=-1)
        
        # Apply 1D convolution
        x = self.conv(x)
        
        # Apply activation and selective scan
        x = F.silu(x)
        y = self.scan(x, * self.compute_params(x))
        
        # Combine with residual and project output
        return self.out_proj(y * F.silu(res))
    
    def compute_params(self, x):
        """
        Compute input-dependent parameters for selective scan.
        Args:
            x: Input tensor (batch, length, dim)
        Returns:
            delta: Time step scaling (batch, length, dim)
            A: State transition matrix (dim, state_dim)
            B: Input projection matrix (batch, length, state_dim)
            C: Output projection matrix (batch, length, state_dim)
            D: Skip connection (dim)
        """
        # Project input to delta, B, and C
        delta, B, C = self.x_proj(x).split([self.dt_proj.in_features, self.A_log.shape[1], self.A_log.shape[1]], dim=-1)
        
        # Compute delta and A
        delta = F.softplus(self.dt_proj(delta))  # Time step scaling
        A = -torch.exp(self.A_log)  # State transition matrix
        
        return delta, A, B, C, self.D
    
def generate_text(model, tokenizer, prompt, max_length=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
       
# Training and Running Code
def train(model, dataloader, optimizer, criterion, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, labels = batch
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            
def run():
    # Example usage
    model = MambaModel(d_model=128, n_layers=4, vocab_size=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy dataset
    class DummyDataset(Dataset):
        def __len__(self): return 100
        def __getitem__(self, idx): return torch.randint(0, 1000, (10,)), torch.randint(0, 1000, (10,))
    
    dataloader = DataLoader(DummyDataset(), batch_size=2)
    train(model, dataloader, optimizer, criterion)
    
if __name__ == "__main__":
    run()
                       