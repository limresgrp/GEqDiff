import torch
from e3nn.util.jit import compile_mode


@compile_mode("script")
class SinusoidalPositionEmbedding(torch.nn.Module):
    """
    Creates sinusoidal positional embeddings for a batch of timesteps.
    The output dimension of this module is `t_embedding_dim`.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # The output dimension must be an even number
        if embedding_dim % 2 != 0:
            raise ValueError(f"embedding_dim must be even, but got {embedding_dim}")
            
        self.embedding_dim = embedding_dim
        self.half_dim = embedding_dim // 2

    def forward(self, t: torch.Tensor):
        # t is expected to be a 1D tensor of shape (batch_size,)
        device = t.device
        
        # Calculate frequencies 
        # The original paper uses 10000 as the base.
        # Shape: (half_dim,)
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(self.half_dim, device=device) / self.half_dim
        )
        
        # Project timesteps onto the frequencies
        # t shape: (batch_size, 1)
        # freqs shape: (1, half_dim)
        # Result `x` shape: (batch_size, half_dim)
        x = t.float() * freqs.unsqueeze(0)
        
        # Concatenate sine and cosine components
        # The final embedding `emb` will have shape (batch_size, embedding_dim)
        emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return emb