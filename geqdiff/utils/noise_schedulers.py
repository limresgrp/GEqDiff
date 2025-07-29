import torch
from e3nn.util.jit import compile_mode

# --- Schedule Functions ---

def linear_schedule(Tmax, beta_min=1e-4, beta_max=2e-2):
    """Generates a linear beta schedule."""
    betas = torch.linspace(start=beta_min, end=beta_max, steps=Tmax)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars, alphas, betas

def cosine_schedule(Tmax, s=0.008, raise_to_power: float = 1.0):
    """Generates a cosine beta schedule."""
    steps = Tmax + 1
    t = torch.linspace(0, Tmax, steps)
    alphas_cumprod = torch.cos(((t / Tmax) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.99)
    alphas = 1.0 - betas
    
    # Recalculate alpha_bars from the clipped betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    if raise_to_power != 1:
        alpha_bars = torch.pow(alpha_bars, raise_to_power)

    return alpha_bars, alphas, betas


# --- Base and Subclasses Implementation ---
@compile_mode("script")
class NoiseScheduler(torch.nn.Module):
    """
    Holds the noise schedule parameters (alphas, betas).
    This class is responsible for the FORWARD (noising) process.
    """

    schedule_type: str

    def __init__(self, T: int, schedule_type: str = 'linear', **kwargs):
        super().__init__()
        self.T = T

        self.schedule_type = schedule_type
        if schedule_type == 'linear':
            alpha_bar, alphas, betas = linear_schedule(T, **kwargs)
        elif schedule_type == 'cosine':
            alpha_bar, alphas, betas = cosine_schedule(T, **kwargs)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        # Register schedule tensors as buffers for automatic device management
        self.register_buffer('alpha_bar', alpha_bar)
        self.register_buffer('alphas', alphas)
        self.register_buffer('betas', betas)
        
        # Pre-computed values for the forward process (noising)
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(alpha_bar))
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - alpha_bar))

    def forward(self, t: torch.Tensor):
        """Performs the forward process (noising) for a given timestep t."""
        alpha = self.sqrt_alpha_bar[t]
        sigma = self.sqrt_one_minus_alpha_bar[t]
        return alpha, sigma