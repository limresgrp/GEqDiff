import torch
from abc import ABC, abstractmethod

from geqdiff.utils.noise_schedulers import NoiseScheduler



class Sampler(ABC, torch.nn.Module):
    """Abstract base class for all samplers."""
    def __init__(self, scheduler: NoiseScheduler):
        super().__init__()
        self.scheduler = scheduler
        self.T = scheduler.T

    @abstractmethod
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_pred: torch.Tensor, **kwargs):
        """Performs one reverse step from x_t to x_{t-1}."""
        raise NotImplementedError

class DDPMSampler(Sampler):
    """Implements the standard DDPM sampling algorithm."""
    
    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_pred: torch.Tensor):
        s = self.scheduler
        _sqrt_alpha_t = torch.sqrt(s.alphas[t])
        _one_minus_alpha_t = 1.0 - s.alphas[t]
        _sqrt_one_minus_alpha_bar_t = s.sqrt_one_minus_alpha_bar[t]
        _sqrt_beta_t = torch.sqrt(s.betas[t])

        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        
        term1 = 1.0 / _sqrt_alpha_t
        term2 = (x_t - (_one_minus_alpha_t / _sqrt_one_minus_alpha_bar_t) * eps_pred)
        term3 = _sqrt_beta_t * z
        
        x_t_minus_1 = term1 * term2 + term3
        return x_t_minus_1

class DDIMSampler(Sampler):
    """Implements the DDIM sampling algorithm for accelerated generation."""

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, t_prev: int, eps_pred: torch.Tensor, eta: float = 0.0):
        s = self.scheduler
        alpha_bar_t = s.alpha_bar[t]
        alpha_bar_t_prev = s.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Equation (12) from the DDIM paper
        x0_pred = (x_t - s.sqrt_one_minus_alpha_bar[t] * eps_pred) / s.sqrt_alpha_bar[t]
        
        # Equation (16) from the DDIM paper
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev))
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred
        
        # Sample noise, if not at the last step and eta > 0
        z = torch.randn_like(x_t) if t > 0 and eta > 0 else torch.zeros_like(x_t)

        x_t_minus_1 = torch.sqrt(alpha_bar_t_prev) * x0_pred + dir_xt + sigma_t * z
        return x_t_minus_1