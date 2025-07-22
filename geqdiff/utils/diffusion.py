from typing import Optional
import torch
from geqtrain.utils.pytorch_scatter import scatter_mean

def center_pos(pos: torch.Tensor, batch: Optional[torch.Tensor] = None):
    """
    Centers the coordinates of each molecule in pos according to batch indices.

    Args:
        pos (torch.Tensor): Tensor of shape (N_atoms, 3) with atom coordinates.
        batch (torch.Tensor): Tensor of shape (N_atoms,) with batch indices.

    Returns:
        torch.Tensor: Centered coordinates of shape (N_atoms, 3).
    """

    if batch is None:
        mean = pos.mean(dim=0, keepdim=True)
    else:
        mean = scatter_mean(pos, batch, dim=0)[batch]
    centered_pos = pos - mean
    return centered_pos