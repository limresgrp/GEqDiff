from typing import Optional

import torch

from geqtrain.utils.pytorch_scatter import scatter_mean, scatter_sum


def _normalize_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dim() > 1 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    return mask.to(device=reference.device) > 0.5


def compute_reference_mean(
    pos: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    mask = _normalize_mask(mask, pos)

    if batch is None:
        if mask is None:
            return pos.mean(dim=0, keepdim=True)
        if bool(mask.any()):
            return pos[mask].mean(dim=0, keepdim=True)
        return pos.mean(dim=0, keepdim=True)

    if batch.numel() == 0:
        return torch.zeros((0, pos.shape[-1]), dtype=pos.dtype, device=pos.device)

    full_mean = scatter_mean(pos, batch, dim=0)
    if mask is None:
        return full_mean[batch]

    mask_f = mask.to(dtype=pos.dtype).unsqueeze(-1)
    masked_sum = scatter_sum(pos * mask_f, batch, dim=0)
    masked_count = scatter_sum(mask_f, batch, dim=0)
    masked_mean = masked_sum / masked_count.clamp(min=1.0)
    use_masked = masked_count.squeeze(-1) > 0
    graph_mean = torch.where(use_masked.unsqueeze(-1), masked_mean, full_mean)
    return graph_mean[batch]


def center_pos(
    pos: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
):
    """
    Centers coordinates according to batch indices and an optional reference mask.

    Args:
        pos (torch.Tensor): Tensor of shape (N_atoms, 3) with atom coordinates.
        batch (torch.Tensor): Tensor of shape (N_atoms,) with batch indices.
        mask (torch.Tensor): Optional boolean mask selecting the nodes that define
            the centering reference for each graph.

    Returns:
        torch.Tensor: Centered coordinates of shape (N_atoms, 3).
    """
    mean = compute_reference_mean(pos, batch=batch, mask=mask)
    return pos - mean
