from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from geqdiff.data import AtomicDataDict

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def _center_graph_points(points: torch.Tensor) -> torch.Tensor:
    return points - points.mean(dim=1, keepdim=True)


def _batched_kabsch_align(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source = _center_graph_points(source)
    target = _center_graph_points(target)

    covariance = source.transpose(-1, -2) @ target
    u, _, vh = torch.linalg.svd(covariance)

    det = torch.det(u @ vh)
    correction = torch.eye(3, device=source.device, dtype=source.dtype).unsqueeze(0).repeat(source.shape[0], 1, 1)
    correction[:, -1, -1] = torch.where(det < 0, -torch.ones_like(det), torch.ones_like(det))

    rotation = u @ correction @ vh
    return source @ rotation


def _pairwise_aligned_rmsd(
    target_graphs: torch.Tensor,
    source_graphs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_graphs, num_atoms, _ = target_graphs.shape
    target_pairs = target_graphs.unsqueeze(1).expand(num_graphs, num_graphs, num_atoms, 3).reshape(-1, num_atoms, 3)
    source_pairs = source_graphs.unsqueeze(0).expand(num_graphs, num_graphs, num_atoms, 3).reshape(-1, num_atoms, 3)

    aligned_source_pairs = _batched_kabsch_align(source_pairs, target_pairs)
    squared_distances = (aligned_source_pairs - target_pairs).pow(2).sum(dim=-1)
    rmsd = torch.sqrt(squared_distances.mean(dim=-1).clamp_min(0.0))

    return rmsd.view(num_graphs, num_graphs), aligned_source_pairs.view(num_graphs, num_graphs, num_atoms, 3)


def _graph_signature(
    graph_index: int,
    atom_count: int,
    data: AtomicDataDict.Type,
    batch: torch.Tensor,
    corrupt_mask: Optional[torch.Tensor] = None,
) -> Tuple[int, Optional[Tuple[int, ...]]]:
    node_types = data.get(AtomicDataDict.NODE_TYPE_KEY)
    if node_types is None:
        return atom_count, None

    graph_mask = batch == graph_index
    if corrupt_mask is not None:
        graph_mask = graph_mask & corrupt_mask
    graph_node_types = node_types[graph_mask]
    if graph_node_types.dim() > 1 and graph_node_types.shape[-1] == 1:
        graph_node_types = graph_node_types.squeeze(-1)
    if graph_node_types.numel() != atom_count:
        raise ValueError(
            f"Node-type tensor for graph {graph_index} has {graph_node_types.numel()} entries, "
            f"expected {atom_count}."
        )
    return atom_count, tuple(int(v) for v in graph_node_types.detach().cpu().tolist())


def sample_ot_aligned_noise(
    x: torch.Tensor,
    data: AtomicDataDict.Type,
    batch: torch.Tensor,
    corrupt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if linear_sum_assignment is None:
        raise ImportError("scipy is required to use AOT. Please install it: `pip install scipy`")

    num_graphs = int(batch.max().item()) + 1
    atom_counts = torch.bincount(batch, minlength=num_graphs)
    grouped_graph_indices: Dict[Tuple[int, Optional[Tuple[int, ...]]], List[int]] = defaultdict(list)
    matched_noise = torch.randn_like(x)

    if corrupt_mask is not None:
        if corrupt_mask.dim() > 1 and corrupt_mask.shape[-1] == 1:
            corrupt_mask = corrupt_mask.squeeze(-1)
        corrupt_mask = corrupt_mask.to(device=x.device, dtype=torch.bool)
        if corrupt_mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"Corruption mask shape {tuple(corrupt_mask.shape)} is incompatible with coordinates {tuple(x.shape)}."
            )

    for graph_index, atom_count in enumerate(atom_counts.tolist()):
        if corrupt_mask is None:
            subset_size = atom_count
        else:
            subset_size = int(corrupt_mask[batch == graph_index].sum().item())
        if subset_size <= 0:
            continue
        grouped_graph_indices[_graph_signature(graph_index, subset_size, data, batch, corrupt_mask)].append(graph_index)

    for graph_indices in grouped_graph_indices.values():
        graph_targets = []
        graph_selection_masks = []
        for graph_index in graph_indices:
            graph_mask = batch == graph_index
            if corrupt_mask is not None:
                graph_mask = graph_mask & corrupt_mask
            graph_selection_masks.append(graph_mask)
            graph_targets.append(x[graph_mask])
        graph_targets = torch.stack(graph_targets, dim=0)
        graph_sources = _center_graph_points(torch.randn_like(graph_targets))

        cost_matrix, aligned_sources = _pairwise_aligned_rmsd(graph_targets, graph_sources)
        _, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        col_ind_tensor = torch.as_tensor(col_ind, device=x.device, dtype=torch.long)

        matched_group_sources = aligned_sources[torch.arange(len(graph_indices), device=x.device), col_ind_tensor]
        for group_offset, graph_mask in enumerate(graph_selection_masks):
            matched_noise[graph_mask] = matched_group_sources[group_offset]

    return matched_noise
