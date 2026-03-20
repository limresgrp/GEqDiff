from __future__ import annotations

from typing import Iterable, List, Sequence
from pathlib import Path

import numpy as np
import torch

try:
    from lego.utils import DEFAULT_IRREPS, DEFAULT_LMAX, irrep_signature
except ModuleNotFoundError:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from lego.utils import DEFAULT_IRREPS, DEFAULT_LMAX, irrep_signature


def canonical_sort_vectors(vectors: np.ndarray, decimals: int = 5) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors.reshape(0, 3)
    rounded = np.round(vectors, decimals=decimals)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return vectors[order]


def pad_vectors(vectors: np.ndarray, max_vectors: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    padded = np.zeros((max_vectors, 3), dtype=np.float32)
    if vectors.size == 0 or max_vectors <= 0:
        return padded
    count = min(int(vectors.shape[0]), int(max_vectors))
    padded[:count] = vectors[:count]
    return padded


def sh_feature_from_ports(ports: np.ndarray, lmax: int = DEFAULT_LMAX) -> np.ndarray:
    ports = canonical_sort_vectors(ports)
    if ports.size == 0:
        return np.zeros(((lmax + 1) ** 2,), dtype=np.float32)
    return irrep_signature(ports, lmax=lmax).astype(np.float32)


def build_type_vocab(samples: Sequence[dict]) -> List[str]:
    vocab = sorted({str(brick_type) for sample in samples for brick_type in np.asarray(sample["brick_types"])})
    return vocab


def encode_type_names(type_names: Iterable[str], vocab: Sequence[str]) -> np.ndarray:
    lookup = {name: idx for idx, name in enumerate(vocab)}
    encoded = [lookup[str(name)] for name in type_names]
    return np.asarray(encoded, dtype=np.int64)


def irreps_string() -> str:
    return str(DEFAULT_IRREPS)


SH_IRREP_BLOCK_DIMS = (1, 3, 5, 7)


def sh_block_slices(block_dims: Sequence[int] = SH_IRREP_BLOCK_DIMS):
    slices = []
    start = 0
    for width in block_dims:
        stop = start + int(width)
        slices.append(slice(start, stop))
        start = stop
    return tuple(slices)


def fit_sh_block_normalization(
    values: np.ndarray,
    block_dims: Sequence[int] = SH_IRREP_BLOCK_DIMS,
    eps: float = 1e-8,
    center_scalar: bool = True,
) -> dict:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot fit SH normalization on an empty array.")
    flat = array.reshape(-1, array.shape[-1])
    expected_dim = int(sum(block_dims))
    if flat.shape[-1] != expected_dim:
        raise ValueError(f"Expected SH feature dimension {expected_dim}, got {flat.shape[-1]}.")

    means = np.zeros((len(block_dims),), dtype=np.float32)
    stds = np.ones((len(block_dims),), dtype=np.float32)
    for block_index, slc in enumerate(sh_block_slices(block_dims)):
        block = flat[:, slc]
        if int(block_dims[block_index]) == 1:
            means[block_index] = float(block.mean()) if center_scalar else 0.0
            centered = block - means[block_index]
            std_val = float(centered.std())
        else:
            std_val = float(np.sqrt(np.mean(np.square(block))))
        stds[block_index] = 1.0 if std_val < eps else std_val

    return {
        "block_dims": np.asarray(block_dims, dtype=np.int32),
        "means": means,
        "stds": stds,
    }


def _prepare_sh_stats(values, stats: dict):
    block_dims = tuple(int(dim) for dim in np.asarray(stats["block_dims"]).tolist())
    means = stats["means"]
    stds = stats["stds"]
    if torch.is_tensor(values):
        means = torch.as_tensor(means, device=values.device, dtype=values.dtype)
        stds = torch.as_tensor(stds, device=values.device, dtype=values.dtype)
    else:
        means = np.asarray(means, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
    return block_dims, means, stds


def apply_sh_block_normalization(values, stats: dict):
    block_dims, means, stds = _prepare_sh_stats(values, stats)
    out = values.clone() if torch.is_tensor(values) else np.array(values, dtype=np.float32, copy=True)
    for block_index, slc in enumerate(sh_block_slices(block_dims)):
        if int(block_dims[block_index]) == 1:
            out[..., slc] = (out[..., slc] - means[block_index]) / stds[block_index]
        else:
            out[..., slc] = out[..., slc] / stds[block_index]
    return out


def invert_sh_block_normalization(values, stats: dict):
    block_dims, means, stds = _prepare_sh_stats(values, stats)
    out = values.clone() if torch.is_tensor(values) else np.array(values, dtype=np.float32, copy=True)
    for block_index, slc in enumerate(sh_block_slices(block_dims)):
        if int(block_dims[block_index]) == 1:
            out[..., slc] = out[..., slc] * stds[block_index] + means[block_index]
        else:
            out[..., slc] = out[..., slc] * stds[block_index]
    return out


def fit_scalar_normalization(values: np.ndarray, eps: float = 1e-8) -> dict:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot fit scalar normalization on an empty array.")
    flat = array.reshape(-1, array.shape[-1])
    means = flat.mean(axis=0).astype(np.float32)
    stds = flat.std(axis=0).astype(np.float32)
    stds = np.where(stds < eps, 1.0, stds).astype(np.float32)
    return {
        "means": means,
        "stds": stds,
    }


def apply_scalar_normalization(values, stats: dict):
    means = stats["means"]
    stds = stats["stds"]
    if torch.is_tensor(values):
        means = torch.as_tensor(means, device=values.device, dtype=values.dtype)
        stds = torch.as_tensor(stds, device=values.device, dtype=values.dtype)
        return (values - means) / stds
    return ((np.asarray(values, dtype=np.float32) - np.asarray(means, dtype=np.float32)) / np.asarray(stds, dtype=np.float32)).astype(np.float32)


def invert_scalar_normalization(values, stats: dict):
    means = stats["means"]
    stds = stats["stds"]
    if torch.is_tensor(values):
        means = torch.as_tensor(means, device=values.device, dtype=values.dtype)
        stds = torch.as_tensor(stds, device=values.device, dtype=values.dtype)
        return values * stds + means
    return (np.asarray(values, dtype=np.float32) * np.asarray(stds, dtype=np.float32) + np.asarray(means, dtype=np.float32)).astype(np.float32)
