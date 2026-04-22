from __future__ import annotations

from typing import Dict

import numpy as np

from geqdiff.utils.contact_utils import build_brick_geometries, detect_brick_contacts


def validate_topology(
    *,
    parent_id: np.ndarray,
    branch_id: np.ndarray,
    seq_index_in_branch: np.ndarray,
) -> None:
    parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    n = int(parent_id.shape[0])
    if n == 0:
        raise ValueError("Empty scaffold.")
    if np.any(parent_id >= n):
        raise ValueError("parent_id contains out-of-range indices.")
    roots = np.flatnonzero(parent_id < 0)
    if roots.size != 1:
        raise ValueError(f"Expected exactly one root node, got {roots.size}.")
    for branch in np.unique(branch_id).tolist():
        nodes = np.flatnonzero(branch_id == int(branch)).astype(np.int32)
        if nodes.size == 0:
            continue
        order = seq_index[nodes]
        if np.any(order < 0):
            raise ValueError("seq_index_in_branch must be non-negative.")
        if np.unique(order).shape[0] != nodes.shape[0]:
            raise ValueError(f"Duplicate seq_index_in_branch values inside branch {branch}.")


def validate_roles(
    *,
    degree_topology: np.ndarray,
    role_names: np.ndarray,
) -> None:
    degree = np.asarray(degree_topology, dtype=np.int32).reshape(-1)
    role_names = np.asarray(role_names).astype(str).reshape(-1)
    junction_nodes = np.flatnonzero(degree >= 3).astype(np.int32)
    for node in junction_nodes.tolist():
        if str(role_names[int(node)]) != "JUNCTION_T":
            raise ValueError(f"Node {node} has degree>=3 but role {role_names[int(node)]} != JUNCTION_T.")


def validate_geometry(sample: Dict) -> Dict[str, np.ndarray]:
    geometries = build_brick_geometries(sample)
    contact_data = detect_brick_contacts(geometries)
    return {
        "num_contacts": np.asarray(int(np.asarray(contact_data["contact_pairs"]).shape[0]), dtype=np.int64),
        "num_components": np.asarray(int(np.asarray(contact_data["component_id"]).max(initial=-1) + 1), dtype=np.int64),
    }


def validate_sample(
    *,
    parent_id: np.ndarray,
    branch_id: np.ndarray,
    seq_index_in_branch: np.ndarray,
    degree_topology: np.ndarray,
    role_names: np.ndarray,
    sample_for_geometry: Dict,
) -> Dict[str, np.ndarray]:
    validate_topology(parent_id=parent_id, branch_id=branch_id, seq_index_in_branch=seq_index_in_branch)
    validate_roles(degree_topology=degree_topology, role_names=role_names)
    return validate_geometry(sample_for_geometry)

