from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _build_branch_orders(branch_id: np.ndarray, seq_index_in_branch: np.ndarray) -> Dict[int, np.ndarray]:
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index_in_branch = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    orders: Dict[int, np.ndarray] = {}
    for branch in sorted(np.unique(branch_id).tolist()):
        nodes = np.flatnonzero(branch_id == int(branch)).astype(np.int32)
        if nodes.size == 0:
            continue
        local_order = nodes[np.argsort(seq_index_in_branch[nodes], kind="stable")]
        orders[int(branch)] = local_order.astype(np.int32)
    return orders


def _prev_next_same_branch(
    branch_id: np.ndarray,
    seq_index_in_branch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    n = int(np.asarray(branch_id).shape[0])
    prev_idx = np.full((n,), fill_value=-1, dtype=np.int32)
    next_idx = np.full((n,), fill_value=-1, dtype=np.int32)
    orders = _build_branch_orders(branch_id=branch_id, seq_index_in_branch=seq_index_in_branch)
    for _, order in orders.items():
        if order.size <= 1:
            continue
        prev_idx[order[1:]] = order[:-1]
        next_idx[order[:-1]] = order[1:]
    return prev_idx, next_idx, orders


def _neighbor_covariance(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return np.zeros((3,), dtype=np.float32), np.eye(3, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1.0, float(points.shape[0] - 1))
    try:
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(eigvals)[::-1]
        eigvals = np.asarray(eigvals[order], dtype=np.float32)
        eigvecs = np.asarray(eigvecs[:, order], dtype=np.float32)
        return eigvals.astype(np.float32), eigvecs.astype(np.float32)
    except np.linalg.LinAlgError:
        return np.zeros((3,), dtype=np.float32), np.eye(3, dtype=np.float32)


def compute_descriptors(
    pos: np.ndarray,
    parent_id: np.ndarray,
    branch_id: np.ndarray,
    seq_index_in_branch: np.ndarray,
    degree_topology: np.ndarray,
    helix_phase_period: int = 4,
    neighborhood_radius: float = 2.5,
) -> Dict[str, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
    branch_id = np.asarray(branch_id, dtype=np.int32).reshape(-1)
    seq_index_in_branch = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    degree_topology = np.asarray(degree_topology, dtype=np.int32).reshape(-1)
    n = int(pos.shape[0])

    prev_idx, next_idx, branch_orders = _prev_next_same_branch(
        branch_id=branch_id,
        seq_index_in_branch=seq_index_in_branch,
    )
    tangent = np.zeros((n, 3), dtype=np.float32)
    curvature_mag = np.zeros((n,), dtype=np.float32)
    bend_axis = np.zeros((n, 3), dtype=np.float32)
    branch_terminal = np.ones((n,), dtype=bool)

    for node in range(n):
        prev_node = int(prev_idx[node])
        next_node = int(next_idx[node])
        if next_node >= 0:
            branch_terminal[node] = False
        if prev_node >= 0 and next_node >= 0:
            v1 = pos[node] - pos[prev_node]
            v2 = pos[next_node] - pos[node]
            tangent[node] = _normalize(pos[next_node] - pos[prev_node])
            curvature_mag[node] = float(np.linalg.norm(pos[next_node] - 2.0 * pos[node] + pos[prev_node]))
            bend_axis[node] = np.cross(v1, v2).astype(np.float32)
        elif next_node >= 0:
            tangent[node] = _normalize(pos[next_node] - pos[node])
            curvature_mag[node] = 0.0
        elif prev_node >= 0:
            tangent[node] = _normalize(pos[node] - pos[prev_node])
            curvature_mag[node] = 0.0
        else:
            tangent[node] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            curvature_mag[node] = 0.0

    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1).astype(np.float32)
    eigvals = np.zeros((n, 3), dtype=np.float32)
    eigvecs = np.zeros((n, 3, 3), dtype=np.float32)
    linearity = np.zeros((n,), dtype=np.float32)
    planarity = np.zeros((n,), dtype=np.float32)
    branch_local_normal = np.zeros((n, 3), dtype=np.float32)
    for node in range(n):
        neighborhood = np.flatnonzero(dmat[node] <= float(neighborhood_radius)).astype(np.int32)
        if neighborhood.size < 3:
            neighborhood = np.asarray([node], dtype=np.int32)
            if prev_idx[node] >= 0:
                neighborhood = np.append(neighborhood, prev_idx[node])
            if next_idx[node] >= 0:
                neighborhood = np.append(neighborhood, next_idx[node])
        local_points = pos[np.unique(neighborhood)]
        vals, vecs = _neighbor_covariance(local_points)
        eigvals[node] = vals
        eigvecs[node] = vecs
        l1, l2, l3 = float(vals[0]), float(vals[1]), float(vals[2])
        denom = max(l1, 1e-8)
        linearity[node] = float((l1 - l2) / denom)
        planarity[node] = float((l2 - l3) / denom)
        normal = vecs[:, 2]
        if float(np.linalg.norm(normal)) <= 1e-8:
            if float(np.linalg.norm(bend_axis[node])) > 1e-8:
                normal = _normalize(bend_axis[node])
            else:
                normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        branch_local_normal[node] = _normalize(normal)

    boundary_flag = (seq_index_in_branch == 0) | branch_terminal
    junction_flag = degree_topology >= 3
    phase_index = (seq_index_in_branch % int(max(1, helix_phase_period))).astype(np.int32)

    # Resolve sign ambiguity of local normals along each branch.
    for _, order in branch_orders.items():
        if order.size <= 1:
            continue
        ref = branch_local_normal[int(order[0])]
        for node in order[1:]:
            idx = int(node)
            cur = branch_local_normal[idx]
            if float(np.dot(ref, cur)) < 0.0:
                branch_local_normal[idx] = -cur
            ref = branch_local_normal[idx]

    return {
        "tangent": tangent.astype(np.float32),
        "curvature_mag": curvature_mag.astype(np.float32),
        "bend_axis": bend_axis.astype(np.float32),
        "local_cov_eigs": eigvals.astype(np.float32),
        "local_cov_vecs": eigvecs.astype(np.float32),
        "planarity_score": planarity.astype(np.float32),
        "linearity_score": linearity.astype(np.float32),
        "boundary_flag": boundary_flag.astype(bool),
        "junction_flag": junction_flag.astype(bool),
        "phase_index": phase_index.astype(np.int32),
        "branch_local_normal": branch_local_normal.astype(np.float32),
        "prev_same_branch": prev_idx.astype(np.int32),
        "next_same_branch": next_idx.astype(np.int32),
    }

