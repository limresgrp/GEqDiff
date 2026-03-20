from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


LOCAL_DIPOLE_STATES = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DipoleAssignmentConfig:
    attraction_reward: float = -1.0
    repulsion_penalty: float = 1.25
    neutral_contact_penalty: float = 0.2
    polar_cost: float = 0.24
    restarts: int = 8
    sweeps: int = 16


def normalize_rows(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    out = np.zeros_like(vectors, dtype=np.float32)
    mask = norms > eps
    out = np.divide(vectors, np.maximum(norms, eps), out=out, where=mask)
    return out


def dipole_strengths(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32)
    return np.linalg.norm(directions, axis=-1, keepdims=True).astype(np.float32)


def candidate_world_dipoles(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotations with shape [N, 3, 3], got {rotations.shape}.")
    local = np.broadcast_to(LOCAL_DIPOLE_STATES[None, :, :], (rotations.shape[0], LOCAL_DIPOLE_STATES.shape[0], 3))
    return np.einsum("nij,nkj->nki", rotations, local, dtype=np.float32).astype(np.float32)


def _face_projection(
    dipole: np.ndarray,
    direction: np.ndarray,
) -> float:
    direction = np.asarray(direction, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-8:
        return 0.0
    return float(np.dot(np.asarray(dipole, dtype=np.float32), direction / direction_norm))


def _pair_energy(
    dipole_i: np.ndarray,
    dipole_j: np.ndarray,
    direction_ij: np.ndarray,
    config: DipoleAssignmentConfig,
) -> float:
    q_i = _face_projection(dipole_i, direction_ij)
    q_j = _face_projection(dipole_j, -np.asarray(direction_ij, dtype=np.float32))
    coupling = float(q_i * q_j)
    contact_energy = 0.0
    if coupling > 0.0:
        contact_energy += float(config.repulsion_penalty) * coupling
    elif coupling < 0.0:
        contact_energy += float(config.attraction_reward) * (-coupling)

    occupancy = abs(q_i) * abs(q_j)
    contact_energy += float(config.neutral_contact_penalty) * (1.0 - occupancy)
    return float(contact_energy)


def _node_local_energy(
    node_index: int,
    state_indices: np.ndarray,
    candidates: np.ndarray,
    adjacency_pairs: Sequence[tuple[int, np.ndarray]],
    config: DipoleAssignmentConfig,
) -> float:
    dipole = candidates[node_index, int(state_indices[node_index])]
    energy = float(config.polar_cost) if np.linalg.norm(dipole) > 1e-6 else 0.0
    for neighbor_index, direction in adjacency_pairs:
        energy += _pair_energy(
            dipole,
            candidates[int(neighbor_index), int(state_indices[int(neighbor_index)])],
            np.asarray(direction, dtype=np.float32),
            config=config,
        )
    return energy


def assign_discrete_dipoles(
    rotations: np.ndarray,
    contact_pairs: np.ndarray,
    contact_face_dirs: np.ndarray,
    rng: np.random.Generator | None = None,
    config: DipoleAssignmentConfig = DipoleAssignmentConfig(),
    all_face_contact_pairs: np.ndarray | None = None,
    all_face_contact_dirs: np.ndarray | None = None,
) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=np.float32)
    num_nodes = int(rotations.shape[0])
    if rng is None:
        rng = np.random.default_rng()

    if all_face_contact_pairs is None or all_face_contact_dirs is None:
        all_face_contact_pairs = np.asarray(contact_pairs, dtype=np.int64)
        all_face_contact_dirs = np.asarray(contact_face_dirs, dtype=np.float32)
    else:
        all_face_contact_pairs = np.asarray(all_face_contact_pairs, dtype=np.int64)
        all_face_contact_dirs = np.asarray(all_face_contact_dirs, dtype=np.float32)

    candidates = candidate_world_dipoles(rotations)
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_nodes)]
    contact_terms: list[tuple[int, int, np.ndarray]] = []
    for (src, dst), direction in zip(all_face_contact_pairs, all_face_contact_dirs):
        src_i = int(src)
        dst_i = int(dst)
        direction = np.asarray(direction, dtype=np.float32)
        adjacency[src_i].append((dst_i, direction))
        adjacency[dst_i].append((src_i, -direction))
        contact_terms.append((src_i, dst_i, direction))

    if all(len(neighbors) == 0 for neighbors in adjacency):
        return np.zeros((num_nodes, 3), dtype=np.float32)

    best_energy = None
    best_state = None
    neutral_index = 0
    for restart in range(int(config.restarts)):
        state_indices = np.full((num_nodes,), fill_value=neutral_index, dtype=np.int64)
        if restart > 0:
            state_indices = rng.integers(0, candidates.shape[1], size=(num_nodes,), dtype=np.int64)
        for _ in range(int(config.sweeps)):
            changed = False
            for node_index in rng.permutation(num_nodes).tolist():
                current_choice = int(state_indices[node_index])
                best_choice = current_choice
                best_local_energy = None
                for candidate_index in range(candidates.shape[1]):
                    state_indices[node_index] = int(candidate_index)
                    local_energy = _node_local_energy(
                        node_index=node_index,
                        state_indices=state_indices,
                        candidates=candidates,
                        adjacency_pairs=adjacency[node_index],
                        config=config,
                    )
                    if best_local_energy is None or local_energy < best_local_energy - 1e-8:
                        best_local_energy = float(local_energy)
                        best_choice = int(candidate_index)
                state_indices[node_index] = best_choice
                changed = changed or (best_choice != current_choice)
            if not changed:
                break

        total_energy = 0.0
        for src in range(num_nodes):
            dipole_src = candidates[src, int(state_indices[src])]
            if np.linalg.norm(dipole_src) > 1e-6:
                total_energy += float(config.polar_cost)
        for src, dst, direction in contact_terms:
            total_energy += _pair_energy(
                candidates[int(src), int(state_indices[int(src)])],
                candidates[int(dst), int(state_indices[int(dst)])],
                np.asarray(direction, dtype=np.float32),
                config=config,
            )

        if best_energy is None or total_energy < best_energy:
            best_energy = float(total_energy)
            best_state = state_indices.copy()

    if best_state is None:
        return np.zeros((num_nodes, 3), dtype=np.float32)
    return candidates[np.arange(num_nodes), best_state].astype(np.float32)


def evaluate_contact_energy(
    dipoles: np.ndarray,
    all_face_contact_pairs: np.ndarray,
    all_face_contact_dirs: np.ndarray,
    config: DipoleAssignmentConfig = DipoleAssignmentConfig(),
) -> dict[str, float | int]:
    dipoles = np.asarray(dipoles, dtype=np.float32)
    all_face_contact_pairs = np.asarray(all_face_contact_pairs, dtype=np.int64)
    all_face_contact_dirs = np.asarray(all_face_contact_dirs, dtype=np.float32)

    polar_mask = np.linalg.norm(dipoles, axis=-1) > 1e-6
    polar_cost = float(config.polar_cost) * int(polar_mask.sum())
    contact_energy = 0.0
    attractive_contacts = 0
    repulsive_contacts = 0
    neutral_contacts = 0

    for (src, dst), direction in zip(all_face_contact_pairs, all_face_contact_dirs):
        src_i = int(src)
        dst_i = int(dst)
        direction = np.asarray(direction, dtype=np.float32)
        q_i = _face_projection(dipoles[src_i], direction)
        q_j = _face_projection(dipoles[dst_i], -direction)
        coupling = float(q_i * q_j)
        if abs(q_i) <= 1e-8 or abs(q_j) <= 1e-8:
            neutral_contacts += 1
        elif coupling > 0.0:
            repulsive_contacts += 1
        elif coupling < 0.0:
            attractive_contacts += 1
        else:
            neutral_contacts += 1
        contact_energy += _pair_energy(dipoles[src_i], dipoles[dst_i], direction, config=config)

    total_energy = polar_cost + contact_energy
    num_faces = int(all_face_contact_pairs.shape[0])
    return {
        "total_energy": float(total_energy),
        "polar_cost": float(polar_cost),
        "contact_energy": float(contact_energy),
        "num_face_contacts": num_faces,
        "num_attractive_contacts": int(attractive_contacts),
        "num_repulsive_contacts": int(repulsive_contacts),
        "num_neutral_contacts": int(neutral_contacts),
        "mean_energy_per_face": float(contact_energy / num_faces) if num_faces > 0 else 0.0,
    }


def split_shape_irreps(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != 16:
        raise ValueError(f"Expected SH feature dimension 16, got {values.shape[-1]}.")

    l0 = values[..., 0:1]
    l1 = values[..., 1:4]
    l2 = values[..., 4:9]
    l3 = values[..., 9:16]

    l1_norm = np.linalg.norm(l1, axis=-1, keepdims=True).astype(np.float32)
    l2_norm = np.linalg.norm(l2, axis=-1, keepdims=True).astype(np.float32)
    l3_norm = np.linalg.norm(l3, axis=-1, keepdims=True).astype(np.float32)

    l1_dir = np.divide(l1, np.maximum(l1_norm, 1e-8), where=l1_norm > 1e-8, out=np.zeros_like(l1))
    l2_dir = np.divide(l2, np.maximum(l2_norm, 1e-8), where=l2_norm > 1e-8, out=np.zeros_like(l2))
    l3_dir = np.divide(l3, np.maximum(l3_norm, 1e-8), where=l3_norm > 1e-8, out=np.zeros_like(l3))

    shape_scalars = np.concatenate([l0, l1_norm, l2_norm, l3_norm], axis=-1).astype(np.float32)
    shape_equivariants = np.concatenate([l1_dir, l2_dir, l3_dir], axis=-1).astype(np.float32)
    return shape_scalars, shape_equivariants


def combine_shape_irreps(shape_scalars: np.ndarray, shape_equivariants: np.ndarray) -> np.ndarray:
    shape_scalars = np.asarray(shape_scalars, dtype=np.float32)
    shape_equivariants = np.asarray(shape_equivariants, dtype=np.float32)
    if shape_scalars.shape[-1] != 4:
        raise ValueError(f"Expected shape scalars with dim 4, got {shape_scalars.shape[-1]}.")
    if shape_equivariants.shape[-1] != 15:
        raise ValueError(f"Expected shape equivariants with dim 15, got {shape_equivariants.shape[-1]}.")

    l0 = shape_scalars[..., 0:1]
    l1_mag = np.maximum(shape_scalars[..., 1:2], 0.0)
    l2_mag = np.maximum(shape_scalars[..., 2:3], 0.0)
    l3_mag = np.maximum(shape_scalars[..., 3:4], 0.0)

    l1_dir = normalize_rows(shape_equivariants[..., 0:3].reshape(-1, 3)).reshape(shape_equivariants.shape[:-1] + (3,))
    l2_dir = normalize_rows(shape_equivariants[..., 3:8].reshape(-1, 5)).reshape(shape_equivariants.shape[:-1] + (5,))
    l3_dir = normalize_rows(shape_equivariants[..., 8:15].reshape(-1, 7)).reshape(shape_equivariants.shape[:-1] + (7,))

    l1 = l1_dir * l1_mag
    l2 = l2_dir * l2_mag
    l3 = l3_dir * l3_mag
    return np.concatenate([l0, l1, l2, l3], axis=-1).astype(np.float32)


def normalize_dipole_directions(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32)
    if directions.shape[-1] != 3:
        raise ValueError(f"Expected dipole directions with dim 3, got {directions.shape[-1]}.")
    return normalize_rows(directions.reshape(-1, 3)).reshape(directions.shape).astype(np.float32)
