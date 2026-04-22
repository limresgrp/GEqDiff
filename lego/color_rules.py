from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


ROLE_MAGNITUDE: Dict[str, float] = {
    "CAP_START": 0.22,
    "CAP_END": 0.22,
    "STRAIGHT": 1.00,
    "BEND_LEFT": 0.90,
    "BEND_RIGHT": 0.90,
    "PLANAR": 0.80,
    "SHEET_EDGE": 0.70,
    "JUNCTION_T": 0.15,
    "JUNCTION_BRANCH_LEFT": 0.65,
    "JUNCTION_BRANCH_RIGHT": 0.65,
    "HELIX_PHASE_0": 1.00,
    "HELIX_PHASE_1": 1.00,
    "HELIX_PHASE_2": 1.00,
    "HELIX_PHASE_3": 1.00,
}


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def assign_color_and_dipole(
    *,
    role_names: np.ndarray,
    seq_index_in_branch: np.ndarray,
    descriptors: Dict[str, np.ndarray],
    dipole_noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    role_names = np.asarray(role_names).astype(str).reshape(-1)
    seq_index = np.asarray(seq_index_in_branch, dtype=np.int32).reshape(-1)
    tangent = np.asarray(descriptors["tangent"], dtype=np.float32)
    bend_axis = np.asarray(descriptors["bend_axis"], dtype=np.float32)
    branch_normal = np.asarray(descriptors["branch_local_normal"], dtype=np.float32)
    phase = np.asarray(descriptors["phase_index"], dtype=np.int32).reshape(-1)

    n = int(role_names.shape[0])
    color_class = np.zeros((n,), dtype=np.int32)
    dipoles = np.zeros((n, 3), dtype=np.float32)

    for idx in range(n):
        role = str(role_names[idx])
        mag = float(ROLE_MAGNITUDE.get(role, 0.0))
        if role in {"STRAIGHT", "HELIX_PHASE_0", "HELIX_PHASE_1", "HELIX_PHASE_2", "HELIX_PHASE_3"}:
            sign = 1.0 if (int(phase[idx]) % 2 == 0) else -1.0
            direction = sign * _normalize(tangent[idx])
        elif role in {"PLANAR", "SHEET_EDGE"}:
            sign = 1.0 if (int(seq_index[idx]) % 2 == 0) else -1.0
            direction = sign * _normalize(branch_normal[idx])
        elif role == "BEND_LEFT":
            direction = _normalize(_normalize(tangent[idx]) + 0.8 * _normalize(bend_axis[idx]))
        elif role == "BEND_RIGHT":
            direction = _normalize(_normalize(tangent[idx]) - 0.8 * _normalize(bend_axis[idx]))
        elif role in {"JUNCTION_BRANCH_LEFT", "JUNCTION_BRANCH_RIGHT"}:
            direction = _normalize(0.6 * _normalize(tangent[idx]) + 0.4 * _normalize(branch_normal[idx]))
        elif role == "JUNCTION_T":
            direction = _normalize(branch_normal[idx])
        else:
            # Caps and fallback
            sign = -1.0 if role == "CAP_END" else 1.0
            direction = sign * _normalize(tangent[idx])

        if dipole_noise_scale > 0.0 and rng is not None and mag > 0.0:
            direction = direction + rng.normal(scale=float(dipole_noise_scale), size=(3,)).astype(np.float32)
            direction = _normalize(direction)

        dipoles[idx] = (mag * direction).astype(np.float32)
        if mag <= 1e-8:
            color_class[idx] = 0  # apolar
        else:
            color_class[idx] = 1 if float(np.dot(dipoles[idx], _normalize(direction))) >= 0.0 else 2
            if role == "JUNCTION_T":
                color_class[idx] = 3
            elif role.startswith("CAP_"):
                color_class[idx] = 4
    return color_class.astype(np.int32), dipoles.astype(np.float32)

