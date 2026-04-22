from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from lego.lego_blocks import GRID_ROTATIONS, LEGO_LIBRARY, rotated_offsets
    from lego.utils import irrep_signature
except ModuleNotFoundError:  # pragma: no cover
    from lego_blocks import GRID_ROTATIONS, LEGO_LIBRARY, rotated_offsets
    from utils import irrep_signature


ROLE_TO_BLOCK_TYPE: Dict[str, str] = {
    "CAP_START": "1x1",
    "CAP_END": "1x1",
    "STRAIGHT": "1x1",
    "BEND_LEFT": "1x1",
    "BEND_RIGHT": "1x1",
    "PLANAR": "1x2",
    "SHEET_EDGE": "1x1",
    "JUNCTION_T": "T-shape",
    "JUNCTION_BRANCH_LEFT": "1x1",
    "JUNCTION_BRANCH_RIGHT": "1x1",
    "HELIX_PHASE_0": "L-shape",
    "HELIX_PHASE_1": "1x2",
    "HELIX_PHASE_2": "L-shape",
    "HELIX_PHASE_3": "1x2",
}

ROLE_TO_PROTOTYPE: Dict[str, str] = {
    "CAP_START": "cap",
    "CAP_END": "cap",
    "STRAIGHT": "rod",
    "BEND_LEFT": "corner",
    "BEND_RIGHT": "corner",
    "PLANAR": "plate",
    "SHEET_EDGE": "plate",
    "JUNCTION_T": "tjunction",
    "JUNCTION_BRANCH_LEFT": "rod",
    "JUNCTION_BRANCH_RIGHT": "rod",
    "HELIX_PHASE_0": "rod",
    "HELIX_PHASE_1": "corner",
    "HELIX_PHASE_2": "rod",
    "HELIX_PHASE_3": "corner",
}

PROTOTYPE_LOCAL_POINTS: Dict[str, np.ndarray] = {
    "rod": np.asarray(
        [
            [1.4, 0.0, 0.0],
            [0.8, 0.22, 0.0],
            [-0.5, -0.08, 0.0],
            [0.25, 0.0, 0.18],
        ],
        dtype=np.float32,
    ),
    "corner": np.asarray(
        [
            [1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [-0.5, -0.4, 0.0],
            [0.0, 0.0, 0.8],
            [0.2, 0.0, -0.3],
        ],
        dtype=np.float32,
    ),
    "plate": np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-0.8, 0.1, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, -0.9, 0.0],
            [0.7, 0.6, 0.0],
            [-0.5, -0.7, 0.0],
            [0.15, 0.0, 0.22],
        ],
        dtype=np.float32,
    ),
    "tjunction": np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-0.9, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -0.8, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.0, -0.2],
        ],
        dtype=np.float32,
    ),
    "cap": np.asarray(
        [
            [0.8, 0.0, 0.0],
            [-0.4, 0.0, 0.0],
            [0.0, 0.35, 0.0],
            [0.0, 0.0, 0.35],
        ],
        dtype=np.float32,
    ),
}


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _build_local_frame(
    role_name: str,
    tangent: np.ndarray,
    bend_axis: np.ndarray,
    branch_local_normal: np.ndarray,
) -> np.ndarray:
    x = _normalize(tangent)
    if float(np.linalg.norm(x)) <= 1e-8:
        x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    role_name = str(role_name)
    if role_name in {"PLANAR", "SHEET_EDGE"}:
        y_guess = _normalize(branch_local_normal)
    elif role_name in {"BEND_LEFT", "BEND_RIGHT"}:
        y_guess = _normalize(bend_axis)
    else:
        y_guess = _normalize(np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), x))
    if float(np.linalg.norm(y_guess)) <= 1e-8:
        y_guess = _normalize(np.cross(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), x))
    z = _normalize(np.cross(x, y_guess))
    if float(np.linalg.norm(z)) <= 1e-8:
        z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    y = _normalize(np.cross(z, x))
    frame = np.stack([x, y, z], axis=1).astype(np.float32)
    return frame.astype(np.float32)


def _nearest_grid_rotation(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float32)
    best_idx = 0
    best_score = -np.inf
    for idx, rotation in enumerate(GRID_ROTATIONS):
        rot = np.asarray(rotation, dtype=np.float32)
        score = float(np.trace(rot.T @ frame))
        if score > best_score:
            best_score = score
            best_idx = idx
    return np.asarray(GRID_ROTATIONS[best_idx], dtype=np.float32)


def _placement_world_keys(anchor: np.ndarray, block_type: str, rotation: np.ndarray) -> List[Tuple[int, int, int]]:
    offsets = np.asarray(LEGO_LIBRARY[str(block_type)]["offsets"], dtype=np.int32)
    rotated = rotated_offsets(offsets, np.rint(np.asarray(rotation, dtype=np.float32)).astype(np.int32))
    anchor_i = np.rint(np.asarray(anchor, dtype=np.float32)).astype(np.int32)
    world = rotated + anchor_i[None, :]
    return [tuple(int(v) for v in row.tolist()) for row in world]


def _prototype_signature(role_name: str, frame: np.ndarray) -> np.ndarray:
    role_name = str(role_name)
    proto_name = ROLE_TO_PROTOTYPE.get(role_name, "rod")
    local_points = PROTOTYPE_LOCAL_POINTS[proto_name]
    world_points = (np.asarray(frame, dtype=np.float32) @ local_points.T).T.astype(np.float32)
    coeff = irrep_signature(world_points, lmax=3).astype(np.float32)
    return coeff


def map_roles_to_shapes(
    *,
    anchors: np.ndarray,
    role_names: np.ndarray,
    descriptors: Dict[str, np.ndarray],
    shape_noise_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Dict[str, np.ndarray]:
    anchors = np.asarray(anchors, dtype=np.float32)
    role_names = np.asarray(role_names).astype(str).reshape(-1)
    tangent = np.asarray(descriptors["tangent"], dtype=np.float32)
    bend_axis = np.asarray(descriptors["bend_axis"], dtype=np.float32)
    branch_local_normal = np.asarray(descriptors["branch_local_normal"], dtype=np.float32)
    n = int(anchors.shape[0])

    requested_block_types = np.asarray(
        [ROLE_TO_BLOCK_TYPE.get(str(role), "1x1") for role in role_names.tolist()],
        dtype=object,
    )
    requested_rotations = np.zeros((n, 3, 3), dtype=np.float32)
    placed_rotations = np.zeros((n, 3, 3), dtype=np.float32)
    placed_types = np.empty((n,), dtype=object)
    frame_axes = np.zeros((n, 3, 3), dtype=np.float32)

    occupied = set()
    for idx in range(n):
        frame = _build_local_frame(
            role_name=str(role_names[idx]),
            tangent=tangent[idx],
            bend_axis=bend_axis[idx],
            branch_local_normal=branch_local_normal[idx],
        )
        frame_axes[idx] = frame
        desired_rot = _nearest_grid_rotation(frame)
        requested_rotations[idx] = desired_rot
        desired_type = str(requested_block_types[idx])

        candidate_rotations = [desired_rot] + [
            np.asarray(rotation, dtype=np.float32)
            for rotation in GRID_ROTATIONS
            if not np.allclose(rotation, desired_rot)
        ]
        placed = False
        for candidate_rotation in candidate_rotations:
            world_keys = _placement_world_keys(anchors[idx], desired_type, candidate_rotation)
            if any(key in occupied for key in world_keys):
                continue
            placed_types[idx] = desired_type
            placed_rotations[idx] = candidate_rotation
            occupied.update(world_keys)
            placed = True
            break
        if placed:
            continue

        fallback_rotation = np.eye(3, dtype=np.float32)
        fallback_offsets = (
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -2.0], dtype=np.float32),
        )
        fallback_world_keys = None
        placed_anchor = None
        for delta in fallback_offsets:
            trial_anchor = (anchors[idx] + delta).astype(np.float32)
            trial_keys = _placement_world_keys(trial_anchor, "1x1", fallback_rotation)
            if any(key in occupied for key in trial_keys):
                continue
            placed_anchor = trial_anchor
            fallback_world_keys = trial_keys
            break
        if placed_anchor is None or fallback_world_keys is None:
            raise RuntimeError("Failed to place a non-overlapping fallback 1x1 brick.")
        anchors[idx] = placed_anchor
        placed_types[idx] = "1x1"
        placed_rotations[idx] = fallback_rotation
        occupied.update(fallback_world_keys)

    features = []
    for idx in range(n):
        coeff = _prototype_signature(str(role_names[idx]), frame_axes[idx]).astype(np.float32)
        if shape_noise_scale > 0.0 and rng is not None:
            coeff = coeff + rng.normal(scale=float(shape_noise_scale), size=coeff.shape).astype(np.float32)
        features.append(coeff.astype(np.float32))
    shape_features = np.asarray(features, dtype=np.float32)

    return {
        "anchors": anchors.astype(np.float32),
        "requested_block_types": np.asarray(requested_block_types, dtype=object),
        "requested_rotations": requested_rotations.astype(np.float32),
        "brick_types": np.asarray(placed_types, dtype=object),
        "brick_rotations": placed_rotations.astype(np.float32),
        "shape_features": shape_features.astype(np.float32),
        "local_frames": frame_axes.astype(np.float32),
    }
