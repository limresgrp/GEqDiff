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
    "BEND_LEFT": "L-shape",
    "BEND_RIGHT": "L-shape",
    "PLANAR": "1x2",
    "SHEET_EDGE": "1x1",
    "JUNCTION_T": "T-shape",
    "JUNCTION_BRANCH_LEFT": "T-shape",
    "JUNCTION_BRANCH_RIGHT": "T-shape",
    "HELIX_PHASE_0": "1x2",
    "HELIX_PHASE_1": "1x1",
    "HELIX_PHASE_2": "1x2",
    "HELIX_PHASE_3": "1x1",
}

ROLE_TO_PROTOTYPE: Dict[str, str] = {
    "CAP_START": "cap",
    "CAP_END": "cap",
    "STRAIGHT": "rod",
    "BEND_LEFT": "corner",
    "BEND_RIGHT": "corner",
    "PLANAR": "plate",
    "SHEET_EDGE": "plate",
    "JUNCTION_T": "cap",
    "JUNCTION_BRANCH_LEFT": "tjunction",
    "JUNCTION_BRANCH_RIGHT": "tjunction",
    "HELIX_PHASE_0": "rod",
    "HELIX_PHASE_1": "cap",
    "HELIX_PHASE_2": "rod",
    "HELIX_PHASE_3": "cap",
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


SHEET_ROLE_NAMES = {"BEND_LEFT", "BEND_RIGHT", "PLANAR", "SHEET_EDGE"}


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _project_perpendicular(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    axis = _normalize(axis)
    return (vector - float(np.dot(vector, axis)) * axis).astype(np.float32)


def _principal_lattice_direction(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if float(np.linalg.norm(vector)) <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    axis = int(np.argmax(np.abs(vector)))
    out = np.zeros((3,), dtype=np.float32)
    out[axis] = 1.0 if float(vector[axis]) >= 0.0 else -1.0
    return out.astype(np.float32)


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


def _rotation_from_axes(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    x = _principal_lattice_direction(x_axis)
    y = _principal_lattice_direction(_project_perpendicular(y_axis, x))
    if abs(float(np.dot(x, y))) > 1e-6:
        # fallback: choose any orthogonal cardinal direction
        basis = [
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, -1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ]
        y = None
        for cand in basis:
            if abs(float(np.dot(x, cand))) < 1e-6:
                y = cand
                break
        if y is None:
            y = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    z = np.cross(x, y).astype(np.float32)
    if float(np.linalg.norm(z)) <= 1e-8:
        return np.eye(3, dtype=np.float32)
    z = _principal_lattice_direction(z)
    rot = np.stack([x, y, z], axis=1).astype(np.float32)
    if round(float(np.linalg.det(rot))) != 1:
        z = -z
        rot = np.stack([x, y, z], axis=1).astype(np.float32)
    return rot.astype(np.float32)


def _placement_world_keys(anchor: np.ndarray, block_type: str, rotation: np.ndarray) -> List[Tuple[int, int, int]]:
    offsets = np.asarray(LEGO_LIBRARY[str(block_type)]["offsets"], dtype=np.int32)
    rotated = rotated_offsets(offsets, np.rint(np.asarray(rotation, dtype=np.float32)).astype(np.int32))
    anchor_i = np.rint(np.asarray(anchor, dtype=np.float32)).astype(np.int32)
    world = rotated + anchor_i[None, :]
    return [tuple(int(v) for v in row.tolist()) for row in world]


def _prototype_signature(role_name: str, frame: np.ndarray) -> np.ndarray:
    proto_name = ROLE_TO_PROTOTYPE.get(str(role_name), "rod")
    local_points = PROTOTYPE_LOCAL_POINTS[proto_name]
    world_points = (np.asarray(frame, dtype=np.float32) @ local_points.T).T.astype(np.float32)
    coeff = irrep_signature(world_points, lmax=3).astype(np.float32)
    return coeff


def _default_frame(
    role_name: str,
    tangent: np.ndarray,
    bend_axis: np.ndarray,
    branch_local_normal: np.ndarray,
) -> np.ndarray:
    role_name = str(role_name)
    x = _normalize(tangent)
    if float(np.linalg.norm(x)) <= 1e-8:
        x = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    if role_name.startswith("HELIX_PHASE_"):
        z = x
        radial = _normalize(branch_local_normal)
        radial = _project_perpendicular(radial, z)
        radial = _normalize(radial)
        if float(np.linalg.norm(radial)) <= 1e-8:
            radial = _normalize(np.cross(z, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))
        if float(np.linalg.norm(radial)) <= 1e-8:
            radial = _normalize(np.cross(z, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)))
        azimuth = _normalize(np.cross(z, radial))
        try:
            phase = int(role_name.split("_")[-1]) % 4
        except (ValueError, IndexError):
            phase = 0
        cycle = (radial, azimuth, -radial, -azimuth)
        xh = _normalize(cycle[phase])
        yh = _normalize(np.cross(z, xh))
        return np.stack([xh, yh, z], axis=1).astype(np.float32)

    if role_name in {"JUNCTION_BRANCH_LEFT", "JUNCTION_BRANCH_RIGHT"}:
        z = x
        radial = _normalize(branch_local_normal)
        radial = _project_perpendicular(radial, z)
        radial = _normalize(radial)
        if float(np.linalg.norm(radial)) <= 1e-8:
            radial = _normalize(np.cross(z, np.asarray([0.0, 0.0, 1.0], dtype=np.float32)))
        if float(np.linalg.norm(radial)) <= 1e-8:
            radial = _normalize(np.cross(z, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)))
        if role_name == "JUNCTION_BRANCH_RIGHT":
            radial = -radial
        y = _normalize(np.cross(z, radial))
        return np.stack([radial, y, z], axis=1).astype(np.float32)

    if role_name in {"PLANAR", "SHEET_EDGE"}:
        y_guess = _normalize(branch_local_normal)
    elif role_name in {"BEND_LEFT", "BEND_RIGHT"}:
        turn_normal = _normalize(bend_axis)
        if float(np.linalg.norm(turn_normal)) <= 1e-8:
            turn_normal = _normalize(branch_local_normal)
        y_guess = _normalize(np.cross(turn_normal, x))
        if role_name == "BEND_RIGHT":
            y_guess = -y_guess
    else:
        y_guess = _normalize(np.cross(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), x))
    if float(np.linalg.norm(y_guess)) <= 1e-8:
        y_guess = _normalize(np.cross(np.asarray([0.0, 1.0, 0.0], dtype=np.float32), x))
    z = _normalize(np.cross(x, y_guess))
    if float(np.linalg.norm(z)) <= 1e-8:
        z = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    y = _normalize(np.cross(z, x))
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _sheet_line_rotation(
    idx: int,
    anchors: np.ndarray,
    prev_same_branch: np.ndarray,
    next_same_branch: np.ndarray,
    branch_local_normal: np.ndarray,
) -> np.ndarray:
    prev_idx = int(prev_same_branch[idx])
    next_idx = int(next_same_branch[idx])

    if next_idx >= 0:
        forward = anchors[next_idx] - anchors[idx]
    elif prev_idx >= 0:
        forward = anchors[idx] - anchors[prev_idx]
    else:
        forward = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    x_axis = _principal_lattice_direction(forward)
    y_axis = branch_local_normal[idx]
    return _rotation_from_axes(x_axis=x_axis, y_axis=y_axis)


def _sheet_turn_rotation(
    idx: int,
    anchors: np.ndarray,
    prev_same_branch: np.ndarray,
    next_same_branch: np.ndarray,
) -> np.ndarray:
    prev_idx = int(prev_same_branch[idx])
    next_idx = int(next_same_branch[idx])

    # local L-shape offsets are {0, +x, +y}. At a corner anchor we want one arm
    # to point back toward the previous straight segment and the other arm to
    # point forward into the next straight segment.
    if prev_idx >= 0:
        incoming = _principal_lattice_direction(anchors[idx] - anchors[prev_idx])
        x_axis = -incoming
    else:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    if next_idx >= 0:
        outgoing = _principal_lattice_direction(anchors[next_idx] - anchors[idx])
        y_axis = outgoing
    else:
        # choose something orthogonal to x_axis
        candidates = [
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        ]
        y_axis = None
        for cand in candidates:
            if abs(float(np.dot(cand, x_axis))) < 1e-6:
                y_axis = cand
                break
        if y_axis is None:
            y_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    return _rotation_from_axes(x_axis=x_axis, y_axis=y_axis)


def _desired_rotation_and_frame(
    idx: int,
    role_name: str,
    anchors: np.ndarray,
    descriptors: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    tangent = np.asarray(descriptors["tangent"], dtype=np.float32)
    bend_axis = np.asarray(descriptors["bend_axis"], dtype=np.float32)
    branch_local_normal = np.asarray(descriptors["branch_local_normal"], dtype=np.float32)
    prev_same_branch = np.asarray(descriptors["prev_same_branch"], dtype=np.int32)
    next_same_branch = np.asarray(descriptors["next_same_branch"], dtype=np.int32)

    if role_name in {"PLANAR", "SHEET_EDGE"}:
        rotation = _sheet_line_rotation(idx, anchors, prev_same_branch, next_same_branch, branch_local_normal)
        frame = rotation.astype(np.float32)
        return rotation, frame

    if role_name in {"BEND_LEFT", "BEND_RIGHT"}:
        rotation = _sheet_turn_rotation(idx, anchors, prev_same_branch, next_same_branch)
        frame = rotation.astype(np.float32)
        return rotation, frame

    frame = _default_frame(
        role_name=role_name,
        tangent=tangent[idx],
        bend_axis=bend_axis[idx],
        branch_local_normal=branch_local_normal[idx],
    )
    rotation = _nearest_grid_rotation(frame)
    return rotation.astype(np.float32), frame.astype(np.float32)


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
        role_name = str(role_names[idx])
        desired_type = str(requested_block_types[idx])
        desired_rot, frame = _desired_rotation_and_frame(idx, role_name, anchors, descriptors)
        frame_axes[idx] = frame
        requested_rotations[idx] = desired_rot

        world_keys = _placement_world_keys(anchors[idx], desired_type, desired_rot)
        if not any(key in occupied for key in world_keys):
            placed_types[idx] = desired_type
            placed_rotations[idx] = desired_rot
            occupied.update(world_keys)
            continue

        # For sheet roles, overlap indicates a generator bug. Do not silently
        # rotate away from the line, because that recreates the "zipper" artefact.
        if role_name in SHEET_ROLE_NAMES:
            raise RuntimeError(
                f"Sheet placement overlap at index {idx} for role {role_name} and block {desired_type}. "
                f"This usually means scaffold anchors are inconsistent with brick lengths."
            )

        # For non-sheet motifs (especially thin helices), keep a soft fallback.
        candidate_rotations = [
            np.asarray(rotation, dtype=np.float32)
            for rotation in GRID_ROTATIONS
            if not np.allclose(rotation, desired_rot)
        ]
        placed = False
        for candidate_rotation in candidate_rotations:
            candidate_keys = _placement_world_keys(anchors[idx], desired_type, candidate_rotation)
            if any(key in occupied for key in candidate_keys):
                continue
            placed_types[idx] = desired_type
            placed_rotations[idx] = candidate_rotation
            occupied.update(candidate_keys)
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
        placed_anchor = None
        fallback_world_keys = None
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
