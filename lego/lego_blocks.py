"""Discrete LEGO-like voxel blocks and rotation helpers."""

from itertools import permutations, product

import numpy as np


LEGO_LIBRARY = {
    "1x1": {
        "offsets": [[0, 0, 0]],
        "color": "#d64f4f",
    },
    "1x2": {
        "offsets": [[0, 0, 0], [1, 0, 0]],
        "color": "#4f7bd6",
    },
    "L-shape": {
        "offsets": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "color": "#4ea866",
    },
    "T-shape": {
        "offsets": [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0]],
        "color": "#8b5fd6",
    },
}

NEIGHBOR_DIRS = np.array(
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ],
    dtype=int,
)


def _canonicalize_offsets(offsets: np.ndarray) -> np.ndarray:
    order = np.lexsort((offsets[:, 2], offsets[:, 1], offsets[:, 0]))
    return offsets[order]


def _generate_grid_rotations():
    rotations = []
    seen = set()
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rot = np.zeros((3, 3), dtype=int)
            for row, axis in enumerate(perm):
                rot[row, axis] = signs[row]
            if round(np.linalg.det(rot)) != 1:
                continue
            key = tuple(rot.reshape(-1).tolist())
            if key not in seen:
                seen.add(key)
                rotations.append(rot)
    return tuple(rotations)


GRID_ROTATIONS = _generate_grid_rotations()


def rotated_offsets(offsets, rotation):
    """Rotate integer voxel offsets by a 90-degree grid rotation."""
    offsets = np.asarray(offsets, dtype=int)
    rotation = np.asarray(rotation, dtype=int)
    rotated = (rotation @ offsets.T).T.astype(int)
    return _canonicalize_offsets(rotated)


def iter_rotated_offsets(offsets):
    """Yield unique rotated variants of a block."""
    base_offsets = np.asarray(offsets, dtype=int)
    seen = set()
    for rotation in GRID_ROTATIONS:
        rotated = rotated_offsets(base_offsets, rotation)
        key = tuple(map(tuple, rotated.tolist()))
        if key in seen:
            continue
        seen.add(key)
        yield rotation.astype(np.float32), rotated


def get_exposed_faces(offsets):
    """Centers of exposed voxel faces for a block described by integer offsets."""
    offsets = np.asarray(offsets, dtype=int)
    occupied = {tuple(offset) for offset in offsets.tolist()}
    faces = []
    for offset in offsets:
        for direction in NEIGHBOR_DIRS:
            neighbor = tuple((offset + direction).tolist())
            if neighbor not in occupied:
                faces.append(offset.astype(np.float32) + 0.5 * direction.astype(np.float32))
    if not faces:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(faces, dtype=np.float32)


def world_voxels(anchor, offsets, rotation):
    """World-space voxel centers for a block placed at an integer anchor."""
    anchor = np.asarray(anchor, dtype=np.float32)
    local = rotated_offsets(offsets, rotation).astype(np.float32)
    return local + anchor[None, :]
