"""Shared spherical-harmonic geometry and dataset helpers for LEGO blocks."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

try:
    from lego.lego_blocks import LEGO_LIBRARY, NEIGHBOR_DIRS, get_exposed_faces, iter_rotated_offsets, rotated_offsets
except ModuleNotFoundError:
    from lego_blocks import LEGO_LIBRARY, NEIGHBOR_DIRS, get_exposed_faces, iter_rotated_offsets, rotated_offsets


DEFAULT_LMAX = 3
DEFAULT_IRREPS = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")


def default_dataset_path():
    return Path(__file__).with_name("lego_dataset_dipoles.npz")


def infer_lmax(num_coefficients):
    lmax = int(round(np.sqrt(num_coefficients))) - 1
    if (lmax + 1) ** 2 != int(num_coefficients):
        raise ValueError(
            f"Expected (lmax + 1)^2 coefficients, got {num_coefficients}."
        )
    return lmax


def normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors, np.zeros((0,), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
    unit = np.zeros_like(vectors, dtype=np.float32)
    mask = norms > 1e-8
    unit[mask] = vectors[mask] / norms[mask, None]
    return unit, norms


def spherical_harmonic_basis(vectors, lmax=DEFAULT_LMAX):
    """Real spherical harmonics up to lmax for unit vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return np.zeros((0, (lmax + 1) ** 2), dtype=np.float32)
    unit, _ = normalize_vectors(vectors)
    basis = o3.spherical_harmonics(
        list(range(lmax + 1)),
        torch.as_tensor(unit, dtype=torch.float32),
        normalize=True,
        normalization="integral",
    )
    return basis.detach().cpu().numpy().astype(np.float32)


def radial_profile(
    coefficients,
    directions,
    base_radius=5.0,
    radial_scale=1.55,
    min_radius=3.5,
    max_radius=10.5,
):
    """Evaluate a radial field r(u) = base + scale * <c, Y(u)> on directions."""
    coefficients = np.asarray(coefficients, dtype=np.float32)
    lmax = infer_lmax(coefficients.shape[-1])
    basis = spherical_harmonic_basis(directions, lmax=lmax)
    radii = base_radius + radial_scale * (basis @ coefficients)
    return np.clip(radii, min_radius, max_radius).astype(np.float32)


def build_surface_mesh(
    coefficients,
    resolution=36,
    base_radius=5.0,
    radial_scale=1.55,
    min_radius=3.5,
    max_radius=10.5,
):
    """Sample the SH radial field on a latitude-longitude grid."""
    theta = np.linspace(0.0, np.pi, resolution, dtype=np.float32)
    phi = np.linspace(0.0, 2.0 * np.pi, 2 * resolution, endpoint=False, dtype=np.float32)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    directions = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)

    radii = radial_profile(
        coefficients,
        directions,
        base_radius=base_radius,
        radial_scale=radial_scale,
        min_radius=min_radius,
        max_radius=max_radius,
    ).reshape(theta_grid.shape)

    x = radii * np.sin(theta_grid) * np.cos(phi_grid)
    y = radii * np.sin(theta_grid) * np.sin(phi_grid)
    z = radii * np.cos(theta_grid)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), radii.astype(np.float32)


def largest_connected_component(voxels):
    voxels = np.asarray(voxels, dtype=np.int32)
    if voxels.size == 0:
        return voxels.reshape(0, 3)

    occupied = {tuple(v) for v in voxels.tolist()}
    visited = set()
    largest = []

    for start in occupied:
        if start in visited:
            continue
        stack = [start]
        component = []
        visited.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            current_arr = np.asarray(current, dtype=np.int32)
            for direction in NEIGHBOR_DIRS:
                neighbor = tuple((current_arr + direction).tolist())
                if neighbor in occupied and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if len(component) > len(largest):
            largest = component

    largest = np.asarray(largest, dtype=np.int32)
    order = np.lexsort((largest[:, 2], largest[:, 1], largest[:, 0]))
    return largest[order]


def _voxel_set_is_connected(occupied):
    if len(occupied) == 0:
        return False
    start = next(iter(occupied))
    visited = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        current_arr = np.asarray(current, dtype=np.int32)
        for direction in NEIGHBOR_DIRS:
            neighbor = tuple((current_arr + direction).tolist())
            if neighbor in occupied and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(occupied)


def thin_connected_voxels(voxels, sparsity=0.0, rng=None):
    voxels = np.asarray(voxels, dtype=np.int32)
    if voxels.size == 0:
        return voxels.reshape(0, 3)
    sparsity = float(np.clip(sparsity, 0.0, 0.95))
    if sparsity <= 0.0 or voxels.shape[0] <= 2:
        return largest_connected_component(voxels)

    if rng is None:
        rng = np.random.default_rng()

    occupied = {tuple(v) for v in voxels.tolist()}
    target_removals = int(round(sparsity * len(occupied)))
    candidates = list(occupied)
    rng.shuffle(candidates)

    removed = 0
    for key in candidates:
        if removed >= target_removals or len(occupied) <= 2:
            break
        occupied.remove(key)
        if _voxel_set_is_connected(occupied):
            removed += 1
        else:
            occupied.add(key)

    kept = np.asarray(sorted(occupied), dtype=np.int32)
    return largest_connected_component(kept)


def voxelize_radial_shape(
    coefficients,
    base_radius=5.0,
    radial_scale=1.55,
    min_radius=3.5,
    max_radius=10.5,
    grid_extent=None,
    voxel_margin=0.35,
    occupancy_mode="solid",
    shell_thickness=1.1,
    shell_sparsity=0.0,
    rng=None,
):
    """Convert the SH radial field into occupied unit voxels.

    `solid` fills the full interior volume.
    `shell` keeps only a surface band and can optionally thin it while
    preserving 6-neighbor connectivity as much as possible.
    """
    if grid_extent is None:
        grid_extent = int(np.ceil(max_radius + voxel_margin)) + 1

    grid = np.arange(-grid_extent, grid_extent + 1, dtype=np.int32)
    centers = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(-1, 3)
    norms = np.linalg.norm(centers.astype(np.float32), axis=1)
    directions = np.zeros_like(centers, dtype=np.float32)
    mask = norms > 1e-8
    directions[mask] = centers[mask].astype(np.float32) / norms[mask, None]

    radii = np.zeros_like(norms, dtype=np.float32)
    radii[mask] = radial_profile(
        coefficients,
        directions[mask],
        base_radius=base_radius,
        radial_scale=radial_scale,
        min_radius=min_radius,
        max_radius=max_radius,
    )
    radii[~mask] = base_radius

    occupancy_mode = str(occupancy_mode).strip().lower()
    solid_mask = norms <= (radii + voxel_margin)
    if occupancy_mode == "solid":
        occupied = centers[solid_mask]
    elif occupancy_mode == "shell":
        shell_thickness = float(max(shell_thickness, 0.25))
        shell_mask = solid_mask & (norms >= np.maximum(0.0, radii - shell_thickness))
        occupied = centers[shell_mask]
        if occupied.size == 0:
            boundary_mask = np.abs(norms - radii) <= max(voxel_margin, 0.5)
            occupied = centers[boundary_mask]
        occupied = thin_connected_voxels(occupied, sparsity=shell_sparsity, rng=rng)
    else:
        raise ValueError(f"Unsupported occupancy_mode '{occupancy_mode}'. Expected 'solid' or 'shell'.")

    if occupied.size == 0:
        occupied = np.zeros((1, 3), dtype=np.int32)
    return largest_connected_component(occupied)


def irrep_signature(points, lmax=DEFAULT_LMAX):
    """Weighted SH signature of a point cloud expressed around the origin."""
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return np.zeros(((lmax + 1) ** 2,), dtype=np.float32)
    unit, norms = normalize_vectors(points)
    basis = spherical_harmonic_basis(unit, lmax=lmax)
    signature = (basis * norms[:, None]).sum(axis=0)
    return signature.astype(np.float32)


def block_signature(offsets, rotation=None, lmax=DEFAULT_LMAX):
    """Irrep descriptor of a block from the directions of its exposed faces."""
    if rotation is None:
        rotation = np.eye(3, dtype=int)
    local_offsets = rotated_offsets(offsets, rotation).astype(np.float32)
    com = local_offsets.mean(axis=0)
    exposed_faces = get_exposed_faces(local_offsets) - com
    return irrep_signature(exposed_faces, lmax=lmax)


@lru_cache(maxsize=4)
def brick_signature_library(lmax=DEFAULT_LMAX):
    signatures = []
    brick_types = []
    rotations = []
    for brick_type, spec in LEGO_LIBRARY.items():
        for rotation, _ in iter_rotated_offsets(spec["offsets"]):
            signatures.append(block_signature(spec["offsets"], rotation=rotation, lmax=lmax))
            brick_types.append(brick_type)
            rotations.append(np.asarray(rotation, dtype=np.float32))

    if len(signatures) == 0:
        raise ValueError("LEGO library is empty.")

    return {
        "signatures": np.asarray(signatures, dtype=np.float32),
        "brick_types": np.asarray(brick_types),
        "rotations": np.asarray(rotations, dtype=np.float32),
    }


def decode_brick_signatures(signatures: np.ndarray, lmax=DEFAULT_LMAX):
    query = np.asarray(signatures, dtype=np.float32)
    if query.ndim != 2:
        raise ValueError(f"Expected [N, F] signatures, got {query.shape}.")
    library = brick_signature_library(lmax=lmax)
    reference = np.asarray(library["signatures"], dtype=np.float32)
    distances = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=-1)
    nearest = distances.argmin(axis=1)
    return {
        "brick_types": np.asarray(library["brick_types"])[nearest],
        "rotations": np.asarray(library["rotations"], dtype=np.float32)[nearest],
        "distances": distances[np.arange(query.shape[0]), nearest].astype(np.float32),
        "indices": nearest.astype(np.int64),
    }


def _normalize_sample(sample):
    sample = dict(sample)

    brick_anchors = sample.get("brick_anchors", sample.get("pos", np.zeros((0, 3), dtype=np.float32)))
    brick_types = sample.get("brick_types", sample.get("types", np.asarray([], dtype=str)))
    brick_features = sample.get("brick_features", sample.get("features", np.zeros((0, 16), dtype=np.float32)))
    brick_dipoles = sample.get("brick_dipoles", sample.get("dipoles", np.zeros((len(np.asarray(brick_anchors)), 3), dtype=np.float32)))
    brick_dipole_directions = sample.get("brick_dipole_directions", None)
    brick_dipole_strengths = sample.get("brick_dipole_strengths", None)
    brick_rotations = sample.get("brick_rotations", sample.get("rotations"))

    brick_anchors = np.asarray(brick_anchors, dtype=np.float32)
    brick_types = np.asarray(brick_types)
    brick_features = np.asarray(brick_features, dtype=np.float32)
    brick_dipoles = np.asarray(brick_dipoles, dtype=np.float32)
    if brick_dipole_directions is None:
        norms = np.linalg.norm(brick_dipoles, axis=-1, keepdims=True)
        brick_dipole_directions = np.divide(
            brick_dipoles,
            np.maximum(norms, 1e-8),
            out=np.zeros_like(brick_dipoles, dtype=np.float32),
            where=norms > 1e-8,
        )
    else:
        brick_dipole_directions = np.asarray(brick_dipole_directions, dtype=np.float32)
    if brick_dipole_strengths is None:
        brick_dipole_strengths = np.linalg.norm(brick_dipoles, axis=-1, keepdims=True).astype(np.float32)
    else:
        brick_dipole_strengths = np.asarray(brick_dipole_strengths, dtype=np.float32)

    if brick_rotations is None:
        brick_rotations = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], len(brick_anchors), axis=0)
    else:
        brick_rotations = np.asarray(brick_rotations, dtype=np.float32)

    sample["brick_anchors"] = brick_anchors
    sample["brick_types"] = brick_types
    sample["brick_features"] = brick_features
    sample["brick_dipoles"] = brick_dipoles
    sample["brick_dipole_directions"] = brick_dipole_directions
    sample["brick_dipole_strengths"] = brick_dipole_strengths
    sample["brick_rotations"] = brick_rotations

    # Compatibility aliases for older code.
    sample["pos"] = brick_anchors
    sample["types"] = brick_types
    sample["features"] = brick_features
    sample["dipoles"] = brick_dipoles
    sample["rotations"] = brick_rotations

    if "coefficients" in sample:
        sample["coefficients"] = np.asarray(sample["coefficients"], dtype=np.float32)
    if "occupancy_mode" in sample:
        sample["occupancy_mode"] = np.asarray(sample["occupancy_mode"]).astype(str)
    if "shell_thickness" in sample:
        sample["shell_thickness"] = np.asarray(sample["shell_thickness"], dtype=np.float32)
    if "shell_sparsity" in sample:
        sample["shell_sparsity"] = np.asarray(sample["shell_sparsity"], dtype=np.float32)
    if "target_voxels" in sample:
        sample["target_voxels"] = np.asarray(sample["target_voxels"], dtype=np.int32)
    if "mesh_x" in sample:
        sample["mesh_x"] = np.asarray(sample["mesh_x"], dtype=np.float32)
        sample["mesh_y"] = np.asarray(sample["mesh_y"], dtype=np.float32)
        sample["mesh_z"] = np.asarray(sample["mesh_z"], dtype=np.float32)
    if "sampled_brick_mask" in sample:
        sample["sampled_brick_mask"] = np.asarray(sample["sampled_brick_mask"], dtype=bool).reshape(-1)
    if "stage_index" in sample:
        sample["stage_index"] = np.asarray(sample["stage_index"], dtype=np.int64)
    if "stage_label" in sample:
        sample["stage_label"] = np.asarray(sample["stage_label"]).astype(str)
    if "scheduler_step" in sample:
        sample["scheduler_step"] = np.asarray(sample["scheduler_step"], dtype=np.int64)
    if "tau" in sample:
        sample["tau"] = np.asarray(sample["tau"], dtype=np.float32)

    if "original_brick_anchors" in sample or "original_pos" in sample:
        original_brick_anchors = sample.get("original_brick_anchors", sample.get("original_pos"))
        original_brick_types = sample.get("original_brick_types", sample.get("original_types", brick_types))
        original_brick_features = sample.get(
            "original_brick_features",
            sample.get("original_features", brick_features),
        )
        original_brick_dipoles = sample.get("original_brick_dipoles", sample.get("original_dipoles", brick_dipoles))
        original_brick_dipole_directions = sample.get(
            "original_brick_dipole_directions",
            None,
        )
        original_brick_dipole_strengths = sample.get(
            "original_brick_dipole_strengths",
            None,
        )
        original_brick_rotations = sample.get("original_brick_rotations", sample.get("original_rotations", brick_rotations))

        sample["original_brick_anchors"] = np.asarray(original_brick_anchors, dtype=np.float32)
        sample["original_brick_types"] = np.asarray(original_brick_types)
        sample["original_brick_features"] = np.asarray(original_brick_features, dtype=np.float32)
        sample["original_brick_dipoles"] = np.asarray(original_brick_dipoles, dtype=np.float32)
        if original_brick_dipole_directions is None:
            norms = np.linalg.norm(sample["original_brick_dipoles"], axis=-1, keepdims=True)
            sample["original_brick_dipole_directions"] = np.divide(
                sample["original_brick_dipoles"],
                np.maximum(norms, 1e-8),
                out=np.zeros_like(sample["original_brick_dipoles"], dtype=np.float32),
                where=norms > 1e-8,
            )
        else:
            sample["original_brick_dipole_directions"] = np.asarray(original_brick_dipole_directions, dtype=np.float32)
        if original_brick_dipole_strengths is None:
            sample["original_brick_dipole_strengths"] = np.linalg.norm(sample["original_brick_dipoles"], axis=-1, keepdims=True).astype(np.float32)
        else:
            sample["original_brick_dipole_strengths"] = np.asarray(original_brick_dipole_strengths, dtype=np.float32)
        sample["original_brick_rotations"] = np.asarray(original_brick_rotations, dtype=np.float32)

        sample["original_pos"] = sample["original_brick_anchors"]
        sample["original_types"] = sample["original_brick_types"]
        sample["original_features"] = sample["original_brick_features"]
        sample["original_dipoles"] = sample["original_brick_dipoles"]
        sample["original_rotations"] = sample["original_brick_rotations"]

    if "intermediate_states" in sample:
        raw_intermediate_states = sample["intermediate_states"]
        if isinstance(raw_intermediate_states, np.ndarray) and raw_intermediate_states.ndim == 0:
            intermediate_entries = [raw_intermediate_states.item()]
        elif isinstance(raw_intermediate_states, np.ndarray):
            intermediate_entries = list(raw_intermediate_states.tolist())
        else:
            intermediate_entries = list(raw_intermediate_states)
        sample["intermediate_states"] = [_normalize_sample(entry) for entry in intermediate_entries]

    return sample


def _load_object_samples(samples):
    if isinstance(samples, np.ndarray) and samples.ndim == 0:
        entries = [samples.item()]
    else:
        entries = list(samples.tolist())
    return [_normalize_sample(entry) for entry in entries]


def _load_legacy_samples(raw):
    if not {"pos", "types"}.issubset(set(raw.files)):
        raise ValueError(
            f"Unsupported LEGO dataset format in {raw.filename}: keys={raw.files}"
        )
    features_key = "features" if "features" in raw.files else "feats"
    sample = {
        "brick_anchors": raw["pos"],
        "brick_types": raw["types"],
        "brick_dipoles": raw["dipoles"] if "dipoles" in raw.files else np.zeros((len(raw["pos"]), 3), dtype=np.float32),
        "brick_features": raw[features_key] if features_key in raw.files else np.zeros((len(raw["pos"]), 16)),
        "brick_rotations": raw["rotations"] if "rotations" in raw.files else None,
        "source_format": "legacy-flat",
    }
    return [_normalize_sample(sample)]


def load_samples(path):
    """Load the canonical sample list, accepting the previous flat NPZ format."""
    raw = np.load(path, allow_pickle=True)
    if "samples" in raw.files:
        return _load_object_samples(raw["samples"])
    return _load_legacy_samples(raw)


def save_samples(path, samples):
    """Persist samples in the canonical object-array NPZ format."""
    path = Path(path)
    np.savez(
        path,
        samples=np.asarray(list(samples), dtype=object),
        schema_version=np.asarray([2], dtype=np.int32),
        irreps=np.asarray(str(DEFAULT_IRREPS)),
    )


def block_palette():
    return {name: spec["color"] for name, spec in LEGO_LIBRARY.items()}
