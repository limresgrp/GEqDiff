from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.utils.contact_utils import build_brick_geometries, detect_brick_contacts
from geqdiff.utils.dipole_utils import DipoleAssignmentConfig, evaluate_contact_energy


FACE_MATCH_TOLERANCE = 0.35
CONNECTED_FACE_AREA_THRESHOLD = 0.12
TARGET_MATCH_DISTANCE = 0.75
OVERLAP_TOLERANCE_PER_PAIR = 0.01
SEVERE_OVERLAP_THRESHOLD = 0.08
VALIDITY_OVERLAP_WEIGHT = 12.0
VALIDITY_SEVERE_PAIR_WEIGHT = 0.9
VALID_LIKE_EFFECTIVE_OVERLAP = 0.02
EPS = 1e-8


def structure_from_sample(sample: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    return {
        "brick_anchors": np.asarray(sample[f"{prefix}brick_anchors"], dtype=np.float32),
        "brick_rotations": np.asarray(sample[f"{prefix}brick_rotations"], dtype=np.float32),
        "brick_types": np.asarray(sample[f"{prefix}brick_types"]),
        "brick_dipoles": np.asarray(
            sample.get(f"{prefix}brick_dipoles", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ),
    }


def _continuous_face_match(delta: np.ndarray, overlap_volume: float, tolerance: float = FACE_MATCH_TOLERANCE) -> tuple[float, np.ndarray | None]:
    delta = np.asarray(delta, dtype=np.float32)
    axis_overlap = np.maximum(0.0, 1.0 - np.abs(delta))
    best_area = 0.0
    best_direction: np.ndarray | None = None

    for axis in range(3):
        tangential_axes = [idx for idx in range(3) if idx != axis]
        tangential_overlap = float(axis_overlap[tangential_axes[0]] * axis_overlap[tangential_axes[1]])
        if tangential_overlap <= EPS:
            continue
        normal_error = abs(abs(float(delta[axis])) - 1.0)
        normal_proximity = max(0.0, 1.0 - normal_error / float(tolerance))
        if normal_proximity <= EPS:
            continue
        area = tangential_overlap * normal_proximity * float(np.exp(-4.0 * float(overlap_volume)))
        if area <= best_area:
            continue
        direction = np.zeros((3,), dtype=np.float32)
        direction[axis] = 1.0 if float(delta[axis]) >= 0.0 else -1.0
        best_area = float(area)
        best_direction = direction

    return best_area, best_direction


def _connected_components(num_nodes: int, edges: List[Tuple[int, int]]) -> int:
    if num_nodes == 0:
        return 0
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edges:
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))

    visited = np.zeros((num_nodes,), dtype=bool)
    num_components = 0
    for start in range(num_nodes):
        if visited[start]:
            continue
        num_components += 1
        stack = [int(start)]
        visited[start] = True
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                stack.append(int(neighbor))
    return int(num_components)


def _target_fit_metrics(sample: Dict[str, Any], geometries: List[Dict[str, np.ndarray]]) -> Dict[str, float | int]:
    if "target_voxels" not in sample:
        return {}

    target_voxels = np.asarray(sample["target_voxels"], dtype=np.float32).reshape(-1, 3)
    if target_voxels.size == 0:
        return {}

    structure_voxels = np.concatenate(
        [np.asarray(geometry["world_voxels"], dtype=np.float32).reshape(-1, 3) for geometry in geometries],
        axis=0,
    )
    if structure_voxels.size == 0:
        return {}

    deltas = target_voxels[:, None, :] - structure_voxels[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    target_to_structure = distances.min(axis=1)
    structure_to_target = distances.min(axis=0)

    coverage = float((target_to_structure <= TARGET_MATCH_DISTANCE).mean())
    precision = float((structure_to_target <= TARGET_MATCH_DISTANCE).mean())
    if coverage + precision <= EPS:
        f1 = 0.0
    else:
        f1 = float(2.0 * coverage * precision / (coverage + precision))

    return {
        "num_target_voxels": int(target_voxels.shape[0]),
        "target_coverage": coverage,
        "target_precision": precision,
        "target_f1": f1,
        "mean_target_to_structure_distance": float(target_to_structure.mean()),
        "mean_structure_to_target_distance": float(structure_to_target.mean()),
    }


def evaluate_structure_scores(
    sample: Dict[str, Any] | None = None,
    *,
    structure: Dict[str, np.ndarray] | None = None,
    prefix: str = "",
    dipole_config: DipoleAssignmentConfig = DipoleAssignmentConfig(),
) -> Dict[str, Any]:
    if structure is None:
        if sample is None:
            raise ValueError("Either `sample` or `structure` must be provided.")
        structure = structure_from_sample(sample, prefix=prefix)

    geometries = build_brick_geometries(structure)
    num_bricks = int(len(geometries))
    intrinsic_face_count = int(sum(int(np.asarray(geometry["intrinsic_ports"]).shape[0]) for geometry in geometries))
    total_overlap_volume = 0.0
    clashing_brick_pairs = 0
    max_pair_overlap = 0.0
    pair_overlaps: List[float] = []
    matched_face_area = 0.0
    connected_pair_count = 0
    component_edges: List[Tuple[int, int]] = []

    dipoles = np.asarray(structure.get("brick_dipoles", np.zeros((num_bricks, 3), dtype=np.float32)), dtype=np.float32)

    for src_index in range(num_bricks):
        voxels_src = np.asarray(geometries[src_index]["world_voxels"], dtype=np.float32).reshape(-1, 3)
        for dst_index in range(src_index + 1, num_bricks):
            voxels_dst = np.asarray(geometries[dst_index]["world_voxels"], dtype=np.float32).reshape(-1, 3)

            pair_overlap = 0.0
            pair_contact_area = 0.0

            for voxel_src in voxels_src:
                for voxel_dst in voxels_dst:
                    delta = np.asarray(voxel_dst - voxel_src, dtype=np.float32)
                    axis_overlap = np.maximum(0.0, 1.0 - np.abs(delta))
                    overlap_volume = float(axis_overlap[0] * axis_overlap[1] * axis_overlap[2])
                    if overlap_volume > EPS:
                        pair_overlap += overlap_volume

                    contact_area, direction = _continuous_face_match(delta, overlap_volume)
                    if contact_area <= EPS or direction is None:
                        continue

                    pair_contact_area += float(contact_area)

            total_overlap_volume += pair_overlap
            pair_overlaps.append(float(pair_overlap))
            if pair_overlap > EPS:
                clashing_brick_pairs += 1
                max_pair_overlap = max(max_pair_overlap, float(pair_overlap))
            if pair_contact_area > EPS:
                matched_face_area += float(pair_contact_area)
            if pair_contact_area > CONNECTED_FACE_AREA_THRESHOLD:
                connected_pair_count += 1
                component_edges.append((int(src_index), int(dst_index)))

    num_components = _connected_components(num_bricks, component_edges)
    micro_overlapping_pairs = int(sum(1 for overlap in pair_overlaps if overlap > OVERLAP_TOLERANCE_PER_PAIR))
    severe_overlapping_pairs = int(sum(1 for overlap in pair_overlaps if overlap > SEVERE_OVERLAP_THRESHOLD))
    effective_overlap_volume = float(
        sum(max(0.0, float(overlap) - OVERLAP_TOLERANCE_PER_PAIR) for overlap in pair_overlaps)
    )
    matched_face_ratio = float(
        np.clip((2.0 * matched_face_area) / max(float(intrinsic_face_count), 1.0), 0.0, 1.0)
    )

    dipole_eval_error = ""
    try:
        contacts = detect_brick_contacts(geometries)
        energy_eval = evaluate_contact_energy(
            dipoles=dipoles,
            all_face_contact_pairs=np.asarray(contacts["all_face_contact_pairs"], dtype=np.int64),
            all_face_contact_dirs=np.asarray(contacts["all_face_contact_dirs"], dtype=np.float32),
            config=dipole_config,
        )
    except Exception as exc:
        dipole_eval_error = f"{type(exc).__name__}: {exc}"
        energy_eval = {
            "total_energy": 0.0,
            "polar_cost": 0.0,
            "contact_energy": 0.0,
            "num_face_contacts": 0,
            "num_attractive_contacts": 0,
            "num_repulsive_contacts": 0,
            "num_neutral_contacts": 0,
            "mean_energy_per_face": 0.0,
        }

    attractive_contact_count = int(energy_eval["num_attractive_contacts"])
    repulsive_contact_count = int(energy_eval["num_repulsive_contacts"])
    neutral_contact_count = int(energy_eval["num_neutral_contacts"])
    total_contact_count = int(energy_eval["num_face_contacts"])

    validity_score = float(
        100.0
        * np.exp(
            -VALIDITY_OVERLAP_WEIGHT * float(effective_overlap_volume)
            - VALIDITY_SEVERE_PAIR_WEIGHT * float(severe_overlapping_pairs)
        )
        / max(1.0, float(np.sqrt(num_components)))
    )
    fitness_score = float(100.0 * matched_face_ratio)
    dipole_score = float(
        100.0 * (float(attractive_contact_count) + 0.5 * float(neutral_contact_count)) / float(total_contact_count)
    ) if total_contact_count > 0 else 50.0

    metrics: Dict[str, float | int | bool] = {
        "num_bricks": int(num_bricks),
        "intrinsic_face_count": int(intrinsic_face_count),
        "matched_face_area": float(matched_face_area),
        "matched_face_ratio": matched_face_ratio,
        "connected_brick_pairs": int(connected_pair_count),
        "num_components": int(num_components),
        "total_overlap_volume": float(total_overlap_volume),
        "effective_overlap_volume": float(effective_overlap_volume),
        "clashing_brick_pairs": int(clashing_brick_pairs),
        "micro_overlapping_pairs": int(micro_overlapping_pairs),
        "severe_overlapping_pairs": int(severe_overlapping_pairs),
        "max_pair_overlap_volume": float(max_pair_overlap),
        "overlap_tolerance_per_pair": float(OVERLAP_TOLERANCE_PER_PAIR),
        "severe_overlap_threshold": float(SEVERE_OVERLAP_THRESHOLD),
        "attractive_contact_count": int(attractive_contact_count),
        "repulsive_contact_count": int(repulsive_contact_count),
        "neutral_contact_count": int(neutral_contact_count),
        "total_contact_count": int(total_contact_count),
        "dipole_total_energy": float(energy_eval["total_energy"]),
        "dipole_polar_cost": float(energy_eval["polar_cost"]),
        "dipole_contact_energy": float(energy_eval["contact_energy"]),
        "mean_dipole_energy_per_face": float(energy_eval["mean_energy_per_face"]),
        "dipole_eval_error": dipole_eval_error,
        # Backward-compatible aliases used by visualizer/report tables.
        "attractive_contact_area": float(attractive_contact_count),
        "repulsive_contact_area": float(repulsive_contact_count),
        "neutral_contact_area": float(neutral_contact_count),
        "total_contact_area": float(total_contact_count),
        "weighted_dipole_energy": float(energy_eval["total_energy"]),
        "mean_weighted_dipole_energy": float(energy_eval["mean_energy_per_face"]),
        "is_valid_like": bool(
            effective_overlap_volume <= VALID_LIKE_EFFECTIVE_OVERLAP
            and severe_overlapping_pairs == 0
            and num_components == 1
        ),
    }
    if sample is not None:
        metrics.update(_target_fit_metrics(sample, geometries))

    shell_surface_ratio = float(np.clip(1.0 - matched_face_ratio, 0.0, 1.0))
    metrics["shell_surface_ratio"] = shell_surface_ratio

    target_f1 = metrics.get("target_f1", None)
    if isinstance(target_f1, (float, int)) and np.isfinite(float(target_f1)):
        shellness_score = float(100.0 * (0.85 * float(target_f1) + 0.15 * shell_surface_ratio))
    else:
        shellness_score = float(100.0 * shell_surface_ratio)

    return {
        "scores": {
            "validity": float(validity_score),
            "compactness": float(fitness_score),
            "shellness": float(shellness_score),
            "dipoles": float(dipole_score),
            "fitness": float(fitness_score),
        },
        "metrics": metrics,
    }


def _shift_compare(sample: Dict[str, Any]) -> Dict[str, float | int]:
    sampled_anchors = np.asarray(sample["brick_anchors"], dtype=np.float32)
    original_anchors = np.asarray(sample["original_brick_anchors"], dtype=np.float32)
    shifts = np.linalg.norm(sampled_anchors - original_anchors, axis=-1)
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((len(shifts),), dtype=bool)), dtype=bool).reshape(-1)
    fixed_mask = ~mask
    return {
        "num_diffused": int(mask.sum()),
        "diffused_shift_mean": float(shifts[mask].mean()) if mask.any() else 0.0,
        "diffused_shift_max": float(shifts[mask].max()) if mask.any() else 0.0,
        "fixed_shift_mean": float(shifts[fixed_mask].mean()) if fixed_mask.any() else 0.0,
        "fixed_shift_max": float(shifts[fixed_mask].max()) if fixed_mask.any() else 0.0,
    }


def evaluate_sample_scores(sample: Dict[str, Any], dipole_config: DipoleAssignmentConfig = DipoleAssignmentConfig()) -> Dict[str, Any]:
    sampled = evaluate_structure_scores(sample, prefix="", dipole_config=dipole_config)
    payload: Dict[str, Any] = {"sampled": sampled}

    if "original_brick_anchors" not in sample:
        return payload

    original = evaluate_structure_scores(sample, prefix="original_", dipole_config=dipole_config)
    payload["original"] = original
    payload["compare"] = {
        "score_delta": {
            "validity": float(sampled["scores"]["validity"] - original["scores"]["validity"]),
            "compactness": float(sampled["scores"]["compactness"] - original["scores"]["compactness"]),
            "shellness": float(sampled["scores"]["shellness"] - original["scores"]["shellness"]),
            "dipoles": float(sampled["scores"]["dipoles"] - original["scores"]["dipoles"]),
        },
        "metric_delta": {
            "overlap_volume": float(sampled["metrics"]["total_overlap_volume"] - original["metrics"]["total_overlap_volume"]),
            "matched_face_ratio": float(sampled["metrics"]["matched_face_ratio"] - original["metrics"]["matched_face_ratio"]),
            "shell_surface_ratio": float(sampled["metrics"]["shell_surface_ratio"] - original["metrics"]["shell_surface_ratio"]),
            "weighted_dipole_energy": float(
                sampled["metrics"]["weighted_dipole_energy"] - original["metrics"]["weighted_dipole_energy"]
            ),
        },
        "anchor_shift": _shift_compare(sample),
    }
    return payload
