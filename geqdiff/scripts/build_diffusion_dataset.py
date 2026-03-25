from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.utils.contact_utils import build_brick_geometries, detect_brick_contacts
from geqdiff.utils.dipole_utils import (
    assign_discrete_dipoles,
    dipole_strengths,
    normalize_dipole_directions,
    split_shape_irreps,
)
from geqdiff.utils.feature_utils import (
    apply_scalar_normalization,
    build_type_vocab,
    encode_type_names,
    fit_scalar_normalization,
    irreps_string,
)
from lego.utils import default_dataset_path, load_samples


LIGAND_UNKNOWN_TYPE_NAME = "ligand_unknown"
DIRECTION_VALID_EPS = 1e-6


def _adjacency_from_contact_pairs(num_nodes: int, contact_pairs: np.ndarray) -> List[List[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in np.asarray(contact_pairs, dtype=np.int64):
        adjacency[int(src)].append(int(dst))
        adjacency[int(dst)].append(int(src))
    for neighbors in adjacency:
        neighbors.sort()
    return adjacency


def _sample_ligand_size(
    num_nodes: int,
    strategy: str,
    component_id: np.ndarray,
    rng: np.random.Generator,
    ligand_size: int | None,
    ligand_size_min: int,
    ligand_size_max: int | None,
    ligand_fraction: float | None,
) -> int:
    if num_nodes < 2:
        raise ValueError("Each split requires at least two nodes so ligand and pocket are both non-empty.")

    max_allowed = num_nodes - 1
    if strategy == "connected":
        component_sizes = Counter(np.asarray(component_id, dtype=np.int32).tolist())
        max_allowed = min(max_allowed, max(component_sizes.values()))

    if ligand_size is not None:
        return int(np.clip(ligand_size, 1, max_allowed))

    if ligand_fraction is not None:
        target = int(round(float(ligand_fraction) * float(num_nodes)))
        return int(np.clip(target, 1, max_allowed))

    upper = max_allowed if ligand_size_max is None else min(max_allowed, int(ligand_size_max))
    lower = min(max(1, int(ligand_size_min)), upper)
    if lower == upper:
        return lower
    return int(rng.integers(lower, upper + 1))


def _connected_ligand_mask(
    num_nodes: int,
    adjacency: Sequence[Sequence[int]],
    component_id: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    component_members = defaultdict(list)
    for node, comp in enumerate(np.asarray(component_id, dtype=np.int32)):
        component_members[int(comp)].append(int(node))
    candidate_components = [nodes for nodes in component_members.values() if len(nodes) >= target_size]
    if len(candidate_components) == 0:
        raise ValueError(
            f"Cannot sample a connected ligand of size {target_size}; "
            f"largest component has size {max(len(nodes) for nodes in component_members.values())}."
        )

    component_nodes = candidate_components[int(rng.integers(len(candidate_components)))]
    seed = int(component_nodes[int(rng.integers(len(component_nodes)))])
    ligand_nodes = {seed}

    while len(ligand_nodes) < target_size:
        frontier = sorted(
            {
                neighbor
                for node in ligand_nodes
                for neighbor in adjacency[node]
                if neighbor not in ligand_nodes
            }
        )
        if len(frontier) == 0:
            raise ValueError(
                f"Failed to grow a connected ligand to size {target_size}; "
                f"stalled at size {len(ligand_nodes)}."
            )
        ligand_nodes.add(int(frontier[int(rng.integers(len(frontier)))]))

    mask = np.zeros((num_nodes,), dtype=bool)
    mask[np.asarray(sorted(ligand_nodes), dtype=np.int64)] = True
    return mask


def _radius_ligand_mask(pos: np.ndarray, target_size: int, rng: np.random.Generator) -> np.ndarray:
    num_nodes = int(pos.shape[0])
    center = int(rng.integers(num_nodes))
    distances = np.linalg.norm(pos - pos[center], axis=1)
    order = np.lexsort((np.arange(num_nodes), distances))
    chosen = order[:target_size]
    mask = np.zeros((num_nodes,), dtype=bool)
    mask[chosen] = True
    return mask


def _ligand_is_connected(mask: np.ndarray, adjacency: Sequence[Sequence[int]]) -> bool:
    ligand_nodes = np.flatnonzero(mask)
    if ligand_nodes.size <= 1:
        return True
    allowed = set(int(node) for node in ligand_nodes.tolist())
    start = int(ligand_nodes[0])
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in allowed or neighbor in visited:
                continue
            visited.add(int(neighbor))
            stack.append(int(neighbor))
    return len(visited) == len(allowed)


def _extract_or_assign_dipoles(
    sample: Dict,
    rotations: np.ndarray,
    contact_pairs: np.ndarray,
    contact_face_dirs: np.ndarray,
    all_face_contact_pairs: np.ndarray,
    all_face_contact_dirs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if "brick_dipoles" in sample:
        dipoles = np.asarray(sample["brick_dipoles"], dtype=np.float32)
    elif "dipoles" in sample:
        dipoles = np.asarray(sample["dipoles"], dtype=np.float32)
    else:
        dipoles = assign_discrete_dipoles(
            rotations=rotations,
            contact_pairs=contact_pairs,
            contact_face_dirs=contact_face_dirs,
            all_face_contact_pairs=all_face_contact_pairs,
            all_face_contact_dirs=all_face_contact_dirs,
            rng=rng,
        )
    if dipoles.shape != (rotations.shape[0], 3):
        raise ValueError(f"Expected dipoles with shape {(rotations.shape[0], 3)}, got {dipoles.shape}.")
    return dipoles.astype(np.float32)


def _directional_target_metadata(
    shape_scalar_features: np.ndarray,
    dipole_strength: np.ndarray,
    eps: float = DIRECTION_VALID_EPS,
) -> Dict[str, np.ndarray]:
    shape_scalar_features = np.asarray(shape_scalar_features, dtype=np.float32)
    dipole_strength = np.asarray(dipole_strength, dtype=np.float32)

    l1_weight = np.maximum(shape_scalar_features[:, 1], 0.0).astype(np.float32)
    l2_weight = np.maximum(shape_scalar_features[:, 2], 0.0).astype(np.float32)
    l3_weight = np.maximum(shape_scalar_features[:, 3], 0.0).astype(np.float32)
    dipole_weight = np.maximum(dipole_strength.reshape(-1), 0.0).astype(np.float32)

    return {
        "shape_l1_weight": l1_weight,
        "shape_l2_weight": l2_weight,
        "shape_l3_weight": l3_weight,
        "dipole_direction_weight": dipole_weight,
        "shape_l1_valid": (l1_weight > float(eps)),
        "shape_l2_valid": (l2_weight > float(eps)),
        "shape_l3_valid": (l3_weight > float(eps)),
        "dipole_direction_valid": (dipole_weight > float(eps)),
    }


def _validate_example(example: Dict, adjacency: Sequence[Sequence[int]], split_strategy: str) -> None:
    ligand_mask = np.asarray(example["ligand_mask"], dtype=bool)
    pocket_mask = np.asarray(example["pocket_mask"], dtype=bool)
    shape_scalar_features = np.asarray(example["shape_scalar_features"], dtype=np.float32)
    shape_equiv_features = np.asarray(example["shape_equiv_features"], dtype=np.float32)
    dipole_strength = np.asarray(example["dipole_strength"], dtype=np.float32)
    dipole_direction = np.asarray(example["dipole_direction"], dtype=np.float32)
    shape_l1_valid = np.asarray(example["shape_l1_valid"], dtype=bool)
    shape_l2_valid = np.asarray(example["shape_l2_valid"], dtype=bool)
    shape_l3_valid = np.asarray(example["shape_l3_valid"], dtype=bool)
    dipole_direction_valid = np.asarray(example["dipole_direction_valid"], dtype=bool)
    shape_l1_weight = np.asarray(example["shape_l1_weight"], dtype=np.float32)
    shape_l2_weight = np.asarray(example["shape_l2_weight"], dtype=np.float32)
    shape_l3_weight = np.asarray(example["shape_l3_weight"], dtype=np.float32)
    dipole_direction_weight = np.asarray(example["dipole_direction_weight"], dtype=np.float32)
    edge_index = np.asarray(example["edge_index"], dtype=np.int64)
    num_nodes = int(example["num_nodes"])

    assert ligand_mask.sum() > 0
    assert pocket_mask.sum() > 0
    assert not np.any(ligand_mask & pocket_mask)
    assert shape_scalar_features.shape == (num_nodes, 4)
    assert shape_equiv_features.shape == (num_nodes, 15)
    assert dipole_strength.shape == (num_nodes, 1)
    assert dipole_direction.shape == (num_nodes, 3)
    assert shape_l1_valid.shape == (num_nodes,)
    assert shape_l2_valid.shape == (num_nodes,)
    assert shape_l3_valid.shape == (num_nodes,)
    assert dipole_direction_valid.shape == (num_nodes,)
    assert shape_l1_weight.shape == (num_nodes,)
    assert shape_l2_weight.shape == (num_nodes,)
    assert shape_l3_weight.shape == (num_nodes,)
    assert dipole_direction_weight.shape == (num_nodes,)
    assert np.all(dipole_strength >= -1e-6)
    assert np.all(shape_l1_weight >= -1e-6)
    assert np.all(shape_l2_weight >= -1e-6)
    assert np.all(shape_l3_weight >= -1e-6)
    assert np.all(dipole_direction_weight >= -1e-6)
    if split_strategy == "connected":
        assert _ligand_is_connected(ligand_mask, adjacency)

    if edge_index.size > 0:
        assert not np.any(edge_index[0] == edge_index[1])
        directed = {tuple(edge.tolist()) for edge in edge_index.T}
        for src, dst in directed:
            assert (dst, src) in directed


def _build_frame_record(sample: Dict, source_frame_id: int, type_vocab: Sequence[str], rng: np.random.Generator) -> Dict:
    geometries = build_brick_geometries(sample)
    contact_data = detect_brick_contacts(geometries)

    pos = np.asarray(sample["brick_anchors"], dtype=np.float32)
    rotations = np.asarray(sample["brick_rotations"], dtype=np.float32)
    types = np.asarray(sample["brick_types"])
    node_types_true = encode_type_names(types, type_vocab)
    shape_features = np.asarray(sample["brick_features"], dtype=np.float32)
    shape_scalar_features, shape_equiv_features = split_shape_irreps(shape_features)

    edge_index = np.asarray(contact_data["edge_index"], dtype=np.int64)
    edge_types = np.asarray(contact_data["edge_types"], dtype=np.int32)
    contact_pairs = np.asarray(contact_data["contact_pairs"], dtype=np.int64)
    contact_face_dirs = np.asarray(contact_data["contact_face_dirs"], dtype=np.int32)
    all_face_contact_pairs = np.asarray(contact_data["all_face_contact_pairs"], dtype=np.int64)
    all_face_contact_dirs = np.asarray(contact_data["all_face_contact_dirs"], dtype=np.int32)
    component_id = np.asarray(contact_data["component_id"], dtype=np.int32)

    dipoles = _extract_or_assign_dipoles(
        sample=sample,
        rotations=rotations,
        contact_pairs=contact_pairs,
        contact_face_dirs=contact_face_dirs.astype(np.float32),
        all_face_contact_pairs=all_face_contact_pairs,
        all_face_contact_dirs=all_face_contact_dirs.astype(np.float32),
        rng=rng,
    )
    dipole_direction = normalize_dipole_directions(dipoles)
    dipole_strength = dipole_strengths(dipoles)
    directional_metadata = _directional_target_metadata(
        shape_scalar_features=shape_scalar_features,
        dipole_strength=dipole_strength,
    )

    return {
        "source_frame_id": int(source_frame_id),
        "num_nodes": int(pos.shape[0]),
        "pos": pos,
        "rotations": rotations,
        "types": types,
        "node_types_true": node_types_true,
        "type_vocab": np.asarray(type_vocab),
        "shape_features_raw": shape_features,
        "shape_scalar_features": shape_scalar_features,
        "shape_equiv_features": shape_equiv_features,
        "brick_dipoles_raw": dipoles,
        "dipole_direction": dipole_direction,
        "dipole_strength": dipole_strength,
        **directional_metadata,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "contact_pairs": contact_pairs,
        "contact_face_dirs": contact_face_dirs,
        "component_id": component_id,
        "frame_centroid": pos.mean(axis=0).astype(np.float32),
    }


def _build_examples_for_frame(
    frame: Dict,
    split_strategy: str,
    splits_per_frame: int,
    rng: np.random.Generator,
    ligand_size: int | None,
    ligand_size_min: int,
    ligand_size_max: int | None,
    ligand_fraction: float | None,
) -> List[Dict]:
    pos = np.asarray(frame["pos"], dtype=np.float32)
    num_nodes = int(frame["num_nodes"])
    adjacency = _adjacency_from_contact_pairs(num_nodes, frame["contact_pairs"])
    type_vocab = np.asarray(frame["type_vocab"]).astype(str).tolist()
    unknown_type_index = type_vocab.index(LIGAND_UNKNOWN_TYPE_NAME)

    examples: List[Dict] = []
    for split_id in range(splits_per_frame):
        target_size = _sample_ligand_size(
            num_nodes=num_nodes,
            strategy=split_strategy,
            component_id=frame["component_id"],
            rng=rng,
            ligand_size=ligand_size,
            ligand_size_min=ligand_size_min,
            ligand_size_max=ligand_size_max,
            ligand_fraction=ligand_fraction,
        )

        if split_strategy == "connected":
            ligand_mask = _connected_ligand_mask(
                num_nodes=num_nodes,
                adjacency=adjacency,
                component_id=frame["component_id"],
                target_size=target_size,
                rng=rng,
            )
        elif split_strategy == "radius":
            ligand_mask = _radius_ligand_mask(pos=pos, target_size=target_size, rng=rng)
        else:
            raise ValueError(f"Unsupported split strategy '{split_strategy}'.")

        pocket_mask = ~ligand_mask
        ligand_centroid = pos[ligand_mask].mean(axis=0).astype(np.float32)
        input_node_types = frame["node_types_true"].astype(np.int64).copy()
        input_node_types[ligand_mask] = int(unknown_type_index)

        example = {
            "source_frame_id": np.int64(frame["source_frame_id"]),
            "split_id": np.int64(split_id),
            "num_nodes": np.int64(num_nodes),
            "ligand_size": np.int64(ligand_mask.sum()),
            "num_edges": np.int64(frame["edge_index"].shape[1]),
            "pos": frame["pos"].astype(np.float32),
            "rotations": frame["rotations"].astype(np.float32),
            "types": np.asarray(frame["types"]),
            "node_types": input_node_types.astype(np.int64),
            "node_types_true": frame["node_types_true"].astype(np.int64),
            "shape_features_raw": frame["shape_features_raw"].astype(np.float32),
            "shape_scalar_features": frame["shape_scalar_features"].astype(np.float32),
            "shape_equiv_features": frame["shape_equiv_features"].astype(np.float32),
            "brick_dipoles_raw": frame["brick_dipoles_raw"].astype(np.float32),
            "dipole_direction": frame["dipole_direction"].astype(np.float32),
            "dipole_strength": frame["dipole_strength"].astype(np.float32),
            "shape_l1_valid": frame["shape_l1_valid"].astype(bool),
            "shape_l2_valid": frame["shape_l2_valid"].astype(bool),
            "shape_l3_valid": frame["shape_l3_valid"].astype(bool),
            "dipole_direction_valid": frame["dipole_direction_valid"].astype(bool),
            "shape_l1_weight": frame["shape_l1_weight"].astype(np.float32),
            "shape_l2_weight": frame["shape_l2_weight"].astype(np.float32),
            "shape_l3_weight": frame["shape_l3_weight"].astype(np.float32),
            "dipole_direction_weight": frame["dipole_direction_weight"].astype(np.float32),
            "ligand_mask": ligand_mask.astype(bool),
            "pocket_mask": pocket_mask.astype(bool),
            "edge_index": frame["edge_index"].astype(np.int64),
            "edge_types": frame["edge_types"].astype(np.int32),
            "contact_pairs": frame["contact_pairs"].astype(np.int64),
            "contact_face_dirs": frame["contact_face_dirs"].astype(np.int32),
            "component_id": frame["component_id"].astype(np.int32),
            "frame_centroid": frame["frame_centroid"].astype(np.float32),
            "ligand_centroid": ligand_centroid,
        }
        _validate_example(example, adjacency=adjacency, split_strategy=split_strategy)
        examples.append(example)

    return examples


def _normalize_scalar_fields(
    examples: Sequence[Dict],
    normalize_scalars: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    metadata: Dict[str, Dict[str, np.ndarray]] = {}

    for field in ("shape_scalar_features", "dipole_strength"):
        raw_field = f"{field}_raw"
        for example in examples:
            example[raw_field] = np.asarray(example[field], dtype=np.float32).copy()

    for field in ("shape_equiv_features", "dipole_direction"):
        raw_field = f"{field}_raw"
        for example in examples:
            example[raw_field] = np.asarray(example[field], dtype=np.float32).copy()

    if not normalize_scalars:
        shape_stats = {"means": np.zeros((4,), dtype=np.float32), "stds": np.ones((4,), dtype=np.float32)}
        for example in examples:
            example["shape_scalar_norm_means"] = np.asarray(shape_stats["means"], dtype=np.float32)
            example["shape_scalar_norm_stds"] = np.asarray(shape_stats["stds"], dtype=np.float32)
        return metadata

    for field in ("shape_scalar_features", "dipole_strength"):
        stacked = np.concatenate(
            [np.asarray(example[field], dtype=np.float32) for example in examples],
            axis=0,
        )
        stats = fit_scalar_normalization(stacked)
        metadata[field] = stats
        for example in examples:
            example[field] = apply_scalar_normalization(np.asarray(example[field], dtype=np.float32), stats).astype(np.float32)

    shape_stats = metadata.get("shape_scalar_features", {"means": np.zeros((4,), dtype=np.float32), "stds": np.ones((4,), dtype=np.float32)})
    for example in examples:
        example["shape_scalar_norm_means"] = np.asarray(shape_stats["means"], dtype=np.float32)
        example["shape_scalar_norm_stds"] = np.asarray(shape_stats["stds"], dtype=np.float32)

    return metadata


def _pad_node_field(payload: Dict[str, np.ndarray], field: str, examples: Sequence[Dict], max_nodes: int, dtype, tail_shape=()):
    shape = (len(examples), max_nodes) + tuple(tail_shape)
    fill_value = "" if np.dtype(dtype).kind in {"U", "S"} else 0
    payload[field] = np.full(shape, fill_value=fill_value, dtype=dtype)
    payload[f"{field}__mask__"] = np.ones((len(examples), max_nodes), dtype=bool)

    for example_index, example in enumerate(examples):
        values = np.asarray(example[field], dtype=dtype)
        count = int(example["num_nodes"])
        payload[field][example_index, :count] = values
        payload[f"{field}__mask__"][example_index, :count] = False


def _pack_examples(
    examples: Sequence[Dict],
    type_vocab: Sequence[str],
    output: Path,
    split_strategy: str,
    scalar_normalization: Dict[str, Dict[str, np.ndarray]],
) -> None:
    if len(examples) == 0:
        raise ValueError("No examples were generated.")

    max_nodes = max(int(example["num_nodes"]) for example in examples)
    max_edges = max(int(example["edge_index"].shape[1]) for example in examples)
    max_contact_pairs = max(int(example["contact_pairs"].shape[0]) for example in examples)

    max_type_len = max(max(len(str(name)) for name in type_vocab), 1)
    payload: Dict[str, np.ndarray] = {}

    _pad_node_field(payload, "pos", examples, max_nodes, np.float32, tail_shape=(3,))
    _pad_node_field(payload, "rotations", examples, max_nodes, np.float32, tail_shape=(3, 3))
    _pad_node_field(payload, "types", examples, max_nodes, f"<U{max_type_len}")
    _pad_node_field(payload, "node_types", examples, max_nodes, np.int64)
    _pad_node_field(payload, "node_types_true", examples, max_nodes, np.int64)
    _pad_node_field(payload, "shape_features_raw", examples, max_nodes, np.float32, tail_shape=(16,))
    _pad_node_field(payload, "shape_scalar_features", examples, max_nodes, np.float32, tail_shape=(4,))
    _pad_node_field(payload, "shape_scalar_features_raw", examples, max_nodes, np.float32, tail_shape=(4,))
    _pad_node_field(payload, "shape_equiv_features", examples, max_nodes, np.float32, tail_shape=(15,))
    _pad_node_field(payload, "shape_equiv_features_raw", examples, max_nodes, np.float32, tail_shape=(15,))
    _pad_node_field(payload, "brick_dipoles_raw", examples, max_nodes, np.float32, tail_shape=(3,))
    _pad_node_field(payload, "dipole_strength", examples, max_nodes, np.float32, tail_shape=(1,))
    _pad_node_field(payload, "dipole_strength_raw", examples, max_nodes, np.float32, tail_shape=(1,))
    _pad_node_field(payload, "dipole_direction", examples, max_nodes, np.float32, tail_shape=(3,))
    _pad_node_field(payload, "dipole_direction_raw", examples, max_nodes, np.float32, tail_shape=(3,))
    _pad_node_field(payload, "shape_l1_valid", examples, max_nodes, bool)
    _pad_node_field(payload, "shape_l2_valid", examples, max_nodes, bool)
    _pad_node_field(payload, "shape_l3_valid", examples, max_nodes, bool)
    _pad_node_field(payload, "dipole_direction_valid", examples, max_nodes, bool)
    _pad_node_field(payload, "shape_l1_weight", examples, max_nodes, np.float32)
    _pad_node_field(payload, "shape_l2_weight", examples, max_nodes, np.float32)
    _pad_node_field(payload, "shape_l3_weight", examples, max_nodes, np.float32)
    _pad_node_field(payload, "dipole_direction_weight", examples, max_nodes, np.float32)
    _pad_node_field(payload, "ligand_mask", examples, max_nodes, bool)
    _pad_node_field(payload, "pocket_mask", examples, max_nodes, bool)
    _pad_node_field(payload, "component_id", examples, max_nodes, np.int32)

    payload["node_mask"] = np.zeros((len(examples), max_nodes), dtype=bool)
    for example_index, example in enumerate(examples):
        count = int(example["num_nodes"])
        payload["node_mask"][example_index, :count] = True

    payload["edge_index"] = np.full((len(examples), 2, max_edges), fill_value=-1, dtype=np.int64)
    payload["edge_types"] = np.full((len(examples), max_edges), fill_value=-1, dtype=np.int32)
    payload["contact_pairs"] = np.full((len(examples), max_contact_pairs, 2), fill_value=-1, dtype=np.int64)
    payload["contact_face_dirs"] = np.zeros((len(examples), max_contact_pairs, 3), dtype=np.int32)
    for example_index, example in enumerate(examples):
        num_edges = int(example["edge_index"].shape[1])
        if num_edges > 0:
            payload["edge_index"][example_index, :, :num_edges] = example["edge_index"]
            payload["edge_types"][example_index, :num_edges] = example["edge_types"]
        num_pairs = int(example["contact_pairs"].shape[0])
        if num_pairs > 0:
            payload["contact_pairs"][example_index, :num_pairs] = example["contact_pairs"]
            payload["contact_face_dirs"][example_index, :num_pairs] = example["contact_face_dirs"]

    for field in ["source_frame_id", "split_id", "num_nodes", "ligand_size", "num_edges"]:
        payload[field] = np.asarray([example[field] for example in examples], dtype=np.int64)
    for field in ["frame_centroid", "ligand_centroid"]:
        payload[field] = np.asarray([example[field] for example in examples], dtype=np.float32)
    for field in ["shape_scalar_norm_means", "shape_scalar_norm_stds"]:
        payload[field] = np.asarray([example[field] for example in examples], dtype=np.float32)

    payload["type_vocab"] = np.asarray(type_vocab, dtype=f"<U{max_type_len}")
    payload["schema_version"] = np.asarray([4], dtype=np.int32)
    payload["irreps"] = np.asarray(str(irreps_string()))
    payload["split_strategy"] = np.asarray(str(split_strategy))
    payload["scalar_normalization_enabled"] = np.asarray([1 if len(scalar_normalization) > 0 else 0], dtype=np.int32)
    for field, prefix in (("shape_scalar_features", "shape_scalar"), ("dipole_strength", "dipole_strength")):
        stats = scalar_normalization.get(field)
        feature_dim = 4 if field == "shape_scalar_features" else 1
        if stats is None:
            payload[f"{prefix}_means"] = np.zeros((feature_dim,), dtype=np.float32)
            payload[f"{prefix}_stds"] = np.ones((feature_dim,), dtype=np.float32)
        else:
            payload[f"{prefix}_means"] = np.asarray(stats["means"], dtype=np.float32)
            payload[f"{prefix}_stds"] = np.asarray(stats["stds"], dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def _print_stats(examples: Sequence[Dict]) -> None:
    num_examples = len(examples)
    avg_nodes = float(np.mean([example["num_nodes"] for example in examples]))
    avg_ligand = float(np.mean([example["ligand_size"] for example in examples]))
    avg_contacts = float(
        np.mean(
            [
                0.0 if int(example["num_nodes"]) == 0 else int(example["num_edges"]) / float(example["num_nodes"])
                for example in examples
            ]
        )
    )
    polar_hist = Counter(
        int(np.squeeze(np.asarray(example["dipole_strength_raw"], dtype=np.float32)[node_index]) > 1e-6)
        for example in examples
        for node_index in range(int(example["num_nodes"]))
    )
    valid_stats = {
        "dipole_direction_valid": float(
            np.mean(
                [
                    np.asarray(example["dipole_direction_valid"], dtype=np.float32)[: int(example["num_nodes"])].mean()
                    for example in examples
                ]
            )
        ),
        "shape_l1_valid": float(
            np.mean(
                [
                    np.asarray(example["shape_l1_valid"], dtype=np.float32)[: int(example["num_nodes"])].mean()
                    for example in examples
                ]
            )
        ),
        "shape_l2_valid": float(
            np.mean(
                [
                    np.asarray(example["shape_l2_valid"], dtype=np.float32)[: int(example["num_nodes"])].mean()
                    for example in examples
                ]
            )
        ),
        "shape_l3_valid": float(
            np.mean(
                [
                    np.asarray(example["shape_l3_valid"], dtype=np.float32)[: int(example["num_nodes"])].mean()
                    for example in examples
                ]
            )
        ),
    }

    print("--- LEGO Diffusion Dataset ---")
    print(f"Examples: {num_examples}")
    print(f"Average nodes/example: {avg_nodes:.2f}")
    print(f"Average ligand size: {avg_ligand:.2f}")
    print(f"Average directed contacts/node: {avg_contacts:.2f}")
    print("Dipole state histogram:")
    print(f"  neutral: {polar_hist.get(0, 0)}")
    print(f"  polar: {polar_hist.get(1, 0)}")
    print("Directional validity fractions:")
    print(f"  dipole: {valid_stats['dipole_direction_valid']:.3f}")
    print(f"  shape l1: {valid_stats['shape_l1_valid']:.3f}")
    print(f"  shape l2: {valid_stats['shape_l2_valid']:.3f}")
    print(f"  shape l3: {valid_stats['shape_l3_valid']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an offline LEGO flow-matching dataset with dipoles and fixed ligand/pocket splits.")
    parser.add_argument("--input", type=Path, default=default_dataset_path(), help="Input canonical LEGO dataset.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path for the split dataset.")
    parser.add_argument("--splits-per-frame", type=int, default=4, help="Number of deterministic split variants to generate per source frame.")
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="connected",
        choices=["connected", "radius"],
        help="How to choose ligand nodes inside each source frame.",
    )
    parser.add_argument("--ligand-size", type=int, default=None, help="Exact ligand size. Overrides min/max and fraction if provided.")
    parser.add_argument("--ligand-size-min", type=int, default=2, help="Minimum ligand size when sampling ranges.")
    parser.add_argument("--ligand-size-max", type=int, default=8, help="Maximum ligand size when sampling ranges.")
    parser.add_argument("--ligand-fraction", type=float, default=None, help="Optional ligand fraction of total nodes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic split generation.")
    parser.add_argument(
        "--normalize-scalars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply optional normalization to scalar shape magnitudes and dipole strengths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    samples = load_samples(args.input)
    type_vocab = build_type_vocab(samples)
    if LIGAND_UNKNOWN_TYPE_NAME not in type_vocab:
        type_vocab = list(type_vocab) + [LIGAND_UNKNOWN_TYPE_NAME]

    frames = [
        _build_frame_record(
            sample=sample,
            source_frame_id=frame_id,
            type_vocab=type_vocab,
            rng=rng,
        )
        for frame_id, sample in enumerate(samples)
    ]

    examples: List[Dict] = []
    for frame in frames:
        examples.extend(
            _build_examples_for_frame(
                frame=frame,
                split_strategy=args.split_strategy,
                splits_per_frame=args.splits_per_frame,
                rng=rng,
                ligand_size=args.ligand_size,
                ligand_size_min=args.ligand_size_min,
                ligand_size_max=args.ligand_size_max,
                ligand_fraction=args.ligand_fraction,
            )
        )

    scalar_normalization = _normalize_scalar_fields(examples=examples, normalize_scalars=args.normalize_scalars)
    _print_stats(examples)
    print(f"Scalar normalization: {'enabled' if scalar_normalization else 'disabled'}")

    _pack_examples(
        examples=examples,
        type_vocab=type_vocab,
        output=args.output,
        split_strategy=args.split_strategy,
        scalar_normalization=scalar_normalization,
    )
    print(f"Saved dataset to: {args.output}")


if __name__ == "__main__":
    main()
