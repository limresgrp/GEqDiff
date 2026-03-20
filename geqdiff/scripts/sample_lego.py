from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
GEQTRAIN_ROOT = REPO_ROOT / "deps" / "GEqTrain"
if GEQTRAIN_ROOT.exists() and str(GEQTRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(GEQTRAIN_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.data import AtomicDataDict
from geqdiff.utils import FlowMatchingSampler, FlowMatchingScheduler
from geqdiff.utils.diffusion import center_pos, compute_reference_mean
from geqdiff.utils.dipole_utils import (
    combine_shape_irreps,
    normalize_dipole_directions,
    normalize_rows,
)
from geqdiff.utils.feature_utils import invert_scalar_normalization
from geqtrain.train.components.checkpointing import CheckpointHandler
from lego.utils import load_samples, save_samples


STATE_FIELD_SPECS = (
    ("pos", "velocity"),
    ("shape_scalar_features", "shape_scalar_velocity"),
    ("shape_equiv_features", "shape_equiv_velocity"),
    ("dipole_strength", "dipole_strength_velocity"),
    ("dipole_direction", "dipole_direction_velocity"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample masked LEGO ligand bricks from a trained flow-matching checkpoint and "
            "save the results in the canonical LEGO dataset format."
        )
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to a trained GEqTrain checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Input LEGO diffusion NPZ used for conditioning.")
    parser.add_argument("--output", type=Path, required=True, help="Output canonical LEGO NPZ.")
    parser.add_argument(
        "--source-canonical",
        type=Path,
        default=None,
        help="Optional canonical LEGO dataset used to copy smooth target meshes and voxelizations by source_frame_id.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of conditioning examples to subsample from the input diffusion NPZ.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit example indices. If omitted, indices are drawn randomly without replacement.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for subsampling and initialization.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device used for inference.")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of reverse integration steps. Lower values are faster but coarser.",
    )
    return parser.parse_args()


def infer_model_tmax(config: Dict) -> int | None:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict):
        tmax = stack[0].get("Tmax")
        if tmax is not None:
            return int(tmax)
    return None


def validate_flow_checkpoint_config(config: Dict) -> None:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return
    flow_cfg = stack[0]
    if flow_cfg.get("_target_") != "geqdiff.nn.ForwardFlowMatchingModule":
        raise ValueError("Checkpoint is not a flow-matching model.")
    if flow_cfg.get("flow_target_parameterization") != "scheduler_velocity":
        raise ValueError("Checkpoint uses an unsupported flow target parameterization.")
    if flow_cfg.get("flow_time_parameterization") != "tau":
        raise ValueError("Checkpoint uses an unsupported flow time parameterization.")


def infer_position_centering(config: Dict) -> bool:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return True
    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    for spec in corrupt_fields:
        if str(spec.get("field", "")) == AtomicDataDict.POSITIONS_KEY:
            return bool(spec.get("center", True))
    return True


def infer_position_center_mask_field(config: Dict) -> str:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return ""
    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    for spec in corrupt_fields:
        if str(spec.get("field", "")) == AtomicDataDict.POSITIONS_KEY:
            value = spec.get("center_mask_field", "")
            if value in (None, "", "all", "__all__"):
                return ""
            return str(value)
    return ""


def infer_position_noise_centering(config: Dict) -> bool:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return True
    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    for spec in corrupt_fields:
        if str(spec.get("field", "")) == AtomicDataDict.POSITIONS_KEY:
            return bool(spec.get("center_noise", spec.get("center", True)))
    return True


def infer_position_noise_center_mask_field(config: Dict) -> str:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return ""
    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    for spec in corrupt_fields:
        if str(spec.get("field", "")) != AtomicDataDict.POSITIONS_KEY:
            continue
        value = spec.get("noise_center_mask_field", "__auto__")
        if value == "__auto__":
            mask_field = spec.get("mask_field", "")
            if mask_field not in (None, ""):
                return str(mask_field)
            center_value = spec.get("center_mask_field", "")
            if center_value in (None, "", "all", "__all__"):
                return ""
            return str(center_value)
        if value in (None, "", "all", "__all__"):
            return ""
        return str(value)
    return ""


def infer_project_normalized_states(config: Dict) -> bool:
    return bool(config.get("project_normalized_states", True))


def infer_corrupt_field_map(config: Dict) -> Dict[str, str]:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return {field: out_field for field, out_field in STATE_FIELD_SPECS}

    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    if not isinstance(corrupt_fields, list) or len(corrupt_fields) == 0:
        return {field: out_field for field, out_field in STATE_FIELD_SPECS}

    mapping: Dict[str, str] = {}
    for spec in corrupt_fields:
        field_name = str(spec.get("field", ""))
        if field_name == "":
            continue
        default_out = "velocity" if field_name == AtomicDataDict.POSITIONS_KEY else f"{field_name}_velocity"
        mapping[field_name] = str(spec.get("out_field", default_out))
    return mapping


def build_edge_index(positions: torch.Tensor, cutoff: float) -> torch.Tensor:
    distances = torch.cdist(positions, positions)
    mask = (distances <= cutoff) & (distances > 0)
    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=positions.device)
    return edge_index


def choose_example_indices(args: argparse.Namespace, num_examples: int) -> List[int]:
    if args.indices is not None and len(args.indices) > 0:
        indices = [int(index) for index in args.indices]
        for index in indices:
            if index < 0 or index >= num_examples:
                raise IndexError(f"Requested index {index} but the input dataset has {num_examples} examples.")
        return indices

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1.")
    if args.num_samples > num_examples:
        raise ValueError(
            f"Requested {args.num_samples} samples but the input dataset only has {num_examples} examples."
        )
    rng = np.random.default_rng(args.seed)
    return rng.choice(num_examples, size=args.num_samples, replace=False).tolist()


def _slice_dense_field(data: Dict[str, np.ndarray], field: str, example_index: int, num_nodes: int) -> np.ndarray:
    values = np.asarray(data[field][example_index])
    return values[:num_nodes]


def extract_example(data: Dict[str, np.ndarray], example_index: int) -> Dict[str, np.ndarray]:
    num_nodes = int(np.asarray(data["num_nodes"])[example_index])
    example: Dict[str, np.ndarray] = {
        "num_nodes": np.asarray(num_nodes, dtype=np.int64),
        "pos": _slice_dense_field(data, "pos", example_index, num_nodes).astype(np.float32),
        "rotations": _slice_dense_field(data, "rotations", example_index, num_nodes).astype(np.float32),
        "shape_scalar_features": _slice_dense_field(data, "shape_scalar_features", example_index, num_nodes).astype(np.float32),
        "shape_equiv_features": _slice_dense_field(data, "shape_equiv_features", example_index, num_nodes).astype(np.float32),
        "dipole_strength": _slice_dense_field(data, "dipole_strength", example_index, num_nodes).astype(np.float32),
        "dipole_direction": _slice_dense_field(data, "dipole_direction", example_index, num_nodes).astype(np.float32),
        "ligand_mask": _slice_dense_field(data, "ligand_mask", example_index, num_nodes).astype(bool).reshape(num_nodes),
        "pocket_mask": _slice_dense_field(data, "pocket_mask", example_index, num_nodes).astype(bool).reshape(num_nodes),
        "source_frame_id": np.asarray(data["source_frame_id"][example_index]).astype(np.int64),
        "split_id": np.asarray(data["split_id"][example_index]).astype(np.int64),
    }
    for field in (
        "shape_features_raw",
        "shape_scalar_features_raw",
        "shape_equiv_features_raw",
        "brick_dipoles_raw",
        "dipole_strength_raw",
        "dipole_direction_raw",
    ):
        if field in data:
            example[field] = _slice_dense_field(data, field, example_index, num_nodes).astype(np.float32)

    if "types" in data:
        example["types"] = _slice_dense_field(data, "types", example_index, num_nodes).astype(str)
    elif "type_vocab" in data:
        type_vocab = np.asarray(data["type_vocab"]).astype(str).tolist()
        node_types = _slice_dense_field(data, "node_types", example_index, num_nodes).astype(np.int64).reshape(num_nodes)
        example["types"] = np.asarray([type_vocab[int(index)] for index in node_types], dtype=f"<U{max(len(v) for v in type_vocab)}")
    else:
        raise KeyError("Input diffusion dataset is missing both `types` and `type_vocab`.")

    if "node_types" in data:
        example["node_types"] = _slice_dense_field(data, "node_types", example_index, num_nodes).astype(np.int64).reshape(num_nodes)
    else:
        raise KeyError("Input diffusion dataset is missing `node_types`.")
    return example


def extract_scalar_normalization(data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    enabled = bool(int(np.asarray(data.get("scalar_normalization_enabled", [0])).reshape(-1)[0]))
    if not enabled:
        return {}

    output = {}
    for field, prefix in (("shape_scalar_features", "shape_scalar"), ("dipole_strength", "dipole_strength")):
        means_key = f"{prefix}_means"
        stds_key = f"{prefix}_stds"
        if means_key not in data or stds_key not in data:
            continue
        output[field] = {
            "means": np.asarray(data[means_key], dtype=np.float32),
            "stds": np.asarray(data[stds_key], dtype=np.float32),
        }
    return output


def _random_unit_vectors(shape: tuple[int, int], generator: torch.Generator, device: torch.device) -> torch.Tensor:
    noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    norms = torch.linalg.norm(noise, dim=-1, keepdim=True)
    return noise / torch.clamp(norms, min=1e-8)


def _random_shape_equiv_noise(num_nodes: int, generator: torch.Generator, device: torch.device) -> torch.Tensor:
    parts = [
        _random_unit_vectors((num_nodes, 3), generator=generator, device=device),
        _random_unit_vectors((num_nodes, 5), generator=generator, device=device),
        _random_unit_vectors((num_nodes, 7), generator=generator, device=device),
    ]
    return torch.cat(parts, dim=-1)


def _project_shape_equiv(values: torch.Tensor) -> torch.Tensor:
    parts = []
    for start, stop in ((0, 3), (3, 8), (8, 15)):
        block = values[..., start:stop]
        norm = torch.linalg.norm(block, dim=-1, keepdim=True)
        parts.append(block / torch.clamp(norm, min=1e-8))
    return torch.cat(parts, dim=-1)


def _project_dipole_direction(values: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(values, dim=-1, keepdim=True)
    return values / torch.clamp(norm, min=1e-8)


def initial_masked_state(
    example: Dict[str, np.ndarray],
    generator: torch.Generator,
    device: torch.device,
    center_positions: bool,
    position_center_mask_field: str,
    center_position_noise: bool,
    position_noise_center_mask_field: str,
    corrupt_field_map: Dict[str, str],
):
    pos_data = torch.as_tensor(example["pos"], dtype=torch.float32, device=device)
    center_mask = None
    if position_center_mask_field != "":
        if position_center_mask_field not in example:
            raise KeyError(
                f"Sampling requested position center mask field '{position_center_mask_field}' but it is missing from the example."
            )
        center_mask = torch.as_tensor(example[position_center_mask_field], dtype=torch.bool, device=device)
    if center_positions:
        centroid = compute_reference_mean(pos_data, mask=center_mask)
        pos_model = pos_data - centroid
    else:
        centroid = torch.zeros((1, 3), dtype=pos_data.dtype, device=device)
        pos_model = pos_data

    noise_center_mask = None
    if position_noise_center_mask_field != "":
        if position_noise_center_mask_field not in example:
            raise KeyError(
                f"Sampling requested position noise center mask field '{position_noise_center_mask_field}' "
                "but it is missing from the example."
            )
        noise_center_mask = torch.as_tensor(example[position_noise_center_mask_field], dtype=torch.bool, device=device)

    shape_scalar_data = torch.as_tensor(example["shape_scalar_features"], dtype=torch.float32, device=device)
    shape_equiv_data = torch.as_tensor(example["shape_equiv_features"], dtype=torch.float32, device=device)
    dipole_strength_data = torch.as_tensor(example["dipole_strength"], dtype=torch.float32, device=device)
    dipole_direction_data = torch.as_tensor(example["dipole_direction"], dtype=torch.float32, device=device)

    ligand_mask = torch.as_tensor(example["ligand_mask"], dtype=torch.bool, device=device)
    pocket_mask = torch.as_tensor(example["pocket_mask"], dtype=torch.bool, device=device)

    pos_noise = torch.randn(pos_model.shape, generator=generator, device=device, dtype=torch.float32)
    if center_positions and center_position_noise:
        pos_noise = center_pos(pos_noise, mask=noise_center_mask)
    shape_scalar_noise = torch.randn(shape_scalar_data.shape, generator=generator, device=device, dtype=torch.float32)
    shape_equiv_noise = _random_shape_equiv_noise(int(example["num_nodes"]), generator=generator, device=device)
    dipole_strength_noise = torch.randn(dipole_strength_data.shape, generator=generator, device=device, dtype=torch.float32)
    dipole_direction_noise = _random_unit_vectors((int(example["num_nodes"]), 3), generator=generator, device=device)

    pos_state = pos_model.clone()
    if "pos" in corrupt_field_map:
        pos_state[ligand_mask] = pos_noise[ligand_mask]

    shape_scalar_state = shape_scalar_data.clone()
    if "shape_scalar_features" in corrupt_field_map:
        shape_scalar_state[ligand_mask] = shape_scalar_noise[ligand_mask]

    shape_equiv_state = shape_equiv_data.clone()
    if "shape_equiv_features" in corrupt_field_map:
        shape_equiv_state[ligand_mask] = shape_equiv_noise[ligand_mask]

    dipole_strength_state = dipole_strength_data.clone()
    if "dipole_strength" in corrupt_field_map:
        dipole_strength_state[ligand_mask] = dipole_strength_noise[ligand_mask]

    dipole_direction_state = dipole_direction_data.clone()
    if "dipole_direction" in corrupt_field_map:
        dipole_direction_state[ligand_mask] = dipole_direction_noise[ligand_mask]

    return {
        "centroid": centroid,
        "pos_fixed": pos_model,
        "shape_scalar_fixed": shape_scalar_data,
        "shape_equiv_fixed": shape_equiv_data,
        "dipole_strength_fixed": dipole_strength_data,
        "dipole_direction_fixed": dipole_direction_data,
        "pos": pos_state,
        "shape_scalar_features": shape_scalar_state,
        "shape_equiv_features": shape_equiv_state,
        "dipole_strength": dipole_strength_state,
        "dipole_direction": dipole_direction_state,
        "ligand_mask": ligand_mask,
        "pocket_mask": pocket_mask,
    }


@torch.no_grad()
def sample_example(
    model: torch.nn.Module,
    sampler: FlowMatchingSampler,
    example: Dict[str, np.ndarray],
    r_max: float,
    steps: int,
    device: torch.device,
    seed: int,
    scalar_normalization: Dict[str, Dict[str, np.ndarray]],
    center_positions: bool,
    position_center_mask_field: str,
    center_position_noise: bool,
    position_noise_center_mask_field: str,
    project_normalized_states: bool,
    corrupt_field_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    state = initial_masked_state(
        example=example,
        generator=generator,
        device=device,
        center_positions=center_positions,
        position_center_mask_field=position_center_mask_field,
        center_position_noise=center_position_noise,
        position_noise_center_mask_field=position_noise_center_mask_field,
        corrupt_field_map=corrupt_field_map,
    )
    ligand_mask = state["ligand_mask"]
    ligand_mask_column = ligand_mask.unsqueeze(-1)
    batch = torch.zeros((int(example["num_nodes"]),), dtype=torch.long, device=device)
    node_types = torch.as_tensor(example["node_types"], dtype=torch.long, device=device).unsqueeze(-1)
    rotations = torch.as_tensor(example["rotations"], dtype=torch.float32, device=device)

    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    start_step = sampler.T - 1
    time_steps = np.linspace(0, start_step, num=steps, dtype=int)[::-1].copy()

    for step_idx, t in enumerate(time_steps):
        t_prev = int(time_steps[step_idx + 1]) if step_idx < len(time_steps) - 1 else -1
        tau = torch.full((1, 1), float(sampler.scheduler.tau[int(t)].item()), device=device, dtype=torch.float32)
        edge_index = build_edge_index(state["pos"], cutoff=r_max)
        batch_input = {
            AtomicDataDict.POSITIONS_KEY: state["pos"],
            AtomicDataDict.NODE_TYPE_KEY: node_types,
            AtomicDataDict.BATCH_KEY: batch,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.T_SAMPLED_KEY: tau,
            AtomicDataDict.SHAPE_SCALAR_FEATURES_KEY: state["shape_scalar_features"],
            AtomicDataDict.SHAPE_EQUIV_FEATURES_KEY: state["shape_equiv_features"],
            AtomicDataDict.DIPOLE_STRENGTH_KEY: state["dipole_strength"],
            AtomicDataDict.DIPOLE_DIRECTION_KEY: state["dipole_direction"],
            AtomicDataDict.LIGAND_MASK_KEY: ligand_mask_column.to(dtype=torch.float32),
            AtomicDataDict.POCKET_MASK_KEY: state["pocket_mask"].unsqueeze(-1).to(dtype=torch.float32),
            AtomicDataDict.ROTATIONS_KEY: rotations,
        }
        output = model(batch_input)

        next_states = {
            "pos": state["pos"],
            "shape_scalar_features": state["shape_scalar_features"],
            "shape_equiv_features": state["shape_equiv_features"],
            "dipole_strength": state["dipole_strength"],
            "dipole_direction": state["dipole_direction"],
        }
        for field_name, out_field in STATE_FIELD_SPECS:
            if field_name not in corrupt_field_map:
                continue
            if out_field not in output:
                raise KeyError(
                    f"Checkpoint declares corrupt field '{field_name}' but model output is missing '{out_field}'."
                )
            next_states[field_name] = sampler.step(state[field_name], int(t), t_prev, output[out_field])

        if project_normalized_states:
            next_states["shape_equiv_features"] = _project_shape_equiv(next_states["shape_equiv_features"])
            next_states["dipole_direction"] = _project_dipole_direction(next_states["dipole_direction"])

        state["pos"] = torch.where(ligand_mask_column, next_states["pos"], state["pos_fixed"])
        state["shape_scalar_features"] = torch.where(ligand_mask_column, next_states["shape_scalar_features"], state["shape_scalar_fixed"])
        state["shape_equiv_features"] = torch.where(ligand_mask_column, next_states["shape_equiv_features"], state["shape_equiv_fixed"])
        state["dipole_strength"] = torch.where(ligand_mask_column, next_states["dipole_strength"], state["dipole_strength_fixed"])
        state["dipole_direction"] = torch.where(ligand_mask_column, next_states["dipole_direction"], state["dipole_direction_fixed"])

    sampled_pos = (state["pos"] + state["centroid"]).detach().cpu().numpy().astype(np.float32)
    sampled_shape_scalar_model = state["shape_scalar_features"].detach().cpu().numpy().astype(np.float32)
    sampled_dipole_strength_model = state["dipole_strength"].detach().cpu().numpy().astype(np.float32)
    sampled_shape_equiv = _project_shape_equiv(state["shape_equiv_features"]).detach().cpu().numpy().astype(np.float32)
    sampled_dipole_direction = _project_dipole_direction(state["dipole_direction"]).detach().cpu().numpy().astype(np.float32)

    sampled_shape_scalar = (
        invert_scalar_normalization(sampled_shape_scalar_model, scalar_normalization["shape_scalar_features"]).astype(np.float32)
        if "shape_scalar_features" in scalar_normalization
        else sampled_shape_scalar_model
    )
    sampled_dipole_strength = (
        invert_scalar_normalization(sampled_dipole_strength_model, scalar_normalization["dipole_strength"]).astype(np.float32)
        if "dipole_strength" in scalar_normalization
        else sampled_dipole_strength_model
    )
    sampled_dipole_strength = np.clip(sampled_dipole_strength, 0.0, None).astype(np.float32)
    sampled_dipole_direction = normalize_dipole_directions(sampled_dipole_direction).astype(np.float32)

    sampled_shape = combine_shape_irreps(sampled_shape_scalar, sampled_shape_equiv).astype(np.float32)
    sampled_dipoles = (sampled_dipole_direction * sampled_dipole_strength).astype(np.float32)

    original_shape = np.asarray(
        example.get("shape_features_raw", combine_shape_irreps(example["shape_scalar_features"], example["shape_equiv_features"])),
        dtype=np.float32,
    )
    original_dipoles = np.asarray(
        example.get("brick_dipoles_raw", example["dipole_direction"] * example["dipole_strength"]),
        dtype=np.float32,
    )
    original_dipole_strength = np.asarray(
        example.get("dipole_strength_raw", example["dipole_strength"]),
        dtype=np.float32,
    )
    original_dipole_direction = normalize_dipole_directions(
        np.asarray(example.get("dipole_direction_raw", example["dipole_direction"]), dtype=np.float32)
    ).astype(np.float32)

    return {
        "brick_anchors": sampled_pos,
        "brick_types": np.asarray(example["types"]).astype(str),
        "brick_rotations": np.asarray(example["rotations"], dtype=np.float32),
        "brick_features": sampled_shape,
        "brick_dipoles": sampled_dipoles,
        "brick_dipole_strengths": sampled_dipole_strength,
        "brick_dipole_directions": sampled_dipole_direction,
        "original_brick_anchors": np.asarray(example["pos"], dtype=np.float32),
        "original_brick_types": np.asarray(example["types"]).astype(str),
        "original_brick_rotations": np.asarray(example["rotations"], dtype=np.float32),
        "original_brick_features": original_shape,
        "original_brick_dipoles": original_dipoles,
        "original_brick_dipole_strengths": original_dipole_strength,
        "original_brick_dipole_directions": original_dipole_direction,
        "sampled_brick_mask": np.asarray(example["ligand_mask"], dtype=bool),
        "ligand_mask": np.asarray(example["ligand_mask"], dtype=bool),
        "pocket_mask": np.asarray(example["pocket_mask"], dtype=bool),
        "source_frame_id": np.asarray(example["source_frame_id"], dtype=np.int64),
        "split_id": np.asarray(example["split_id"], dtype=np.int64),
    }


def enrich_from_canonical_source(
    sample: Dict[str, np.ndarray],
    source_samples: Sequence[Dict[str, np.ndarray]] | None,
) -> Dict[str, np.ndarray]:
    if source_samples is None:
        return sample

    source_frame_id = int(np.asarray(sample["source_frame_id"]).item())
    if source_frame_id < 0 or source_frame_id >= len(source_samples):
        return sample

    enriched = dict(sample)
    source = source_samples[source_frame_id]
    for key in (
        "coefficients",
        "mesh_x",
        "mesh_y",
        "mesh_z",
        "target_voxels",
        "occupancy_mode",
        "shell_thickness",
        "shell_sparsity",
    ):
        if key in source:
            enriched[key] = np.asarray(source[key]).copy()
    return enriched


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    model, config, _ = CheckpointHandler.load_model(str(args.model), device=args.device)
    validate_flow_checkpoint_config(config)
    center_positions = infer_position_centering(config)
    position_center_mask_field = infer_position_center_mask_field(config)
    center_position_noise = infer_position_noise_centering(config)
    position_noise_center_mask_field = infer_position_noise_center_mask_field(config)
    project_normalized_states = infer_project_normalized_states(config)
    corrupt_field_map = infer_corrupt_field_map(config)

    try:
        r_max = float(config[AtomicDataDict.R_MAX_KEY])
    except KeyError as exc:
        raise KeyError(f"Could not find '{AtomicDataDict.R_MAX_KEY}' in the checkpoint config.") from exc

    tmax = infer_model_tmax(config)
    if tmax is None:
        raise ValueError("Could not determine Tmax from the checkpoint config.")

    sampler = FlowMatchingSampler(FlowMatchingScheduler(T=tmax)).to(args.device)
    model.to(args.device)
    model.eval()

    with np.load(args.input, allow_pickle=True) as raw:
        data = {key: raw[key] for key in raw.files}
    scalar_normalization = extract_scalar_normalization(data)

    source_samples = load_samples(args.source_canonical) if args.source_canonical is not None else None
    selected_indices = choose_example_indices(args, num_examples=int(np.asarray(data["num_nodes"]).shape[0]))

    samples = []
    for output_index, example_index in enumerate(tqdm(selected_indices, desc="Sampling LEGO")):
        example = extract_example(data, int(example_index))
        sample = sample_example(
            model=model,
            sampler=sampler,
            example=example,
            r_max=r_max,
            steps=args.steps,
            device=torch.device(args.device),
            seed=args.seed + output_index,
            scalar_normalization=scalar_normalization,
            center_positions=center_positions,
            position_center_mask_field=position_center_mask_field,
            center_position_noise=center_position_noise,
            position_noise_center_mask_field=position_noise_center_mask_field,
            project_normalized_states=project_normalized_states,
            corrupt_field_map=corrupt_field_map,
        )
        sample["conditioning_example_index"] = np.asarray(example_index, dtype=np.int64)
        samples.append(enrich_from_canonical_source(sample, source_samples=source_samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_samples(args.output, samples)
    print(f"Saved {len(samples)} sampled LEGO assemblies to {args.output}")


if __name__ == "__main__":
    main()
