from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

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
from geqdiff.utils import FlowMatchingHeunSampler, FlowMatchingSampler, FlowMatchingScheduler
from geqdiff.utils.diffusion import center_pos, compute_reference_mean
from geqdiff.utils.dipole_utils import (
    dipole_strengths,
    normalize_dipole_directions,
)
from geqdiff.scripts.evaluate_lego_samples import build_evaluation_report
from geqtrain.train.components.checkpointing import CheckpointHandler
from lego.utils import load_samples, save_samples


STATE_FIELD_SPECS = (
    ("pos", "velocity"),
    ("shape_features", "shape_features_velocity"),
    ("dipole_direction", "dipole_direction_velocity"),
)


def _default_noise_like(
    field_name: str,
    data: torch.Tensor,
    *,
    num_nodes: int,
    generator: torch.Generator,
    device: torch.device,
    unit_norm_noise: bool = False,
) -> torch.Tensor:
    _ = field_name
    _ = num_nodes
    _ = unit_norm_noise
    return torch.randn(data.shape, generator=generator, device=device, dtype=torch.float32)


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
    parser.add_argument(
        "--start-step",
        type=int,
        default=-1,
        help=(
            "Optional initial scheduler step to start from (0..T-1). "
            "If set, masked fields are initialized as x_tau=(1-tau)x_data+tau*noise at that step. "
            "Default -1 disables this and starts from full-noise (T-1)."
        ),
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        choices=("heun", "euler"),
        help="Reverse integrator to use for flow matching. `heun` is slower but usually more accurate.",
    )
    parser.add_argument(
        "--late-refine-from-step",
        type=int,
        default=-1,
        help="If >= 0, subdivide all reverse intervals from this discrete scheduler step downward.",
    )
    parser.add_argument(
        "--late-refine-factor",
        type=int,
        default=1,
        help="Subdivision factor for late refinement. 1 disables it.",
    )
    parser.add_argument(
        "--linger-step",
        type=int,
        default=-1,
        help="Experimental: discrete scheduler step at which to apply extra frozen-tau micro-steps before continuing.",
    )
    parser.add_argument(
        "--linger-count",
        type=int,
        default=0,
        help="Experimental number of extra frozen-tau micro-steps to apply at --linger-step.",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save the full sampled trajectory for each example, including the initial noisy state and all reverse steps.",
    )
    parser.add_argument(
        "--save-velocity-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When saving intermediates, also save per-step position displacement vectors "
            "(predicted and guided). Ignored for models that do not output `velocity`."
        ),
    )
    parser.add_argument(
        "--clash-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply optional clash guidance from the predicted final structure during reverse integration.",
    )
    parser.add_argument(
        "--clash-guidance-strength",
        type=float,
        default=0.05,
        help="Relative scale of the clash guidance correction added to the position velocity field.",
    )
    parser.add_argument(
        "--clash-guidance-max-norm",
        type=float,
        default=1.0,
        help="Per-node max norm for the clash guidance correction before timestep weighting.",
    )
    parser.add_argument(
        "--clash-guidance-weight-schedule",
        type=str,
        default="late_quadratic",
        choices=("late_linear", "late_quadratic", "late_cubic", "flat", "tau"),
        help="Timestep weighting for clash guidance. Late-weighted schedules emphasize end-of-trajectory refinement.",
    )
    parser.add_argument(
        "--clash-guidance-auto-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically scale guidance relative to the model velocity magnitude on each step.",
    )
    parser.add_argument(
        "--clash-guidance-auto-scale-min",
        type=float,
        default=0.2,
        help="Lower clamp for the auto guidance scale factor.",
    )
    parser.add_argument(
        "--clash-guidance-auto-scale-max",
        type=float,
        default=5.0,
        help="Upper clamp for the auto guidance scale factor.",
    )
    parser.add_argument(
        "--save-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an evaluation JSON report next to the sampled NPZ.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional explicit path for the saved evaluation JSON.",
    )
    parser.add_argument(
        "--shape-decode-mode",
        type=str,
        default="input_knn",
        choices=("input_knn", "keep_original"),
        help=(
            "How to convert predicted continuous shape features to discrete brick type/rotation. "
            "`input_knn` decodes against shape/type exemplars from the input dataset."
        ),
    )
    parser.add_argument(
        "--shape-decode-max-candidates",
        type=int,
        default=2048,
        help="Maximum number of candidates used by `input_knn` shape decoding.",
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


def infer_corrupt_field_settings(config: Dict) -> Dict[str, Dict[str, Any]]:
    model_cfg = config.get("model", {})
    stack = model_cfg.get("stack", [])
    default_specs = {
        field: {
            "field": field,
            "out_field": out_field,
            "center": field == AtomicDataDict.POSITIONS_KEY,
            "center_mask_field": "",
            "center_noise": field == AtomicDataDict.POSITIONS_KEY,
            "noise_center_mask_field": "",
            "mask_field": "ligand_mask",
            "unit_norm_noise": False,
        }
        for field, out_field in STATE_FIELD_SPECS
    }
    if not (isinstance(stack, list) and len(stack) > 0 and isinstance(stack[0], dict)):
        return default_specs

    flow_cfg = stack[0]
    corrupt_fields = flow_cfg.get("corrupt_fields", [])
    if not isinstance(corrupt_fields, list) or len(corrupt_fields) == 0:
        return default_specs

    parsed: Dict[str, Dict[str, Any]] = {}
    for spec in corrupt_fields:
        field_name = str(spec.get("field", ""))
        if field_name == "":
            continue
        default_out = "velocity" if field_name == AtomicDataDict.POSITIONS_KEY else f"{field_name}_velocity"
        center_field = bool(spec.get("center", field_name == AtomicDataDict.POSITIONS_KEY))
        center_mask_field = spec.get("center_mask_field", "")
        if center_mask_field in (None, "", "all", "__all__"):
            center_mask_field = ""
        else:
            center_mask_field = str(center_mask_field)
        center_noise = bool(spec.get("center_noise", center_field))
        unit_norm_noise = spec.get("unit_norm_noise", "__auto__")
        if unit_norm_noise == "__auto__":
            unit_norm_noise = False
        unit_norm_noise = bool(unit_norm_noise)
        noise_center_mask_field = spec.get("noise_center_mask_field", "__auto__")
        if noise_center_mask_field == "__auto__":
            mask_field = spec.get("mask_field", "")
            if field_name == AtomicDataDict.POSITIONS_KEY and mask_field not in (None, ""):
                noise_center_mask_field = str(mask_field)
            else:
                noise_center_mask_field = center_mask_field
        elif noise_center_mask_field in (None, "", "all", "__all__"):
            noise_center_mask_field = ""
        else:
            noise_center_mask_field = str(noise_center_mask_field)
        parsed[field_name] = {
            "field": field_name,
            "out_field": str(spec.get("out_field", default_out)),
            "center": center_field,
            "center_mask_field": center_mask_field,
            "center_noise": center_noise,
            "noise_center_mask_field": noise_center_mask_field,
            "mask_field": "" if spec.get("mask_field", "") in (None, "") else str(spec.get("mask_field", "")),
            "unit_norm_noise": unit_norm_noise,
        }
    return parsed if parsed else default_specs


def infer_corrupt_field_map(config: Dict) -> Dict[str, str]:
    settings = infer_corrupt_field_settings(config)
    return {field_name: str(spec["out_field"]) for field_name, spec in settings.items()}


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
    if "shape_features" in data:
        shape_features = _slice_dense_field(data, "shape_features", example_index, num_nodes).astype(np.float32)
    elif "shape_features_raw" in data:
        shape_features = _slice_dense_field(data, "shape_features_raw", example_index, num_nodes).astype(np.float32)
    else:
        raise KeyError("Input diffusion dataset is missing `shape_features` (or `shape_features_raw`).")

    example: Dict[str, np.ndarray] = {
        "num_nodes": np.asarray(num_nodes, dtype=np.int64),
        "pos": _slice_dense_field(data, "pos", example_index, num_nodes).astype(np.float32),
        "rotations": _slice_dense_field(data, "rotations", example_index, num_nodes).astype(np.float32),
        "shape_features": shape_features,
        "dipole_direction": _slice_dense_field(data, "dipole_direction", example_index, num_nodes).astype(np.float32),
        "ligand_mask": _slice_dense_field(data, "ligand_mask", example_index, num_nodes).astype(bool).reshape(num_nodes),
        "pocket_mask": _slice_dense_field(data, "pocket_mask", example_index, num_nodes).astype(bool).reshape(num_nodes),
        "source_frame_id": np.asarray(data["source_frame_id"][example_index]).astype(np.int64),
        "split_id": np.asarray(data["split_id"][example_index]).astype(np.int64),
    }
    if "sequence_position" in data:
        seq = _slice_dense_field(data, "sequence_position", example_index, num_nodes).astype(np.int64)
        if seq.ndim == 1:
            seq = seq[:, None]
        if seq.shape != (num_nodes, 1):
            raise ValueError(
                f"Expected sequence_position shape {(num_nodes, 1)}, got {seq.shape}."
            )
        example["sequence_position"] = seq
    else:
        # Backward-compatible fallback for older diffusion datasets.
        example["sequence_position"] = np.arange(num_nodes, dtype=np.int64)[:, None]
    for field in (
        "shape_features_raw",
        "brick_dipoles_raw",
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


def _dataset_types_matrix(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "types" in data:
        return np.asarray(data["types"]).astype(str)
    if "type_vocab" in data and "node_types" in data:
        type_vocab = np.asarray(data["type_vocab"]).astype(str).tolist()
        node_types = np.asarray(data["node_types"], dtype=np.int64)
        out = np.empty(node_types.shape, dtype=f"<U{max(len(v) for v in type_vocab)}")
        for idx, token in enumerate(type_vocab):
            out[node_types == idx] = str(token)
        return out
    raise KeyError("Input diffusion dataset is missing both `types` and (`type_vocab` + `node_types`).")


def _dataset_shape_features_matrix(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "shape_features_raw" in data:
        return np.asarray(data["shape_features_raw"], dtype=np.float32)
    if "shape_features" in data:
        return np.asarray(data["shape_features"], dtype=np.float32)
    raise KeyError(
        "Input diffusion dataset is missing shape features. Expected one of "
        "`shape_features_raw` or `shape_features`."
    )


def _decode_with_input_knn(
    signatures: np.ndarray,
    decode_library: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    x = np.asarray(signatures, dtype=np.float32)
    y = np.asarray(decode_library["features"], dtype=np.float32)
    if x.ndim != 2 or x.shape[-1] != 16:
        raise ValueError(f"Expected signatures shape [N,16], got {x.shape}.")
    if y.ndim != 2 or y.shape[-1] != 16:
        raise ValueError(f"Expected decode-library features shape [M,16], got {y.shape}.")
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1)[None, :]
    distances2 = np.maximum(0.0, x2 + y2 - 2.0 * (x @ y.T))
    indices = np.argmin(distances2, axis=1).astype(np.int64)
    distances = np.sqrt(np.take_along_axis(distances2, indices[:, None], axis=1).reshape(-1)).astype(np.float32)
    return {
        "brick_types": np.asarray(decode_library["types"])[indices].astype(str),
        "rotations": np.asarray(decode_library["rotations"], dtype=np.float32)[indices],
        "indices": indices,
        "distances": distances,
    }


def _build_input_knn_decode_library(
    data: Dict[str, np.ndarray],
    *,
    max_candidates: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    max_candidates = int(max(max_candidates, 1))
    num_nodes = np.asarray(data["num_nodes"], dtype=np.int64).reshape(-1)
    rotations_all = np.asarray(data["rotations"], dtype=np.float32)
    shape_all = _dataset_shape_features_matrix(data)
    types_all = _dataset_types_matrix(data)
    ligand_all = np.asarray(data.get("ligand_mask"), dtype=bool) if "ligand_mask" in data else None

    features_list: List[np.ndarray] = []
    types_list: List[np.ndarray] = []
    rotations_list: List[np.ndarray] = []
    for example_idx in range(int(num_nodes.shape[0])):
        nn = int(num_nodes[example_idx])
        if nn <= 0:
            continue
        select = np.ones((nn,), dtype=bool)
        if ligand_all is not None:
            select = np.asarray(ligand_all[example_idx, :nn], dtype=bool).reshape(nn)
            if not np.any(select):
                continue
        features_list.append(np.asarray(shape_all[example_idx, :nn], dtype=np.float32)[select])
        types_list.append(np.asarray(types_all[example_idx, :nn]).astype(str)[select])
        rotations_list.append(np.asarray(rotations_all[example_idx, :nn], dtype=np.float32)[select])

    if len(features_list) == 0:
        raise ValueError("Could not build shape decode library from input dataset: no candidate nodes found.")

    features = np.concatenate(features_list, axis=0).astype(np.float32)
    types = np.concatenate(types_list, axis=0).astype(str)
    rotations = np.concatenate(rotations_list, axis=0).astype(np.float32)

    if features.shape[0] > max_candidates:
        rng = np.random.default_rng(int(seed))
        keep = rng.choice(features.shape[0], size=max_candidates, replace=False)
        features = features[keep]
        types = types[keep]
        rotations = rotations[keep]

    return {
        "features": features.astype(np.float32),
        "types": types.astype(str),
        "rotations": rotations.astype(np.float32),
    }


def _decode_sampled_structure(
    state: Dict[str, torch.Tensor],
    example: Dict[str, np.ndarray],
    shape_decode_mode: str,
    shape_decode_library: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    sampled_pos = (state["pos"] + state["centroid"]).detach().cpu().numpy().astype(np.float32)
    sampled_shape = state["shape_features"].detach().cpu().numpy().astype(np.float32)
    sampled_dipoles = state["dipole_direction"].detach().cpu().numpy().astype(np.float32)
    sampled_dipole_strength = dipole_strengths(sampled_dipoles).astype(np.float32)
    sampled_dipole_direction = normalize_dipole_directions(sampled_dipoles).astype(np.float32)

    mode = str(shape_decode_mode).strip().lower()
    if mode == "input_knn":
        if shape_decode_library is None:
            raise ValueError("shape_decode_mode='input_knn' requires a non-empty decode library.")
        decoded_library = _decode_with_input_knn(sampled_shape, shape_decode_library)
    elif mode == "keep_original":
        decoded_library = {
            "brick_types": np.asarray(example["types"]).astype(str),
            "rotations": np.asarray(example["rotations"], dtype=np.float32),
            "indices": np.full((sampled_shape.shape[0],), -1, dtype=np.int64),
            "distances": np.zeros((sampled_shape.shape[0],), dtype=np.float32),
        }
    else:
        raise ValueError(f"Unsupported shape decode mode '{shape_decode_mode}'.")
    brick_types = np.asarray(example["types"]).astype(str).copy()
    brick_rotations = np.asarray(example["rotations"], dtype=np.float32).copy()
    ligand_mask = np.asarray(example["ligand_mask"], dtype=bool).reshape(-1)
    brick_types[ligand_mask] = np.asarray(decoded_library["brick_types"])[ligand_mask]
    brick_rotations[ligand_mask] = np.asarray(decoded_library["rotations"], dtype=np.float32)[ligand_mask]
    return {
        "brick_anchors": sampled_pos,
        "brick_types": brick_types,
        "brick_rotations": brick_rotations,
        "brick_features": sampled_shape,
        "brick_dipoles": sampled_dipoles,
        "brick_dipole_strengths": sampled_dipole_strength,
        "brick_dipole_directions": sampled_dipole_direction,
        "decoded_library_index": np.asarray(decoded_library["indices"], dtype=np.int64),
        "decoded_library_distance": np.asarray(decoded_library["distances"], dtype=np.float32),
    }


def _trajectory_snapshot(
    state: Dict[str, torch.Tensor],
    example: Dict[str, np.ndarray],
    shape_decode_mode: str,
    shape_decode_library: Dict[str, np.ndarray] | None,
    *,
    stage_index: int,
    stage_label: str,
    scheduler_step: int,
    tau: float,
    velocity_vectors: torch.Tensor | None = None,
    velocity_raw_vectors: torch.Tensor | None = None,
) -> Dict[str, np.ndarray]:
    snapshot = _decode_sampled_structure(
        state,
        example,
        shape_decode_mode=shape_decode_mode,
        shape_decode_library=shape_decode_library,
    )
    snapshot["stage_index"] = np.asarray(stage_index, dtype=np.int64)
    snapshot["stage_label"] = np.asarray(stage_label)
    snapshot["scheduler_step"] = np.asarray(scheduler_step, dtype=np.int64)
    snapshot["tau"] = np.asarray(tau, dtype=np.float32)
    if velocity_vectors is not None:
        snapshot["velocity_vectors"] = velocity_vectors.detach().cpu().numpy().astype(np.float32)
    if velocity_raw_vectors is not None:
        snapshot["velocity_raw_vectors"] = velocity_raw_vectors.detach().cpu().numpy().astype(np.float32)
    return snapshot


def _build_sampling_schedule(
    sampler: FlowMatchingSampler,
    *,
    steps: int,
    start_scheduler_step: int,
    late_refine_from_step: int,
    late_refine_factor: int,
    linger_step: int,
    linger_count: int,
) -> List[Dict[str, float | int | str]]:
    if steps < 1:
        raise ValueError("--steps must be >= 1.")
    if late_refine_factor < 1:
        raise ValueError("--late-refine-factor must be >= 1.")
    if linger_count < 0:
        raise ValueError("--linger-count must be >= 0.")

    start_step = int(start_scheduler_step)
    if start_step < 0 or start_step >= int(sampler.T):
        raise ValueError(f"start_scheduler_step must be in [0, {int(sampler.T) - 1}], got {start_step}.")
    time_edges = np.linspace(0, start_step, num=steps + 1, dtype=int)[::-1].copy()
    schedule: List[Dict[str, float | int | str]] = []

    for edge_index in range(steps):
        t = int(time_edges[edge_index])
        t_prev_edge = int(time_edges[edge_index + 1])
        tau_t = float(sampler.scheduler.tau[t].item())
        tau_prev = float(sampler.scheduler.tau[t_prev_edge].item()) if t_prev_edge > 0 else 0.0

        if linger_step >= 0 and linger_count > 0 and t == linger_step:
            linger_edges = np.linspace(tau_t, tau_prev, num=linger_count + 2, dtype=np.float32)
            for sub_index in range(linger_count + 1):
                next_tau = float(linger_edges[sub_index + 1])
                schedule.append(
                    {
                        "tau": tau_t,
                        "tau_next": next_tau,
                        "scheduler_step": t,
                        "next_scheduler_step": (t_prev_edge if sub_index == linger_count else t),
                        "mode": "linger" if sub_index < linger_count else "base",
                    }
                )
            continue

        refine_factor = late_refine_factor if (late_refine_from_step >= 0 and late_refine_factor > 1 and t <= late_refine_from_step) else 1
        if refine_factor > 1:
            refine_edges = np.linspace(tau_t, tau_prev, num=refine_factor + 1, dtype=np.float32)
            for sub_index in range(refine_factor):
                schedule.append(
                    {
                        "tau": float(refine_edges[sub_index]),
                        "tau_next": float(refine_edges[sub_index + 1]),
                        "scheduler_step": t,
                        "next_scheduler_step": t_prev_edge if sub_index == refine_factor - 1 else t,
                        "mode": "refine",
                    }
                )
        else:
            schedule.append(
                {
                    "tau": tau_t,
                    "tau_next": tau_prev,
                    "scheduler_step": t,
                    "next_scheduler_step": t_prev_edge,
                    "mode": "base",
                }
            )
    return schedule


def initial_masked_state(
    example: Dict[str, np.ndarray],
    generator: torch.Generator,
    device: torch.device,
    corrupt_field_map: Dict[str, str],
    corrupt_field_settings: Dict[str, Dict[str, Any]],
    start_tau: float = 1.0,
):
    ligand_mask = torch.as_tensor(example["ligand_mask"], dtype=torch.bool, device=device)
    pocket_mask = torch.as_tensor(example["pocket_mask"], dtype=torch.bool, device=device)
    state: Dict[str, torch.Tensor] = {
        "ligand_mask": ligand_mask,
        "pocket_mask": pocket_mask,
        "centroid": torch.zeros((1, 3), dtype=torch.float32, device=device),
    }

    tau = float(np.clip(start_tau, 0.0, 1.0))
    data_scale = 1.0 - tau
    noise_scale = tau

    num_nodes = int(example["num_nodes"])
    for field_name, _ in STATE_FIELD_SPECS:
        if field_name not in example:
            raise KeyError(f"Sampling expected field '{field_name}' in the example.")
        data = torch.as_tensor(example[field_name], dtype=torch.float32, device=device)
        spec = corrupt_field_settings.get(
            field_name,
            {
                "center": field_name == "pos",
                "center_mask_field": "",
                "center_noise": field_name == "pos",
                "noise_center_mask_field": "",
            },
        )

        center_mask = None
        center_mask_field = str(spec.get("center_mask_field", ""))
        if center_mask_field != "":
            if center_mask_field not in example:
                raise KeyError(
                    f"Sampling requested center mask field '{center_mask_field}' for '{field_name}' but it is missing."
                )
            center_mask = torch.as_tensor(example[center_mask_field], dtype=torch.bool, device=device)

        if bool(spec.get("center", False)):
            centered = center_pos(data, mask=center_mask)
            if field_name == "pos":
                state["centroid"] = compute_reference_mean(data, mask=center_mask)
        else:
            centered = data
            if field_name == "pos":
                state["centroid"] = torch.zeros((1, int(data.shape[-1])), dtype=data.dtype, device=device)

        noise = _default_noise_like(
            field_name,
            data,
            num_nodes=num_nodes,
            generator=generator,
            device=device,
            unit_norm_noise=bool(spec.get("unit_norm_noise", True)),
        )
        noise_center_mask = None
        noise_center_mask_field = str(spec.get("noise_center_mask_field", ""))
        if noise_center_mask_field != "":
            if noise_center_mask_field not in example:
                raise KeyError(
                    f"Sampling requested noise center mask field '{noise_center_mask_field}' for '{field_name}' but it is missing."
                )
            noise_center_mask = torch.as_tensor(example[noise_center_mask_field], dtype=torch.bool, device=device)
        if bool(spec.get("center", False)) and bool(spec.get("center_noise", False)):
            noise = center_pos(noise, mask=noise_center_mask)

        current = centered.clone()
        if field_name in corrupt_field_map:
            mixed = data_scale * centered + noise_scale * noise
            current[ligand_mask] = mixed[ligand_mask]

        fixed_key = "pos_fixed" if field_name == "pos" else f"{field_name}_fixed"
        state[fixed_key] = centered
        state[field_name] = current

    return state


def _build_model_input(
    state: Dict[str, torch.Tensor],
    *,
    batch: torch.Tensor,
    node_types: torch.Tensor,
    sequence_position: torch.Tensor | None,
    rotations: torch.Tensor,
    tau_value: float,
    r_max: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ligand_mask_column = state["ligand_mask"].unsqueeze(-1)
    tau = torch.full((1, 1), float(tau_value), device=device, dtype=torch.float32)
    edge_index = build_edge_index(state["pos"], cutoff=r_max)
    payload = {
        AtomicDataDict.POSITIONS_KEY: state["pos"],
        AtomicDataDict.NODE_TYPE_KEY: node_types,
        AtomicDataDict.BATCH_KEY: batch,
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.T_SAMPLED_KEY: tau,
        AtomicDataDict.SHAPE_FEATURES_KEY: state["shape_features"],
        AtomicDataDict.DIPOLE_DIRECTION_KEY: state["dipole_direction"],
        AtomicDataDict.LIGAND_MASK_KEY: ligand_mask_column.to(dtype=torch.float32),
        AtomicDataDict.POCKET_MASK_KEY: state["pocket_mask"].unsqueeze(-1).to(dtype=torch.float32),
        AtomicDataDict.ROTATIONS_KEY: rotations,
    }
    if sequence_position is not None:
        payload["sequence_position"] = sequence_position
    return payload


def _merge_ligand_updates(
    state: Dict[str, torch.Tensor],
    next_states: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    ligand_mask_column = state["ligand_mask"].unsqueeze(-1)
    merged = dict(state)
    merged["pos"] = torch.where(ligand_mask_column, next_states["pos"], state["pos_fixed"])
    for field_name in ("shape_features", "dipole_direction"):
        if field_name not in state or field_name not in next_states:
            continue
        fixed_key = "pos_fixed" if field_name == "pos" else f"{field_name}_fixed"
        if fixed_key not in state:
            raise KeyError(f"Missing fixed state for {field_name}.")
        merged[field_name] = torch.where(ligand_mask_column, next_states[field_name], state[fixed_key])
    return merged


def _support_sample_directions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    norms = torch.linalg.norm(directions, dim=-1, keepdim=True)
    return directions / torch.clamp(norms, min=1e-8)


def _directional_extent(
    shape_coeffs: torch.Tensor,
    directions: torch.Tensor,
    *,
    support_min_extent: float = 0.45,
    support_extent_scale: float = 0.75,
) -> torch.Tensor:
    from e3nn import o3

    basis = o3.spherical_harmonics(
        list(range(4)),
        directions.to(dtype=shape_coeffs.dtype),
        normalize=True,
        normalization="integral",
    )
    support_raw = torch.sum(shape_coeffs * basis, dim=-1)
    return float(support_min_extent) + float(support_extent_scale) * torch.sigmoid(support_raw)


def _shape_aware_clash_energy(
    pred_pos_0: torch.Tensor,
    pred_shape_coeffs_detached: torch.Tensor,
    ligand_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    *,
    pair_cutoff: float = 3.0,
    overlap_margin: float = 0.05,
    clash_sharpness: float = 10.0,
) -> torch.Tensor:
    num_nodes = int(pred_pos_0.shape[0])
    if num_nodes < 2:
        return pred_pos_0.new_zeros(())

    idx_i, idx_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=pred_pos_0.device)
    if idx_i.numel() == 0:
        return pred_pos_0.new_zeros(())

    lig_i = ligand_mask[idx_i]
    lig_j = ligand_mask[idx_j]
    poc_i = pocket_mask[idx_i]
    poc_j = pocket_mask[idx_j]
    pair_mask = (lig_i & lig_j) | (lig_i & poc_j) | (poc_i & lig_j)
    if not torch.any(pair_mask):
        return pred_pos_0.new_zeros(())

    idx_i = idx_i[pair_mask]
    idx_j = idx_j[pair_mask]
    deltas = pred_pos_0[idx_j] - pred_pos_0[idx_i]
    distances = torch.linalg.norm(deltas, dim=-1)
    valid = torch.isfinite(distances) & (distances > 1e-6) & (distances < float(pair_cutoff))
    if not torch.any(valid):
        return pred_pos_0.new_zeros(())

    idx_i = idx_i[valid]
    idx_j = idx_j[valid]
    deltas = deltas[valid]
    distances = distances[valid]
    directions = deltas / distances.unsqueeze(-1)

    extent_i = _directional_extent(pred_shape_coeffs_detached[idx_i], directions)
    extent_j = _directional_extent(pred_shape_coeffs_detached[idx_j], -directions)
    overlap = extent_i + extent_j + float(overlap_margin) - distances
    # Harmonic hinge penalty: zero gradient when no clash, quadratic growth on overlap.
    stiffness = float(clash_sharpness)
    if stiffness <= 0.0:
        return pred_pos_0.new_zeros(())
    overlap_violation = torch.nn.functional.relu(overlap)
    penalties = 0.5 * stiffness * overlap_violation.square()
    if penalties.numel() == 0:
        return pred_pos_0.new_zeros(())
    finite = torch.isfinite(penalties)
    if not torch.any(finite):
        return pred_pos_0.new_zeros(())
    return penalties[finite].mean()


def _apply_clash_guidance(
    output: Dict[str, torch.Tensor],
    state: Dict[str, torch.Tensor],
    *,
    tau_value: float,
    guidance_enabled: bool,
    guidance_strength: float,
    guidance_max_norm: float,
    guidance_weight_schedule: str,
    guidance_auto_scale: bool,
    guidance_auto_scale_min: float,
    guidance_auto_scale_max: float,
) -> Dict[str, torch.Tensor]:
    if (not guidance_enabled) or guidance_strength <= 0.0 or tau_value <= 1e-8:
        return output
    if "velocity" not in output:
        return output

    tau_value = float(tau_value)
    if guidance_weight_schedule == "late_linear":
        schedule_weight = 1.0 - tau_value
    elif guidance_weight_schedule == "late_quadratic":
        schedule_weight = (1.0 - tau_value) ** 2
    elif guidance_weight_schedule == "late_cubic":
        schedule_weight = (1.0 - tau_value) ** 3
    elif guidance_weight_schedule == "flat":
        schedule_weight = 1.0
    else:
        schedule_weight = tau_value
    if schedule_weight <= 1e-8:
        return output

    tau_safe = float(max(tau_value, 1e-6))
    one_minus_tau_safe = float(max(1.0 - tau_value, 1e-6))

    x_tau = state["pos"].detach().to(dtype=torch.float32)
    velocity_ref = output["velocity"].detach().to(dtype=torch.float32)

    with torch.enable_grad():
        # Compute guidance gradient in x_tau space (stop-grad through model velocity).
        # For the linear FM path x_tau=(1-tau)x0+tau*noise and velocity target x_noise-x0:
        #   x0_hat = x_tau - tau * v_theta
        # so dE/dx_tau is directly usable for score-space conditioning.
        x_tau_req = x_tau.clone().requires_grad_(True)
        pred_pos_0 = x_tau_req - tau_value * velocity_ref
        if "shape_features_velocity" in output and "shape_features" in state:
            pred_shape_coeffs = (
                state["shape_features"].detach().to(dtype=torch.float32)
                - tau_value * output["shape_features_velocity"].detach().to(dtype=torch.float32)
            ).detach()
        else:
            return output
        clash_energy = _shape_aware_clash_energy(
            pred_pos_0,
            pred_shape_coeffs,
            state["ligand_mask"].detach(),
            state["pocket_mask"].detach(),
        )
        if float(clash_energy.detach().item()) <= 0.0:
            return output
        grad_x_tau = torch.autograd.grad(clash_energy, x_tau_req, allow_unused=False)[0]

    ligand_mask = state["ligand_mask"].detach().unsqueeze(-1)
    grad_x_tau = torch.where(ligand_mask, grad_x_tau, torch.zeros_like(grad_x_tau))
    if guidance_max_norm > 0.0:
        grad_norm = torch.linalg.norm(grad_x_tau, dim=-1, keepdim=True)
        clip = torch.clamp(float(guidance_max_norm) / torch.clamp(grad_norm, min=1e-8), max=1.0)
        grad_x_tau = grad_x_tau * clip

    scale_multiplier = 1.0
    # Convert model velocity to score for the linear FM path:
    #   s_theta(x_tau, tau) = -(x_tau + (1 - tau) * v_theta) / tau
    score_ref = -(x_tau + (1.0 - tau_value) * velocity_ref) / tau_safe
    score_ref = torch.where(ligand_mask, score_ref, torch.zeros_like(score_ref))

    if guidance_auto_scale:
        ligand_mask_flat = state["ligand_mask"].detach().reshape(-1)
        score_norm = torch.linalg.norm(score_ref[ligand_mask_flat], dim=-1)
        guide_norm = torch.linalg.norm(grad_x_tau[ligand_mask_flat], dim=-1)
        finite_mask = torch.isfinite(score_norm) & torch.isfinite(guide_norm) & (guide_norm > 1e-8)
        if torch.any(finite_mask):
            score_rms = torch.sqrt(torch.mean(score_norm[finite_mask] ** 2))
            guide_rms = torch.sqrt(torch.mean(guide_norm[finite_mask] ** 2))
            scale_multiplier = float((score_rms / torch.clamp(guide_rms, min=1e-8)).item())
            scale_multiplier = float(
                np.clip(
                    scale_multiplier,
                    float(guidance_auto_scale_min),
                    float(guidance_auto_scale_max),
                )
            )

    # ExEnDiff-style conditioning in score space:
    #   s_cond = s_theta - lambda * grad_x_tau E
    # then map back to velocity:
    #   v_cond = -(x_tau + tau * s_cond) / (1 - tau)
    guidance_scale = float(guidance_strength) * float(schedule_weight) * float(scale_multiplier)
    score_cond = score_ref - guidance_scale * grad_x_tau
    velocity_cond = -(x_tau + tau_value * score_cond) / one_minus_tau_safe
    velocity_cond = torch.where(ligand_mask, velocity_cond, velocity_ref)

    guided = dict(output)
    guided["velocity"] = velocity_cond.to(device=output["velocity"].device, dtype=output["velocity"].dtype)
    return guided


def _euler_candidate_states(
    state: Dict[str, torch.Tensor],
    *,
    sampler: FlowMatchingSampler,
    tau_t: float,
    tau_prev: float,
    output: Dict[str, torch.Tensor],
    corrupt_field_map: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    dtau = sampler.dtau_from_values(x_t=state["pos"], tau_t=tau_t, tau_prev=tau_prev)
    next_states = {"pos": state["pos"]}
    for field_name in ("shape_features", "dipole_direction"):
        if field_name in state:
            next_states[field_name] = state[field_name]
    for field_name, out_field in STATE_FIELD_SPECS:
        if field_name not in corrupt_field_map:
            continue
        if out_field not in output:
            raise KeyError(
                f"Checkpoint declares corrupt field '{field_name}' but model output is missing '{out_field}'."
            )
        next_states[field_name] = state[field_name] + dtau * output[out_field]
    return next_states


@torch.no_grad()
def sample_example(
    model: torch.nn.Module,
    sampler: FlowMatchingSampler,
    example: Dict[str, np.ndarray],
    r_max: float,
    steps: int,
    device: torch.device,
    seed: int,
    start_scheduler_step: int,
    shape_decode_mode: str,
    shape_decode_library: Dict[str, np.ndarray] | None,
    corrupt_field_map: Dict[str, str],
    corrupt_field_settings: Dict[str, Dict[str, Any]],
    save_intermediates: bool,
    save_velocity_vectors: bool,
    sampler_name: str,
    late_refine_from_step: int,
    late_refine_factor: int,
    linger_step: int,
    linger_count: int,
    clash_guidance: bool,
    clash_guidance_strength: float,
    clash_guidance_max_norm: float,
    clash_guidance_weight_schedule: str,
    clash_guidance_auto_scale: bool,
    clash_guidance_auto_scale_min: float,
    clash_guidance_auto_scale_max: float,
) -> Dict[str, np.ndarray]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    if start_scheduler_step < 0 or start_scheduler_step >= int(sampler.T):
        raise ValueError(
            f"start_scheduler_step must be in [0, {int(sampler.T) - 1}], got {start_scheduler_step}."
        )
    start_tau = float(sampler.scheduler.tau[int(start_scheduler_step)].item())

    state = initial_masked_state(
        example=example,
        generator=generator,
        device=device,
        corrupt_field_map=corrupt_field_map,
        corrupt_field_settings=corrupt_field_settings,
        start_tau=start_tau,
    )
    batch = torch.zeros((int(example["num_nodes"]),), dtype=torch.long, device=device)
    node_types = torch.as_tensor(example["node_types"], dtype=torch.long, device=device).unsqueeze(-1)
    sequence_position = torch.as_tensor(example.get("sequence_position"), dtype=torch.long, device=device)
    if sequence_position.ndim == 1:
        sequence_position = sequence_position.unsqueeze(-1)
    rotations = torch.as_tensor(example["rotations"], dtype=torch.float32, device=device)

    schedule = _build_sampling_schedule(
        sampler,
        steps=steps,
        start_scheduler_step=int(start_scheduler_step),
        late_refine_from_step=late_refine_from_step,
        late_refine_factor=late_refine_factor,
        linger_step=linger_step,
        linger_count=linger_count,
    )
    trajectory: List[Dict[str, np.ndarray]] = []

    if save_intermediates:
        initial_stage = schedule[0]
        trajectory.append(
            _trajectory_snapshot(
                state,
                example,
                shape_decode_mode=shape_decode_mode,
                shape_decode_library=shape_decode_library,
                stage_index=0,
                stage_label="noise",
                scheduler_step=int(initial_stage["scheduler_step"]),
                tau=float(initial_stage["tau"]),
            )
        )

    for step_idx, stage in enumerate(schedule):
        t = int(stage["scheduler_step"])
        t_prev_edge = int(stage["next_scheduler_step"])
        tau_value = float(stage["tau"])
        tau_next_value = float(stage["tau_next"])
        batch_input = _build_model_input(
            state,
            batch=batch,
            node_types=node_types,
            sequence_position=sequence_position,
            rotations=rotations,
            tau_value=tau_value,
            r_max=r_max,
            device=device,
        )
        raw_output = model(batch_input)
        output = _apply_clash_guidance(
            raw_output,
            state,
            tau_value=tau_value,
            guidance_enabled=clash_guidance,
            guidance_strength=clash_guidance_strength,
            guidance_max_norm=clash_guidance_max_norm,
            guidance_weight_schedule=clash_guidance_weight_schedule,
            guidance_auto_scale=clash_guidance_auto_scale,
            guidance_auto_scale_min=clash_guidance_auto_scale_min,
            guidance_auto_scale_max=clash_guidance_auto_scale_max,
        )
        euler_states = _euler_candidate_states(
            state,
            sampler=sampler,
            tau_t=tau_value,
            tau_prev=tau_next_value,
            output=output,
            corrupt_field_map=corrupt_field_map,
        )

        if sampler_name == "heun" and tau_next_value < tau_value - 1e-12:
            provisional_state = _merge_ligand_updates(
                state,
                euler_states,
            )
            batch_input_next = _build_model_input(
                provisional_state,
                batch=batch,
                node_types=node_types,
                sequence_position=sequence_position,
                rotations=rotations,
                tau_value=tau_next_value,
                r_max=r_max,
                device=device,
            )
            raw_output_next = model(batch_input_next)
            output_next = _apply_clash_guidance(
                raw_output_next,
                provisional_state,
                tau_value=tau_next_value,
                guidance_enabled=clash_guidance,
                guidance_strength=clash_guidance_strength,
                guidance_max_norm=clash_guidance_max_norm,
                guidance_weight_schedule=clash_guidance_weight_schedule,
                guidance_auto_scale=clash_guidance_auto_scale,
                guidance_auto_scale_min=clash_guidance_auto_scale_min,
                guidance_auto_scale_max=clash_guidance_auto_scale_max,
            )
            next_states = {"pos": state["pos"]}
            for field_name in ("shape_features", "dipole_direction"):
                if field_name in state:
                    next_states[field_name] = state[field_name]
            for field_name, out_field in STATE_FIELD_SPECS:
                if field_name not in corrupt_field_map:
                    continue
                if out_field not in output_next:
                    raise KeyError(
                        f"Checkpoint declares corrupt field '{field_name}' but model output is missing '{out_field}' on Heun correction."
                    )
                next_states[field_name] = sampler.step_tau(
                    state[field_name],
                    tau_t=tau_value,
                    tau_prev=tau_next_value,
                    velocity_pred=output[out_field],
                    velocity_pred_next=output_next[out_field],
                )
        else:
            next_states = euler_states

        state = _merge_ligand_updates(state, next_states)
        if save_intermediates:
            velocity_vectors = None
            velocity_raw_vectors = None
            if save_velocity_vectors and "velocity" in output:
                if tau_next_value <= 0.0:
                    velocity_vectors = torch.zeros_like(output["velocity"])
                else:
                    velocity_vectors = sampler.dtau_from_values(
                        x_t=state["pos"],
                        tau_t=tau_value,
                        tau_prev=tau_next_value,
                    ).to(dtype=state["pos"].dtype) * output["velocity"]
            if save_velocity_vectors and "velocity" in raw_output:
                if tau_next_value <= 0.0:
                    velocity_raw_vectors = torch.zeros_like(raw_output["velocity"])
                else:
                    velocity_raw_vectors = sampler.dtau_from_values(
                        x_t=state["pos"],
                        tau_t=tau_value,
                        tau_prev=tau_next_value,
                    ).to(dtype=state["pos"].dtype) * raw_output["velocity"]
            if tau_next_value > 0.0:
                stage_label = f"step {step_idx + 1}/{len(schedule)}"
                tau_stage_value = tau_next_value
                scheduler_step = int(t_prev_edge)
            else:
                stage_label = "final"
                tau_stage_value = 0.0
                scheduler_step = -1
            trajectory.append(
                _trajectory_snapshot(
                    state,
                    example,
                    shape_decode_mode=shape_decode_mode,
                    shape_decode_library=shape_decode_library,
                    stage_index=step_idx + 1,
                    stage_label=stage_label,
                    scheduler_step=scheduler_step,
                    tau=tau_stage_value,
                    velocity_vectors=velocity_vectors,
                    velocity_raw_vectors=velocity_raw_vectors,
                )
            )

    decoded_sample = _decode_sampled_structure(
        state,
        example,
        shape_decode_mode=shape_decode_mode,
        shape_decode_library=shape_decode_library,
    )

    original_shape = np.asarray(
        example.get("shape_features_raw", example["shape_features"]),
        dtype=np.float32,
    )
    original_dipoles = np.asarray(
        example.get(
            "brick_dipoles_raw",
            example["dipole_direction"],
        ),
        dtype=np.float32,
    )
    original_dipole_strength = dipole_strengths(original_dipoles).astype(np.float32)
    original_dipole_direction = normalize_dipole_directions(original_dipoles).astype(np.float32)

    sample = {
        **decoded_sample,
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
        "sampling_sampler": np.asarray(str(sampler_name)),
        "sampling_steps": np.asarray(int(steps), dtype=np.int64),
        "sampling_start_step": np.asarray(int(start_scheduler_step), dtype=np.int64),
        "sampling_start_tau": np.asarray(float(start_tau), dtype=np.float32),
        "sampling_late_refine_from_step": np.asarray(int(late_refine_from_step), dtype=np.int64),
        "sampling_late_refine_factor": np.asarray(int(late_refine_factor), dtype=np.int64),
        "sampling_linger_step": np.asarray(int(linger_step), dtype=np.int64),
        "sampling_linger_count": np.asarray(int(linger_count), dtype=np.int64),
        "sampling_clash_guidance": np.asarray(bool(clash_guidance)),
        "sampling_clash_guidance_strength": np.asarray(float(clash_guidance_strength), dtype=np.float32),
        "sampling_clash_guidance_max_norm": np.asarray(float(clash_guidance_max_norm), dtype=np.float32),
        "sampling_clash_guidance_weight_schedule": np.asarray(str(clash_guidance_weight_schedule)),
        "sampling_clash_guidance_auto_scale": np.asarray(bool(clash_guidance_auto_scale)),
        "sampling_clash_guidance_auto_scale_min": np.asarray(float(clash_guidance_auto_scale_min), dtype=np.float32),
        "sampling_clash_guidance_auto_scale_max": np.asarray(float(clash_guidance_auto_scale_max), dtype=np.float32),
        "sampling_shape_decode_mode": np.asarray(str(shape_decode_mode)),
    }
    if save_intermediates:
        sample["intermediate_states"] = np.asarray(trajectory, dtype=object)
    return sample


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
    corrupt_field_settings = infer_corrupt_field_settings(config)
    corrupt_field_map = infer_corrupt_field_map(config)

    try:
        r_max = float(config[AtomicDataDict.R_MAX_KEY])
    except KeyError as exc:
        raise KeyError(f"Could not find '{AtomicDataDict.R_MAX_KEY}' in the checkpoint config.") from exc

    tmax = infer_model_tmax(config)
    if tmax is None:
        raise ValueError("Could not determine Tmax from the checkpoint config.")

    if args.sampler == "heun":
        sampler = FlowMatchingHeunSampler(FlowMatchingScheduler(T=tmax)).to(args.device)
    else:
        sampler = FlowMatchingSampler(FlowMatchingScheduler(T=tmax)).to(args.device)
    if int(args.start_step) < 0:
        start_scheduler_step = int(tmax) - 1
    else:
        start_scheduler_step = int(args.start_step)
    if start_scheduler_step < 0 or start_scheduler_step >= int(tmax):
        raise ValueError(f"--start-step must be in [0, {int(tmax) - 1}] or -1, got {args.start_step}.")

    model.to(args.device)
    model.eval()

    with np.load(args.input, allow_pickle=True) as raw:
        data = {key: raw[key] for key in raw.files}

    requested_decode_mode = str(args.shape_decode_mode).strip().lower()
    shape_decode_library = None
    if requested_decode_mode == "input_knn":
        shape_decode_mode = "input_knn"
        shape_decode_library = _build_input_knn_decode_library(
            data,
            max_candidates=int(args.shape_decode_max_candidates),
            seed=int(args.seed),
        )
        print(f"Shape decode mode: input_knn (candidates={int(shape_decode_library['features'].shape[0])}).")
    elif requested_decode_mode == "keep_original":
        shape_decode_mode = requested_decode_mode
        print(f"Shape decode mode: {shape_decode_mode}.")
    else:
        raise ValueError(f"Unsupported --shape-decode-mode: {requested_decode_mode}")

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
            start_scheduler_step=start_scheduler_step,
            shape_decode_mode=shape_decode_mode,
            shape_decode_library=shape_decode_library,
            corrupt_field_map=corrupt_field_map,
            corrupt_field_settings=corrupt_field_settings,
            save_intermediates=bool(args.save_intermediates),
            save_velocity_vectors=bool(args.save_velocity_vectors),
            sampler_name=str(args.sampler),
            late_refine_from_step=int(args.late_refine_from_step),
            late_refine_factor=int(args.late_refine_factor),
            linger_step=int(args.linger_step),
            linger_count=int(args.linger_count),
            clash_guidance=bool(args.clash_guidance),
            clash_guidance_strength=float(args.clash_guidance_strength),
            clash_guidance_max_norm=float(args.clash_guidance_max_norm),
            clash_guidance_weight_schedule=str(args.clash_guidance_weight_schedule),
            clash_guidance_auto_scale=bool(args.clash_guidance_auto_scale),
            clash_guidance_auto_scale_min=float(args.clash_guidance_auto_scale_min),
            clash_guidance_auto_scale_max=float(args.clash_guidance_auto_scale_max),
        )
        sample["conditioning_example_index"] = np.asarray(example_index, dtype=np.int64)
        samples.append(enrich_from_canonical_source(sample, source_samples=source_samples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_samples(args.output, samples)
    print(f"Saved {len(samples)} sampled LEGO assemblies to {args.output}")

    if args.save_metrics:
        metrics_path = args.metrics_json
        if metrics_path is None:
            metrics_path = args.output.with_name(f"{args.output.stem}_metrics.json")
        report = build_evaluation_report(samples)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Saved sampling metrics to {metrics_path}")


if __name__ == "__main__":
    main()
