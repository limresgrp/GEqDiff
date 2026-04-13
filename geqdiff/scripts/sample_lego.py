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
    combine_shape_irreps,
    normalize_dipole_directions,
    normalize_rows,
)
from geqdiff.utils.feature_utils import invert_scalar_normalization
from geqdiff.scripts.evaluate_lego_samples import build_evaluation_report
from geqtrain.train.components.checkpointing import CheckpointHandler
from lego.utils import decode_brick_signatures, load_samples, save_samples


STATE_FIELD_SPECS = (
    ("pos", "velocity"),
    ("shape_scalar_features", "shape_scalar_velocity"),
    ("shape_equiv_features", "shape_equiv_velocity"),
    ("dipole_strength", "dipole_strength_velocity"),
    ("dipole_direction", "dipole_direction_velocity"),
)


def _default_noise_like(
    field_name: str,
    data: torch.Tensor,
    *,
    num_nodes: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if field_name == "shape_equiv_features":
        return _random_shape_equiv_noise(num_nodes, generator=generator, device=device)
    if field_name == "dipole_direction":
        return _random_unit_vectors((num_nodes, int(data.shape[-1])), generator=generator, device=device)
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
        "--clash-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        default=True,
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
        "--cohesion-guidance-strength",
        type=float,
        default=0.0,
        help=(
            "Optional additional guidance weight for a soft contact-count cohesion term. "
            "This helps avoid reducing clashes by simply fragmenting the diffused assembly."
        ),
    )
    parser.add_argument(
        "--cohesion-guidance-target-contacts",
        type=float,
        default=1.5,
        help="Desired soft contact count per diffused brick for the cohesion guidance term.",
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


def _decode_sampled_structure(
    state: Dict[str, torch.Tensor],
    example: Dict[str, np.ndarray],
    scalar_normalization: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
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
    decoded_library = decode_brick_signatures(sampled_shape)
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
    scalar_normalization: Dict[str, Dict[str, np.ndarray]],
    *,
    stage_index: int,
    stage_label: str,
    scheduler_step: int,
    tau: float,
) -> Dict[str, np.ndarray]:
    snapshot = _decode_sampled_structure(state, example, scalar_normalization)
    snapshot["stage_index"] = np.asarray(stage_index, dtype=np.int64)
    snapshot["stage_label"] = np.asarray(stage_label)
    snapshot["scheduler_step"] = np.asarray(scheduler_step, dtype=np.int64)
    snapshot["tau"] = np.asarray(tau, dtype=np.float32)
    return snapshot


def _build_sampling_schedule(
    sampler: FlowMatchingSampler,
    *,
    steps: int,
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

    start_step = sampler.T - 1
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
):
    ligand_mask = torch.as_tensor(example["ligand_mask"], dtype=torch.bool, device=device)
    pocket_mask = torch.as_tensor(example["pocket_mask"], dtype=torch.bool, device=device)
    state: Dict[str, torch.Tensor] = {
        "ligand_mask": ligand_mask,
        "pocket_mask": pocket_mask,
        "centroid": torch.zeros((1, 3), dtype=torch.float32, device=device),
    }

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

        noise = _default_noise_like(field_name, data, num_nodes=num_nodes, generator=generator, device=device)
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
            current[ligand_mask] = noise[ligand_mask]

        fixed_key = "pos_fixed" if field_name == "pos" else f"{field_name}_fixed"
        state[fixed_key] = centered
        state[field_name] = current

    return state


def _build_model_input(
    state: Dict[str, torch.Tensor],
    *,
    batch: torch.Tensor,
    node_types: torch.Tensor,
    rotations: torch.Tensor,
    tau_value: float,
    r_max: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    ligand_mask_column = state["ligand_mask"].unsqueeze(-1)
    tau = torch.full((1, 1), float(tau_value), device=device, dtype=torch.float32)
    edge_index = build_edge_index(state["pos"], cutoff=r_max)
    return {
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


def _apply_state_projection(
    next_states: Dict[str, torch.Tensor],
    *,
    project_normalized_states: bool,
) -> Dict[str, torch.Tensor]:
    projected = dict(next_states)
    if project_normalized_states:
        projected["shape_equiv_features"] = _project_shape_equiv(projected["shape_equiv_features"])
        projected["dipole_direction"] = _project_dipole_direction(projected["dipole_direction"])
    return projected


def _merge_ligand_updates(
    state: Dict[str, torch.Tensor],
    next_states: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    ligand_mask_column = state["ligand_mask"].unsqueeze(-1)
    merged = dict(state)
    merged["pos"] = torch.where(ligand_mask_column, next_states["pos"], state["pos_fixed"])
    merged["shape_scalar_features"] = torch.where(ligand_mask_column, next_states["shape_scalar_features"], state["shape_scalar_fixed"])
    merged["shape_equiv_features"] = torch.where(ligand_mask_column, next_states["shape_equiv_features"], state["shape_equiv_fixed"])
    merged["dipole_strength"] = torch.where(ligand_mask_column, next_states["dipole_strength"], state["dipole_strength_fixed"])
    merged["dipole_direction"] = torch.where(ligand_mask_column, next_states["dipole_direction"], state["dipole_direction_fixed"])
    return merged


def _normalize_feature_rows_torch(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.numel() == 0:
        return values
    norms = torch.linalg.norm(values, dim=-1, keepdim=True)
    safe = torch.clamp(norms, min=eps)
    normalized = values / safe
    return torch.where(norms > eps, normalized, torch.zeros_like(values))


def _combine_shape_irreps_torch(
    shape_scalars: torch.Tensor,
    shape_equivariants: torch.Tensor,
) -> torch.Tensor:
    l0 = shape_scalars[..., 0:1]
    l1_mag = torch.clamp(shape_scalars[..., 1:2], min=0.0)
    l2_mag = torch.clamp(shape_scalars[..., 2:3], min=0.0)
    l3_mag = torch.clamp(shape_scalars[..., 3:4], min=0.0)

    l1_dir = _normalize_feature_rows_torch(shape_equivariants[..., 0:3])
    l2_dir = _normalize_feature_rows_torch(shape_equivariants[..., 3:8])
    l3_dir = _normalize_feature_rows_torch(shape_equivariants[..., 8:15])
    return torch.cat([l0, l1_dir * l1_mag, l2_dir * l2_mag, l3_dir * l3_mag], dim=-1)


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
    return _normalize_feature_rows_torch(directions)


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
    penalties = torch.nn.functional.softplus(float(clash_sharpness) * overlap) / float(clash_sharpness)
    if penalties.numel() == 0:
        return pred_pos_0.new_zeros(())
    finite = torch.isfinite(penalties)
    if not torch.any(finite):
        return pred_pos_0.new_zeros(())
    return penalties[finite].mean()


def _shape_aware_cohesion_energy(
    pred_pos_0: torch.Tensor,
    pred_shape_coeffs_detached: torch.Tensor,
    ligand_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    *,
    pair_cutoff: float = 3.5,
    contact_margin: float = 0.20,
    contact_sharpness: float = 8.0,
    target_contacts: float = 1.5,
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
    lig_i = lig_i[pair_mask]
    lig_j = lig_j[pair_mask]
    deltas = pred_pos_0[idx_j] - pred_pos_0[idx_i]
    distances = torch.linalg.norm(deltas, dim=-1)
    valid = torch.isfinite(distances) & (distances > 1e-6) & (distances < float(pair_cutoff))
    if not torch.any(valid):
        return pred_pos_0.new_zeros(())

    idx_i = idx_i[valid]
    idx_j = idx_j[valid]
    lig_i = lig_i[valid]
    lig_j = lig_j[valid]
    deltas = deltas[valid]
    distances = distances[valid]
    directions = deltas / distances.unsqueeze(-1)

    extent_i = _directional_extent(pred_shape_coeffs_detached[idx_i], directions)
    extent_j = _directional_extent(pred_shape_coeffs_detached[idx_j], -directions)
    contact_logits = float(contact_sharpness) * (extent_i + extent_j + float(contact_margin) - distances)
    contact_scores = torch.sigmoid(contact_logits)
    finite = torch.isfinite(contact_scores)
    if not torch.any(finite):
        return pred_pos_0.new_zeros(())

    idx_i = idx_i[finite]
    idx_j = idx_j[finite]
    lig_i = lig_i[finite]
    lig_j = lig_j[finite]
    contact_scores = contact_scores[finite]

    per_node_contacts = pred_pos_0.new_zeros((num_nodes,), dtype=torch.float32)
    if torch.any(lig_i):
        per_node_contacts.scatter_add_(0, idx_i[lig_i], contact_scores[lig_i])
    if torch.any(lig_j):
        per_node_contacts.scatter_add_(0, idx_j[lig_j], contact_scores[lig_j])

    ligand_contacts = per_node_contacts[ligand_mask]
    if ligand_contacts.numel() == 0:
        return pred_pos_0.new_zeros(())

    deficits = torch.nn.functional.relu(float(target_contacts) - ligand_contacts)
    return torch.mean(deficits.square())


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
    cohesion_guidance_strength: float,
    cohesion_guidance_target_contacts: float,
) -> Dict[str, torch.Tensor]:
    if (not guidance_enabled) or guidance_strength <= 0.0 or tau_value <= 1e-8:
        return output
    if not all(field in output for field in ("velocity", "shape_scalar_velocity", "shape_equiv_velocity")):
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

    with torch.enable_grad():
        pred_pos_0 = (
            state["pos"].detach().to(dtype=torch.float32)
            - tau_value * output["velocity"].detach().to(dtype=torch.float32)
        ).clone().requires_grad_(True)
        pred_shape_scalar = (
            state["shape_scalar_features"].detach().to(dtype=torch.float32)
            - tau_value * output["shape_scalar_velocity"].detach().to(dtype=torch.float32)
        )
        pred_shape_equiv = (
            state["shape_equiv_features"].detach().to(dtype=torch.float32)
            - tau_value * output["shape_equiv_velocity"].detach().to(dtype=torch.float32)
        )
        pred_shape_coeffs = _combine_shape_irreps_torch(pred_shape_scalar, pred_shape_equiv).detach()
        clash_energy = _shape_aware_clash_energy(
            pred_pos_0,
            pred_shape_coeffs,
            state["ligand_mask"].detach(),
            state["pocket_mask"].detach(),
        )
        guidance_energy = clash_energy
        if float(cohesion_guidance_strength) > 0.0:
            cohesion_energy = _shape_aware_cohesion_energy(
                pred_pos_0,
                pred_shape_coeffs,
                state["ligand_mask"].detach(),
                state["pocket_mask"].detach(),
                target_contacts=float(cohesion_guidance_target_contacts),
            )
            guidance_energy = guidance_energy + float(cohesion_guidance_strength) * cohesion_energy
        if float(guidance_energy.detach().item()) <= 0.0:
            return output
        grad_pos = torch.autograd.grad(guidance_energy, pred_pos_0, allow_unused=False)[0]

    ligand_mask = state["ligand_mask"].detach().unsqueeze(-1)
    grad_pos = torch.where(ligand_mask, grad_pos, torch.zeros_like(grad_pos))
    if guidance_max_norm > 0.0:
        grad_norm = torch.linalg.norm(grad_pos, dim=-1, keepdim=True)
        clip = torch.clamp(float(guidance_max_norm) / torch.clamp(grad_norm, min=1e-8), max=1.0)
        grad_pos = grad_pos * clip

    scale_multiplier = 1.0
    if guidance_auto_scale:
        ligand_mask_flat = state["ligand_mask"].detach().reshape(-1)
        velocity_ref = output["velocity"].detach().to(dtype=torch.float32)
        velocity_ref = torch.where(ligand_mask_flat.unsqueeze(-1), velocity_ref, torch.zeros_like(velocity_ref))
        vel_norm = torch.linalg.norm(velocity_ref[ligand_mask_flat], dim=-1)
        guide_norm = torch.linalg.norm(grad_pos[ligand_mask_flat], dim=-1)
        finite_mask = torch.isfinite(vel_norm) & torch.isfinite(guide_norm) & (guide_norm > 1e-8)
        if torch.any(finite_mask):
            vel_rms = torch.sqrt(torch.mean(vel_norm[finite_mask] ** 2))
            guide_rms = torch.sqrt(torch.mean(guide_norm[finite_mask] ** 2))
            scale_multiplier = float((vel_rms / torch.clamp(guide_rms, min=1e-8)).item())
            scale_multiplier = float(
                np.clip(
                    scale_multiplier,
                    float(guidance_auto_scale_min),
                    float(guidance_auto_scale_max),
                )
            )

    guided = dict(output)
    guided["velocity"] = output["velocity"] + float(guidance_strength) * float(schedule_weight) * float(scale_multiplier) * grad_pos.to(
        device=output["velocity"].device,
        dtype=output["velocity"].dtype,
    )
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
    scalar_normalization: Dict[str, Dict[str, np.ndarray]],
    project_normalized_states: bool,
    corrupt_field_map: Dict[str, str],
    corrupt_field_settings: Dict[str, Dict[str, Any]],
    save_intermediates: bool,
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
    cohesion_guidance_strength: float,
    cohesion_guidance_target_contacts: float,
) -> Dict[str, np.ndarray]:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    state = initial_masked_state(
        example=example,
        generator=generator,
        device=device,
        corrupt_field_map=corrupt_field_map,
        corrupt_field_settings=corrupt_field_settings,
    )
    batch = torch.zeros((int(example["num_nodes"]),), dtype=torch.long, device=device)
    node_types = torch.as_tensor(example["node_types"], dtype=torch.long, device=device).unsqueeze(-1)
    rotations = torch.as_tensor(example["rotations"], dtype=torch.float32, device=device)

    schedule = _build_sampling_schedule(
        sampler,
        steps=steps,
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
                scalar_normalization,
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
            rotations=rotations,
            tau_value=tau_value,
            r_max=r_max,
            device=device,
        )
        output = _apply_clash_guidance(
            model(batch_input),
            state,
            tau_value=tau_value,
            guidance_enabled=clash_guidance,
            guidance_strength=clash_guidance_strength,
            guidance_max_norm=clash_guidance_max_norm,
            guidance_weight_schedule=clash_guidance_weight_schedule,
            guidance_auto_scale=clash_guidance_auto_scale,
            guidance_auto_scale_min=clash_guidance_auto_scale_min,
            guidance_auto_scale_max=clash_guidance_auto_scale_max,
            cohesion_guidance_strength=cohesion_guidance_strength,
            cohesion_guidance_target_contacts=cohesion_guidance_target_contacts,
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
                _apply_state_projection(
                    euler_states,
                    project_normalized_states=project_normalized_states,
                ),
            )
            batch_input_next = _build_model_input(
                provisional_state,
                batch=batch,
                node_types=node_types,
                rotations=rotations,
                tau_value=tau_next_value,
                r_max=r_max,
                device=device,
            )
            output_next = _apply_clash_guidance(
                model(batch_input_next),
                provisional_state,
                tau_value=tau_next_value,
                guidance_enabled=clash_guidance,
                guidance_strength=clash_guidance_strength,
                guidance_max_norm=clash_guidance_max_norm,
                guidance_weight_schedule=clash_guidance_weight_schedule,
                guidance_auto_scale=clash_guidance_auto_scale,
                guidance_auto_scale_min=clash_guidance_auto_scale_min,
                guidance_auto_scale_max=clash_guidance_auto_scale_max,
                cohesion_guidance_strength=cohesion_guidance_strength,
                cohesion_guidance_target_contacts=cohesion_guidance_target_contacts,
            )
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

        state = _merge_ligand_updates(
            state,
            _apply_state_projection(
                next_states,
                project_normalized_states=project_normalized_states,
            ),
        )
        if save_intermediates:
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
                    scalar_normalization,
                    stage_index=step_idx + 1,
                    stage_label=stage_label,
                    scheduler_step=scheduler_step,
                    tau=tau_stage_value,
                )
            )

    decoded_sample = _decode_sampled_structure(state, example, scalar_normalization)

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
        "sampling_cohesion_guidance_strength": np.asarray(float(cohesion_guidance_strength), dtype=np.float32),
        "sampling_cohesion_guidance_target_contacts": np.asarray(float(cohesion_guidance_target_contacts), dtype=np.float32),
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
    project_normalized_states = infer_project_normalized_states(config)
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
            project_normalized_states=project_normalized_states,
            corrupt_field_map=corrupt_field_map,
            corrupt_field_settings=corrupt_field_settings,
            save_intermediates=bool(args.save_intermediates),
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
            cohesion_guidance_strength=float(args.cohesion_guidance_strength),
            cohesion_guidance_target_contacts=float(args.cohesion_guidance_target_contacts),
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
