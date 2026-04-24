from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.utils.contact_utils import build_brick_geometries, detect_brick_contacts
from geqdiff.utils.dipole_utils import (
    DipoleAssignmentConfig,
    dipole_strengths,
    evaluate_contact_energy,
    normalize_dipole_directions,
    split_shape_irreps,
)
from lego.score_utils import evaluate_sample_scores_by_anchor_mode
from lego.utils import load_samples

SHAPE_MISMATCH_THRESHOLD = 65.0
DIPOLE_MISMATCH_THRESHOLD = 65.0
RELATIVE_VALIDITY_THRESHOLD = 55.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate canonical LEGO sampled datasets using the same polar-contact energy "
            "used for dipole assignment, plus geometric comparison against original structures "
            "when available."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Canonical LEGO NPZ to evaluate.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluated samples.")
    parser.add_argument(
        "--large-shift-threshold",
        type=float,
        default=1.8,
        help="Mean diffused-anchor shift threshold used to flag a large placement error.",
    )
    parser.add_argument(
        "--energy-regression-threshold",
        type=float,
        default=0.8,
        help="Sampled-minus-original energy threshold used to flag a polar-energy regression.",
    )
    return parser.parse_args()


def _structure_from_sample(sample: Dict[str, Any], prefix: str = "") -> Dict[str, np.ndarray]:
    return {
        "brick_anchors": np.asarray(sample[f"{prefix}brick_anchors"], dtype=np.float32),
        "brick_rotations": np.asarray(sample[f"{prefix}brick_rotations"], dtype=np.float32),
        "brick_types": np.asarray(sample[f"{prefix}brick_types"]),
        "brick_dipoles": np.asarray(sample.get(f"{prefix}brick_dipoles", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
    }


def _evaluate_structure(structure: Dict[str, np.ndarray], config: DipoleAssignmentConfig) -> Dict[str, Any]:
    geometries = build_brick_geometries(structure)
    contacts = detect_brick_contacts(geometries)
    energy = evaluate_contact_energy(
        dipoles=np.asarray(structure["brick_dipoles"], dtype=np.float32),
        all_face_contact_pairs=np.asarray(contacts["all_face_contact_pairs"], dtype=np.int64),
        all_face_contact_dirs=np.asarray(contacts["all_face_contact_dirs"], dtype=np.float32),
        config=config,
    )
    component_id = np.asarray(contacts["component_id"], dtype=np.int32)
    num_components = int(component_id.max()) + 1 if component_id.size > 0 else 0
    return {
        "num_bricks": int(len(structure["brick_anchors"])),
        "num_contact_pairs": int(np.asarray(contacts["contact_pairs"], dtype=np.int64).shape[0]),
        "num_face_contacts": int(np.asarray(contacts["all_face_contact_pairs"], dtype=np.int64).shape[0]),
        "num_components": num_components,
        **energy,
    }


def _safe_evaluate_structure(structure: Dict[str, np.ndarray], config: DipoleAssignmentConfig) -> Dict[str, Any]:
    try:
        payload = _evaluate_structure(structure, config=config)
        payload["valid_geometry"] = True
        payload["error"] = ""
        return payload
    except Exception as exc:
        return {
            "valid_geometry": False,
            "error": f"{type(exc).__name__}: {exc}",
            "num_bricks": int(len(structure["brick_anchors"])),
            "num_contact_pairs": 0,
            "num_face_contacts": 0,
            "num_components": 0,
            "total_energy": float("nan"),
            "polar_cost": float("nan"),
            "contact_energy": float("nan"),
            "num_attractive_contacts": 0,
            "num_repulsive_contacts": 0,
            "num_neutral_contacts": 0,
            "mean_energy_per_face": float("nan"),
        }


def _compare_against_original(sample: Dict[str, Any], sampled_eval: Dict[str, Any], original_eval: Dict[str, Any]) -> Dict[str, Any]:
    sampled_anchors = np.asarray(sample["brick_anchors"], dtype=np.float32)
    original_anchors = np.asarray(sample["original_brick_anchors"], dtype=np.float32)
    shifts = np.linalg.norm(sampled_anchors - original_anchors, axis=-1)
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((len(shifts),), dtype=bool)), dtype=bool).reshape(-1)
    fixed_mask = ~mask

    diffused_shift_mean = float(shifts[mask].mean()) if mask.any() else 0.0
    diffused_shift_max = float(shifts[mask].max()) if mask.any() else 0.0
    fixed_shift_mean = float(shifts[fixed_mask].mean()) if fixed_mask.any() else 0.0
    fixed_shift_max = float(shifts[fixed_mask].max()) if fixed_mask.any() else 0.0

    energy_delta = float(sampled_eval["total_energy"] - original_eval["total_energy"])
    repulsive_delta = int(sampled_eval["num_repulsive_contacts"] - original_eval["num_repulsive_contacts"])
    attractive_delta = int(sampled_eval["num_attractive_contacts"] - original_eval["num_attractive_contacts"])

    return {
        "num_diffused": int(mask.sum()),
        "diffused_shift_mean": diffused_shift_mean,
        "diffused_shift_max": diffused_shift_max,
        "fixed_shift_mean": fixed_shift_mean,
        "fixed_shift_max": fixed_shift_max,
        "energy_delta": energy_delta,
        "repulsive_delta": repulsive_delta,
        "attractive_delta": attractive_delta,
    }


def _mean_angular_error_deg(sampled: np.ndarray, original: np.ndarray, valid_mask: np.ndarray) -> float:
    if not np.any(valid_mask):
        return 0.0
    sampled_unit = sampled[valid_mask]
    original_unit = original[valid_mask]
    cosine = np.sum(sampled_unit * original_unit, axis=-1)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)).mean())


def _compare_descriptors(sample: Dict[str, Any]) -> Dict[str, Any]:
    sampled_shape = np.asarray(sample["brick_features"], dtype=np.float32)
    original_shape = np.asarray(sample["original_brick_features"], dtype=np.float32)
    sampled_dipoles = np.asarray(sample.get("brick_dipoles", np.zeros_like(sampled_shape[:, :3])), dtype=np.float32)
    original_dipoles = np.asarray(sample.get("original_brick_dipoles", np.zeros_like(sampled_shape[:, :3])), dtype=np.float32)

    if "sampled_brick_mask" in sample:
        compare_mask = np.asarray(sample["sampled_brick_mask"], dtype=bool).reshape(-1)
    else:
        compare_mask = np.ones((sampled_shape.shape[0],), dtype=bool)

    if not np.any(compare_mask):
        compare_mask = np.ones((sampled_shape.shape[0],), dtype=bool)

    sampled_shape = sampled_shape[compare_mask]
    original_shape = original_shape[compare_mask]
    sampled_dipoles = sampled_dipoles[compare_mask]
    original_dipoles = original_dipoles[compare_mask]

    sampled_shape_scalar, sampled_shape_equiv = split_shape_irreps(sampled_shape)
    original_shape_scalar, original_shape_equiv = split_shape_irreps(original_shape)

    sampled_l1 = sampled_shape_equiv[:, 0:3]
    original_l1 = original_shape_equiv[:, 0:3]
    sampled_l2 = sampled_shape_equiv[:, 3:8]
    original_l2 = original_shape_equiv[:, 3:8]
    sampled_l3 = sampled_shape_equiv[:, 8:15]
    original_l3 = original_shape_equiv[:, 8:15]

    sampled_strength = dipole_strengths(sampled_dipoles).reshape(-1)
    original_strength = dipole_strengths(original_dipoles).reshape(-1)
    sampled_direction = normalize_dipole_directions(sampled_dipoles)
    original_direction = normalize_dipole_directions(original_dipoles)

    l1_valid = (np.linalg.norm(sampled_l1, axis=-1) > 1e-6) & (np.linalg.norm(original_l1, axis=-1) > 1e-6)
    l2_valid = (np.linalg.norm(sampled_l2, axis=-1) > 1e-6) & (np.linalg.norm(original_l2, axis=-1) > 1e-6)
    l3_valid = (np.linalg.norm(sampled_l3, axis=-1) > 1e-6) & (np.linalg.norm(original_l3, axis=-1) > 1e-6)
    dipole_valid = (sampled_strength > 1e-6) & (original_strength > 1e-6)

    return {
        "num_compared": int(compare_mask.sum()),
        "shape_mse": float(np.mean((sampled_shape - original_shape) ** 2)),
        "shape_scalar_mse": float(np.mean((sampled_shape_scalar - original_shape_scalar) ** 2)),
        "shape_equiv_mse": float(np.mean((sampled_shape_equiv - original_shape_equiv) ** 2)),
        "shape_l1_mse": float(np.mean((sampled_l1 - original_l1) ** 2)),
        "shape_l2_mse": float(np.mean((sampled_l2 - original_l2) ** 2)),
        "shape_l3_mse": float(np.mean((sampled_l3 - original_l3) ** 2)),
        "shape_l1_angle_deg": _mean_angular_error_deg(sampled_l1, original_l1, l1_valid),
        "shape_l2_angle_deg": _mean_angular_error_deg(sampled_l2, original_l2, l2_valid),
        "shape_l3_angle_deg": _mean_angular_error_deg(sampled_l3, original_l3, l3_valid),
        "dipole_strength_mse": float(np.mean((sampled_strength - original_strength) ** 2)),
        "dipole_direction_angle_deg": _mean_angular_error_deg(sampled_direction, original_direction, dipole_valid),
    }


def _failure_labels(
    sampled_eval: Dict[str, Any],
    score_card: Dict[str, Any] | None,
    compare: Dict[str, Any] | None,
    large_shift_threshold: float,
    energy_regression_threshold: float,
) -> List[str]:
    labels: List[str] = []
    if not sampled_eval["valid_geometry"]:
        labels.append("invalid_geometry")
        return labels
    if sampled_eval["num_repulsive_contacts"] > sampled_eval["num_attractive_contacts"]:
        labels.append("repulsion_dominated")
    if sampled_eval["num_components"] > 1:
        labels.append("disconnected")
    if score_card is not None:
        sampled_scores = score_card.get("sampled", {}).get("scores", {})
        if float(sampled_scores.get("shape", 100.0)) < SHAPE_MISMATCH_THRESHOLD:
            labels.append("shape_mismatch")
        if float(sampled_scores.get("dipoles", 100.0)) < DIPOLE_MISMATCH_THRESHOLD:
            labels.append("dipole_mismatch")
    if compare is not None:
        if compare["fixed_shift_max"] > 1e-4:
            labels.append("fixed_bricks_moved")
        if compare["diffused_shift_mean"] > large_shift_threshold:
            labels.append("large_diffused_shift")
        if compare["energy_delta"] > energy_regression_threshold:
            labels.append("energy_regression")
    return labels


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_evaluation_report(
    samples: List[Dict[str, Any]],
    *,
    large_shift_threshold: float = 1.8,
    energy_regression_threshold: float = 0.8,
) -> Dict[str, Any]:
    config = DipoleAssignmentConfig()
    records: List[Dict[str, Any]] = []
    failure_counts: Dict[str, int] = {}

    for sample_index, sample in enumerate(samples):
        sampled_structure = _structure_from_sample(sample, prefix="")
        sampled_eval = _safe_evaluate_structure(sampled_structure, config=config)
        score_cards = evaluate_sample_scores_by_anchor_mode(sample, dipole_config=config)
        score_card = score_cards["voxelized"]

        record: Dict[str, Any] = {
            "sample_index": int(sample_index),
            "sampled": sampled_eval,
            "score_card": score_card,
            "score_cards": score_cards,
            "score_card_raw": score_cards["raw"],
            "score_card_voxelized": score_cards["voxelized"],
        }
        compare = None
        if "original_brick_anchors" in sample:
            original_structure = _structure_from_sample(sample, prefix="original_")
            original_eval = _safe_evaluate_structure(original_structure, config=config)
            compare = _compare_against_original(sample, sampled_eval, original_eval)
            record["original"] = original_eval
            record["compare"] = compare
        if "original_brick_features" in sample and "brick_features" in sample:
            record["descriptor_compare"] = _compare_descriptors(sample)

        failures = _failure_labels(
            sampled_eval=sampled_eval,
            score_card=score_card,
            compare=compare,
            large_shift_threshold=float(large_shift_threshold),
            energy_regression_threshold=float(energy_regression_threshold),
        )
        record["failures"] = failures
        for label in failures:
            failure_counts[label] = failure_counts.get(label, 0) + 1
        records.append(record)

    sampled_energies = [
        float(record["sampled"]["total_energy"])
        for record in records
        if record["sampled"]["valid_geometry"] and np.isfinite(record["sampled"]["total_energy"])
    ]
    diffused_means = [
        float(record["compare"]["diffused_shift_mean"])
        for record in records
        if "compare" in record
    ]
    energy_deltas = [
        float(record["compare"]["energy_delta"])
        for record in records
        if "compare" in record and np.isfinite(record["compare"]["energy_delta"])
    ]
    shape_mses = [
        float(record["descriptor_compare"]["shape_mse"])
        for record in records
        if "descriptor_compare" in record and np.isfinite(record["descriptor_compare"]["shape_mse"])
    ]
    dipole_angles = [
        float(record["descriptor_compare"]["dipole_direction_angle_deg"])
        for record in records
        if "descriptor_compare" in record and np.isfinite(record["descriptor_compare"]["dipole_direction_angle_deg"])
    ]
    validity_scores = [float(record["score_card"]["sampled"]["scores"]["validity"]) for record in records if "score_card" in record]
    shape_scores = [float(record["score_card"]["sampled"]["scores"].get("shape", float("nan"))) for record in records if "score_card" in record]
    dipole_scores = [float(record["score_card"]["sampled"]["scores"]["dipoles"]) for record in records if "score_card" in record]
    pose_scores = [float(record["score_card"]["sampled"]["scores"].get("pose", float("nan"))) for record in records if "score_card" in record]
    validity_scores_raw = [
        float(record["score_card_raw"]["sampled"]["scores"]["validity"])
        for record in records
        if "score_card_raw" in record
    ]
    shape_scores_raw = [
        float(record["score_card_raw"]["sampled"]["scores"].get("shape", float("nan")))
        for record in records
        if "score_card_raw" in record
    ]
    dipole_scores_raw = [
        float(record["score_card_raw"]["sampled"]["scores"]["dipoles"])
        for record in records
        if "score_card_raw" in record
    ]
    pose_scores_raw = [
        float(record["score_card_raw"]["sampled"]["scores"].get("pose", float("nan")))
        for record in records
        if "score_card_raw" in record
    ]

    parsed_geometries = int(sum(1 for record in records if record["sampled"]["valid_geometry"]))
    valid_relative_geometries = int(
        sum(
            1
            for record in records
            if float(record["score_card"]["sampled"]["scores"].get("validity", 0.0)) >= RELATIVE_VALIDITY_THRESHOLD
        )
    )

    summary = {
        "num_samples": int(len(records)),
        "parsed_geometries": parsed_geometries,
        "valid_relative_geometries": valid_relative_geometries,
        "valid_geometries": valid_relative_geometries,
        "mean_sampled_energy": _mean(sampled_energies),
        "mean_diffused_shift": _mean(diffused_means),
        "mean_energy_delta": _mean(energy_deltas),
        "mean_shape_mse": _mean(shape_mses),
        "mean_dipole_direction_angle_deg": _mean(dipole_angles),
        "mean_validity_score": _mean(validity_scores),
        "mean_shape_score": _mean([v for v in shape_scores if np.isfinite(v)]),
        "mean_dipole_score": _mean(dipole_scores),
        "mean_pose_score": _mean([v for v in pose_scores if np.isfinite(v)]),
        "mean_validity_score_raw": _mean(validity_scores_raw),
        "mean_shape_score_raw": _mean([v for v in shape_scores_raw if np.isfinite(v)]),
        "mean_dipole_score_raw": _mean(dipole_scores_raw),
        "mean_pose_score_raw": _mean([v for v in pose_scores_raw if np.isfinite(v)]),
        "failure_counts": failure_counts,
    }
    return {"summary": summary, "records": records}


def print_evaluation_report(report: Dict[str, Any], *, input_path: Path | str) -> None:
    summary = report["summary"]
    records = report["records"]

    print("--- LEGO Sample Evaluation ---")
    print(f"Input: {input_path}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Parsed geometries: {summary['parsed_geometries']}/{summary['num_samples']}")
    print(f"Relative-valid geometries: {summary['valid_relative_geometries']}/{summary['num_samples']}")
    print(f"Mean sampled energy: {summary['mean_sampled_energy']:.3f}")
    print(
        "Mean validity / shape / dipole / pose scores: "
        f"{summary['mean_validity_score']:.2f} / "
        f"{summary['mean_shape_score']:.2f} / "
        f"{summary['mean_dipole_score']:.2f} / "
        f"{summary['mean_pose_score']:.2f}"
    )
    print(
        "Mean raw validity / shape / dipole / pose: "
        f"{summary['mean_validity_score_raw']:.2f} / "
        f"{summary['mean_shape_score_raw']:.2f} / "
        f"{summary['mean_dipole_score_raw']:.2f} / "
        f"{summary['mean_pose_score_raw']:.2f}"
    )
    if summary["mean_diffused_shift"] is not None:
        print(f"Mean diffused-anchor shift: {summary['mean_diffused_shift']:.3f}")
    if summary["mean_energy_delta"] is not None:
        print(f"Mean sampled-original energy delta: {summary['mean_energy_delta']:.3f}")
    if summary["mean_shape_mse"] is not None:
        print(f"Mean sampled-original shape MSE: {summary['mean_shape_mse']:.4f}")
    if summary["mean_dipole_direction_angle_deg"] is not None:
        print(f"Mean sampled-original dipole angle: {summary['mean_dipole_direction_angle_deg']:.2f} deg")
    if summary["failure_counts"]:
        print("Failures:")
        for label, count in sorted(summary["failure_counts"].items()):
            print(f"  {label}: {count}")

    worst_by_shift = sorted(
        [record for record in records if "compare" in record],
        key=lambda record: float(record["compare"]["diffused_shift_mean"]),
        reverse=True,
    )[:5]
    if worst_by_shift:
        print("Worst diffused-anchor shifts:")
        for record in worst_by_shift:
            compare = record["compare"]
            print(
                f"  sample {record['sample_index']}: "
                f"mean_shift={compare['diffused_shift_mean']:.3f}, "
                f"energy_delta={compare['energy_delta']:.3f}, "
                f"failures={','.join(record['failures']) or 'none'}"
            )


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input)
    if args.max_samples is not None:
        samples = samples[: int(args.max_samples)]

    report = build_evaluation_report(
        samples,
        large_shift_threshold=float(args.large_shift_threshold),
        energy_regression_threshold=float(args.energy_regression_threshold),
    )
    summary = report["summary"]
    print_evaluation_report(report, input_path=args.input)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON report to {args.output_json}")


if __name__ == "__main__":
    main()
