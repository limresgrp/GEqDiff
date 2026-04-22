"""Deterministic scaffold-based LEGO dataset generator.

This replaces legacy SH/shell and heuristic post-hoc placement flows with:
positions + topology -> descriptors -> roles -> shape/color/dipole.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.utils.dipole_utils import dipole_strengths, normalize_dipole_directions
from lego.color_rules import assign_color_and_dipole
from lego.descriptors import compute_descriptors
from lego.role_assignment import assign_roles
from lego.scaffold_sampling import ScaffoldSampler
from lego.shape_prototypes import map_roles_to_shapes
from lego.utils import DEFAULT_IRREPS, default_dataset_path, save_samples
from lego.validation import validate_sample


class LegoDeterministicEngine:
    def __init__(
        self,
        *,
        min_nodes: int = 18,
        max_nodes: int = 40,
        scaffold_family: str = "mixed",
        branch_depth_limit: int = 2,
        bifurcation_probability: float = 0.45,
        chain_helix_probability: float = 0.45,
        chain_curved_probability: float = 0.30,
        helix_radius_min: float = 1.2,
        helix_radius_max: float = 2.0,
        helix_pitch_min: float = 2.2,
        helix_pitch_max: float = 3.6,
        sheet_strands_min: int = 2,
        sheet_strands_max: int = 4,
        sheet_spacing_min: float = 2.0,
        sheet_spacing_max: float = 2.8,
        junction_angle_min_deg: float = 40.0,
        descriptor_neighborhood_radius: float = 2.5,
        tau_straight: float = 0.35,
        tau_planar: float = 0.34,
        tau_junction_degree: int = 3,
        helix_phase_period: int = 4,
        shape_noise_scale: float = 0.0,
        dipole_noise_scale: float = 0.0,
        position_noise_std: float = 0.07,
        sequence_pos_max: int | None = None,
    ) -> None:
        self.irreps = DEFAULT_IRREPS
        self.min_nodes = int(max(8, min_nodes))
        self.max_nodes = int(max(self.min_nodes, max_nodes))
        self.scaffold_family = str(scaffold_family).strip().lower()
        self.descriptor_neighborhood_radius = float(max(0.5, descriptor_neighborhood_radius))
        self.tau_straight = float(max(0.0, tau_straight))
        self.tau_planar = float(max(0.0, tau_planar))
        self.tau_junction_degree = int(max(2, tau_junction_degree))
        self.helix_phase_period = int(max(2, helix_phase_period))
        self.shape_noise_scale = float(max(0.0, shape_noise_scale))
        self.dipole_noise_scale = float(max(0.0, dipole_noise_scale))
        self.sequence_pos_max = int(sequence_pos_max) if sequence_pos_max is not None else int(self.max_nodes)
        self.sequence_pos_max = int(max(2, self.sequence_pos_max))

        self.sampler = ScaffoldSampler(
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes,
            family=self.scaffold_family,
            branch_depth_limit=int(max(1, branch_depth_limit)),
            bifurcation_probability=float(np.clip(bifurcation_probability, 0.0, 1.0)),
            chain_helix_probability=float(np.clip(chain_helix_probability, 0.0, 1.0)),
            chain_curved_probability=float(np.clip(chain_curved_probability, 0.0, 1.0)),
            helix_radius_min=float(helix_radius_min),
            helix_radius_max=float(helix_radius_max),
            helix_pitch_min=float(helix_pitch_min),
            helix_pitch_max=float(helix_pitch_max),
            sheet_strands_min=int(max(2, sheet_strands_min)),
            sheet_strands_max=int(max(sheet_strands_min, sheet_strands_max)),
            sheet_spacing_min=float(sheet_spacing_min),
            sheet_spacing_max=float(sheet_spacing_max),
            junction_angle_min_deg=float(junction_angle_min_deg),
            position_noise_std=float(max(0.0, position_noise_std)),
        )

    def _sequence_positions(self, node_count: int) -> np.ndarray:
        node_count = int(max(1, node_count))
        if node_count == 1:
            return np.zeros((1, 1), dtype=np.int32)
        values = np.linspace(0, self.sequence_pos_max - 1, num=node_count, dtype=np.float32)
        return np.rint(values).astype(np.int32)[:, None]

    def build_sample(self, *, rng: np.random.Generator) -> Dict:
        scaffold = self.sampler.sample(rng=rng)
        pos = np.asarray(scaffold.pos, dtype=np.float32)
        parent_id = np.asarray(scaffold.parent_id, dtype=np.int32)
        branch_id = np.asarray(scaffold.branch_id, dtype=np.int32)
        seq_index = np.asarray(scaffold.seq_index_in_branch, dtype=np.int32)
        degree_topology = np.asarray(scaffold.degree_topology, dtype=np.int32)
        branch_kind = np.asarray(scaffold.branch_kind).astype(str)

        descriptors = compute_descriptors(
            pos=pos,
            parent_id=parent_id,
            branch_id=branch_id,
            seq_index_in_branch=seq_index,
            degree_topology=degree_topology,
            helix_phase_period=self.helix_phase_period,
            neighborhood_radius=self.descriptor_neighborhood_radius,
        )

        role_id, role_name = assign_roles(
            parent_id=parent_id,
            branch_id=branch_id,
            seq_index_in_branch=seq_index,
            branch_kind=branch_kind,
            degree_topology=degree_topology,
            descriptors=descriptors,
            tau_straight=self.tau_straight,
            tau_planar=self.tau_planar,
            junction_degree_threshold=self.tau_junction_degree,
            helix_phase_period=self.helix_phase_period,
        )

        shape_payload = map_roles_to_shapes(
            anchors=pos.copy(),
            role_names=role_name,
            descriptors=descriptors,
            shape_noise_scale=self.shape_noise_scale,
            rng=rng,
        )
        anchors = np.asarray(shape_payload["anchors"], dtype=np.float32)
        brick_types = np.asarray(shape_payload["brick_types"])
        brick_rotations = np.asarray(shape_payload["brick_rotations"], dtype=np.float32)
        brick_features = np.asarray(shape_payload["shape_features"], dtype=np.float32)

        color_class, dipoles = assign_color_and_dipole(
            role_names=role_name,
            seq_index_in_branch=seq_index,
            descriptors=descriptors,
            dipole_noise_scale=self.dipole_noise_scale,
            rng=rng,
        )
        dipole_dirs = normalize_dipole_directions(dipoles)
        dipole_mag = dipole_strengths(dipoles)
        sequence_position = self._sequence_positions(node_count=int(anchors.shape[0]))

        sample = {
            "coefficients": np.zeros((16,), dtype=np.float32),
            "irreps": np.asarray(str(self.irreps)),
            "generation_mode": np.asarray("deterministic_scaffold"),
            "scaffold_family": np.asarray(str(scaffold.family)),
            "hidden_scaffold_label": np.asarray(scaffold.hidden_label),
            "pos": anchors.astype(np.float32),
            "rotations": brick_rotations.astype(np.float32),
            "types": np.asarray(brick_types),
            "features": brick_features.astype(np.float32),
            "dipoles": dipoles.astype(np.float32),
            "brick_anchors": anchors.astype(np.float32),
            "brick_rotations": brick_rotations.astype(np.float32),
            "brick_types": np.asarray(brick_types),
            "brick_features": brick_features.astype(np.float32),
            "brick_dipoles": dipoles.astype(np.float32),
            "brick_dipole_directions": dipole_dirs.astype(np.float32),
            "brick_dipole_strengths": dipole_mag.astype(np.float32),
            "brick_sequence_position": sequence_position.astype(np.int32),
            "sequence_position": sequence_position.astype(np.int32),
            "sequence_pos_max": np.asarray(self.sequence_pos_max, dtype=np.int32),
            "node_id": np.asarray(scaffold.node_id, dtype=np.int32),
            "parent_id": parent_id.astype(np.int32),
            "branch_id": branch_id.astype(np.int32),
            "seq_index_in_branch": seq_index.astype(np.int32),
            "degree_topology": degree_topology.astype(np.int32),
            "branch_kind": np.asarray(branch_kind),
            "role_id": role_id.astype(np.int32),
            "role": np.asarray(role_name),
            "color_class": np.asarray(color_class, dtype=np.int32),
            "requested_block_types": np.asarray(shape_payload["requested_block_types"]),
            "requested_rotations": np.asarray(shape_payload["requested_rotations"], dtype=np.float32),
            "local_frames": np.asarray(shape_payload["local_frames"], dtype=np.float32),
            "descriptor_tangent": np.asarray(descriptors["tangent"], dtype=np.float32),
            "descriptor_curvature": np.asarray(descriptors["curvature_mag"], dtype=np.float32),
            "descriptor_planarity": np.asarray(descriptors["planarity_score"], dtype=np.float32),
            "descriptor_linearity": np.asarray(descriptors["linearity_score"], dtype=np.float32),
            "descriptor_phase_index": np.asarray(descriptors["phase_index"], dtype=np.int32),
            "descriptor_boundary_flag": np.asarray(descriptors["boundary_flag"], dtype=bool),
            "descriptor_junction_flag": np.asarray(descriptors["junction_flag"], dtype=bool),
        }

        validation = validate_sample(
            parent_id=parent_id,
            branch_id=branch_id,
            seq_index_in_branch=seq_index,
            degree_topology=degree_topology,
            role_names=role_name,
            sample_for_geometry={
                "brick_anchors": sample["brick_anchors"],
                "brick_rotations": sample["brick_rotations"],
                "brick_types": sample["brick_types"],
            },
        )
        sample["validation_num_contacts"] = np.asarray(validation["num_contacts"], dtype=np.int64)
        sample["validation_num_components"] = np.asarray(validation["num_components"], dtype=np.int64)
        return sample

    def generate_task(self, coefficients=None, seed: int | None = None):  # compatibility
        _ = coefficients
        rng = np.random.default_rng(seed)
        return self.build_sample(rng=rng)

    def generate_dataset(self, n_samples: int = 1, seed: int | None = None) -> List[Dict]:
        rng = np.random.default_rng(seed)
        return [self.build_sample(rng=rng) for _ in range(int(n_samples))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic scaffold-based LEGO samples (chain/sheet/junction)."
    )
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--path", type=str, default=str(default_dataset_path()), help="Output canonical dataset path.")
    parser.add_argument(
        "--scaffold-family",
        type=str,
        default="mixed",
        choices=["mixed", "chain", "alpha_helix", "sheet", "junction"],
    )
    parser.add_argument("--min-nodes", type=int, default=18)
    parser.add_argument("--max-nodes", type=int, default=40)
    parser.add_argument("--branch-depth-limit", type=int, default=2)
    parser.add_argument("--bifurcation-probability", type=float, default=0.45)
    parser.add_argument("--chain-helix-probability", type=float, default=0.45)
    parser.add_argument("--chain-curved-probability", type=float, default=0.30)
    parser.add_argument("--helix-radius-min", type=float, default=1.2)
    parser.add_argument("--helix-radius-max", type=float, default=2.0)
    parser.add_argument("--helix-pitch-min", type=float, default=2.2)
    parser.add_argument("--helix-pitch-max", type=float, default=3.6)
    parser.add_argument("--sheet-strands-min", type=int, default=2)
    parser.add_argument("--sheet-strands-max", type=int, default=4)
    parser.add_argument("--sheet-spacing-min", type=float, default=2.0)
    parser.add_argument("--sheet-spacing-max", type=float, default=2.8)
    parser.add_argument("--junction-angle-min-deg", type=float, default=40.0)
    parser.add_argument("--descriptor-neighborhood-radius", type=float, default=2.5)
    parser.add_argument("--tau-straight", type=float, default=0.35)
    parser.add_argument("--tau-planar", type=float, default=0.34)
    parser.add_argument("--tau-junction-degree", type=int, default=3)
    parser.add_argument("--helix-phase-period", type=int, default=4)
    parser.add_argument("--shape-noise-scale", type=float, default=0.0)
    parser.add_argument("--dipole-noise-scale", type=float, default=0.0)
    parser.add_argument("--position-noise-std", type=float, default=0.07)
    parser.add_argument("--sequence-pos-max", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = LegoDeterministicEngine(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        scaffold_family=args.scaffold_family,
        branch_depth_limit=args.branch_depth_limit,
        bifurcation_probability=args.bifurcation_probability,
        chain_helix_probability=args.chain_helix_probability,
        chain_curved_probability=args.chain_curved_probability,
        helix_radius_min=args.helix_radius_min,
        helix_radius_max=args.helix_radius_max,
        helix_pitch_min=args.helix_pitch_min,
        helix_pitch_max=args.helix_pitch_max,
        sheet_strands_min=args.sheet_strands_min,
        sheet_strands_max=args.sheet_strands_max,
        sheet_spacing_min=args.sheet_spacing_min,
        sheet_spacing_max=args.sheet_spacing_max,
        junction_angle_min_deg=args.junction_angle_min_deg,
        descriptor_neighborhood_radius=args.descriptor_neighborhood_radius,
        tau_straight=args.tau_straight,
        tau_planar=args.tau_planar,
        tau_junction_degree=args.tau_junction_degree,
        helix_phase_period=args.helix_phase_period,
        shape_noise_scale=args.shape_noise_scale,
        dipole_noise_scale=args.dipole_noise_scale,
        position_noise_std=args.position_noise_std,
        sequence_pos_max=args.sequence_pos_max,
    )
    samples = engine.generate_dataset(n_samples=args.samples, seed=args.seed)
    save_samples(args.path, samples)

    print("--- LEGO Dataset Generated ---")
    print(f"Samples: {len(samples)}")
    print(f"Irreps: {engine.irreps}")
    print(f"Saved to: {args.path}")
    for idx, sample in enumerate(samples):
        print(
            f"Sample {idx}: nodes={len(sample['brick_types'])}, "
            f"family={str(np.asarray(sample['scaffold_family']).reshape(-1)[0])}, "
            f"contacts={int(np.asarray(sample['validation_num_contacts']).reshape(-1)[0])}, "
            f"components={int(np.asarray(sample['validation_num_components']).reshape(-1)[0])}"
        )


if __name__ == "__main__":
    main()


# Backward compatibility aliases used by older scripts.
LegoDipoleEngine = LegoDeterministicEngine
LegoMultiPortEngine = LegoDeterministicEngine
