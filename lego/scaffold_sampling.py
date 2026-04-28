from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


@dataclass
class Scaffold:
    pos: np.ndarray
    node_id: np.ndarray
    parent_id: np.ndarray
    branch_id: np.ndarray
    seq_index_in_branch: np.ndarray
    degree_topology: np.ndarray
    family: str
    branch_kind: np.ndarray
    hidden_label: np.ndarray
    turn_mask: np.ndarray
    segment_id: np.ndarray


class ScaffoldSampler:
    """Clean scaffold sampler for the LEGO tensor-flow benchmark.

    Key design choice:
    positions are BRICK ANCHORS, not unit-voxel backbone points.

    This matters for sheets. If a straight segment alternates 1x1 and 1x2, the
    1x2 brick must replace two consecutive 1x1 bricks *along the same line*.
    Therefore the anchor spacing must advance by 1 or 2 lattice units depending
    on the selected brick length. The previous version sampled one point per
    lattice step and only later tried to assign 1x2 bricks, which forced the
    placement code to rotate bricks away from the line to avoid overlap.
    """

    def __init__(
        self,
        min_nodes: int = 18,
        max_nodes: int = 40,
        family: str = "mixed",
        branch_depth_limit: int = 2,
        bifurcation_probability: float = 0.45,
        chain_helix_probability: float = 0.45,
        chain_curved_probability: float = 0.30,
        helix_radius_min: float = 1.2,
        helix_radius_max: float = 2.0,
        helix_pitch_min: float = 2.2,
        helix_pitch_max: float = 3.6,
        alpha_helix_radius_scale: float = 1.35,
        alpha_helix_pitch_scale: float = 1.35,
        sheet_run_length_min: int = 4,
        sheet_run_length_max: int = 8,
        sheet_turn_step: int = 1,
        junction_angle_min_deg: float = 40.0,
        position_noise_std: float = 0.07,
    ) -> None:
        self.min_nodes = int(max(8, min_nodes))
        self.max_nodes = int(max(self.min_nodes, max_nodes))
        self.family = str(family).strip().lower()
        self.branch_depth_limit = int(max(1, branch_depth_limit))
        self.bifurcation_probability = float(np.clip(bifurcation_probability, 0.0, 1.0))
        self.chain_helix_probability = float(np.clip(chain_helix_probability, 0.0, 1.0))
        self.chain_curved_probability = float(np.clip(chain_curved_probability, 0.0, 1.0))
        self.helix_radius_min = float(min(helix_radius_min, helix_radius_max))
        self.helix_radius_max = float(max(helix_radius_min, helix_radius_max))
        self.helix_pitch_min = float(min(helix_pitch_min, helix_pitch_max))
        self.helix_pitch_max = float(max(helix_pitch_min, helix_pitch_max))
        self.alpha_helix_radius_scale = float(max(1.0, alpha_helix_radius_scale))
        self.alpha_helix_pitch_scale = float(max(1.0, alpha_helix_pitch_scale))
        self.sheet_run_length_min = int(max(2, min(sheet_run_length_min, sheet_run_length_max)))
        self.sheet_run_length_max = int(max(self.sheet_run_length_min, sheet_run_length_max))
        self.sheet_turn_step = int(max(1, sheet_turn_step))
        self.junction_angle_min_rad = float(np.deg2rad(max(5.0, junction_angle_min_deg)))
        self.position_noise_std = float(max(0.0, position_noise_std))
        if self.family not in {"mixed", "chain", "sheet", "alpha_helix"}:
            raise ValueError(f"Unsupported scaffold family '{family}'.")

    def _choose_family(self, rng: np.random.Generator) -> str:
        if self.family != "mixed":
            return self.family
        sheet_prob = float(np.clip(self.bifurcation_probability, 0.10, 0.60))
        alpha_prob = float(np.clip(0.5 * self.chain_helix_probability, 0.10, 0.35))
        chain_prob = float(max(0.05, 1.0 - sheet_prob - alpha_prob))
        probs = np.asarray([chain_prob, sheet_prob, alpha_prob], dtype=np.float32)
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        return str(rng.choice(np.asarray(["chain", "sheet", "alpha_helix"]), p=probs))

    @staticmethod
    def _build_tree_degree(parent_id: np.ndarray) -> np.ndarray:
        parent_id = np.asarray(parent_id, dtype=np.int32).reshape(-1)
        n = int(parent_id.shape[0])
        degree = np.zeros((n,), dtype=np.int32)
        for child, parent in enumerate(parent_id.tolist()):
            if int(parent) >= 0:
                degree[child] += 1
                degree[int(parent)] += 1
        return degree.astype(np.int32)

    def _sample_sheet_like_anchors(
        self,
        n_nodes: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample brick-level anchors for sheet/chain grammars.

        Returns
        -------
        anchors:
            [N, 3] anchor positions, one per brick.
        turn_mask:
            [N] True only for explicit L-turn corner nodes.
        segment_id:
            [N] Straight-segment identifier; turn nodes carry -1.
        """
        n_nodes = int(max(8, n_nodes))

        current = np.asarray([0, 0, 0], dtype=np.int32)
        # forward direction
        current_axis = int(rng.integers(0, 3))
        current_sign = -1 if float(rng.random()) < 0.5 else 1

        anchors: List[np.ndarray] = []
        turn_mask: List[bool] = []
        segment_id: List[int] = []

        occupied = set()
        current_segment = 0
        straight_parity = 0  # even->1x1, odd->1x2

        while len(anchors) < n_nodes:
            run_len = int(rng.integers(self.sheet_run_length_min, self.sheet_run_length_max + 1))
            placed_any_straight = False

            for _ in range(run_len):
                if len(anchors) >= n_nodes:
                    break

                # Determine intended straight brick length.
                brick_len = 1 if (straight_parity % 2 == 0) else 2
                step = np.zeros((3,), dtype=np.int32)
                step[current_axis] = current_sign

                # Occupancy of the straight brick along the line.
                brick_cells = [tuple(int(v) for v in current.tolist())]
                if brick_len == 2:
                    brick_cells.append(tuple(int(v) for v in (current + step).tolist()))

                if any(cell in occupied for cell in brick_cells):
                    break

                anchors.append(current.copy())
                turn_mask.append(False)
                segment_id.append(current_segment)
                occupied.update(brick_cells)
                placed_any_straight = True

                current = current + brick_len * step
                straight_parity += 1

            if len(anchors) >= n_nodes:
                break
            if not placed_any_straight and len(anchors) > 0:
                break

            # Explicit L-turn at the current corner anchor. Local rule:
            # the L brick occupies the current corner cell, one cell backward
            # along the previous segment, and one cell forward into the new segment.
            prev_step = np.zeros((3,), dtype=np.int32)
            prev_step[current_axis] = current_sign
            prev_leg = current - prev_step

            candidates: List[Tuple[int, int, np.ndarray, Tuple[int, int, int]]] = []
            for axis in range(3):
                if axis == current_axis:
                    continue
                for sign in (-1, 1):
                    new_step = np.zeros((3,), dtype=np.int32)
                    new_step[axis] = sign * self.sheet_turn_step
                    next_leg = current + new_step
                    turn_cells = {
                        tuple(int(v) for v in current.tolist()),
                        tuple(int(v) for v in prev_leg.tolist()),
                        tuple(int(v) for v in next_leg.tolist()),
                    }
                    if any(cell in occupied for cell in turn_cells):
                        continue
                    candidates.append((axis, sign, next_leg, tuple(int(v) for v in next_leg.tolist())))

            if len(candidates) == 0:
                break
            turn_axis, turn_sign, next_leg, _ = candidates[int(rng.integers(0, len(candidates)))]

            anchors.append(current.copy())
            turn_mask.append(True)
            segment_id.append(-1)
            occupied.update(
                {
                    tuple(int(v) for v in current.tolist()),
                    tuple(int(v) for v in prev_leg.tolist()),
                    tuple(int(v) for v in next_leg.tolist()),
                }
            )

            # After the corner, the next straight brick starts at the outgoing leg.
            current = next_leg.copy()
            current_axis = int(turn_axis)
            current_sign = int(turn_sign)
            current_segment += 1
            straight_parity = 0

        if len(anchors) == 0:
            anchors = [np.zeros((3,), dtype=np.int32)]
            turn_mask = [False]
            segment_id = [0]

        return (
            np.asarray(anchors[:n_nodes], dtype=np.float32),
            np.asarray(turn_mask[:n_nodes], dtype=bool),
            np.asarray(segment_id[:n_nodes], dtype=np.int32),
        )

    def _sample_alpha_helix_centers(
        self,
        n_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_points = int(max(8, n_points))
        radius = float(rng.uniform(self.helix_radius_min, self.helix_radius_max)) * self.alpha_helix_radius_scale
        pitch_turn = float(rng.uniform(self.helix_pitch_min, self.helix_pitch_max)) * self.alpha_helix_pitch_scale

        delta_theta = float(2.0 * math.pi / 3.6)
        theta0 = float(rng.uniform(0.0, 2.0 * math.pi))
        handedness = -1.0 if float(rng.random()) < 0.5 else 1.0
        k = np.arange(n_points, dtype=np.float32)
        theta = theta0 + handedness * delta_theta * k
        points = np.stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                (pitch_turn / (2.0 * math.pi)) * theta,
            ],
            axis=-1,
        ).astype(np.float32)
        return points.astype(np.float32)

    def sample(self, rng: np.random.Generator) -> Scaffold:
        for _ in range(256):
            family = self._choose_family(rng)
            target_nodes = int(rng.integers(self.min_nodes, self.max_nodes + 1))

            if family == "alpha_helix":
                pos = self._sample_alpha_helix_centers(target_nodes, rng)
                # tiny noise only; preserve the clean thin helix.
                if self.position_noise_std > 0.0:
                    pos = pos + rng.normal(scale=min(self.position_noise_std, 0.02), size=pos.shape).astype(np.float32)
                n = int(pos.shape[0])
                parent = np.full((n,), fill_value=-1, dtype=np.int32)
                parent[1:] = np.arange(0, n - 1, dtype=np.int32)
                branch_id = np.zeros((n,), dtype=np.int32)
                seq_index = np.arange(n, dtype=np.int32)
                degree = self._build_tree_degree(parent)
                branch_kind = np.asarray(["alpha_helix"] * n, dtype=str)
                hidden = np.asarray(["alpha_helix_thin"] * n, dtype=str)
                turn_mask = np.zeros((n,), dtype=bool)
                segment_id = np.zeros((n,), dtype=np.int32)
                return Scaffold(
                    pos=pos.astype(np.float32),
                    node_id=np.arange(n, dtype=np.int32),
                    parent_id=parent,
                    branch_id=branch_id,
                    seq_index_in_branch=seq_index,
                    degree_topology=degree,
                    family=family,
                    branch_kind=branch_kind,
                    hidden_label=hidden,
                    turn_mask=turn_mask,
                    segment_id=segment_id,
                )

            anchors, turn_mask, segment_id = self._sample_sheet_like_anchors(target_nodes, rng)
            if anchors.shape[0] < self.min_nodes:
                continue

            if family == "sheet":
                noise_std = min(self.position_noise_std, 0.01)
                hidden_label = "orthogonal_sheet"
            else:
                noise_std = min(self.position_noise_std, 0.02)
                hidden_label = "orthogonal_chain"

            if noise_std > 0.0:
                anchors = anchors + rng.normal(scale=noise_std, size=anchors.shape).astype(np.float32)

            n = int(anchors.shape[0])
            parent = np.full((n,), fill_value=-1, dtype=np.int32)
            parent[1:] = np.arange(0, n - 1, dtype=np.int32)
            degree = self._build_tree_degree(parent)
            if int(np.max(degree)) > 2:
                continue

            return Scaffold(
                pos=anchors.astype(np.float32),
                node_id=np.arange(n, dtype=np.int32),
                parent_id=parent,
                branch_id=np.zeros((n,), dtype=np.int32),
                seq_index_in_branch=np.arange(n, dtype=np.int32),
                degree_topology=degree.astype(np.int32),
                family=family,
                branch_kind=np.asarray([family] * n, dtype=str),
                hidden_label=np.asarray([hidden_label] * n, dtype=str),
                turn_mask=turn_mask.astype(bool),
                segment_id=segment_id.astype(np.int32),
            )

        raise RuntimeError("Failed to sample a valid clean scaffold.")
