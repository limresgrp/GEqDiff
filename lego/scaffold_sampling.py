from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np

try:
    from lego.lego_blocks import NEIGHBOR_DIRS
except ModuleNotFoundError:  # pragma: no cover
    from lego_blocks import NEIGHBOR_DIRS


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= eps:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    count = int(max(2, count))
    if points.shape[0] <= 1:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1).astype(np.float32)
    total = float(seg.sum())
    if total <= 1e-8:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg)])
    query = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.zeros((count, 3), dtype=np.float32)
    for idx, value in enumerate(query.tolist()):
        right = int(np.searchsorted(cumulative, value, side="right"))
        seg_idx = int(np.clip(right - 1, 0, points.shape[0] - 2))
        left_s = float(cumulative[seg_idx])
        right_s = float(cumulative[seg_idx + 1])
        frac = 0.0 if right_s - left_s <= 1e-8 else (float(value) - left_s) / (right_s - left_s)
        out[idx] = (1.0 - frac) * points[seg_idx] + frac * points[seg_idx + 1]
    return out.astype(np.float32)


def _connect_lattice_points(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    start = np.asarray(start, dtype=np.int32).copy()
    end = np.asarray(end, dtype=np.int32).copy()
    path = [start.copy()]
    current = start.copy()
    while not np.array_equal(current, end):
        delta = end - current
        axis = int(np.argmax(np.abs(delta)))
        step = int(np.sign(delta[axis]))
        if step == 0:
            remaining_axes = np.flatnonzero(delta != 0)
            if remaining_axes.size == 0:
                break
            axis = int(remaining_axes[0])
            step = int(np.sign(delta[axis]))
        current = current.copy()
        current[axis] += step
        path.append(current.copy())
    return np.asarray(path, dtype=np.int32)


def polyline_to_connected_voxels(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    lattice = np.rint(points).astype(np.int32)
    ordered: List[np.ndarray] = [lattice[0].copy()]
    for idx in range(1, lattice.shape[0]):
        segment = _connect_lattice_points(ordered[-1], lattice[idx])
        for voxel in segment[1:]:
            if not np.array_equal(voxel, ordered[-1]):
                ordered.append(voxel.copy())
    unique: List[np.ndarray] = []
    seen = set()
    for voxel in ordered:
        key = tuple(int(v) for v in voxel.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(np.asarray(voxel, dtype=np.int32))
    return np.asarray(unique, dtype=np.int32)


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
        # Mixed mode intentionally excludes branching junctions to keep a
        # deterministic single-chain ordering and avoid sequence ambiguities.
        sheet_prob = float(np.clip(self.bifurcation_probability, 0.10, 0.60))
        alpha_prob = float(np.clip(0.5 * self.chain_helix_probability, 0.10, 0.35))
        chain_prob = float(max(0.05, 1.0 - sheet_prob - alpha_prob))
        probs = np.asarray([chain_prob, sheet_prob, alpha_prob], dtype=np.float32)
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        return str(rng.choice(np.asarray(["chain", "sheet", "alpha_helix"]), p=probs))

    def _sample_chain_polyline(
        self,
        n_points: int,
        rng: np.random.Generator,
        force_regime: str | None = None,
        helix_handedness: float | None = None,
    ) -> Tuple[np.ndarray, str]:
        n_points = int(max(8, n_points))
        if force_regime is not None:
            regime = str(force_regime)
        else:
            draw = float(rng.random())
            if draw < self.chain_helix_probability:
                regime = "helix"
            elif draw < self.chain_helix_probability + self.chain_curved_probability:
                regime = "curved"
            else:
                regime = "straight"

        if regime == "helix":
            radius = float(rng.uniform(self.helix_radius_min, self.helix_radius_max))
            pitch_turn = float(rng.uniform(self.helix_pitch_min, self.helix_pitch_max))
            if force_regime == "helix" and helix_handedness is not None:
                # Alpha-helix-only family defaults should be larger/sparser.
                radius *= self.alpha_helix_radius_scale
                pitch_turn *= self.alpha_helix_pitch_scale
            turns = float(max(1.0, n_points / 8.0))
            theta = np.linspace(0.0, 2.0 * math.pi * turns, n_points, dtype=np.float32)
            if helix_handedness is None:
                handedness = -1.0 if float(rng.random()) < 0.5 else 1.0
            else:
                handedness = -1.0 if float(helix_handedness) < 0.0 else 1.0
            points = np.stack(
                [
                    radius * np.cos(handedness * theta),
                    radius * np.sin(handedness * theta),
                    pitch_turn * theta / (2.0 * math.pi),
                ],
                axis=-1,
            ).astype(np.float32)
        elif regime == "curved":
            t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
            amp_y = float(rng.uniform(1.0, 2.4))
            amp_z = float(rng.uniform(0.4, 1.2))
            freq = float(rng.uniform(0.8, 1.5))
            points = np.stack(
                [
                    10.0 * t,
                    amp_y * np.sin(2.0 * math.pi * freq * t),
                    amp_z * np.sin(math.pi * freq * t + 0.5),
                ],
                axis=-1,
            ).astype(np.float32)
        else:
            # Use integer-spaced points so voxelization preserves node count.
            x = np.arange(n_points, dtype=np.float32)
            points = np.stack(
                [
                    x,
                    np.zeros_like(x),
                    np.zeros_like(x),
                ],
                axis=-1,
            ).astype(np.float32)
        return points.astype(np.float32), regime

    def _sample_sheet_polyline(self, n_points: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        # Simplified sheet generator: connected 3D serpentine line with explicit turns.
        n_points = int(max(12, n_points))
        run_min = int(self.sheet_run_length_min)
        run_max = int(self.sheet_run_length_max)
        turn_step = int(self.sheet_turn_step)

        current = np.asarray([0, 0, 0], dtype=np.int32)
        current_axis = 0
        current_sign = 1
        line_id = 0
        points: List[np.ndarray] = [current.copy()]
        labels: List[int] = [line_id]
        turns: List[bool] = [False]

        while len(points) < n_points:
            run_len = int(rng.integers(run_min, run_max + 1))
            for _ in range(run_len):
                if len(points) >= n_points:
                    break
                step = np.zeros((3,), dtype=np.int32)
                step[current_axis] = current_sign
                current = current + step
                points.append(current.copy())
                labels.append(line_id)
                turns.append(False)
            if len(points) >= n_points:
                break
            remaining_axes = [axis for axis in (0, 1, 2) if axis != current_axis]
            turn_axis = int(remaining_axes[int(rng.integers(0, len(remaining_axes)))])
            turn_sign = -1 if float(rng.random()) < 0.5 else 1
            current = current + np.eye(3, dtype=np.int32)[turn_axis] * int(turn_sign * turn_step)
            points.append(current.copy())
            labels.append(-1)
            turns.append(True)
            next_axis = [axis for axis in (0, 1, 2) if axis not in {current_axis, turn_axis}]
            current_axis = int(next_axis[0]) if len(next_axis) == 1 else int(next_axis[int(rng.integers(0, len(next_axis)))])
            current_sign = -current_sign if float(rng.random()) < 0.5 else current_sign
            line_id += 1

        raw_points = np.asarray(points[:n_points], dtype=np.float32)
        raw_labels = np.asarray(labels[:n_points], dtype=np.int32)
        raw_turns = np.asarray(turns[:n_points], dtype=bool)
        return raw_points, np.stack([raw_labels, raw_turns.astype(np.int32)], axis=-1)

    def _build_tree_degree(self, parent_id: np.ndarray) -> np.ndarray:
        n = int(parent_id.shape[0])
        degree = np.zeros((n,), dtype=np.int32)
        for node_idx, parent in enumerate(parent_id.tolist()):
            if int(parent) < 0:
                continue
            degree[node_idx] += 1
            degree[int(parent)] += 1
        return degree.astype(np.int32)

    def sample(self, rng: np.random.Generator) -> Scaffold:
        for _ in range(96):
            target_nodes = int(rng.integers(self.min_nodes, self.max_nodes + 1))
            family = self._choose_family(rng)
            if family in {"chain", "alpha_helix"}:
                force_regime = "helix" if family == "alpha_helix" else "straight"
                helix_handedness = 1.0 if family == "alpha_helix" else None
                if family == "alpha_helix":
                    # Helical curves become longer after voxel connectivity stitching;
                    # downsample control points so final connected-voxel count stays in range.
                    scale = max(1.0, float(self.alpha_helix_radius_scale) * float(self.alpha_helix_pitch_scale) ** 0.5)
                    requested_points = max(6, int(round((0.52 / scale) * float(target_nodes))))
                else:
                    # Keep chain sampling direct and deterministic.
                    requested_points = max(8, target_nodes)
                points, regime = self._sample_chain_polyline(
                    n_points=requested_points,
                    rng=rng,
                    force_regime=force_regime,
                    helix_handedness=helix_handedness,
                )
                if family == "alpha_helix":
                    # Keep alpha-helix axes stable in lattice space; random full rotations
                    # inflate Manhattan path length and make node counts unstable.
                    mild_noise = min(self.position_noise_std, 0.03)
                    if mild_noise > 0.0:
                        points = points + rng.normal(scale=mild_noise, size=points.shape).astype(np.float32)
                else:
                    # Keep chain motifs stable and connected for the periodic T/1x1 program.
                    mild_noise = min(self.position_noise_std, 0.02)
                    if mild_noise > 0.0:
                        points = points + rng.normal(scale=mild_noise, size=points.shape).astype(np.float32)
                vox = polyline_to_connected_voxels(points)
                if not (self.min_nodes <= int(vox.shape[0]) <= self.max_nodes):
                    continue
                n = int(vox.shape[0])
                parent = np.full((n,), fill_value=-1, dtype=np.int32)
                parent[1:] = np.arange(0, n - 1, dtype=np.int32)
                branch_id = np.zeros((n,), dtype=np.int32)
                seq_index = np.arange(n, dtype=np.int32)
                branch_kind = np.asarray(["chain"] * n, dtype=object)
                if regime == "helix":
                    branch_kind[:] = "helix"
                elif regime == "curved":
                    branch_kind[:] = "chain"
                hidden = np.asarray([regime] * n, dtype=object)
                degree = self._build_tree_degree(parent)
                return Scaffold(
                    pos=vox.astype(np.float32),
                    node_id=np.arange(n, dtype=np.int32),
                    parent_id=parent,
                    branch_id=branch_id,
                    seq_index_in_branch=seq_index,
                    degree_topology=degree,
                    family=family,
                    branch_kind=branch_kind.astype(str),
                    hidden_label=hidden.astype(str),
                    turn_mask=np.zeros((n,), dtype=bool),
                    segment_id=np.full((n,), fill_value=-1, dtype=np.int32),
                )
            if family == "sheet":
                requested_points = max(10, target_nodes)
                points, sheet_meta = self._sample_sheet_polyline(n_points=requested_points, rng=rng)
                # Keep simplified sheets axis-aligned and low-noise to preserve
                # deterministic 1D line semantics with sparse turns.
                mild_noise = min(self.position_noise_std, 0.02)
                if mild_noise > 0.0:
                    points = points + rng.normal(scale=mild_noise, size=points.shape).astype(np.float32)
                vox = polyline_to_connected_voxels(points)
                if not (self.min_nodes <= int(vox.shape[0]) <= self.max_nodes):
                    continue
                n = int(vox.shape[0])
                parent = np.full((n,), fill_value=-1, dtype=np.int32)
                parent[1:] = np.arange(0, n - 1, dtype=np.int32)
                nearest = np.argmin(np.sum((vox[:, None, :] - points[None, :, :]) ** 2, axis=-1), axis=1)
                raw_segment = sheet_meta[:, 0]
                raw_turn = sheet_meta[:, 1].astype(bool)
                branch_id = np.zeros((n,), dtype=np.int32)
                seq_index = np.arange(n, dtype=np.int32)
                turn_mask = np.asarray(raw_turn[nearest], dtype=bool)
                segment_id = np.asarray(raw_segment[nearest], dtype=np.int32)
                degree = self._build_tree_degree(parent)
                hidden = np.asarray(raw_segment[nearest].tolist(), dtype=np.int32)
                return Scaffold(
                    pos=vox.astype(np.float32),
                    node_id=np.arange(n, dtype=np.int32),
                    parent_id=parent,
                    branch_id=branch_id,
                    seq_index_in_branch=seq_index,
                    degree_topology=degree,
                    family=family,
                    branch_kind=np.asarray(["sheet"] * n, dtype=str),
                    hidden_label=hidden.astype(np.int32),
                    turn_mask=turn_mask.astype(bool),
                    segment_id=segment_id.astype(np.int32),
                )
        raise RuntimeError("Failed to sample a valid scaffold after multiple attempts.")
