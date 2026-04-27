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


def _sample_random_rotation(rng: np.random.Generator) -> np.ndarray:
    axis = _normalize(rng.normal(size=(3,)).astype(np.float32))
    if float(np.linalg.norm(axis)) <= 1e-8:
        axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    theta = float(rng.uniform(-math.pi, math.pi))
    x, y, z = axis.tolist()
    K = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float32,
    )
    I = np.eye(3, dtype=np.float32)
    return (I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)).astype(np.float32)


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
        sheet_strands_min: int = 2,
        sheet_strands_max: int = 4,
        sheet_spacing_min: float = 2.0,
        sheet_spacing_max: float = 2.8,
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
        self.sheet_strands_min = int(max(2, min(sheet_strands_min, sheet_strands_max)))
        self.sheet_strands_max = int(max(self.sheet_strands_min, sheet_strands_max))
        self.sheet_spacing_min = float(min(sheet_spacing_min, sheet_spacing_max))
        self.sheet_spacing_max = float(max(sheet_spacing_min, sheet_spacing_max))
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
            turns = float(max(1.0, n_points / 8.0))
            theta = np.linspace(0.0, 2.0 * math.pi * turns, n_points, dtype=np.float32)
            handedness = -1.0 if float(rng.random()) < 0.5 else 1.0
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
            t = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
            points = np.stack(
                [
                    10.0 * t,
                    0.35 * np.sin(2.0 * math.pi * t),
                    0.20 * np.sin(4.0 * math.pi * t + 0.4),
                ],
                axis=-1,
            ).astype(np.float32)
        return points.astype(np.float32), regime

    def _sample_sheet_polyline(self, n_points: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        n_points = int(max(12, n_points))
        n_strands = int(rng.integers(self.sheet_strands_min, self.sheet_strands_max + 1))
        spacing = float(rng.uniform(self.sheet_spacing_min, self.sheet_spacing_max))
        loop_n = max(4, n_points // (4 * n_strands))
        strand_total = max(n_points - (n_strands - 1) * loop_n, n_strands * 4)
        base_len = strand_total // n_strands
        strand_lengths = [base_len] * n_strands
        for extra in range(strand_total - base_len * n_strands):
            strand_lengths[extra % n_strands] += 1
        length = float(rng.uniform(8.0, 12.0))
        pleat = float(rng.uniform(0.2, 0.45))

        pieces: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        prev = None
        for strand_idx, length_i in enumerate(strand_lengths):
            x = np.linspace(-0.5 * length, 0.5 * length, int(length_i), dtype=np.float32)
            if strand_idx % 2 == 1:
                x = x[::-1]
            y = np.full_like(x, fill_value=float(strand_idx) * spacing)
            z = pleat * np.cos(np.arange(int(length_i), dtype=np.float32) * math.pi).astype(np.float32)
            strand = np.stack([x, y, z], axis=-1).astype(np.float32)
            if prev is not None:
                start = prev[-1]
                end = strand[0]
                mid = 0.5 * (start + end) + np.asarray([0.0, 0.0, float(rng.uniform(0.6, 1.2))], dtype=np.float32)
                loop = _resample_polyline(np.stack([start, mid, end], axis=0).astype(np.float32), loop_n)
                if loop.shape[0] > 2:
                    pieces.append(loop[1:-1].astype(np.float32))
                    labels.append(np.full((loop.shape[0] - 2,), fill_value=-1, dtype=np.int32))
            pieces.append(strand.astype(np.float32))
            labels.append(np.full((strand.shape[0],), fill_value=int(strand_idx), dtype=np.int32))
            prev = strand
        raw_points = np.concatenate(pieces, axis=0).astype(np.float32)
        raw_labels = np.concatenate(labels, axis=0).astype(np.int32)
        points = _resample_polyline(raw_points, n_points)
        nearest = np.argmin(np.sum((points[:, None, :] - raw_points[None, :, :]) ** 2, axis=-1), axis=1)
        strand_labels = raw_labels[nearest].astype(np.int32)
        return points.astype(np.float32), strand_labels

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
                force_regime = "helix" if family == "alpha_helix" else None
                requested_points = max(10, target_nodes)
                if family == "alpha_helix":
                    # Helical curves become longer after voxel connectivity stitching;
                    # downsample control points so final connected-voxel count stays in range.
                    requested_points = max(8, int(round(0.6 * float(target_nodes))))
                points, regime = self._sample_chain_polyline(
                    n_points=requested_points,
                    rng=rng,
                    force_regime=force_regime,
                )
                if family != "alpha_helix":
                    points = points @ _sample_random_rotation(rng).T
                    if self.position_noise_std > 0.0:
                        points = points + rng.normal(scale=self.position_noise_std, size=points.shape).astype(np.float32)
                else:
                    # Keep alpha-helix axes stable in lattice space; random full rotations
                    # inflate Manhattan path length and make node counts unstable.
                    mild_noise = min(self.position_noise_std, 0.03)
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
                )
            if family == "sheet":
                requested_points = max(10, int(round(0.75 * float(target_nodes))))
                points, strand_labels = self._sample_sheet_polyline(n_points=requested_points, rng=rng)
                points = points @ _sample_random_rotation(rng).T
                if self.position_noise_std > 0.0:
                    points = points + rng.normal(scale=self.position_noise_std, size=points.shape).astype(np.float32)
                vox = polyline_to_connected_voxels(points)
                if not (self.min_nodes <= int(vox.shape[0]) <= self.max_nodes):
                    continue
                n = int(vox.shape[0])
                parent = np.full((n,), fill_value=-1, dtype=np.int32)
                parent[1:] = np.arange(0, n - 1, dtype=np.int32)
                branch_id = np.zeros((n,), dtype=np.int32)
                seq_index = np.arange(n, dtype=np.int32)
                degree = self._build_tree_degree(parent)
                nearest = np.argmin(np.sum((vox[:, None, :] - points[None, :, :]) ** 2, axis=-1), axis=1)
                hidden = np.asarray(strand_labels[nearest].tolist(), dtype=np.int32)
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
                )
        raise RuntimeError("Failed to sample a valid scaffold after multiple attempts.")
