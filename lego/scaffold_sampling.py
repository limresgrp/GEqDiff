from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

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
        if self.family not in {"mixed", "chain", "sheet", "junction", "alpha_helix"}:
            raise ValueError(f"Unsupported scaffold family '{family}'.")

    def _choose_family(self, rng: np.random.Generator) -> str:
        if self.family != "mixed":
            return self.family
        probs = np.asarray(
            [
                max(0.05, 1.0 - self.bifurcation_probability),
                0.35,
                max(0.05, self.bifurcation_probability),
            ],
            dtype=np.float32,
        )
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        return str(rng.choice(np.asarray(["chain", "sheet", "junction"]), p=probs))

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

    def _sample_junction_branches(
        self,
        target_nodes: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        target_nodes = int(max(15, target_nodes))
        parent_len = int(max(6, target_nodes // 3))
        remaining = int(max(8, target_nodes - parent_len))
        child_len_1 = int(max(4, remaining // 2))
        child_len_2 = int(max(4, remaining - child_len_1))

        parent_vox = np.stack(
            [
                np.arange(parent_len, dtype=np.int32),
                np.zeros((parent_len,), dtype=np.int32),
                np.zeros((parent_len,), dtype=np.int32),
            ],
            axis=-1,
        ).astype(np.int32)
        junction = parent_vox[-1].astype(np.int32)

        candidate_dirs = [
            np.asarray([0, 1, 0], dtype=np.int32),
            np.asarray([0, -1, 0], dtype=np.int32),
            np.asarray([0, 0, 1], dtype=np.int32),
            np.asarray([0, 0, -1], dtype=np.int32),
        ]
        valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
        for i, d1 in enumerate(candidate_dirs):
            for d2 in candidate_dirs[i + 1 :]:
                u1 = _normalize(d1.astype(np.float32))
                u2 = _normalize(d2.astype(np.float32))
                angle = float(math.acos(float(np.clip(np.dot(u1, u2), -1.0, 1.0))))
                if angle >= self.junction_angle_min_rad:
                    valid_pairs.append((d1, d2))
        if len(valid_pairs) == 0:
            valid_pairs.append((np.asarray([0, 1, 0], dtype=np.int32), np.asarray([0, 0, 1], dtype=np.int32)))
        dir_1, dir_2 = valid_pairs[int(rng.integers(len(valid_pairs)))]

        child_1 = np.asarray([junction + (i + 1) * dir_1 for i in range(child_len_1)], dtype=np.int32)
        child_2 = np.asarray([junction + (i + 1) * dir_2 for i in range(child_len_2)], dtype=np.int32)
        occupied = {tuple(v.tolist()) for v in parent_vox}
        if any(tuple(v.tolist()) in occupied for v in child_1):
            raise RuntimeError("Child branch 1 intersects parent branch.")
        occupied.update(tuple(v.tolist()) for v in child_1)
        if any(tuple(v.tolist()) in occupied for v in child_2):
            raise RuntimeError("Child branch 2 intersects existing branches.")

        child_kind_1 = "helix" if float(rng.random()) < 0.5 else "sheet"
        child_kind_2 = "sheet" if child_kind_1 == "helix" else "helix"

        pos = np.concatenate([parent_vox, child_1, child_2], axis=0).astype(np.int32)
        branch_id = np.concatenate(
            [
                np.zeros((parent_vox.shape[0],), dtype=np.int32),
                np.full((child_1.shape[0],), fill_value=1, dtype=np.int32),
                np.full((child_2.shape[0],), fill_value=2, dtype=np.int32),
            ],
            axis=0,
        )
        seq_index = np.concatenate(
            [
                np.arange(parent_vox.shape[0], dtype=np.int32),
                np.arange(child_1.shape[0], dtype=np.int32),
                np.arange(child_2.shape[0], dtype=np.int32),
            ],
            axis=0,
        )
        parent = np.full((pos.shape[0],), fill_value=-1, dtype=np.int32)
        parent[1:parent_vox.shape[0]] = np.arange(parent_vox.shape[0] - 1, dtype=np.int32)
        child1_start = int(parent_vox.shape[0])
        child2_start = int(parent_vox.shape[0] + child_1.shape[0])
        parent[child1_start] = int(parent_vox.shape[0] - 1)
        parent[child2_start] = int(parent_vox.shape[0] - 1)
        if child_1.shape[0] > 1:
            parent[child1_start + 1 : child1_start + child_1.shape[0]] = np.arange(
                child1_start, child1_start + child_1.shape[0] - 1, dtype=np.int32
            )
        if child_2.shape[0] > 1:
            parent[child2_start + 1 : child2_start + child_2.shape[0]] = np.arange(
                child2_start, child2_start + child_2.shape[0] - 1, dtype=np.int32
            )
        branch_kind = {0: "chain", 1: child_kind_1, 2: child_kind_2}
        return pos.astype(np.int32), parent.astype(np.int32), branch_id.astype(np.int32), seq_index.astype(np.int32), branch_kind

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
                points, regime = self._sample_chain_polyline(
                    n_points=max(10, target_nodes),
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
                points, strand_labels = self._sample_sheet_polyline(n_points=max(12, target_nodes), rng=rng)
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
            # junction family
            try:
                pos, parent, branch_id, seq_index, branch_kind_map = self._sample_junction_branches(
                    target_nodes=target_nodes,
                    rng=rng,
                )
            except RuntimeError:
                continue
            if not (self.min_nodes <= int(pos.shape[0]) <= self.max_nodes):
                continue
            n = int(pos.shape[0])
            degree = self._build_tree_degree(parent)
            branch_kind = np.asarray([branch_kind_map[int(b)] for b in branch_id.tolist()], dtype=str)
            hidden = np.asarray([f"{branch_kind_map[int(b)]}_b{int(b)}" for b in branch_id.tolist()], dtype=str)
            return Scaffold(
                pos=pos.astype(np.float32),
                node_id=np.arange(n, dtype=np.int32),
                parent_id=parent.astype(np.int32),
                branch_id=branch_id.astype(np.int32),
                seq_index_in_branch=seq_index.astype(np.int32),
                degree_topology=degree.astype(np.int32),
                family="junction",
                branch_kind=branch_kind,
                hidden_label=hidden,
            )
        raise RuntimeError("Failed to sample a valid scaffold after multiple attempts.")
