"""Generate SH-defined target shapes and their discrete LEGO approximations."""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

from lego_blocks import LEGO_LIBRARY, NEIGHBOR_DIRS, iter_rotated_offsets
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from geqdiff.utils.contact_utils import build_brick_geometries, detect_brick_contacts
from geqdiff.utils.dipole_utils import assign_discrete_dipoles, dipole_strengths, normalize_dipole_directions
from utils import (
    DEFAULT_IRREPS,
    block_signature,
    build_surface_mesh,
    default_dataset_path,
    save_samples,
    voxelize_radial_shape,
)


class LegoDipoleEngine:
    def __init__(
        self,
        lmax=3,
        mesh_resolution=36,
        base_radius=5.0,
        radial_scale=0.55,
        min_radius=0.75,
        max_radius=4.0,
        occupancy_mode="solid",
        shell_thickness=1.1,
        shell_sparsity=0.0,
        structure_mode="sh",
        secondary_motif="mixed",
        secondary_min_bricks=16,
        secondary_max_bricks=32,
        secondary_nonlocal_min_sep=4,
        sequence_pos_max=None,
    ):
        if lmax != 3:
            raise ValueError("This demo currently supports lmax=3 only.")

        self.lmax = lmax
        self.irreps = DEFAULT_IRREPS
        self.library = LEGO_LIBRARY
        self.mesh_resolution = mesh_resolution
        self.base_radius = base_radius
        self.radial_scale = radial_scale
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.occupancy_mode = str(occupancy_mode).strip().lower()
        self.shell_thickness = float(shell_thickness)
        self.shell_sparsity = float(shell_sparsity)
        self.structure_mode = str(structure_mode).strip().lower()
        if self.structure_mode not in {"sh", "secondary"}:
            raise ValueError(f"Unsupported structure_mode '{structure_mode}'. Expected 'sh' or 'secondary'.")
        self.secondary_motif = str(secondary_motif).strip().lower()
        if self.secondary_motif not in {"mixed", "helix", "sheet"}:
            raise ValueError(f"Unsupported secondary_motif '{secondary_motif}'. Expected 'mixed', 'helix', or 'sheet'.")
        self.secondary_min_bricks = int(max(4, secondary_min_bricks))
        self.secondary_max_bricks = int(max(self.secondary_min_bricks, secondary_max_bricks))
        self.secondary_nonlocal_min_sep = int(max(1, secondary_nonlocal_min_sep))
        self.sequence_pos_max = int(sequence_pos_max) if sequence_pos_max is not None else int(self.secondary_max_bricks)
        self.sequence_pos_max = int(max(2, self.sequence_pos_max))

        self.block_order = sorted(
            self.library,
            key=lambda name: len(self.library[name]["offsets"]),
            reverse=True,
        )
        self.rotated_library = {
            name: list(iter_rotated_offsets(spec["offsets"]))
            for name, spec in self.library.items()
        }
        self.all_variants = []
        for name in sorted(self.rotated_library.keys()):
            for rotation, rotated_offsets in self.rotated_library[name]:
                self.all_variants.append((name, rotation.astype(np.float32), rotated_offsets.astype(np.int32)))

    def _sample_coefficients(self, rng):
        coefficients = np.zeros(((self.lmax + 1) ** 2,), dtype=np.float32)
        cursor = 0
        for l in range(self.lmax + 1):
            width = 2 * l + 1
            coefficients[cursor : cursor + width] = rng.normal(
                loc=0.0,
                scale=0.9 / (l + 1),
                size=width,
            )
            cursor += width
        coefficients[0] += 1.0
        return coefficients.astype(np.float32)

    def _choose_secondary_motif(self, rng):
        if self.secondary_motif != "mixed":
            return self.secondary_motif
        return "helix" if float(rng.random()) < 0.5 else "sheet"

    def _normalize_vec(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return np.zeros_like(vector, dtype=np.float32)
        return (vector / norm).astype(np.float32)

    def _resample_polyline(self, points, count):
        points = np.asarray(points, dtype=np.float32)
        count = int(max(2, count))
        if points.shape[0] <= 1:
            return np.repeat(points[:1], count, axis=0).astype(np.float32)
        seg = np.linalg.norm(points[1:] - points[:-1], axis=1).astype(np.float32)
        total = float(seg.sum())
        if total <= 1e-8:
            return np.repeat(points[:1], count, axis=0).astype(np.float32)
        cumulative = np.concatenate([np.zeros((1,), dtype=np.float32), np.cumsum(seg)]).astype(np.float32)
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

    def _bezier_curve(self, p0, p1, p2, p3, count):
        count = int(max(2, count))
        t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
        omt = 1.0 - t
        curve = (
            (omt ** 3) * np.asarray(p0, dtype=np.float32)[None, :]
            + 3.0 * (omt ** 2) * t * np.asarray(p1, dtype=np.float32)[None, :]
            + 3.0 * omt * (t ** 2) * np.asarray(p2, dtype=np.float32)[None, :]
            + (t ** 3) * np.asarray(p3, dtype=np.float32)[None, :]
        )
        return curve.astype(np.float32)

    def _build_helix_loop_helix(self, point_count, rng):
        point_count = int(max(18, point_count))
        loop_n = max(8, point_count // 7)
        h_total = point_count - loop_n
        h1_n = max(6, h_total // 2)
        h2_n = max(6, h_total - h1_n)

        radius = float(rng.uniform(1.7, 2.3))
        pitch_turn = float(rng.uniform(1.4, 1.9))
        turns_1 = float(max(1.3, h1_n / 7.2))
        turns_2 = float(max(1.3, h2_n / 7.2))
        theta_1 = np.linspace(0.0, 2.0 * math.pi * turns_1, h1_n, dtype=np.float32)
        h1 = np.stack(
            [
                radius * np.cos(theta_1),
                radius * np.sin(theta_1),
                pitch_turn * theta_1 / (2.0 * math.pi),
            ],
            axis=-1,
        ).astype(np.float32)

        axis_shift = np.asarray(
            [
                float(rng.uniform(2.0 * radius, 2.6 * radius)),
                float(rng.uniform(-0.7, 0.7)),
                float(rng.uniform(0.3, 1.2)) + 0.55 * pitch_turn * turns_1,
            ],
            dtype=np.float32,
        )
        theta_2 = np.linspace(0.3 * math.pi, 0.3 * math.pi + 2.0 * math.pi * turns_2, h2_n, dtype=np.float32)
        h2 = np.stack(
            [
                axis_shift[0] + radius * np.cos(-theta_2),
                axis_shift[1] + radius * np.sin(-theta_2),
                axis_shift[2] + pitch_turn * (turns_2 - theta_2 / (2.0 * math.pi)),
            ],
            axis=-1,
        ).astype(np.float32)

        start = h1[-1]
        end = h2[0]
        tan_start = self._normalize_vec(h1[-1] - h1[-2])
        tan_end = self._normalize_vec(h2[1] - h2[0])
        chord = end - start
        chord_norm = float(np.linalg.norm(chord))
        if chord_norm <= 1e-8:
            loop = np.repeat(start[None, :], loop_n, axis=0).astype(np.float32)
        else:
            normal = np.cross(tan_start, tan_end)
            if float(np.linalg.norm(normal)) <= 1e-6:
                normal = np.cross(tan_start, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
            normal = self._normalize_vec(normal)
            bend = float(rng.uniform(0.5, 1.2))
            c1 = start + tan_start * (0.45 * chord_norm) + normal * bend
            c2 = end - tan_end * (0.45 * chord_norm) + normal * bend
            loop = self._bezier_curve(start, c1, c2, end, loop_n)

        points = np.concatenate([h1, loop[1:-1], h2], axis=0).astype(np.float32)
        points = self._resample_polyline(points, point_count)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32)

    def _build_beta_sheet_with_loops(self, point_count, rng):
        point_count = int(max(16, point_count))
        n_strands = 3 if point_count >= 34 else 2
        loop_n = max(6, point_count // 11)
        strand_points_total = point_count - (n_strands - 1) * loop_n
        strand_n = max(6, strand_points_total // n_strands)
        strand_lengths = [strand_n for _ in range(n_strands)]
        for extra in range(strand_points_total - strand_n * n_strands):
            strand_lengths[extra % n_strands] += 1

        length = float(rng.uniform(7.5, 11.5))
        spacing = float(rng.uniform(2.0, 2.6))
        pleat = float(rng.uniform(0.25, 0.45))
        pieces = []
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
                tan_start = self._normalize_vec(prev[-1] - prev[-2])
                tan_end = self._normalize_vec(strand[1] - strand[0])
                chord = end - start
                chord_norm = float(np.linalg.norm(chord))
                if chord_norm <= 1e-8:
                    loop = np.repeat(start[None, :], loop_n, axis=0).astype(np.float32)
                else:
                    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
                    c1 = start + tan_start * (0.3 * chord_norm) + normal * 0.9
                    c2 = end - tan_end * (0.3 * chord_norm) + normal * 0.9
                    loop = self._bezier_curve(start, c1, c2, end, loop_n)
                pieces.append(loop[1:-1])
            pieces.append(strand)
            prev = strand

        points = np.concatenate(pieces, axis=0).astype(np.float32)
        points = self._resample_polyline(points, point_count)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32)

    def _secondary_backbone_points(self, rng, point_count, motif=None):
        motif = self._choose_secondary_motif(rng) if motif is None else str(motif)
        if motif == "helix":
            points = self._build_helix_loop_helix(point_count=point_count, rng=rng)
        else:
            points = self._build_beta_sheet_with_loops(point_count=point_count, rng=rng)
        return points.astype(np.float32), motif

    def _connect_lattice_points(self, start, end):
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

    def _polyline_to_connected_voxels(self, points):
        points = np.asarray(points, dtype=np.float32)
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.int32)
        lattice = np.rint(points).astype(np.int32)
        ordered = [lattice[0].copy()]
        for idx in range(1, lattice.shape[0]):
            segment = self._connect_lattice_points(ordered[-1], lattice[idx])
            for voxel in segment[1:]:
                if not np.array_equal(voxel, ordered[-1]):
                    ordered.append(voxel.copy())

        unique = []
        seen = set()
        for voxel in ordered:
            key = tuple(int(v) for v in voxel.tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(np.asarray(voxel, dtype=np.int32))
        return np.asarray(unique, dtype=np.int32)

    def _candidate_anchor_offsets(self):
        offsets = []
        for radius in range(0, 3):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        offsets.append(np.asarray([dx, dy, dz], dtype=np.int32))
        return offsets

    def _place_bricks_from_backbone(self, lattice_points, rng):
        lattice_points = np.asarray(lattice_points, dtype=np.int32)
        occupied_voxels = set()
        placements = []
        anchor_offsets = self._candidate_anchor_offsets()

        variant_order = []
        for name, rotation, rotated in self.all_variants:
            if name == "1x1":
                weight = 0.55
            elif name == "1x2":
                weight = 0.30
            elif name == "L-shape":
                weight = 0.10
            else:
                weight = 0.05
            variant_order.append((weight, name, rotation, rotated))
        weights = np.asarray([v[0] for v in variant_order], dtype=np.float32)
        weights = weights / np.clip(weights.sum(), 1e-8, None)

        for seq_idx, base_point in enumerate(lattice_points):
            placed = None
            candidate_indices = rng.choice(len(variant_order), size=min(24, len(variant_order)), replace=True, p=weights)
            for variant_idx in candidate_indices.tolist():
                _, name, rotation, rotated_offsets = variant_order[int(variant_idx)]
                for anchor_delta in anchor_offsets:
                    anchor = base_point + anchor_delta
                    world = rotated_offsets + anchor[None, :]
                    world_keys = tuple(tuple(int(v) for v in row) for row in world.tolist())
                    if any(key in occupied_voxels for key in world_keys):
                        continue
                    contact_score = sum(
                        tuple((np.asarray(key, dtype=np.int32) + direction).tolist()) in occupied_voxels
                        for key in world_keys
                        for direction in NEIGHBOR_DIRS
                    )
                    placed = {
                        "name": str(name),
                        "anchor": anchor.astype(np.float32),
                        "rotation": rotation.astype(np.float32),
                        "world_keys": world_keys,
                        "sequence_index": int(seq_idx),
                        "score": int(contact_score),
                    }
                    break
                if placed is not None:
                    break

            if placed is None:
                for anchor_delta in anchor_offsets:
                    anchor = base_point + anchor_delta
                    key = tuple(anchor.tolist())
                    if key in occupied_voxels:
                        continue
                    placed = {
                        "name": "1x1",
                        "anchor": anchor.astype(np.float32),
                        "rotation": np.eye(3, dtype=np.float32),
                        "world_keys": (key,),
                        "sequence_index": int(seq_idx),
                        "score": 0,
                    }
                    break

            if placed is None:
                raise RuntimeError("Failed to place a non-overlapping brick on secondary backbone.")

            occupied_voxels.update(placed["world_keys"])
            placements.append(placed)

        return placements

    def _sequence_positions(self, count):
        count = int(count)
        if count <= 1:
            return np.zeros((count, 1), dtype=np.int32)
        values = np.linspace(0, self.sequence_pos_max - 1, num=count, dtype=np.float32)
        return np.rint(values).astype(np.int32).reshape(count, 1)

    def _assign_secondary_dipoles(self, sample, sequence_positions):
        sequence_positions = np.asarray(sequence_positions, dtype=np.int32).reshape(-1)
        geometries = build_brick_geometries(sample)
        contact_data = detect_brick_contacts(geometries)
        all_pairs = np.asarray(contact_data["all_face_contact_pairs"], dtype=np.int64)
        all_dirs = np.asarray(contact_data["all_face_contact_dirs"], dtype=np.float32)
        num_nodes = int(len(sample["brick_types"]))
        if all_pairs.shape[0] == 0:
            return np.zeros((num_nodes, 3), dtype=np.float32)

        preferences = np.zeros((num_nodes, 3), dtype=np.float32)
        for (src, dst), face_dir in zip(all_pairs, all_dirs):
            src_i = int(src)
            dst_i = int(dst)
            if abs(int(sequence_positions[src_i]) - int(sequence_positions[dst_i])) < self.secondary_nonlocal_min_sep:
                continue
            direction = np.asarray(face_dir, dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                continue
            direction = direction / norm
            preferences[src_i] += direction
            preferences[dst_i] += direction

        dipoles = np.zeros((num_nodes, 3), dtype=np.float32)
        norms = np.linalg.norm(preferences, axis=-1, keepdims=True)
        valid = (norms.reshape(-1) > 1e-6)
        if np.any(valid):
            dipoles[valid] = preferences[valid] / np.clip(norms[valid], 1e-8, None)
        return dipoles.astype(np.float32)

    def _contact_score(self, world_keys, occupied):
        world_set = set(world_keys)
        contacts = 0
        for key in world_keys:
            key_arr = np.asarray(key, dtype=np.int32)
            for direction in NEIGHBOR_DIRS:
                neighbor = tuple((key_arr + direction).tolist())
                if neighbor in occupied and neighbor not in world_set:
                    contacts += 1
        return contacts

    def _pick_brick(self, uncovered, occupied):
        uncovered_anchors = sorted(uncovered)
        best = None

        for name in self.block_order:
            for rotation, rotated_offsets in self.rotated_library[name]:
                for anchor_key in uncovered_anchors:
                    anchor = np.asarray(anchor_key, dtype=np.int32)
                    world = rotated_offsets + anchor[None, :]
                    world_keys = tuple(tuple(int(v) for v in row) for row in world.tolist())
                    if not all(key in uncovered for key in world_keys):
                        continue

                    score = (
                        len(world_keys),
                        self._contact_score(world_keys, occupied),
                        -int(np.ptp(world, axis=0).sum()),
                    )
                    if best is None or score > best["score"]:
                        best = {
                            "score": score,
                            "name": name,
                            "anchor": anchor.astype(np.float32),
                            "rotation": rotation.astype(np.float32),
                            "world_keys": world_keys,
                        }

        return best

    def approximate_voxels(self, target_voxels):
        occupied = {tuple(v) for v in np.asarray(target_voxels, dtype=np.int32).tolist()}
        uncovered = set(occupied)
        placements = []

        while uncovered:
            placement = self._pick_brick(uncovered, occupied)
            if placement is None:
                anchor = np.asarray(sorted(uncovered)[0], dtype=np.float32)
                placement = {
                    "name": "1x1",
                    "anchor": anchor,
                    "rotation": np.eye(3, dtype=np.float32),
                    "world_keys": (tuple(anchor.astype(int).tolist()),),
                }

            uncovered.difference_update(placement["world_keys"])
            placements.append(placement)

        return placements

    def build_sample(self, coefficients=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.structure_mode == "secondary":
            return self._build_secondary_sample(rng=rng)

        if coefficients is None:
            coefficients = self._sample_coefficients(rng)
        coefficients = np.asarray(coefficients, dtype=np.float32)

        mesh_x, mesh_y, mesh_z, _ = build_surface_mesh(
            coefficients,
            resolution=self.mesh_resolution,
            base_radius=self.base_radius,
            radial_scale=self.radial_scale,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
        )
        target_voxels = voxelize_radial_shape(
            coefficients,
            base_radius=self.base_radius,
            radial_scale=self.radial_scale,
            min_radius=self.min_radius,
            max_radius=self.max_radius,
            occupancy_mode=self.occupancy_mode,
            shell_thickness=self.shell_thickness,
            shell_sparsity=self.shell_sparsity,
            rng=rng,
        )
        placements = self.approximate_voxels(target_voxels)

        anchors = []
        rotations = []
        types = []
        features = []
        for placement in placements:
            name = placement["name"]
            rotation = placement["rotation"]
            anchors.append(placement["anchor"])
            rotations.append(rotation)
            types.append(name)
            features.append(
                block_signature(
                    self.library[name]["offsets"],
                    rotation=rotation,
                    lmax=self.lmax,
                )
            )

        anchors = np.asarray(anchors, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)
        types = np.asarray(types)
        features = np.asarray(features, dtype=np.float32)
        sample = {
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": types,
        }
        geometries = build_brick_geometries(sample)
        contact_data = detect_brick_contacts(geometries)
        dipoles = assign_discrete_dipoles(
            rotations=rotations,
            contact_pairs=np.asarray(contact_data["contact_pairs"], dtype=np.int64),
            contact_face_dirs=np.asarray(contact_data["contact_face_dirs"], dtype=np.float32),
            all_face_contact_pairs=np.asarray(contact_data["all_face_contact_pairs"], dtype=np.int64),
            all_face_contact_dirs=np.asarray(contact_data["all_face_contact_dirs"], dtype=np.float32),
            rng=rng,
        )
        dipole_dirs = normalize_dipole_directions(dipoles)
        dipole_mag = dipole_strengths(dipoles)

        return {
            "coefficients": coefficients,
            "irreps": str(self.irreps),
            "mesh_x": mesh_x,
            "mesh_y": mesh_y,
            "mesh_z": mesh_z,
            "target_voxels": np.asarray(target_voxels, dtype=np.int32),
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": types,
            "brick_features": features,
            "brick_dipoles": dipoles,
            "brick_dipole_directions": dipole_dirs,
            "brick_dipole_strengths": dipole_mag,
            "base_radius": np.float32(self.base_radius),
            "radial_scale": np.float32(self.radial_scale),
            "min_radius": np.float32(self.min_radius),
            "max_radius": np.float32(self.max_radius),
            "occupancy_mode": np.asarray(self.occupancy_mode),
            "shell_thickness": np.float32(self.shell_thickness),
            "shell_sparsity": np.float32(self.shell_sparsity),
            # Compatibility aliases.
            "pos": anchors,
            "rotations": rotations,
            "types": types,
            "features": features,
            "dipoles": dipoles,
            "generation_mode": np.asarray("sh"),
        }

    def _build_secondary_sample(self, rng):
        target_bricks = int(rng.integers(self.secondary_min_bricks, self.secondary_max_bricks + 1))
        point_count = int(max(20, target_bricks * 3))
        motif_choice = self._choose_secondary_motif(rng)
        best = None

        for _ in range(8):
            backbone_points, motif = self._secondary_backbone_points(rng, point_count=point_count, motif=motif_choice)
            backbone_voxels = self._polyline_to_connected_voxels(backbone_points)
            if backbone_voxels.shape[0] == 0:
                point_count = int(max(20, point_count + 6))
                continue
            placements = self.approximate_voxels(backbone_voxels)
            n_placed = int(len(placements))

            diff_to_target = abs(n_placed - target_bricks)
            candidate = {
                "motif": motif,
                "backbone_points": backbone_points,
                "backbone_voxels": backbone_voxels,
                "placements": placements,
                "n_placed": n_placed,
                "diff": diff_to_target,
            }
            if best is None or int(candidate["diff"]) < int(best["diff"]):
                best = candidate

            if self.secondary_min_bricks <= n_placed <= self.secondary_max_bricks:
                best = candidate
                break
            if n_placed < self.secondary_min_bricks:
                point_count = int(point_count * 1.20) + 2
            else:
                point_count = max(20, int(point_count * 0.85))

        if best is None:
            raise RuntimeError("Failed to generate a secondary scaffold.")

        motif = str(best["motif"])
        backbone_points = np.asarray(best["backbone_points"], dtype=np.float32)
        target_voxels = np.asarray(best["backbone_voxels"], dtype=np.int32)
        placements = list(best["placements"])

        # Sequence order follows progression along the mathematical backbone.
        backbone_order = np.asarray(backbone_points, dtype=np.float32)
        if backbone_order.shape[0] > 0:
            scored_placements = []
            for placement in placements:
                world = np.asarray(placement["world_keys"], dtype=np.float32)
                center = world.mean(axis=0)
                d2 = np.sum((backbone_order - center[None, :]) ** 2, axis=1)
                scored_placements.append((int(np.argmin(d2)), center, placement))
            scored_placements.sort(key=lambda item: (item[0], float(item[1][0]), float(item[1][1]), float(item[1][2])))
            placements = [item[2] for item in scored_placements]

        anchors = []
        rotations = []
        types = []
        features = []
        for placement in placements:
            name = placement["name"]
            rotation = placement["rotation"]
            anchors.append(placement["anchor"])
            rotations.append(rotation)
            types.append(name)
            features.append(
                block_signature(
                    self.library[name]["offsets"],
                    rotation=rotation,
                    lmax=self.lmax,
                )
            )
        anchors = np.asarray(anchors, dtype=np.float32)
        rotations = np.asarray(rotations, dtype=np.float32)
        types = np.asarray(types)
        features = np.asarray(features, dtype=np.float32)

        sequence_positions = self._sequence_positions(len(types))
        sample = {
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": types,
        }
        geometries = build_brick_geometries(sample)
        all_world_voxels = np.concatenate([np.asarray(g["world_voxels"], dtype=np.float32) for g in geometries], axis=0)
        occupied_voxels = np.unique(np.rint(all_world_voxels).astype(np.int32), axis=0)
        dipoles = self._assign_secondary_dipoles(sample=sample, sequence_positions=sequence_positions)
        dipole_dirs = normalize_dipole_directions(dipoles)
        dipole_mag = dipole_strengths(dipoles)

        return {
            "coefficients": np.zeros(((self.lmax + 1) ** 2,), dtype=np.float32),
            "irreps": str(self.irreps),
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": types,
            "brick_features": features,
            "brick_dipoles": dipoles,
            "brick_dipole_directions": dipole_dirs,
            "brick_dipole_strengths": dipole_mag,
            "target_voxels": occupied_voxels.astype(np.int32),
            "secondary_backbone_voxels": target_voxels.astype(np.int32),
            "secondary_backbone_points": backbone_points.astype(np.float32),
            "brick_sequence_position": sequence_positions.astype(np.int32),
            "sequence_pos_max": np.asarray(self.sequence_pos_max, dtype=np.int32),
            "secondary_motif": np.asarray(str(motif)),
            "occupancy_mode": np.asarray("secondary"),
            "generation_mode": np.asarray("secondary"),
            # Compatibility aliases.
            "pos": anchors,
            "rotations": rotations,
            "types": types,
            "features": features,
            "dipoles": dipoles,
            "sequence_position": sequence_positions.astype(np.int32),
        }

    def generate_task(self, coefficients=None, seed=None):
        rng = np.random.default_rng(seed)
        return self.build_sample(coefficients=coefficients, rng=rng)

    def generate_dataset(self, n_samples=1, seed=None):
        rng = np.random.default_rng(seed)
        return [self.build_sample(rng=rng) for _ in range(n_samples)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1, help="Number of LEGO samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for repeatable coefficients.")
    parser.add_argument(
        "--path",
        type=str,
        default=str(default_dataset_path()),
        help="Output dataset path.",
    )
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=36,
        help="Latitude resolution for the target SH mesh.",
    )
    parser.add_argument("--base-radius", type=float, default=5.0, help="Base radius of the SH radial field.")
    parser.add_argument("--radial-scale", type=float, default=0.55, help="Scale applied to the SH radial perturbation.")
    parser.add_argument("--min-radius", type=float, default=0.75, help="Lower clip for the SH radial field.")
    parser.add_argument("--max-radius", type=float, default=4.0, help="Upper clip for the SH radial field.")
    parser.add_argument(
        "--structure-mode",
        type=str,
        default="sh",
        choices=["sh", "secondary"],
        help="Generation family: SH voxelization or secondary-structure motifs.",
    )
    parser.add_argument(
        "--occupancy-mode",
        type=str,
        default="solid",
        choices=["solid", "shell"],
        help="Voxelization mode for the SH target.",
    )
    parser.add_argument(
        "--shell-thickness",
        type=float,
        default=1.1,
        help="Thickness of the occupied surface band when --occupancy-mode shell.",
    )
    parser.add_argument(
        "--shell-sparsity",
        type=float,
        default=0.0,
        help="Connectivity-preserving thinning fraction for shell occupancy in [0, 0.95].",
    )
    parser.add_argument(
        "--secondary-motif",
        type=str,
        default="mixed",
        choices=["mixed", "helix", "sheet"],
        help="Motif family used when --structure-mode secondary.",
    )
    parser.add_argument(
        "--secondary-min-bricks",
        type=int,
        default=16,
        help="Minimum number of bricks per sample in secondary mode.",
    )
    parser.add_argument(
        "--secondary-max-bricks",
        type=int,
        default=32,
        help="Maximum number of bricks per sample in secondary mode.",
    )
    parser.add_argument(
        "--sequence-pos-max",
        type=int,
        default=None,
        help="Maximum sequence index for brick positions (0..max-1). Default uses secondary-max-bricks.",
    )
    parser.add_argument(
        "--secondary-nonlocal-min-sep",
        type=int,
        default=4,
        help="Minimum sequence separation to consider contacts for dipole assignment in secondary mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    engine = LegoDipoleEngine(
        mesh_resolution=args.mesh_resolution,
        base_radius=args.base_radius,
        radial_scale=args.radial_scale,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        occupancy_mode=args.occupancy_mode,
        shell_thickness=args.shell_thickness,
        shell_sparsity=args.shell_sparsity,
        structure_mode=args.structure_mode,
        secondary_motif=args.secondary_motif,
        secondary_min_bricks=args.secondary_min_bricks,
        secondary_max_bricks=args.secondary_max_bricks,
        secondary_nonlocal_min_sep=args.secondary_nonlocal_min_sep,
        sequence_pos_max=args.sequence_pos_max,
    )
    samples = engine.generate_dataset(n_samples=args.samples, seed=args.seed)
    save_samples(args.path, samples)

    print("--- LEGO Dataset Generated ---")
    print(f"Samples: {len(samples)}")
    print(f"Irreps: {engine.irreps}")
    print(f"Saved to: {args.path}")
    for idx, sample in enumerate(samples):
        mode = sample.get("generation_mode", sample.get("occupancy_mode", engine.occupancy_mode))
        if isinstance(mode, np.ndarray):
            mode = str(mode.reshape(-1)[0])
        print(
            f"Sample {idx}: voxels={len(sample.get('target_voxels', []))}, "
            f"bricks={len(sample['brick_types'])}, "
            f"mode={mode}, "
            f"coeff_norm={np.linalg.norm(sample['coefficients']):.3f}"
        )


LegoMultiPortEngine = LegoDipoleEngine
