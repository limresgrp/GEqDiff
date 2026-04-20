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
        secondary_target_polar_fraction=0.34,
        secondary_min_polar_fraction=0.24,
        secondary_max_consecutive_polar=2,
        secondary_polar_min_spacing=3,
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
        self.secondary_target_polar_fraction = float(np.clip(secondary_target_polar_fraction, 0.05, 0.95))
        self.secondary_min_polar_fraction = float(
            np.clip(secondary_min_polar_fraction, 0.0, self.secondary_target_polar_fraction)
        )
        self.secondary_max_consecutive_polar = int(max(1, secondary_max_consecutive_polar))
        self.secondary_polar_min_spacing = int(max(1, secondary_polar_min_spacing))
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
        return "mixed"

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

    def _rotation_from_to(self, source, target):
        source = self._normalize_vec(source)
        target = self._normalize_vec(target)
        if float(np.linalg.norm(source)) <= 1e-8 or float(np.linalg.norm(target)) <= 1e-8:
            return np.eye(3, dtype=np.float32)
        cross = np.cross(source, target).astype(np.float32)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm <= 1e-8:
            if dot > 0.0:
                return np.eye(3, dtype=np.float32)
            # 180-degree rotation around any axis orthogonal to source.
            aux = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(aux, source))) > 0.9:
                aux = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            axis = self._normalize_vec(np.cross(source, aux))
            x, y, z = axis.tolist()
            # Rodrigues with theta=pi: R = -I + 2 uu^T
            return np.asarray(
                [
                    [-1.0 + 2.0 * x * x, 2.0 * x * y, 2.0 * x * z],
                    [2.0 * y * x, -1.0 + 2.0 * y * y, 2.0 * y * z],
                    [2.0 * z * x, 2.0 * z * y, -1.0 + 2.0 * z * z],
                ],
                dtype=np.float32,
            )

        vx, vy, vz = (cross / cross_norm).tolist()
        K = np.asarray(
            [
                [0.0, -vz, vy],
                [vz, 0.0, -vx],
                [-vy, vx, 0.0],
            ],
            dtype=np.float32,
        )
        angle = float(math.acos(dot))
        I = np.eye(3, dtype=np.float32)
        R = I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
        return R.astype(np.float32)

    def _point_labels_from_nearest(self, query_points, reference_points, reference_labels):
        query_points = np.asarray(query_points, dtype=np.float32)
        reference_points = np.asarray(reference_points, dtype=np.float32)
        reference_labels = np.asarray(reference_labels, dtype=np.int32).reshape(-1)
        if reference_points.shape[0] == 0:
            return np.full((query_points.shape[0],), fill_value=-1, dtype=np.int32)
        d2 = np.sum((query_points[:, None, :] - reference_points[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(d2, axis=1)
        return reference_labels[nearest].astype(np.int32)

    def _build_single_helix(self, point_count, rng):
        point_count = int(max(8, point_count))
        # Favor taller turns so lattice discretization keeps the helical climb visible.
        radius = float(rng.uniform(1.35, 1.95))
        pitch_turn = float(rng.uniform(2.55, 3.45))
        turns = float(max(1.0, point_count / 7.0))
        theta = np.linspace(0.0, 2.0 * math.pi * turns, point_count, dtype=np.float32)
        handedness = -1.0 if float(rng.random()) < 0.5 else 1.0
        helix = np.stack(
            [
                radius * np.cos(handedness * theta),
                radius * np.sin(handedness * theta),
                pitch_turn * theta / (2.0 * math.pi),
            ],
            axis=-1,
        ).astype(np.float32)
        helix -= helix.mean(axis=0, keepdims=True)
        labels = np.full((helix.shape[0],), fill_value=-1, dtype=np.int32)
        return helix.astype(np.float32), labels

    def _build_helix_loop_helix(self, point_count, rng):
        point_count = int(max(18, point_count))
        loop_n = max(8, point_count // 7)
        h_total = point_count - loop_n
        h1_n = max(6, h_total // 2)
        h2_n = max(6, h_total - h1_n)

        # Increase vertical rise per turn to avoid flat/cylindrical-looking helices.
        radius = float(rng.uniform(1.45, 2.05))
        pitch_turn = float(rng.uniform(2.45, 3.35))
        turns_1 = float(max(1.3, h1_n / 6.8))
        turns_2 = float(max(1.3, h2_n / 6.8))
        handedness_1 = -1.0 if float(rng.random()) < 0.5 else 1.0
        handedness_2 = -1.0 if float(rng.random()) < 0.5 else 1.0
        theta_1 = np.linspace(0.0, 2.0 * math.pi * turns_1, h1_n, dtype=np.float32)
        h1 = np.stack(
            [
                radius * np.cos(handedness_1 * theta_1),
                radius * np.sin(handedness_1 * theta_1),
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
                axis_shift[0] + radius * np.cos(-handedness_2 * theta_2),
                axis_shift[1] + radius * np.sin(-handedness_2 * theta_2),
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

        raw_points = np.concatenate([h1, loop[1:-1], h2], axis=0).astype(np.float32)
        raw_labels = np.full((raw_points.shape[0],), fill_value=-1, dtype=np.int32)
        points = self._resample_polyline(raw_points, point_count)
        labels = self._point_labels_from_nearest(points, raw_points, raw_labels)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32), labels.astype(np.int32)

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
        piece_labels = []
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
                loop_points = loop[1:-1].astype(np.float32)
                pieces.append(loop_points)
                piece_labels.append(np.full((loop_points.shape[0],), fill_value=-1, dtype=np.int32))
            pieces.append(strand)
            piece_labels.append(np.full((strand.shape[0],), fill_value=int(strand_idx), dtype=np.int32))
            prev = strand

        raw_points = np.concatenate(pieces, axis=0).astype(np.float32)
        raw_labels = np.concatenate(piece_labels, axis=0).astype(np.int32)
        points = self._resample_polyline(raw_points, point_count)
        labels = self._point_labels_from_nearest(points, raw_points, raw_labels)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32), labels.astype(np.int32)

    def _build_mixed_helix_sheet(self, point_count, rng):
        point_count = int(max(28, point_count))
        n_h1 = max(8, point_count // 4)
        n_sheet = max(10, point_count // 2)
        n_h2 = max(8, point_count - n_h1 - n_sheet)
        n_loop = max(6, point_count // 10)

        helix_a, labels_a = self._build_single_helix(n_h1, rng)
        sheet, labels_sheet = self._build_beta_sheet_with_loops(n_sheet, rng)
        helix_b, labels_b = self._build_single_helix(n_h2, rng)

        sheet += np.asarray([5.8, 0.5, 0.6], dtype=np.float32)
        helix_b += np.asarray([10.9, -0.4, 0.4], dtype=np.float32)

        loop1 = self._bezier_curve(
            helix_a[-1],
            helix_a[-1] + np.asarray([1.2, 0.2, 1.0], dtype=np.float32),
            sheet[0] + np.asarray([-1.2, 0.1, 1.0], dtype=np.float32),
            sheet[0],
            n_loop,
        ).astype(np.float32)
        loop2 = self._bezier_curve(
            sheet[-1],
            sheet[-1] + np.asarray([1.2, -0.2, 0.9], dtype=np.float32),
            helix_b[0] + np.asarray([-1.2, 0.1, 1.0], dtype=np.float32),
            helix_b[0],
            n_loop,
        ).astype(np.float32)

        raw_points = np.concatenate([helix_a, loop1[1:-1], sheet, loop2[1:-1], helix_b], axis=0).astype(np.float32)
        raw_labels = np.concatenate(
            [
                labels_a.astype(np.int32),
                np.full((max(0, loop1.shape[0] - 2),), fill_value=-1, dtype=np.int32),
                labels_sheet.astype(np.int32),
                np.full((max(0, loop2.shape[0] - 2),), fill_value=-1, dtype=np.int32),
                labels_b.astype(np.int32),
            ],
            axis=0,
        ).astype(np.int32)

        points = self._resample_polyline(raw_points, point_count)
        labels = self._point_labels_from_nearest(points, raw_points, raw_labels)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32), labels.astype(np.int32)

    def _segment_min_points(self, segment_kind):
        return 9 if str(segment_kind) == "helix" else 12

    def _recipe_min_points(self, recipe):
        recipe = [str(kind) for kind in recipe]
        if len(recipe) == 0:
            return 0
        return int(sum(self._segment_min_points(kind) for kind in recipe) + max(0, len(recipe) - 1) * 5)

    def _sample_mixed_recipe(self, point_count, rng, target_bricks=None):
        if target_bricks is None:
            max_segments = 4
        else:
            target_bricks = int(target_bricks)
            if target_bricks <= 22:
                max_segments = 2
            elif target_bricks <= 26:
                max_segments = 3
            else:
                max_segments = 4
        for _ in range(64):
            n_alpha = int(rng.integers(0, 3))
            n_beta = int(rng.integers(0, 3))
            if n_alpha + n_beta == 0:
                continue

            if n_alpha == 0:
                recipe = ["sheet"] * n_beta
            elif n_beta == 0:
                recipe = ["helix"] * n_alpha
            else:
                recipe = []
                remaining_alpha = n_alpha
                remaining_beta = n_beta
                current = "helix" if float(rng.random()) < 0.5 else "sheet"
                while remaining_alpha > 0 or remaining_beta > 0:
                    if current == "helix":
                        if remaining_alpha > 0:
                            recipe.append("helix")
                            remaining_alpha -= 1
                        elif remaining_beta > 0:
                            recipe.append("sheet")
                            remaining_beta -= 1
                        current = "sheet"
                    else:
                        if remaining_beta > 0:
                            recipe.append("sheet")
                            remaining_beta -= 1
                        elif remaining_alpha > 0:
                            recipe.append("helix")
                            remaining_alpha -= 1
                        current = "helix"
                if len(recipe) >= 3 and float(rng.random()) < 0.35:
                    swap_idx = int(rng.integers(0, len(recipe) - 1))
                    recipe[swap_idx], recipe[swap_idx + 1] = recipe[swap_idx + 1], recipe[swap_idx]
            if len(recipe) > max_segments:
                continue
            return recipe
        return ["helix"]

    def _build_segment_curve(self, segment_kind, point_count, rng):
        segment_kind = str(segment_kind)
        point_count = int(max(self._segment_min_points(segment_kind), point_count))
        if segment_kind == "helix":
            return self._build_single_helix(point_count, rng)
        return self._build_beta_sheet_with_loops(point_count, rng)

    def _build_mixed_random_combo(self, point_count, rng, recipe=None):
        point_count = int(max(24, point_count))
        recipe = self._sample_mixed_recipe(point_count=point_count, rng=rng) if recipe is None else list(recipe)
        if len(recipe) == 1:
            pts, labels = self._build_segment_curve(recipe[0], point_count=point_count, rng=rng)
            return pts.astype(np.float32), labels.astype(np.int32), "+".join(recipe)

        loop_n = max(5, point_count // (8 + 2 * len(recipe)))
        loop_total = loop_n * (len(recipe) - 1)
        min_total = sum(self._segment_min_points(kind) for kind in recipe)
        if point_count - loop_total < min_total:
            loop_n = max(3, (point_count - min_total) // max(1, len(recipe) - 1))
            loop_total = loop_n * (len(recipe) - 1)
        alloc_total = max(min_total, point_count - loop_total)
        mins = np.asarray([self._segment_min_points(kind) for kind in recipe], dtype=np.int32)
        extras = int(max(0, alloc_total - int(mins.sum())))
        weights = np.asarray([1.0 if kind == "helix" else 1.15 for kind in recipe], dtype=np.float32)
        if float(weights.sum()) <= 1e-8:
            weights = np.ones_like(weights, dtype=np.float32)
        raw_extra = extras * (weights / float(weights.sum()))
        add = np.floor(raw_extra).astype(np.int32)
        remainder = int(extras - int(add.sum()))
        if remainder > 0:
            frac = raw_extra - add.astype(np.float32)
            order = np.argsort(-frac)
            for idx in order[:remainder]:
                add[int(idx)] += 1
        seg_counts = (mins + add).astype(np.int32).tolist()

        built_segments = []
        strand_offset = 0
        for kind, seg_count in zip(recipe, seg_counts):
            seg_points, seg_labels = self._build_segment_curve(kind, point_count=int(seg_count), rng=rng)
            seg_labels = np.asarray(seg_labels, dtype=np.int32).reshape(-1)
            if kind == "sheet":
                mask = seg_labels >= 0
                if np.any(mask):
                    seg_labels = seg_labels.copy()
                    seg_labels[mask] += int(strand_offset)
                    strand_offset = int(seg_labels[mask].max()) + 1
            built_segments.append((np.asarray(seg_points, dtype=np.float32), seg_labels.astype(np.int32), str(kind)))

        placed_points = []
        placed_labels = []
        first_points, first_labels, _ = built_segments[0]
        placed_points.append(first_points.astype(np.float32))
        placed_labels.append(first_labels.astype(np.int32))
        prev_points = first_points.astype(np.float32)
        prev_end = prev_points[-1]
        prev_tangent = self._normalize_vec(prev_points[-1] - prev_points[-2]) if prev_points.shape[0] >= 2 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        for segment_points, segment_labels, _ in built_segments[1:]:
            segment_points = np.asarray(segment_points, dtype=np.float32)
            local_tangent = self._normalize_vec(segment_points[1] - segment_points[0]) if segment_points.shape[0] >= 2 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            desired = self._normalize_vec(prev_tangent + rng.normal(loc=0.0, scale=0.2, size=(3,)).astype(np.float32))
            if float(np.linalg.norm(desired)) <= 1e-8:
                desired = prev_tangent
            R = self._rotation_from_to(local_tangent, desired)
            seg_rot = (segment_points @ R.T).astype(np.float32)

            gap = float(rng.uniform(1.5, 2.4))
            start_target = prev_end + gap * desired
            jitter = rng.normal(loc=0.0, scale=0.28, size=(3,)).astype(np.float32)
            jitter[2] *= 0.6
            start_target = start_target + jitter
            shift = start_target - seg_rot[0]
            seg_global = seg_rot + shift[None, :]

            seg_tangent_start = self._normalize_vec(seg_global[1] - seg_global[0]) if seg_global.shape[0] >= 2 else desired
            chord = seg_global[0] - prev_end
            chord_norm = float(np.linalg.norm(chord))
            if chord_norm <= 1e-8:
                loop = np.repeat(prev_end[None, :], max(2, loop_n), axis=0).astype(np.float32)
            else:
                bend_axis = np.cross(prev_tangent, seg_tangent_start)
                if float(np.linalg.norm(bend_axis)) <= 1e-6:
                    bend_axis = np.cross(prev_tangent, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
                bend_axis = self._normalize_vec(bend_axis)
                bend = float(rng.uniform(0.35, 0.9))
                c1 = prev_end + prev_tangent * (0.35 * chord_norm) + bend_axis * bend
                c2 = seg_global[0] - seg_tangent_start * (0.35 * chord_norm) + bend_axis * bend
                loop = self._bezier_curve(prev_end, c1, c2, seg_global[0], max(2, loop_n)).astype(np.float32)

            if loop.shape[0] > 2:
                placed_points.append(loop[1:-1].astype(np.float32))
                placed_labels.append(np.full((loop.shape[0] - 2,), fill_value=-1, dtype=np.int32))
            placed_points.append(seg_global.astype(np.float32))
            placed_labels.append(np.asarray(segment_labels, dtype=np.int32))

            prev_end = seg_global[-1]
            prev_tangent = self._normalize_vec(seg_global[-1] - seg_global[-2]) if seg_global.shape[0] >= 2 else prev_tangent

        raw_points = np.concatenate(placed_points, axis=0).astype(np.float32)
        raw_labels = np.concatenate(placed_labels, axis=0).astype(np.int32)
        points = self._resample_polyline(raw_points, point_count)
        labels = self._point_labels_from_nearest(points, raw_points, raw_labels)
        points -= points.mean(axis=0, keepdims=True)
        return points.astype(np.float32), labels.astype(np.int32), "+".join(recipe)

    def _secondary_backbone_points(self, rng, point_count, motif=None, mixed_recipe=None):
        motif = self._choose_secondary_motif(rng) if motif is None else str(motif)
        if motif == "helix":
            points, strand_labels = self._build_helix_loop_helix(point_count=point_count, rng=rng)
            recipe = "helix"
        elif motif == "sheet":
            points, strand_labels = self._build_beta_sheet_with_loops(point_count=point_count, rng=rng)
            recipe = "sheet"
        else:
            points, strand_labels, recipe = self._build_mixed_random_combo(
                point_count=point_count,
                rng=rng,
                recipe=mixed_recipe,
            )
        return points.astype(np.float32), motif, np.asarray(strand_labels, dtype=np.int32), str(recipe)

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

    def _sequence_order(self, sequence_positions):
        sequence_positions = np.asarray(sequence_positions, dtype=np.int32).reshape(-1)
        indices = np.arange(sequence_positions.shape[0], dtype=np.int32)
        order = np.lexsort((indices, sequence_positions)).astype(np.int32)
        return order

    def _max_consecutive_selected(self, selected_mask, order):
        selected_mask = np.asarray(selected_mask, dtype=bool).reshape(-1)
        order = np.asarray(order, dtype=np.int32).reshape(-1)
        run = 0
        max_run = 0
        for idx in order.tolist():
            if bool(selected_mask[int(idx)]):
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return int(max_run)

    def _select_polar_nodes(
        self,
        scores,
        candidate_mask,
        sequence_order,
        target_count,
        min_count,
        max_consecutive,
        min_spacing,
        centers,
        closeness_weight,
        closeness_distance_scale,
    ):
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        n = int(scores.shape[0])
        selected = np.zeros((n,), dtype=bool)
        if n == 0:
            return selected

        sequence_order = np.asarray(sequence_order, dtype=np.int32).reshape(-1)
        sequence_rank = np.empty((n,), dtype=np.int32)
        sequence_rank[sequence_order] = np.arange(n, dtype=np.int32)

        def _respects_spacing(node_i: int, spacing: int) -> bool:
            if spacing <= 1:
                return True
            selected_idx = np.flatnonzero(selected)
            if selected_idx.size == 0:
                return True
            rank = int(sequence_rank[int(node_i)])
            selected_ranks = sequence_rank[selected_idx]
            return bool(np.all(np.abs(selected_ranks - rank) >= int(spacing)))

        def _close_kernel(distance: float) -> float:
            scale = float(max(1e-3, closeness_distance_scale))
            d = float(max(distance, 0.0))
            return float(1.0 / (1.0 + (d / scale) ** 2))

        def _max_consecutive_if_added(node_i: int) -> int:
            selected[node_i] = True
            value = self._max_consecutive_selected(selected, sequence_order)
            selected[node_i] = False
            return int(value)

        def _pick_best_next(allow_all_candidates: bool, run_limit: int, spacing: int):
            current_selected = np.flatnonzero(selected).astype(np.int32)
            best_idx = -1
            best_objective = -np.inf
            for node_i in range(n):
                if selected[int(node_i)]:
                    continue
                if not allow_all_candidates and not bool(candidate_mask[int(node_i)]):
                    continue
                if scores[int(node_i)] <= 1e-8 and not allow_all_candidates:
                    continue
                if not _respects_spacing(int(node_i), spacing=spacing):
                    continue
                if run_limit > 0 and _max_consecutive_if_added(int(node_i)) > int(run_limit):
                    continue
                base_score = float(scores[int(node_i)])
                if current_selected.size == 0:
                    proximity_bonus = 0.0
                else:
                    rel = centers[current_selected] - centers[int(node_i)][None, :]
                    dists = np.linalg.norm(rel, axis=-1)
                    proximity_bonus = float(np.max([_close_kernel(float(d)) for d in dists.tolist()]))
                objective = base_score + float(closeness_weight) * proximity_bonus
                if objective > best_objective:
                    best_objective = objective
                    best_idx = int(node_i)
            return int(best_idx)

        def _pass(allow_all_candidates: bool, run_limit: int, spacing: int, limit_count: int):
            while int(selected.sum()) < int(limit_count):
                node_i = _pick_best_next(
                    allow_all_candidates=allow_all_candidates,
                    run_limit=run_limit,
                    spacing=spacing,
                )
                if node_i < 0:
                    break
                selected[int(node_i)] = True

        _pass(
            allow_all_candidates=False,
            run_limit=max_consecutive,
            spacing=min_spacing,
            limit_count=target_count,
        )
        if int(selected.sum()) < int(target_count):
            _pass(
                allow_all_candidates=True,
                run_limit=max_consecutive,
                spacing=min_spacing,
                limit_count=target_count,
            )
        if int(selected.sum()) < int(min_count):
            _pass(
                allow_all_candidates=True,
                run_limit=max_consecutive,
                spacing=min_spacing,
                limit_count=min_count,
            )
        return selected.astype(bool)

    def _assign_secondary_dipoles(self, sample, sequence_positions, brick_sheet_strand_ids=None):
        def _distance_weight(distance: float, scale: float = 3.8) -> float:
            d = float(max(distance, 0.0))
            s = float(max(1e-3, scale))
            return float(1.0 / (1.0 + (d / s) ** 2))

        sequence_positions = np.asarray(sequence_positions, dtype=np.int32).reshape(-1)
        if brick_sheet_strand_ids is None:
            brick_sheet_strand_ids = np.full((sequence_positions.shape[0],), fill_value=-1, dtype=np.int32)
        brick_sheet_strand_ids = np.asarray(brick_sheet_strand_ids, dtype=np.int32).reshape(-1)
        geometries = build_brick_geometries(sample)
        contact_data = detect_brick_contacts(geometries)
        all_pairs = np.asarray(contact_data["all_face_contact_pairs"], dtype=np.int64)
        all_dirs = np.asarray(contact_data["all_face_contact_dirs"], dtype=np.float32)
        num_nodes = int(len(sample["brick_types"]))
        if all_pairs.shape[0] == 0:
            return np.zeros((num_nodes, 3), dtype=np.float32)

        preferences = np.zeros((num_nodes, 3), dtype=np.float32)
        contact_scores = np.zeros((num_nodes,), dtype=np.float32)
        promoted_scores = np.zeros((num_nodes,), dtype=np.float32)
        for (src, dst), face_dir in zip(all_pairs, all_dirs):
            src_i = int(src)
            dst_i = int(dst)
            direction = np.asarray(face_dir, dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-8:
                continue
            direction = direction / norm

            seq_sep = abs(int(sequence_positions[src_i]) - int(sequence_positions[dst_i]))
            src_strand = int(brick_sheet_strand_ids[src_i]) if src_i < brick_sheet_strand_ids.shape[0] else -1
            dst_strand = int(brick_sheet_strand_ids[dst_i]) if dst_i < brick_sheet_strand_ids.shape[0] else -1

            weight = 0.0
            promoted = False
            if src_strand >= 0 and dst_strand >= 0:
                strand_delta = abs(src_strand - dst_strand)
                if src_strand == dst_strand:
                    if seq_sep < self.secondary_nonlocal_min_sep:
                        continue
                    # Along-strand contacts are less informative than across-strand packing.
                    weight = 0.45
                elif strand_delta == 1:
                    # Strongly promote adjacent-strand pairing.
                    weight = 3.1
                    promoted = True
                else:
                    # Still favor non-adjacent strands if they face each other.
                    weight = 2.0
                    promoted = True
            else:
                min_sep = self.secondary_nonlocal_min_sep
                if (src_strand >= 0) ^ (dst_strand >= 0):
                    # T/L bridge contacts are useful to couple neighboring strands.
                    if seq_sep >= max(2, min_sep - 1):
                        weight = 1.45
                        promoted = True
                    elif seq_sep >= 2:
                        weight = 0.8
                else:
                    if seq_sep >= min_sep:
                        weight = 1.35
                        promoted = True
                    elif seq_sep >= 2:
                        weight = 0.7

            if weight <= 1e-8:
                continue

            preferences[src_i] += weight * direction
            preferences[dst_i] += weight * direction
            contact_scores[src_i] += float(weight)
            contact_scores[dst_i] += float(weight)
            if promoted:
                promoted_scores[src_i] += float(weight)
                promoted_scores[dst_i] += float(weight)

        dipoles = np.zeros((num_nodes, 3), dtype=np.float32)
        vec_norms = np.linalg.norm(preferences, axis=-1).astype(np.float32)
        if float(vec_norms.max(initial=0.0)) <= 1e-8:
            return dipoles.astype(np.float32)

        # Prefer bricks involved in promoted non-local/inter-strand contacts,
        # while keeping approximately 2/3 apolar nodes.
        node_scores = (
            1.55 * promoted_scores
            + 0.75 * contact_scores
            + 0.55 * vec_norms
        ).astype(np.float32)
        candidate_mask = (promoted_scores > 1e-6) | (node_scores > 0.05)

        target_count = int(round(self.secondary_target_polar_fraction * float(num_nodes)))
        min_count = int(round(self.secondary_min_polar_fraction * float(num_nodes)))
        target_count = int(np.clip(target_count, 1, max(1, num_nodes)))
        min_count = int(np.clip(min_count, 0, target_count))
        sequence_order = self._sequence_order(sequence_positions)
        centers = np.asarray([g["world_center"] for g in geometries], dtype=np.float32)
        selected = self._select_polar_nodes(
            scores=node_scores,
            candidate_mask=candidate_mask,
            sequence_order=sequence_order,
            target_count=target_count,
            min_count=min_count,
            max_consecutive=self.secondary_max_consecutive_polar,
            min_spacing=self.secondary_polar_min_spacing,
            centers=centers,
            closeness_weight=1.1,
            closeness_distance_scale=3.8,
        )

        selected_idx = np.flatnonzero(selected).astype(np.int32)
        if selected_idx.size == 0:
            return dipoles.astype(np.float32)

        pair_terms = {}
        for src, dst in all_pairs.tolist():
            src_i = int(src)
            dst_i = int(dst)
            if not (selected[src_i] and selected[dst_i]):
                continue
            key = (min(src_i, dst_i), max(src_i, dst_i))
            vec = centers[key[1]] - centers[key[0]]
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-8:
                continue
            u = (vec / dist).astype(np.float32)
            w = 1.8 * _distance_weight(dist) + 0.6
            if key not in pair_terms:
                pair_terms[key] = {"u": u, "w": float(w)}
            else:
                prev_w = float(pair_terms[key]["w"])
                pair_terms[key]["u"] = (
                    (prev_w * pair_terms[key]["u"] + float(w) * u) / max(1e-8, prev_w + float(w))
                ).astype(np.float32)
                pair_terms[key]["w"] = prev_w + float(w)

        for i_pos, i in enumerate(selected_idx.tolist()):
            for j in selected_idx[i_pos + 1 :].tolist():
                key = (int(min(i, j)), int(max(i, j)))
                if key in pair_terms:
                    continue
                rel = centers[int(j)] - centers[int(i)]
                dist = float(np.linalg.norm(rel))
                if dist <= 1e-8 or dist > 6.8:
                    continue
                seq_sep = abs(int(sequence_positions[int(i)]) - int(sequence_positions[int(j)]))
                if seq_sep < max(2, self.secondary_nonlocal_min_sep - 2):
                    continue
                u = (rel / dist).astype(np.float32)
                w = 1.25 * _distance_weight(dist)
                pair_terms[key] = {"u": u, "w": float(w)}

        adjacency = {int(i): [] for i in selected_idx.tolist()}
        for (i, j), payload in pair_terms.items():
            u = np.asarray(payload["u"], dtype=np.float32)
            w = float(payload["w"])
            adjacency[int(i)].append((int(j), u, w))
            adjacency[int(j)].append((int(i), (-u).astype(np.float32), w))

        norms = np.linalg.norm(preferences, axis=-1, keepdims=True)
        for node_i in selected_idx.tolist():
            n = float(norms[int(node_i), 0])
            if n > 1e-8:
                dipoles[int(node_i)] = preferences[int(node_i)] / n
            else:
                dipoles[int(node_i)] = np.zeros((3,), dtype=np.float32)

        for node_i in selected_idx.tolist():
            if np.linalg.norm(dipoles[int(node_i)]) > 1e-8:
                continue
            neighbors = adjacency.get(int(node_i), [])
            if len(neighbors) == 0:
                continue
            vec = np.zeros((3,), dtype=np.float32)
            for _, u_ij, w_ij in neighbors:
                vec += float(w_ij) * np.asarray(u_ij, dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                dipoles[int(node_i)] = vec / norm

        prior_weight = 0.22
        for _ in range(8):
            changed = False
            for node_i in selected_idx.tolist():
                neighbors = adjacency.get(int(node_i), [])
                if len(neighbors) == 0:
                    continue
                field = np.zeros((3,), dtype=np.float32)
                for nbr_j, u_ij, w_ij in neighbors:
                    d_j = np.asarray(dipoles[int(nbr_j)], dtype=np.float32)
                    q_j = float(np.dot(d_j, -np.asarray(u_ij, dtype=np.float32)))
                    field += float(w_ij) * q_j * np.asarray(u_ij, dtype=np.float32)
                p_i = np.asarray(preferences[int(node_i)], dtype=np.float32)
                p_norm = float(np.linalg.norm(p_i))
                if p_norm > 1e-8:
                    field -= float(prior_weight) * (p_i / p_norm)
                f_norm = float(np.linalg.norm(field))
                if f_norm <= 1e-8:
                    continue
                new_d = (-field / f_norm).astype(np.float32)
                if float(np.dot(new_d, dipoles[int(node_i)])) < 0.999:
                    changed = True
                dipoles[int(node_i)] = new_d
            if not changed:
                break
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
        motif_choice = self._choose_secondary_motif(rng)
        if motif_choice == "mixed":
            point_count = int(max(18, round(target_bricks * 2.2)))
        else:
            point_count = int(max(20, target_bricks * 3))
        recipe_choice = (
            self._sample_mixed_recipe(point_count=point_count, rng=rng, target_bricks=target_bricks)
            if motif_choice == "mixed"
            else None
        )
        best = None

        for _ in range(8):
            backbone_points, motif, backbone_sheet_labels, recipe = self._secondary_backbone_points(
                rng,
                point_count=point_count,
                motif=motif_choice,
                mixed_recipe=recipe_choice,
            )
            backbone_voxels = self._polyline_to_connected_voxels(backbone_points)
            if backbone_voxels.shape[0] == 0:
                point_count = int(max(20, point_count + 6))
                continue
            placements = self.approximate_voxels(backbone_voxels)
            n_placed = int(len(placements))

            diff_to_target = abs(n_placed - target_bricks)
            candidate = {
                "motif": motif,
                "recipe": str(recipe),
                "backbone_points": backbone_points,
                "backbone_sheet_labels": np.asarray(backbone_sheet_labels, dtype=np.int32),
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
                point_count = int(point_count * 1.18) + 2
            else:
                floor_points = 16 if motif_choice == "mixed" else 20
                point_count = max(floor_points, int(point_count * 0.72) - 2)

        if best is None:
            raise RuntimeError("Failed to generate a secondary scaffold.")

        motif = str(best["motif"])
        secondary_recipe = str(best.get("recipe", motif))
        backbone_points = np.asarray(best["backbone_points"], dtype=np.float32)
        backbone_sheet_labels = np.asarray(best["backbone_sheet_labels"], dtype=np.int32).reshape(-1)
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
            brick_backbone_indices = np.asarray([item[0] for item in scored_placements], dtype=np.int32)
        else:
            brick_backbone_indices = np.zeros((len(placements),), dtype=np.int32)

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
        brick_sheet_strand_ids = np.full((len(types),), fill_value=-1, dtype=np.int32)
        if backbone_sheet_labels.shape[0] > 0 and brick_backbone_indices.shape[0] == len(types):
            clipped = np.clip(brick_backbone_indices, 0, max(0, backbone_sheet_labels.shape[0] - 1))
            brick_sheet_strand_ids = backbone_sheet_labels[clipped].astype(np.int32)
        sample = {
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": types,
        }
        geometries = build_brick_geometries(sample)
        all_world_voxels = np.concatenate([np.asarray(g["world_voxels"], dtype=np.float32) for g in geometries], axis=0)
        occupied_voxels = np.unique(np.rint(all_world_voxels).astype(np.int32), axis=0)
        dipoles = self._assign_secondary_dipoles(
            sample=sample,
            sequence_positions=sequence_positions,
            brick_sheet_strand_ids=brick_sheet_strand_ids,
        )
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
            "secondary_backbone_sheet_strand": backbone_sheet_labels.astype(np.int32),
            "brick_backbone_index": brick_backbone_indices.astype(np.int32),
            "brick_sheet_strand_id": brick_sheet_strand_ids.astype(np.int32),
            "brick_sequence_position": sequence_positions.astype(np.int32),
            "sequence_pos_max": np.asarray(self.sequence_pos_max, dtype=np.int32),
            "secondary_motif": np.asarray(str(motif)),
            "secondary_recipe": np.asarray(str(secondary_recipe)),
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
    parser.add_argument(
        "--secondary-target-polar-fraction",
        type=float,
        default=0.34,
        help="Target fraction of polar bricks in secondary mode.",
    )
    parser.add_argument(
        "--secondary-min-polar-fraction",
        type=float,
        default=0.24,
        help="Minimum fraction of polar bricks in secondary mode.",
    )
    parser.add_argument(
        "--secondary-max-consecutive-polar",
        type=int,
        default=2,
        help="Maximum number of consecutive sequence bricks allowed to be polar.",
    )
    parser.add_argument(
        "--secondary-polar-min-spacing",
        type=int,
        default=3,
        help="Minimum spacing in sequence order between polar bricks (3 means at most one polar every three).",
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
        secondary_target_polar_fraction=args.secondary_target_polar_fraction,
        secondary_min_polar_fraction=args.secondary_min_polar_fraction,
        secondary_max_consecutive_polar=args.secondary_max_consecutive_polar,
        secondary_polar_min_spacing=args.secondary_polar_min_spacing,
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
