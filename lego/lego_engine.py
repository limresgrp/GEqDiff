"""Generate SH-defined target shapes and their discrete LEGO approximations."""

import argparse
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

        self.block_order = sorted(
            self.library,
            key=lambda name: len(self.library[name]["offsets"]),
            reverse=True,
        )
        self.rotated_library = {
            name: list(iter_rotated_offsets(spec["offsets"]))
            for name, spec in self.library.items()
        }

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
        }

    def generate_task(self, coefficients=None, seed=None):
        rng = np.random.default_rng(seed)
        return self.build_sample(coefficients=coefficients, rng=rng)

    def generate_dataset(self, n_samples=1, seed=None):
        rng = np.random.default_rng(seed)
        return [self.build_sample(rng=rng) for _ in range(n_samples)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1, help="Number of SH blocks to generate.")
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
    )
    samples = engine.generate_dataset(n_samples=args.samples, seed=args.seed)
    save_samples(args.path, samples)

    print("--- SH LEGO Dataset Generated ---")
    print(f"Samples: {len(samples)}")
    print(f"Irreps: {engine.irreps}")
    print(f"Saved to: {args.path}")
    for idx, sample in enumerate(samples):
        print(
            f"Sample {idx}: voxels={len(sample['target_voxels'])}, "
            f"bricks={len(sample['brick_types'])}, "
            f"mode={sample.get('occupancy_mode', engine.occupancy_mode)}, "
            f"coeff_norm={np.linalg.norm(sample['coefficients']):.3f}"
        )


LegoMultiPortEngine = LegoDipoleEngine
