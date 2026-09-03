#!/usr/bin/env python3
"""Plot a periodic Ramachandran KDE from an Alanine NPZ or sampled XYZ."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _dihedral(points: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = np.moveaxis(points, -2, 0)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    if np.any(b1_norm == 0.0):
        raise ValueError("Cannot compute a dihedral with a zero-length central bond.")
    b1 = b1 / b1_norm
    v = b0 - np.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=-1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, axis=-1), np.sum(v * w, axis=-1))


def _angles_from_coords(coords: np.ndarray, dihedral_indices: np.ndarray) -> np.ndarray:
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Coordinates must have shape (frames, atoms, 3), got {coords.shape}.")
    if dihedral_indices.shape != (2, 4):
        raise ValueError(f"Expected two four-atom dihedrals, got {dihedral_indices.shape}.")
    if int(dihedral_indices.max()) >= coords.shape[1]:
        raise ValueError("Dihedral indices exceed the number of atoms in the coordinates.")
    return np.stack(
        [_dihedral(coords[:, indices]) for indices in dihedral_indices],
        axis=1,
    )


def _read_xyz(path: Path) -> np.ndarray:
    frames = []
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                num_atoms = int(line)
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ atom-count line in {path}: {line.strip()!r}") from exc
            if not handle.readline():
                raise ValueError(f"Missing XYZ comment line in {path}.")
            frame = []
            for _ in range(num_atoms):
                fields = handle.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"Incomplete XYZ frame in {path}.")
                frame.append([float(value) for value in fields[1:4]])
            frames.append(frame)
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}.")
    atom_counts = {len(frame) for frame in frames}
    if len(atom_counts) != 1:
        raise ValueError("All XYZ frames must have the same atom count.")
    return np.asarray(frames, dtype=np.float64)


def load_angles(input_path: Path, dihedral_source: Path) -> np.ndarray:
    if input_path.suffix.lower() == ".npz":
        with np.load(input_path) as data:
            if "dihedrals" in data.files:
                angles = np.asarray(data["dihedrals"], dtype=np.float64)
                if angles.ndim != 2 or angles.shape[1] != 2:
                    raise ValueError(f"Expected dihedrals with shape (frames, 2), got {angles.shape}.")
                return np.rad2deg(angles)
            if "coords" not in data.files or "dihedral_idcs" not in data.files:
                raise KeyError("NPZ must contain dihedrals or both coords and dihedral_idcs.")
            return np.rad2deg(_angles_from_coords(data["coords"], data["dihedral_idcs"]))

    if input_path.suffix.lower() != ".xyz":
        raise ValueError("Input must be an Alanine dataset NPZ or a multi-frame XYZ file.")
    with np.load(dihedral_source) as data:
        if "dihedral_idcs" not in data.files:
            raise KeyError(f"{dihedral_source} does not contain dihedral_idcs.")
        indices = np.asarray(data["dihedral_idcs"], dtype=np.int64)
    return np.rad2deg(_angles_from_coords(_read_xyz(input_path), indices))


def periodic_kde(angles: np.ndarray, grid: np.ndarray, bandwidth: float):
    """Evaluate a wrapped-Gaussian KDE and nearest-sample distance."""
    if bandwidth <= 0.0:
        raise ValueError("KDE bandwidth must be positive.")
    phi_grid, psi_grid = np.meshgrid(grid, grid, indexing="xy")
    density = np.zeros_like(phi_grid, dtype=np.float64)
    nearest_distance_sq = np.full_like(phi_grid, np.inf, dtype=np.float64)
    for start in range(0, len(angles), 32):
        chunk = angles[start:start + 32]
        dphi = (phi_grid[..., None] - chunk[:, 0] + 180.0) % 360.0 - 180.0
        dpsi = (psi_grid[..., None] - chunk[:, 1] + 180.0) % 360.0 - 180.0
        distance_sq = dphi * dphi + dpsi * dpsi
        density += np.exp(-0.5 * distance_sq / bandwidth**2).sum(axis=-1)
        nearest_distance_sq = np.minimum(nearest_distance_sq, distance_sq.min(axis=-1))
    density /= len(angles) * 2.0 * np.pi * bandwidth**2
    return density, np.sqrt(nearest_distance_sq)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Training NPZ or sampled multi-frame XYZ.")
    parser.add_argument(
        "--dihedral-source",
        type=Path,
        default=Path("data/train_750.npz"),
        help="NPZ providing dihedral_idcs for XYZ input (default: data/train_750.npz).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output image path.")
    parser.add_argument("--bandwidth", type=float, default=12.0, help="Wrapped Gaussian bandwidth in degrees.")
    parser.add_argument(
        "--support-radius",
        type=float,
        default=1.0,
        help="Mask density farther than this many bandwidths from every sample (default: 3).",
    )
    parser.add_argument("--cmap", default="plasma", help="Matplotlib sequential color map (default: plasma).")
    parser.add_argument("--grid-size", type=int, default=181, help="Points along each angular axis.")
    parser.add_argument("--title", default="Alanine dipeptide Ramachandran density")
    args = parser.parse_args(argv)

    if args.grid_size < 20:
        parser.error("--grid-size must be at least 20.")
    if args.support_radius <= 0.0:
        parser.error("--support-radius must be positive.")
    angles = load_angles(args.input, args.dihedral_source)
    angles = (angles + 180.0) % 360.0 - 180.0
    grid = np.linspace(-180.0, 180.0, args.grid_size)
    density, nearest_distance = periodic_kde(angles, grid, args.bandwidth)
    supported_density = np.ma.masked_where(
        nearest_distance > args.support_radius * args.bandwidth,
        density,
    )

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.set_facecolor("white")
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad("white")
    contour = ax.contourf(grid, grid, supported_density, levels=28, cmap=cmap)
    contour_levels = np.linspace(float(supported_density.min()), float(supported_density.max()), 9)[1:-1]
    ax.contour(
        grid,
        grid,
        supported_density,
        levels=contour_levels,
        colors="#263238",
        linewidths=0.55,
        alpha=0.7,
    )
    fig.colorbar(contour, ax=ax, label="Probability density (degree^-2)")
    ax.set(xlabel="Phi (degrees)", ylabel="Psi (degrees)", title=args.title)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Plotted {len(angles)} conformations to {args.output}")


if __name__ == "__main__":
    main()
