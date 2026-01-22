#!/usr/bin/env python3
"""
Plot a Ramachandran density map (phi/psi) from a trajectory.

Example:
  python geqdiff/scripts/ramachandran.py --top ref.pdb --traj generated.xyz --reorder_to_top --out rama.png
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran
    from MDAnalysis.coordinates.memory import MemoryReader
    try:
        from MDAnalysis.topology.guessers import guess_atom_element
    except ImportError:
        guess_atom_element = None
except ImportError as exc:
    raise ImportError(
        "MDAnalysis is required. Install it with `pip install MDAnalysis`."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "matplotlib is required. Install it with `pip install matplotlib`."
    ) from exc

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

_warned_no_scipy = False


def _flatten_angles(angles: np.ndarray) -> np.ndarray:
    if angles.size == 0:
        raise ValueError("No phi/psi angles were computed for the selection.")
    if angles.ndim == 3 and angles.shape[-1] == 2:
        return angles.reshape(-1, 2)
    if angles.ndim == 2 and angles.shape[1] == 2:
        return angles
    raise ValueError(f"Unexpected angles array shape: {angles.shape}")


def _normalize_element(label: str) -> str:
    label = (label or "").strip()
    if not label:
        return "X"
    if len(label) >= 2:
        candidate = label[0].upper() + label[1].lower()
        if candidate.isalpha():
            return candidate
    return label[0].upper()


def _atom_elements(atoms) -> np.ndarray:
    elements = []
    for atom in atoms:
        element = atom.element
        if not element:
            if guess_atom_element is not None:
                element = guess_atom_element(atom.name)
            else:
                element = atom.name
        elements.append(_normalize_element(element))
    return np.array(elements, dtype=object)


def _distance_fingerprints(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist_sorted = np.sort(dist, axis=1)[:, 1:]
    return dist_sorted


def _greedy_assignment(cost: np.ndarray) -> np.ndarray:
    n_rows, n_cols = cost.shape
    if n_rows != n_cols:
        raise ValueError("Greedy assignment requires a square cost matrix.")
    assignment = -np.ones(n_rows, dtype=int)
    used = set()
    pairs = [(i, j, cost[i, j]) for i in range(n_rows) for j in range(n_cols)]
    pairs.sort(key=lambda x: x[2])
    for i, j, _ in pairs:
        if assignment[i] == -1 and j not in used:
            assignment[i] = j
            used.add(j)
        if len(used) == n_rows:
            break
    if (assignment == -1).any():
        raise ValueError("Greedy assignment failed to find a full matching.")
    return assignment


def _compute_mapping(
    ref_positions: np.ndarray,
    ref_elements: np.ndarray,
    target_positions: np.ndarray,
    target_elements: np.ndarray,
) -> np.ndarray:
    if ref_positions.shape != target_positions.shape:
        raise ValueError("Reference and target positions must have the same shape.")
    mapping = np.empty(ref_positions.shape[0], dtype=int)

    ref_fp = _distance_fingerprints(ref_positions)
    target_fp = _distance_fingerprints(target_positions)

    for element in np.unique(ref_elements):
        ref_idx = np.where(ref_elements == element)[0]
        tgt_idx = np.where(target_elements == element)[0]
        if len(ref_idx) != len(tgt_idx):
            raise ValueError(
                f"Element '{element}' count mismatch: {len(ref_idx)} (ref) vs {len(tgt_idx)} (target)."
            )
        cost = np.linalg.norm(
            ref_fp[ref_idx][:, None, :] - target_fp[tgt_idx][None, :, :],
            axis=-1,
        )
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            global _warned_no_scipy
            if not _warned_no_scipy:
                print("Warning: scipy not found; using greedy assignment for reordering.")
                _warned_no_scipy = True
            row_ind = np.arange(len(ref_idx))
            col_ind = _greedy_assignment(cost)
        mapping[ref_idx[row_ind]] = tgt_idx[col_ind]
    return mapping


def _bond_indices(universe) -> Optional[np.ndarray]:
    try:
        bonds = universe.bonds
    except AttributeError:
        return None
    if bonds is None or len(bonds) == 0:
        return None
    return bonds.indices.astype(int, copy=False)


def _bond_lengths(positions: np.ndarray, bond_indices: np.ndarray) -> np.ndarray:
    if bond_indices is None or bond_indices.size == 0:
        return np.array([], dtype=np.float32)
    diff = positions[bond_indices[:, 0]] - positions[bond_indices[:, 1]]
    return np.linalg.norm(diff, axis=1)


def _passes_bond_checks(
    lengths: np.ndarray,
    ref_lengths: np.ndarray,
    tolerance: float,
    bond_min: Optional[float],
    bond_max: Optional[float],
) -> bool:
    if lengths.size == 0:
        return True
    if np.any(~np.isfinite(lengths)):
        return False
    if bond_min is not None and np.any(lengths < bond_min):
        return False
    if bond_max is not None and np.any(lengths > bond_max):
        return False
    if tolerance is not None and tolerance >= 0:
        lower = ref_lengths * (1.0 - tolerance)
        upper = ref_lengths * (1.0 + tolerance)
        if np.any(lengths < lower) or np.any(lengths > upper):
            return False
    return True


def _write_xyz_frames(path: str, elements: np.ndarray, frames: list, comment: str) -> None:
    if not frames:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_atoms = len(elements)
    with open(out_path, "w") as handle:
        for idx, coords in enumerate(frames, start=1):
            handle.write(f"{num_atoms}\n")
            handle.write(f"{comment} {idx}\n")
            for element, pos in zip(elements, coords):
                handle.write(f"{element}   {pos[0]:.4f}   {pos[1]:.4f}   {pos[2]:.4f}\n")


def plot_ramachandran(
    top_path: str,
    traj_path: str,
    reorder_to_top: bool,
    bond_check: bool,
    bond_tolerance: float,
    bond_min: Optional[float],
    bond_max: Optional[float],
    discarded_xyz: Optional[str],
    selection: str,
    out_path: str,
    bins: int,
    start: Optional[int],
    stop: Optional[int],
    step: int,
    guess_bonds: bool,
    show: bool,
) -> None:
    if reorder_to_top:
        reference = mda.Universe(top_path, guess_bonds=guess_bonds)
        trajectory = mda.Universe(traj_path, guess_bonds=guess_bonds)
        if len(reference.atoms) != len(trajectory.atoms):
            raise ValueError("Reference and trajectory must have the same atom count.")
        ref_elements = _atom_elements(reference.atoms)
        traj_elements = _atom_elements(trajectory.atoms)
        bond_indices = _bond_indices(reference) if bond_check else None
        if bond_check and bond_indices is None:
            print("Warning: no bonds found in reference topology; skipping bond checks.")
            bond_check = False
        ref_lengths = _bond_lengths(reference.atoms.positions, bond_indices)

        reordered = []
        discarded = 0
        discarded_frames = []
        for ts in trajectory.trajectory:
            mapping = _compute_mapping(
                reference.atoms.positions,
                ref_elements,
                trajectory.atoms.positions,
                traj_elements,
            )
            coords = trajectory.atoms.positions[mapping].copy()
            if bond_check:
                lengths = _bond_lengths(coords, bond_indices)
                if not _passes_bond_checks(lengths, ref_lengths, bond_tolerance, bond_min, bond_max):
                    discarded += 1
                    if discarded_xyz:
                        discarded_frames.append(coords)
                    continue
            reordered.append(coords)
        if bond_check:
            total = len(trajectory.trajectory)
            print(f"Discarded {discarded}/{total} frames due to bond sanity checks.")
            if discarded_xyz and discarded_frames:
                _write_xyz_frames(
                    discarded_xyz,
                    ref_elements,
                    discarded_frames,
                    "Discarded frame",
                )
        coords = np.asarray(reordered, dtype=np.float32)
        reference.load_new(coords, format=MemoryReader)
        universe = reference
    else:
        universe = mda.Universe(top_path, traj_path, guess_bonds=guess_bonds)
        if bond_check:
            bond_indices = _bond_indices(universe)
            if bond_indices is None:
                print("Warning: no bonds found in topology; skipping bond checks.")
                bond_check = False
            else:
                elements = _atom_elements(universe.atoms)
                ref_lengths = _bond_lengths(universe.atoms.positions, bond_indices)
                filtered = []
                discarded = 0
                discarded_frames = []
                for ts in universe.trajectory:
                    coords = universe.atoms.positions.copy()
                    lengths = _bond_lengths(coords, bond_indices)
                    if not _passes_bond_checks(lengths, ref_lengths, bond_tolerance, bond_min, bond_max):
                        discarded += 1
                        if discarded_xyz:
                            discarded_frames.append(coords)
                        continue
                    filtered.append(coords)
                total = len(universe.trajectory)
                print(f"Discarded {discarded}/{total} frames due to bond sanity checks.")
                if discarded_xyz and discarded_frames:
                    _write_xyz_frames(
                        discarded_xyz,
                        elements,
                        discarded_frames,
                        "Discarded frame",
                    )
                universe = mda.Universe(top_path, guess_bonds=guess_bonds)
                universe.load_new(np.asarray(filtered, dtype=np.float32), format=MemoryReader)
    sel = universe.select_atoms(selection)
    if len(sel) == 0 and selection == "protein":
        print("Warning: selection 'protein' returned 0 atoms; falling back to 'all'.")
        sel = universe.select_atoms("all")
    if len(sel) == 0:
        raise ValueError(f"Selection '{selection}' returned 0 atoms.")
    if len(sel.residues) == 0:
        raise ValueError(
            "Selection has no residues. Ramachandran requires protein residue topology "
            "(e.g., PDB/PSF with N/CA/C)."
        )

    rama = Ramachandran(sel, check_protein=False).run(start=start, stop=stop, step=step)
    angles = _flatten_angles(np.asarray(rama.angles))

    phi = angles[:, 0]
    psi = angles[:, 1]

    hist, xedges, yedges = np.histogram2d(
        phi,
        psi,
        bins=bins,
        range=[[-180, 180], [-180, 180]],
        density=True,
    )

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xedges, yedges, hist.T, cmap="viridis", shading="auto")
    plt.xlabel("Phi (deg)")
    plt.ylabel("Psi (deg)")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(label="Density")
    plt.title("Ramachandran Density")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    if show:
        plt.show()
    plt.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot a Ramachandran density map from a multi-frame XYZ trajectory."
    )
    parser.add_argument("--top", required=True, help="Topology file (e.g., XYZ, PDB).")
    parser.add_argument("--traj", required=True, help="Trajectory file (e.g., multi-frame XYZ).")
    parser.add_argument(
        "--bond_check",
        action="store_true",
        help="Discard frames with broken bonds based on reference topology.",
    )
    parser.add_argument(
        "--bond_tolerance",
        type=float,
        default=0.8,
        help="Relative bond length tolerance vs reference (default: 0.8).",
    )
    parser.add_argument(
        "--bond_min",
        type=float,
        default=None,
        help="Absolute minimum bond length (Angstrom).",
    )
    parser.add_argument(
        "--bond_max",
        type=float,
        default=None,
        help="Absolute maximum bond length (Angstrom).",
    )
    parser.add_argument(
        "--discarded_xyz",
        default=None,
        help="Write discarded frames to a multi-frame XYZ file.",
    )
    parser.add_argument(
        "--selection",
        default="protein",
        help="MDAnalysis selection string (default: protein).",
    )
    parser.add_argument(
        "--out",
        default="ramachandran_density.png",
        help="Output image path.",
    )
    parser.add_argument("--bins", type=int, default=72, help="Number of histogram bins.")
    parser.add_argument("--start", type=int, default=None, help="Start frame index.")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index.")
    parser.add_argument("--step", type=int, default=1, help="Frame stride.")
    parser.add_argument("--show", action="store_true", help="Show plot window.")
    if argv is not None:
        return parser.parse_args(argv)
    return parser.parse_args()


def main():
    args = parse_args()
    plot_ramachandran(
        top_path=args.top,
        traj_path=args.traj,
        reorder_to_top=True,
        bond_check=args.bond_check,
        bond_tolerance=args.bond_tolerance,
        bond_min=args.bond_min,
        bond_max=args.bond_max,
        discarded_xyz=args.discarded_xyz,
        selection=args.selection,
        out_path=args.out,
        bins=args.bins,
        start=args.start,
        stop=args.stop,
        step=args.step,
        guess_bonds=True,
        show=args.show,
    )


if __name__ == "__main__":
    main()
