#!/usr/bin/env python3
"""
Plot a Ramachandran density map (phi/psi) from a trajectory.

Example:
  python geqdiff/scripts/ramachandran.py --top ref.pdb --traj generated.xyz --rdkit_reorder --noh --out rama.png
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
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
except ImportError:
    Chem = None
    rdMolAlign = None

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
        try:
            element = atom.element
        except:
            element = None
        if not element:
            if guess_atom_element is not None:
                element = guess_atom_element(atom.name)
            else:
                element = atom.name
        elements.append(_normalize_element(element))
    return np.array(elements, dtype=object)


def _atom_numbers(atoms) -> np.ndarray:
    if Chem is None:
        raise ImportError("rdkit is required for --rdkit_reorder.")
    numbers = []
    pt = Chem.GetPeriodicTable()
    for atom in atoms:
        num = None
        try:
            num = atom.atomic_number
        except Exception:
            num = None
        if not num:
            try:
                element = atom.element
            except Exception:
                element = None
            if not element:
                if guess_atom_element is not None:
                    element = guess_atom_element(atom.name)
                else:
                    element = atom.name
            element = _normalize_element(element)
            num = pt.GetAtomicNumber(element)
        if not num:
            raise ValueError(f"Could not determine atomic number for atom '{atom.name}'.")
        numbers.append(int(num))
    return np.array(numbers, dtype=int)


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


def _rdkit_ref_mol(atoms, removeHs: bool, sanitize: bool = True) -> "Chem.Mol":
    atomic_num = _atom_numbers(atoms)
    coords = atoms.positions
    return _rdkit_mol_from_xyz(
        atomic_num,
        coords,
        removeHs=removeHs,
        sanitize=sanitize,
    )


def _xyz_block(elements: np.ndarray, coords: np.ndarray) -> str:
    lines = [str(len(elements)), "frame"]
    for element, pos in zip(elements, coords):
        lines.append(f"{element} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    return "\n".join(lines) + "\n"

def _rdkit_mol_from_xyz(
    atomic_num: np.ndarray,
    coords: np.ndarray,
    removeHs: bool = True,
    sanitize: bool = True,
) -> Chem.Mol:
    """
    Convert coordinates and atomic numbers to an RDKit molecule.

    Args:
        coords: Tensor of shape (N, 3) containing 3D coordinates
        atomic_num: Tensor of shape (N,) containing atomic numbers
        removeHs: Whether to remove hydrogen atoms from the molecule

    Returns:
        RDKit molecule object
    """
    # Create empty molecule
    mol = Chem.RWMol()

    # Add atoms to molecule
    for atomic_num_val in atomic_num:
        atom = Chem.Atom(int(atomic_num_val))
        mol.AddAtom(atom)

    # Create conformer and set coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i].tolist()
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)

    # Add bonds based on distance (simple heuristic)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            dist = np.linalg.norm(coords[i] - coords[j]).item()
            # Simple distance-based bonding heuristic
            if dist < 1.8:  # Typical single bond length
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    # Convert to regular molecule and sanitize
    mol = mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(mol)

    # Handle hydrogens
    # Print initial number of atoms before hydrogen manipulation
    # print(f"Initial number of atoms: {mol.GetNumAtoms()}")

    if removeHs:
        # Remove hydrogen atoms and print new count
        mol = Chem.RemoveHs(mol)
        # print(f"Number of atoms after removing hydrogens: {mol.GetNumAtoms()}")

    # Print final number of atoms for verification
    # print(f"Final number of atoms: {mol.GetNumAtoms()}")
    return mol


def _rdkit_atom_map(probe_mol: "Chem.Mol", ref_mol: "Chem.Mol") -> np.ndarray:
    if rdMolAlign is None:
        raise ImportError("rdkit is required for --rdkit_reorder.")
    result = None
    try:
        result = rdMolAlign.GetBestAlignmentTransform(
            probe_mol, ref_mol, reflect=False, maxIters=1000
        )
    except TypeError:
        result = rdMolAlign.GetBestAlignmentTransform(probe_mol, ref_mol)
    atom_map = None
    if isinstance(result, tuple) and len(result) == 3:
        atom_map = result[2]
    if atom_map is None:
        get_best_rms = getattr(rdMolAlign, "GetBestRMS", None)
        if get_best_rms is None:
            raise ValueError("RDKit does not expose atom maps for alignment.")
        try:
            _, atom_map = get_best_rms(probe_mol, ref_mol, returnAtomMap=True)
        except TypeError as exc:
            raise ValueError("RDKit version does not return atom maps; please update RDKit.") from exc
    atom_map_list = list(atom_map)
    if not atom_map_list:
        raise ValueError("RDKit alignment returned an empty atom map.")
    atom_map_sorted = sorted(atom_map_list, key=lambda x: x[1])
    return np.array([probe_idx for probe_idx, ref_idx in atom_map_sorted], dtype=int)


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


def _make_universe(*args, guess_bonds: bool) -> "mda.Universe":
    if guess_bonds:
        to_guess = ("bonds", "angles", "dihedrals")
        try:
            return mda.Universe(*args, to_guess=to_guess)
        except TypeError:
            return mda.Universe(*args, guess_bonds=True)
    try:
        return mda.Universe(*args)
    except TypeError:
        return mda.Universe(*args, guess_bonds=False)


def _filter_phi_psi_residues(atomgroup):
    residues = atomgroup.residues
    if len(residues) == 0:
        return residues
    try:
        prev = residues._get_prev_residues_by_resid()
        nxt = residues._get_next_residues_by_resid()
    except AttributeError:
        return residues
    keep = np.array([r is not None for r in prev]) & np.array([r is not None for r in nxt])
    if not np.all(keep):
        dropped = len(residues) - int(keep.sum())
        print(f"Note: dropping {dropped} residues without phi/psi definitions.")
        residues = residues[keep]
    return residues


def plot_ramachandran(
    top_path: str,
    traj_path: str,
    reorder_to_top: bool,
    rdkit_reorder: bool,
    discarded_xyz: Optional[str],
    noh: bool,
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
        reference = _make_universe(top_path, guess_bonds=guess_bonds)
        trajectory = _make_universe(traj_path, guess_bonds=guess_bonds)
        if noh:
            ref_atoms = reference.select_atoms("not name H*")
            traj_atoms = trajectory.select_atoms("not name H*")
        else:
            ref_atoms = reference.atoms
            traj_atoms = trajectory.atoms
        if len(ref_atoms) != len(traj_atoms):
            raise ValueError("Reference and trajectory must have the same atom count.")
        ref_elements = _atom_elements(ref_atoms)
        traj_elements = _atom_elements(traj_atoms)
        if rdkit_reorder:
            traj_atomic_num = _atom_numbers(traj_atoms)
            rdkit_ref = _rdkit_ref_mol(ref_atoms, noh)
        else:
            traj_atomic_num = None
            rdkit_ref = None

        reordered = []
        discarded = 0
        discarded_frames = []
        for ts in trajectory.trajectory:
            if rdkit_reorder:
                try:
                    probe_mol = _rdkit_mol_from_xyz(
                        traj_atomic_num,
                        traj_atoms.positions,
                        removeHs=noh,
                    )
                    mapping = _rdkit_atom_map(probe_mol, rdkit_ref)
                except Exception as e:
                    discarded += 1
                    if discarded_xyz:
                        discarded_frames.append(traj_atoms.positions.copy())
                    continue
            else:
                mapping = _compute_mapping(
                    ref_atoms.positions,
                    ref_elements,
                    traj_atoms.positions,
                    traj_elements,
                )
            coords = traj_atoms.positions[mapping].copy()
            reordered.append(coords)
        if rdkit_reorder:
            total = len(trajectory.trajectory)
            print(f"Discarded {discarded}/{total} frames due to RDKit alignment failures.")
            if discarded_xyz and discarded_frames:
                _write_xyz_frames(
                    discarded_xyz,
                    traj_elements,
                    discarded_frames,
                    "Discarded frame",
                )
        if not reordered:
            raise ValueError("No frames left after RDKit alignment filtering.")
        if noh:
            reference = mda.Merge(ref_atoms)
        coords = np.asarray(reordered, dtype=np.float32)
        reference.load_new(coords, format=MemoryReader)
        universe = reference
    else:
        universe = _make_universe(top_path, traj_path, guess_bonds=guess_bonds)
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

    residues = _filter_phi_psi_residues(sel)
    if len(residues) == 0:
        raise ValueError("Selection has no residues with phi/psi definitions.")
    rama = Ramachandran(residues, check_protein=False).run(start=start, stop=stop, step=step)
    angles = _flatten_angles(np.asarray(rama.results.angles))

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
        "--rdkit_reorder",
        action="store_true",
        help="Use RDKit alignment to reorder atoms; discards frames that fail alignment.",
    )
    parser.add_argument(
        "--noh",
        action="store_true",
        help="Drop hydrogens before reordering and analysis.",
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
        rdkit_reorder=args.rdkit_reorder,
        discarded_xyz=args.discarded_xyz,
        noh=args.noh,
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
