#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array


AA_TO_TYPE = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
    # common variants
    "MSE": 13,
    "HID": 9,
    "HIE": 9,
    "HIP": 9,
    "HSD": 9,
    "HSE": 9,
    "CYX": 5,
    "ASH": 4,
    "GLH": 7,
}

TYPE_NAMES = [
    "X",
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

FALLBACK_MASSES = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "SE": 78.971,
}


def is_hydrogen(atom) -> bool:
    try:
        element = str(atom.element).strip().upper()
        if element:
            return element == "H"
    except Exception:
        pass

    return atom.name.strip().upper().startswith("H")


def atom_mass(atom) -> float:
    try:
        mass = float(atom.mass)
        if np.isfinite(mass) and mass > 0:
            return mass
    except Exception:
        pass

    try:
        element = str(atom.element).strip().upper()
    except Exception:
        element = atom.name.strip().upper()

    return FALLBACK_MASSES.get(element, 12.011)


def residue_chain_id(residue) -> str:
    atom = residue.atoms[0]

    for attr in ("chainID", "chainid", "segid"):
        try:
            value = getattr(atom, attr)
            if value is not None and str(value).strip():
                return str(value).strip()
        except Exception:
            pass

    return ""


def sidechain_direction(residue, ca_position) -> np.ndarray:
    """
    Unit vector from C-alpha to the center of mass of
    heavy side-chain atoms.

    Glycine has no heavy side-chain atoms -> zero vector.
    """
    atoms = []

    for atom in residue.atoms:
        name = atom.name.strip().upper()

        if name in BACKBONE_ATOMS:
            continue

        if is_hydrogen(atom):
            continue

        atoms.append(atom)

    if not atoms:
        return np.zeros(3, dtype=np.float32)

    coords = np.asarray(
        [atom.position for atom in atoms],
        dtype=np.float64,
    )

    masses = np.asarray(
        [atom_mass(atom) for atom in atoms],
        dtype=np.float64,
    )

    com = np.average(coords, axis=0, weights=masses)
    vector = com - np.asarray(ca_position, dtype=np.float64)

    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    return (vector / norm).astype(np.float32)


def build_edges(
    ca_positions: np.ndarray,
    residues,
    cutoff: float,
):
    """
    Construct directed C-alpha graph.

    edge_index[0] = source
    edge_index[1] = target

    seq_edge_features:
        [1, 0] -> target is previous residue
        [0, 1] -> target is next residue
        [0, 0] -> other spatial contact
    """

    distances = distance_array(
        ca_positions,
        ca_positions,
    )

    valid = (
        (distances <= cutoff)
        & (distances > 1e-8)
    )

    src, dst = np.where(valid)

    edge_index = np.stack(
        [src, dst],
        axis=0,
    ).astype(np.int64)

    seq_features = np.zeros(
        (len(src), 2),
        dtype=np.float32,
    )

    chain_ids = np.asarray(
        [residue_chain_id(r) for r in residues],
        dtype=object,
    )

    same_chain = chain_ids[src] == chain_ids[dst]

    is_previous = (
        same_chain
        & (dst == src - 1)
    )

    is_next = (
        same_chain
        & (dst == src + 1)
    )

    seq_features[is_previous, 0] = 1.0
    seq_features[is_next, 1] = 1.0

    return edge_index, seq_features


def save_dataset(
    output_path: Path,
    positions,
    sidechain_direction,
    edge_index,
    seq_edge_features,
    node_types,
    sequence_position,
    times,
    frame_indices,
    cutoff,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        pos=positions.astype(np.float32),
        sidechain_direction=sidechain_direction.astype(np.float32),
        node_types=node_types.astype(np.int64),
        sequence_position=sequence_position.astype(np.int64),
        edge_index=edge_index.astype(np.int64),
        seq_edge_features=seq_edge_features.astype(np.float32),
        time=times.astype(np.float64),
        frame_index=frame_indices.astype(np.int64),
        type_names=np.asarray(TYPE_NAMES),
        cutoff=np.asarray(cutoff, dtype=np.float32),
    )


def validate_arguments(args, n_total: int) -> None:
    if args.cutoff <= 0:
        raise ValueError("--cutoff must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.start >= n_total:
        raise ValueError(
            f"--start ({args.start}) must be smaller than the number of "
            f"trajectory frames ({n_total})."
        )
    if args.stop is not None and args.stop <= args.start:
        raise ValueError("--stop must be greater than --start.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be strictly between 0 and 1.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--topology",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--trajectory",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=12.0,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--stop",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of final frames used for validation.",
    )

    args = parser.parse_args()

    u = mda.Universe(
        args.topology,
        args.trajectory,
    )

    protein = u.select_atoms("protein")
    residues = list(protein.residues)

    if not residues:
        raise RuntimeError("No protein residues found.")

    ca_indices = []

    for residue in residues:
        ca = residue.atoms.select_atoms("name CA")

        if len(ca) != 1:
            raise RuntimeError(
                f"{residue.resname} {residue.resid}: "
                f"expected exactly one CA, found {len(ca)}"
            )

        ca_indices.append(ca[0].index)

    # Indexing a Universe AtomGroup returns another AtomGroup.  Keeping this as
    # an AtomGroup (rather than a Python list of Atom objects) is important:
    # AtomGroup exposes the vectorized, frame-aware ``positions`` property.
    ca_atoms = u.atoms[np.asarray(ca_indices, dtype=np.int64)]

    n_nodes = len(residues)

    node_types = np.asarray(
        [
            AA_TO_TYPE.get(res.resname.upper(), 0)
            for res in residues
        ],
        dtype=np.int64,
    )

    sequence_position = np.arange(
        n_nodes,
        dtype=np.int64,
    )

    unknown = [
        (res.resname, res.resid)
        for res in residues
        if AA_TO_TYPE.get(res.resname.upper(), 0) == 0
    ]

    if unknown:
        print(
            "WARNING: unknown residue types:",
            unknown[:20],
        )

    n_total = len(u.trajectory)
    validate_arguments(args, n_total)

    if args.stop is None:
        stop = n_total
    else:
        stop = min(args.stop, n_total)

    frame_ids = np.arange(
        args.start,
        stop,
        args.stride,
        dtype=np.int64,
    )

    if len(frame_ids) < 2:
        raise RuntimeError("Too few frames selected.")

    n_frames = len(frame_ids)

    print(f"Residues: {n_nodes}")
    print(f"Frames:   {n_frames}")
    print(f"Cutoff:   {args.cutoff:.2f} Å")

    # ---------------------------------------------------------------
    # First pass: determine maximum edge count.
    # ---------------------------------------------------------------

    max_edges = 0

    for k, frame_id in enumerate(frame_ids):
        u.trajectory[int(frame_id)]

        ca_pos = np.asarray(
            ca_atoms.positions,
            dtype=np.float32,
        )

        edge_index, _ = build_edges(
            ca_pos,
            residues,
            args.cutoff,
        )

        max_edges = max(
            max_edges,
            edge_index.shape[1],
        )

        if (k + 1) % 100 == 0 or k == n_frames - 1:
            print(
                f"Graph pass: {k + 1}/{n_frames}, "
                f"max edges = {max_edges}"
            )

    # ---------------------------------------------------------------
    # Allocate.
    # ---------------------------------------------------------------

    positions = np.zeros(
        (n_frames, n_nodes, 3),
        dtype=np.float32,
    )

    sidechain_vectors = np.zeros(
        (n_frames, n_nodes, 3),
        dtype=np.float32,
    )

    edge_index_all = np.full(
        (n_frames, 2, max_edges),
        -1,
        dtype=np.int64,
    )

    seq_edge_features_all = np.zeros(
        (n_frames, max_edges, 2),
        dtype=np.float32,
    )

    times = np.zeros(
        n_frames,
        dtype=np.float64,
    )

    # ---------------------------------------------------------------
    # Second pass.
    # ---------------------------------------------------------------

    for k, frame_id in enumerate(frame_ids):
        u.trajectory[int(frame_id)]

        ts = u.trajectory.ts

        ca_pos = np.asarray(
            ca_atoms.positions,
            dtype=np.float32,
        ).copy()

        positions[k] = ca_pos
        times[k] = float(ts.time)

        for i, residue in enumerate(residues):
            sidechain_vectors[k, i] = sidechain_direction(
                residue,
                ca_pos[i],
            )

        # Recompute the graph instead of retaining every frame's edges from
        # the first pass.  For long trajectories this avoids a large duplicate
        # in-memory copy of the variable-size edge arrays.
        ei, ef = build_edges(
            ca_pos,
            residues,
            args.cutoff,
        )
        e = ei.shape[1]

        edge_index_all[
            k,
            :,
            :e
        ] = ei

        seq_edge_features_all[
            k,
            :e,
        ] = ef

        if (k + 1) % 100 == 0 or k == n_frames - 1:
            print(
                f"Extraction pass: {k + 1}/{n_frames}"
            )

    # ---------------------------------------------------------------
    # Contiguous validation split.
    # ---------------------------------------------------------------

    n_val = min(
        n_frames - 1,
        max(
            1,
            int(round(args.val_fraction * n_frames)),
        ),
    )

    n_train = n_frames - n_val

    train_slice = slice(
        0,
        n_train,
    )

    val_slice = slice(
        n_train,
        n_frames,
    )

    train_path = args.output_dir / "train.npz"
    val_path = args.output_dir / "val.npz"

    save_dataset(
        train_path,
        positions[train_slice],
        sidechain_vectors[train_slice],
        edge_index_all[train_slice],
        seq_edge_features_all[train_slice],
        node_types,
        sequence_position,
        times[train_slice],
        frame_ids[train_slice],
        args.cutoff,
    )

    save_dataset(
        val_path,
        positions[val_slice],
        sidechain_vectors[val_slice],
        edge_index_all[val_slice],
        seq_edge_features_all[val_slice],
        node_types,
        sequence_position,
        times[val_slice],
        frame_ids[val_slice],
        args.cutoff,
    )

    print()
    print("Saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print()
    print(f"Train frames: {n_train}")
    print(f"Valid frames: {n_val}")
    print(f"Max edges:    {max_edges}")


if __name__ == "__main__":
    main()
