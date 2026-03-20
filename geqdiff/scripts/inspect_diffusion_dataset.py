from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def _valid_nodes(mask: np.ndarray) -> np.ndarray:
    return ~np.asarray(mask, dtype=bool)


def _trim_edge_index(edge_index: np.ndarray) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        return edge_index.reshape(2, 0)
    keep = edge_index[0] >= 0
    return edge_index[:, keep]


def _connected(mask: np.ndarray, edge_index: np.ndarray) -> bool:
    nodes = np.flatnonzero(mask)
    if nodes.size <= 1:
        return True
    adjacency = {int(node): [] for node in nodes.tolist()}
    for src, dst in _trim_edge_index(edge_index).T.tolist():
        if src in adjacency and dst in adjacency:
            adjacency[src].append(dst)
    start = int(nodes[0])
    stack = [start]
    visited = {start}
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor in visited:
                continue
            visited.add(int(neighbor))
            stack.append(int(neighbor))
    return len(visited) == len(adjacency)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a padded LEGO diffusion dataset NPZ.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the split dataset NPZ.")
    parser.add_argument("--index", type=int, default=None, help="Optional example index to print in detail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as data:
        num_examples = int(data["pos"].shape[0])
        pos_mask = np.asarray(data["pos__mask__"], dtype=bool)
        num_nodes = np.asarray(data["num_nodes"], dtype=np.int64)
        ligand_size = np.asarray(data["ligand_size"], dtype=np.int64)
        num_edges = np.asarray(data["num_edges"], dtype=np.int64)
        dipole_strength = np.asarray(data["dipole_strength"], dtype=np.float32)
        dipole_mask = np.asarray(data["dipole_strength__mask__"], dtype=bool)

        avg_nodes = float(num_nodes.mean()) if num_examples > 0 else 0.0
        avg_ligand = float(ligand_size.mean()) if num_examples > 0 else 0.0
        avg_edges_per_node = float(np.mean(num_edges / np.maximum(num_nodes, 1))) if num_examples > 0 else 0.0

        histogram = Counter(
            int(float(value) > 1e-6)
            for strengths, mask in zip(dipole_strength, dipole_mask)
            for value in strengths[_valid_nodes(mask)].reshape(-1).tolist()
        )

        print("--- Dataset Stats ---")
        print(f"Examples: {num_examples}")
        print(f"Average nodes/example: {avg_nodes:.2f}")
        print(f"Average ligand size: {avg_ligand:.2f}")
        print(f"Average directed contacts/node: {avg_edges_per_node:.2f}")
        print("Dipole state histogram:")
        print(f"  neutral: {histogram.get(0, 0)}")
        print(f"  polar: {histogram.get(1, 0)}")

        if args.index is not None:
            index = int(args.index)
            if not (0 <= index < num_examples):
                raise IndexError(f"Example index {index} is out of range for {num_examples} examples.")

            node_mask = _valid_nodes(pos_mask[index])
            ligand_mask = np.asarray(data["ligand_mask"][index], dtype=bool)[node_mask]
            pocket_mask = np.asarray(data["pocket_mask"][index], dtype=bool)[node_mask]
            edge_index = _trim_edge_index(data["edge_index"][index])
            dipole_strengths = np.asarray(data["dipole_strength_raw"][index], dtype=np.float32)[node_mask]

            print(f"\n--- Example {index} ---")
            print(f"Source frame: {int(data['source_frame_id'][index])}")
            print(f"Split id: {int(data['split_id'][index])}")
            print(f"Nodes: {int(num_nodes[index])}")
            print(f"Ligand size: {int(ligand_size[index])}")
            print(f"Edges: {int(num_edges[index])}")
            print(f"Ligand connected: {_connected(ligand_mask, edge_index)}")
            print(f"Ligand/pocket disjoint: {bool(not np.any(ligand_mask & pocket_mask))}")
            print(f"Polar blocks: {int((dipole_strengths > 1e-6).sum())}")


if __name__ == "__main__":
    main()
