from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lego.utils import load_samples


def _field(sample: dict[str, Any], key: str, default=None):
    return np.asarray(sample[key]) if key in sample else default


def _as_flat(values, length: int, fill: str = "n/a") -> list[str]:
    if values is None:
        return [fill] * length
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [str(arr.item())] * length
    arr = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    out: list[str] = []
    for idx in range(length):
        if idx >= arr.shape[0]:
            out.append(fill)
            continue
        val = arr[idx, 0]
        if np.issubdtype(arr.dtype, np.number):
            number = float(val)
            out.append(str(int(round(number))) if abs(number - round(number)) < 1e-6 else f"{number:.4g}")
        else:
            out.append(str(val))
    return out


def inspect_sample(path: Path, index: int, limit: int | None) -> None:
    samples = load_samples(path)
    if index < 0 or index >= len(samples):
        raise IndexError(f"Sample index {index} is out of range for {len(samples)} samples.")
    sample = samples[index]
    num_bricks = len(np.asarray(sample["brick_types"]))
    count = num_bricks if limit is None else min(num_bricks, int(limit))

    branch = _as_flat(_field(sample, "branch_kind"), num_bricks)
    original_branch = _as_flat(_field(sample, "original_branch_kind"), num_bricks)
    sequence = _as_flat(_field(sample, "sequence_position"), num_bricks)
    brick_sequence = _as_flat(_field(sample, "brick_sequence_position"), num_bricks)
    original_sequence = _as_flat(_field(sample, "original_sequence_position"), num_bricks)
    types = np.asarray(sample["brick_types"]).astype(str)
    anchors = np.asarray(sample["brick_anchors"], dtype=np.float32)
    dipoles = np.asarray(sample.get("brick_dipoles", np.zeros((num_bricks, 3), dtype=np.float32)), dtype=np.float32)
    mask = np.asarray(sample.get("sampled_brick_mask", np.zeros((num_bricks,), dtype=bool)), dtype=bool).reshape(-1)
    if mask.shape[0] != num_bricks:
        mask = np.zeros((num_bricks,), dtype=bool)

    print(f"File: {path}")
    print(f"Sample index: {index} / {len(samples) - 1}")
    print(f"Fields present: branch_kind={'branch_kind' in sample}, sequence_position={'sequence_position' in sample}, brick_sequence_position={'brick_sequence_position' in sample}")
    print("idx\tgroup\ttype\tbranch\torig_branch\tseq\tbrick_seq\torig_seq\tanchor\tdipole_norm")
    for idx in range(count):
        group = "diffused" if bool(mask[idx]) else "fixed"
        anchor = anchors[idx]
        strength = float(np.linalg.norm(dipoles[idx]))
        print(
            f"{idx}\t{group}\t{types[idx]}\t{branch[idx]}\t{original_branch[idx]}\t"
            f"{sequence[idx]}\t{brick_sequence[idx]}\t{original_sequence[idx]}\t"
            f"({anchor[0]:.3f},{anchor[1]:.3f},{anchor[2]:.3f})\t{strength:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect branch/sequence fields saved in a sampled LEGO NPZ.")
    parser.add_argument("path", type=Path, help="Sampled LEGO NPZ.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to inspect.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of bricks to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspect_sample(args.path, index=int(args.index), limit=args.limit)


if __name__ == "__main__":
    main()
