"""Deterministic Lattice-based LEGO dataset generator.

This completely replaces the continuous-to-discrete legacy pipeline.
It uses a 3D discrete Turtle (L-system) with an explicit OccupancyGrid to
guarantee exact 6-connectivity, perfect rotational frame alignment, and
zero clashing via rejection sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

# Ensure repository root is in path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from lego.color_rules import assign_color_and_dipole
from lego.lego_blocks import GRID_ROTATIONS, rotated_offsets, LEGO_LIBRARY
from lego.utils import DEFAULT_IRREPS, default_dataset_path, save_samples, irrep_signature


class DiscreteTurtle:
    """A discrete 3D turtle that tracks a rigorous orthogonal frame on Z^3.
    
    Uses standard aeronautical axes:
    Tangent (X) = Forward
    Normal (Y) = Left
    Binormal (Z) = Up
    """
    def __init__(self, start_pos: np.ndarray):
        self.pos = np.asarray(start_pos, dtype=np.int32)
        # Standard basis
        self.tangent = np.array([1, 0, 0], dtype=np.int32)
        self.normal = np.array([0, 1, 0], dtype=np.int32)
        self.binormal = np.array([0, 0, 1], dtype=np.int32)
        
        self.anchors: List[np.ndarray] = []
        self.brick_types: List[str] = []
        self.rotations: List[np.ndarray] = []
        self.roles: List[str] = []
        self.occupied_voxels = set()
        
    def _current_rotation_matrix(self) -> np.ndarray:
        """Constructs a valid SO(3) rotation matrix from the discrete frame."""
        rot = np.stack([self.tangent, self.normal, self.binormal], axis=1).astype(np.float32)
        # Snap to nearest valid grid rotation to avoid floating point drift
        best_rot = GRID_ROTATIONS[0]
        best_trace = -np.inf
        for g_rot in GRID_ROTATIONS:
            trace = np.trace(g_rot.T @ rot)
            if trace > best_trace:
                best_trace = trace
                best_rot = g_rot
        return best_rot.astype(np.float32)

    def place_and_advance(self, brick_type: str, role: str, advance_local: Tuple[int, int, int]) -> bool:
        """
        Attempts to place a volumetric brick and advance the turtle's anchor.
        Returns False if the placement would cause a geometric collision.
        """
        rot = self._current_rotation_matrix()
        offsets = np.asarray(LEGO_LIBRARY[brick_type]["offsets"], dtype=np.int32)
        rotated = rotated_offsets(offsets, np.rint(rot).astype(np.int32))
        world_voxels = rotated + self.pos[None, :]
        
        # 1. Collision detection (Self-Avoiding Walk check)
        for v in world_voxels:
            if tuple(v.tolist()) in self.occupied_voxels:
                return False
                
        # 2. Commit placement
        self.anchors.append(self.pos.copy())
        self.brick_types.append(brick_type)
        self.rotations.append(rot)
        self.roles.append(role)
        for v in world_voxels:
            self.occupied_voxels.add(tuple(v.tolist()))
            
        # 3. Advance anchor to the exit interface of the placed block
        dx = advance_local[0] * self.tangent
        dy = advance_local[1] * self.normal
        dz = advance_local[2] * self.binormal
        self.pos = self.pos + dx + dy + dz
        return True

    def forward(self, steps: int = 1):
        """Translates the turtle along the tangent without placing a brick."""
        self.pos += self.tangent * steps

    def yaw_left(self):
        """Turn left around binormal (Z)."""
        new_tangent = self.normal.copy()
        self.normal = -self.tangent.copy()
        self.tangent = new_tangent

    def yaw_right(self):
        """Turn right around binormal (Z)."""
        new_tangent = -self.normal.copy()
        self.normal = self.tangent.copy()
        self.tangent = new_tangent

    def pitch_up(self):
        """Nose up around normal (Y)."""
        new_tangent = self.binormal.copy()
        self.binormal = -self.tangent.copy()
        self.tangent = new_tangent

    def pitch_down(self):
        """Nose down around normal (Y)."""
        new_tangent = -self.binormal.copy()
        self.binormal = self.tangent.copy()
        self.tangent = new_tangent

    def roll_left(self):
        """Roll 90 degrees counter-clockwise around tangent (X)."""
        new_normal = self.binormal.copy()
        self.binormal = -self.normal.copy()
        self.normal = new_normal

    def roll_right(self):
        """Roll 90 degrees clockwise around tangent (X)."""
        new_normal = -self.binormal.copy()
        self.binormal = self.normal.copy()
        self.normal = new_normal

    def roll(self):
        """Roll 180 degrees around tangent (flips normal and binormal)."""
        self.normal = -self.normal
        self.binormal = -self.binormal


class LegoProceduralEngine:
    def __init__(
        self,
        *,
        min_nodes: int = 18,
        max_nodes: int = 40,
        scaffold_family: str = "mixed",
        dipole_noise_scale: float = 0.0,
        shape_noise_scale: float = 0.0,
    ) -> None:
        self.irreps = DEFAULT_IRREPS
        self.min_nodes = max(8, min_nodes)
        self.max_nodes = max(self.min_nodes, max_nodes)
        self.scaffold_family = scaffold_family.strip().lower()
        self.dipole_noise_scale = dipole_noise_scale
        self.shape_noise_scale = shape_noise_scale

    def _generate_sheet(self, turtle: DiscreteTurtle, target_length: int, rng: np.random.Generator) -> bool:
        current_len = 0
        left_turn = True 
        
        while current_len < target_length:
            run_length = rng.integers(3, 7)
            for i in range(run_length):
                if current_len >= target_length: return True
                
                if i % 2 == 0 and i < run_length - 1 and rng.random() > 0.5:
                    if not turtle.place_and_advance("1x2", "PLANAR", (2, 0, 0)): return False
                    if not turtle.place_and_advance("1x1", "PLANAR", (1, 0, 0)): return False
                    current_len += 2
                else:
                    if not turtle.place_and_advance("1x1", "STRAIGHT", (1, 0, 0)): return False
                    current_len += 1
            
            if current_len >= target_length: return True

            # First Corner
            if left_turn:
                if not turtle.place_and_advance("L-shape", "BEND_LEFT", (0, 2, 0)): return False
                turtle.yaw_left()
            else:
                turtle.roll() 
                if not turtle.place_and_advance("L-shape", "BEND_RIGHT", (0, 2, 0)): return False
                turtle.yaw_left() # Local left always turns into the normal
                turtle.roll()
            current_len += 1
            if current_len >= target_length: return True
            
            gap_length = rng.integers(1, 4) 
            for _ in range(gap_length):
                if not turtle.place_and_advance("1x1", "SHEET_EDGE", (1, 0, 0)): return False
                current_len += 1
                if current_len >= target_length: return True
                
            # Second Corner
            if left_turn:
                if not turtle.place_and_advance("L-shape", "BEND_LEFT", (0, 2, 0)): return False
                turtle.yaw_left()
            else:
                turtle.roll()
                if not turtle.place_and_advance("L-shape", "BEND_RIGHT", (0, 2, 0)): return False
                turtle.yaw_left()
                turtle.roll()
            current_len += 1
            
            left_turn = not left_turn 
        return True

    def _generate_chain(self, turtle: DiscreteTurtle, target_length: int, rng: np.random.Generator) -> bool:
        current_len = 0
        up_state = True
        
        while current_len < target_length:
            run_length = rng.integers(4, 8)
            for _ in range(run_length):
                if current_len >= target_length: return True
                
                if current_len % 2 == 0:
                    turtle.forward(1) # Clear connection bay for the T-shape's rear arm
                    if not up_state: turtle.roll() 
                    if not turtle.place_and_advance("T-shape", "JUNCTION_BRANCH_LEFT" if up_state else "JUNCTION_BRANCH_RIGHT", (2, 0, 0)): return False
                    if not up_state: turtle.roll() 
                else:
                    if not turtle.place_and_advance("1x1", "STRAIGHT", (1, 0, 0)): return False
                
                up_state = not up_state
                current_len += 1

            if current_len >= target_length: return True

            turn_dir = rng.choice(["left", "right"])
            if turn_dir == "left":
                if not turtle.place_and_advance("L-shape", "BEND_LEFT", (0, 2, 0)): return False
                turtle.yaw_left()
            else:
                turtle.roll()
                if not turtle.place_and_advance("L-shape", "BEND_RIGHT", (0, 2, 0)): return False
                turtle.yaw_left()
                turtle.roll()
            current_len += 1
            if current_len >= target_length: return True
            
            gap_length = rng.integers(2, 4)
            for _ in range(gap_length):
                if not turtle.place_and_advance("1x1", "STRAIGHT", (1, 0, 0)): return False
                current_len += 1
                if current_len >= target_length: return True
                
            turn_dir = rng.choice(["left", "right"])
            if turn_dir == "left":
                if not turtle.place_and_advance("L-shape", "BEND_LEFT", (0, 2, 0)): return False
                turtle.yaw_left()
            else:
                turtle.roll()
                if not turtle.place_and_advance("L-shape", "BEND_RIGHT", (0, 2, 0)): return False
                turtle.yaw_left()
                turtle.roll()
            current_len += 1
        return True

    def _generate_alpha_helix(self, turtle: DiscreteTurtle, target_length: int, rng: np.random.Generator) -> bool:
        current_len = 0
        
        while current_len < target_length:
            role = f"HELIX_PHASE_{current_len % 4}"
            
            # Step 1: Draw outward
            if not turtle.place_and_advance("1x2", role, (2, 0, 0)): return False
            current_len += 1
            if current_len >= target_length: return True
            
            # Step 2: Snap parallel to axis
            turtle.yaw_left()
            if not turtle.place_and_advance("1x1", role, (1, 0, 0)): return False
            current_len += 1
            if current_len >= target_length: return True
            
            # Step 3: Jump Z-plane to finalize the discrete helix spiral
            turtle.pitch_up()
            if not turtle.place_and_advance("1x1", role, (1, 0, 0)): return False
            turtle.pitch_down()
            current_len += 1
            
        return True

    def build_sample(self, *, rng: np.random.Generator) -> Dict:
        target_nodes = int(rng.integers(self.min_nodes, self.max_nodes + 1))
        
        for attempt in range(100):
            family = self.scaffold_family
            if family == "mixed":
                probs = [0.35, 0.35, 0.30]
                family = str(rng.choice(["chain", "sheet", "alpha_helix"], p=probs))

            turtle = DiscreteTurtle(start_pos=np.array([0, 0, 0]))
            
            success = False
            if family == "sheet":
                success = self._generate_sheet(turtle, target_nodes, rng)
            elif family == "chain":
                success = self._generate_chain(turtle, target_nodes, rng)
            elif family == "alpha_helix":
                success = self._generate_alpha_helix(turtle, target_nodes, rng)
                
            if success and len(turtle.anchors) >= self.min_nodes:
                break
        else:
            raise RuntimeError("Failed to generate a non-overlapping scaffold after 100 attempts.")
            
        anchors = np.array(turtle.anchors, dtype=np.float32)
        rotations = np.array(turtle.rotations, dtype=np.float32)
        brick_types = np.array(turtle.brick_types)
        role_names = np.array(turtle.roles)
        n = len(anchors)

        descriptors = {
            "tangent": rotations[:, :, 0],
            "branch_local_normal": rotations[:, :, 1],
            "bend_axis": rotations[:, :, 2],
            "phase_index": np.arange(n) % 4
        }
        
        color_class, dipoles = assign_color_and_dipole(
            role_names=role_names,
            seq_index_in_branch=np.arange(n),
            descriptors=descriptors,
            dipole_noise_scale=self.dipole_noise_scale,
            rng=rng,
        )

        shape_features = []
        for i in range(n):
            coeff = np.zeros(((3 + 1)**2,), dtype=np.float32)
            coeff[0] = 1.0 
            if self.shape_noise_scale > 0.0:
                coeff += rng.normal(scale=self.shape_noise_scale, size=coeff.shape).astype(np.float32)
            shape_features.append(coeff)
            
        shape_features = np.array(shape_features, dtype=np.float32)

        return {
            "coefficients": np.zeros((16,), dtype=np.float32),
            "irreps": np.asarray(str(self.irreps)),
            "generation_mode": np.asarray("discrete_turtle"),
            "scaffold_family": np.asarray(family),
            "pos": anchors,
            "rotations": rotations,
            "types": brick_types,
            "features": shape_features,
            "dipoles": dipoles,
            "brick_anchors": anchors,
            "brick_rotations": rotations,
            "brick_types": brick_types,
            "brick_features": shape_features,
            "brick_dipoles": dipoles,
            "role": role_names,
        }

    def generate_dataset(self, n_samples: int = 1, seed: int | None = None) -> List[Dict]:
        rng = np.random.default_rng(seed)
        return [self.build_sample(rng=rng) for _ in range(int(n_samples))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discrete Turtle-based LEGO dataset generator.")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--path", type=str, default=str(default_dataset_path()), help="Output canonical dataset path.")
    parser.add_argument(
        "--scaffold-family", type=str, default="mixed", choices=["mixed", "chain", "alpha_helix", "sheet"]
    )
    parser.add_argument("--min-nodes", type=int, default=18)
    parser.add_argument("--max-nodes", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = LegoProceduralEngine(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        scaffold_family=args.scaffold_family,
    )
    samples = engine.generate_dataset(n_samples=args.samples, seed=args.seed)
    save_samples(args.path, samples)

    print("--- Discrete Procedural LEGO Dataset Generated ---")
    print(f"Samples: {len(samples)}")
    print(f"Saved to: {args.path}")
    for idx, sample in enumerate(samples):
        print(f"Sample {idx}: nodes={len(sample['brick_types'])}, family={sample['scaffold_family']}")


if __name__ == "__main__":
    main()