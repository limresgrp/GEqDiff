# SH-to-LEGO Pipeline

This directory now uses one consistent pipeline:

1. Sample spherical-harmonic coefficients `c` in the irrep basis
   `1x0e + 1x1o + 1x2e + 1x3o`.
2. Turn those coefficients into a radial field
   `r(u) = base_radius + radial_scale * <c, Y(u)>`,
   where `u` is a unit direction and `Y(u)` are the real spherical harmonics up to `l = 3`.
3. Sample that radial field on a latitude/longitude grid to obtain the smooth target surface mesh.
4. Voxelize the implicit shape in either:
   `solid` mode by marking a unit cube centered at `x` as occupied when
   `||x|| <= r(x / ||x||) + voxel_margin`, or
   `shell` mode by keeping only a surface band near `r(x / ||x||)`.
   The shell mode also supports a sparsity parameter that thins the band while
   trying to preserve 6-neighbor connectivity.
5. Greedily cover the occupied voxels with rotated LEGO primitives (`1x1`, `1x2`, `L-shape`, `T-shape`).
6. Store both the smooth target mesh and the discrete brick placement in the same sample record.

## Why these irreps

For directions on the sphere, the spherical harmonics of degree `l` transform under the `SO(3)` irrep of order `l`.
Their parity is `(-1)^l`, so the physically consistent basis through `l = 3` is:

- `l = 0` -> `0e`
- `l = 1` -> `1o`
- `l = 2` -> `2e`
- `l = 3` -> `3o`

That is why the code uses `1x0e + 1x1o + 1x2e + 1x3o` instead of treating every block as even parity.

## How LEGO blocks are obtained from the irreps

The SH coefficients do not directly choose a LEGO piece. They define a continuous target shape first.
The discrete blocks are then derived in two steps:

1. `coefficients -> target_voxels`
   The SH radial field is evaluated on the voxel lattice to decide which unit cubes belong to the shape.
2. `target_voxels -> brick placements`
   The engine searches over all 90-degree rotations of the LEGO primitives and greedily picks the piece that covers the most currently uncovered voxels.

So the irreps control the global shape, while the LEGO library provides the nearest discrete approximation on the voxel grid.

## Brick features and dipoles

Each placed brick now carries two geometric descriptors:

- `brick_features`
  A raw 16D SH signature of the brick surface, using the irreps
  `1x0e + 1x1o + 1x2e + 1x3o`.
- `brick_dipoles`
  A simple charge-proxy vector. Neutral bricks use the zero vector.
  Polar bricks use one discrete axis-aligned dipole in the brick local frame,
  rotated into world coordinates by the saved `brick_rotations`.
  Dipoles are assigned by minimizing a contact energy over all touching voxel
  faces. Same-sign face charges are penalized, opposite-sign face charges are
  rewarded, and neutral face contacts carry a smaller penalty.

For diffusion training, the 16D SH signature is further decomposed into:

- `shape_scalar_features`
  Four scalar magnitudes: one for `l = 0` and one norm for each of `l = 1, 2, 3`.
- `shape_equiv_features`
  The corresponding normalized direction blocks for `l = 1, 2, 3`, concatenated into 15 channels.

The dipole field is decomposed the same way:

- `dipole_strength`
  A scalar magnitude.
- `dipole_direction`
  A normalized 3D direction.

This keeps the learned scalar magnitudes separate from the learned equivariant directions.

## Canonical dataset schema

Datasets are saved as an object array under the `samples` key. Each sample contains:

- `coefficients`: SH coefficients in the 16D irrep basis.
- `mesh_x`, `mesh_y`, `mesh_z`: the smooth target mesh sampled on a spherical grid.
- `target_voxels`: occupied integer voxel centers derived from the target mesh.
- `brick_anchors`, `brick_rotations`, `brick_types`: the LEGO approximation.
- `brick_features`, `brick_dipoles`: SH descriptors and dipole vectors for each placed brick.

Compatibility aliases (`pos`, `rotations`, `types`, `features`, `dipoles`) are kept in each sample so older code does not break immediately.

## Main entry points

- `lego_engine.py`
  Generates SH-defined target blocks, assigns dipoles, and writes the canonical dataset.
- `lego_visualizer.py`
  Loads that dataset and lets you switch between the smooth SH target, full brick meshes, per-node SH surfaces, and dipole overlays.
- `utils.py`
  Shared SH math, voxelization, dipole-aware dataset I/O, and feature extraction.
- `lego_blocks.py`
  Discrete block library plus all 90-degree grid rotations.
