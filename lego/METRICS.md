# LEGO Scoring Metrics

This note describes the structure scores shown in the LEGO visualizer and used by the offline evaluator.

The goal of the score panel is not to estimate real physical energy. It is a compact way to communicate three things:

- Is the assembly geometrically valid?
- Is it compact or loosely packed?
- Does it behave like the intended shell / target shape?
- Are dipoles arranged in a complementary way across contacting faces?

All headline scores are reported on a `0..100` scale. Higher is better.

The viewer also shows an `Overall` badge/summary score. That aggregate is deliberately simple:

```text
Overall = 0.70 * Validity + 0.30 * Dipoles
```

`Compactness` and `Shellness` stay visible as diagnostics, but they do not contribute to `Overall`. For shell datasets they often move in opposite directions, so folding them into one headline number is more misleading than useful.

## Headline Scores

### 1. Validity

`Validity` penalizes geometric clashes and disconnected assemblies.

It is intentionally tolerance-based: tiny continuous interpenetrations from sampling should not collapse the score on their own.

It is computed from:

- `total_overlap_volume`
  - continuous overlap volume between voxels of different bricks
  - `0` means no clashes
- `effective_overlap_volume`
  - overlap that remains after subtracting a small per-pair tolerance
- `micro_overlapping_pairs`
  - number of pairs with overlap above the micro-overlap tolerance
- `severe_overlapping_pairs`
  - number of pairs with clearly non-negligible overlap
- `num_components`
  - number of connected components in the assembly graph

The implementation uses two overlap thresholds:

- `overlap_tolerance_per_pair = 0.01`
- `severe_overlap_threshold = 0.08`

For each brick pair, the first `0.01` of overlap is treated as numerical / micro-overlap tolerance. Only the excess contributes to `effective_overlap_volume`.

The score is:

```text
Validity = 100 * exp(-12 * effective_overlap_volume - 0.9 * severe_overlapping_pairs) / sqrt(num_components)
```

Interpretation:

- `100` means a clean, connected assembly
- values near `0` indicate substantial overlap and/or disconnected pieces
- tiny overlaps can still get a high score if they stay below the tolerance floor

### 2. Compactness

`Compactness` measures how tightly bricks pack against each other through face-to-face contacts.

It is based on:

- `matched_face_area`
  - continuous estimate of how much face area is aligned between neighboring bricks
- `intrinsic_face_count`
  - total number of exposed faces available on all bricks

The normalized ratio is:

```text
matched_face_ratio = clip(2 * matched_face_area / intrinsic_face_count, 0, 1)
```

and the headline score is:

```text
Compactness = 100 * matched_face_ratio
```

Interpretation:

- high score means the assembly is dense and tightly connected
- low score means the assembly is sparse, shell-like, floating, or only weakly fitted

Low compactness is not automatically bad for shell datasets. A thin shell can be intentionally non-compact.

### 3. Shellness

`Shellness` measures how well the structure behaves like a shell rather than a dense packed solid.

It is based on:

- `shell_surface_ratio = 1 - matched_face_ratio`
  - high when the structure keeps many faces exposed and therefore stays shell-like
- `target_f1`, when `target_voxels` are available
  - high when the structure matches the intended target shell

The headline score is:

```text
Shellness = 100 * (0.85 * target_f1 + 0.15 * shell_surface_ratio)
```

when a target is available, and

```text
Shellness = 100 * shell_surface_ratio
```

otherwise.

Interpretation:

- high score means the assembly preserves a shell-like / target-matching organization
- low score means it is drifting toward a dense fill or away from the target shell

### 4. Dipoles

`Dipoles` measures polarity complementarity across matched contacts.

For every approximately matched face contact:

- attractive area: opposite-facing dipole projections
- repulsive area: same-sign facing dipole projections
- neutral area: one or both projections are small

The headline score is:

```text
Dipoles = 100 * (attractive_contact_area + 0.5 * neutral_contact_area) / total_contact_area
```

Interpretation:

- high score means contacts are mostly attractive or at least not repulsive
- low score means many contacts are polarity-conflicting

Neutral contacts count half-positive on purpose: they are less informative than attractive contacts, but not as bad as repulsive ones.

## Detailed Partials

The viewer also shows the raw partial metrics behind each headline score.

### Validity Details

- `total_overlap_volume`
  - continuous clash measure; more informative than hard voxel collisions for continuous sampling
- `effective_overlap_volume`
  - the overlap that is actually penalized after subtracting the per-pair tolerance floor
- `clashing_brick_pairs`
  - number of pairs with overlap
- `micro_overlapping_pairs`
  - pairs above the micro-overlap tolerance
- `severe_overlapping_pairs`
  - pairs above the severe-overlap threshold
- `max_pair_overlap_volume`
  - worst local clash
- `overlap_tolerance_per_pair`
- `severe_overlap_threshold`
- `num_components`
  - connectivity of the assembly
- `is_valid_like`
  - `yes` if the effective overlap stays very small, there are no severe clashes, and the assembly is connected

### Compactness Details

- `matched_face_area`
  - continuous matched-contact area across brick pairs
- `matched_face_ratio`
  - normalized version used in the `Compactness` score
- `connected_brick_pairs`
  - number of pairs with sufficiently strong face contact
- `intrinsic_face_count`
  - total number of available exposed faces on the bricks

### Shellness Details

- `shell_surface_ratio`
  - `1 - matched_face_ratio`
  - high values indicate a more exposed, shell-like arrangement
- `target_coverage`
- `target_precision`
- `target_f1`
- `mean_target_to_structure_distance`
- `mean_structure_to_target_distance`

### Dipole Details

- `attractive_contact_area`
- `repulsive_contact_area`
- `neutral_contact_area`
- `total_contact_area`
- `weighted_dipole_energy`
  - face-area-weighted version of the same local dipole interaction rule used during procedural dipole assignment
- `mean_weighted_dipole_energy`
  - weighted dipole energy divided by total contact area

Lower dipole energy is better. The headline `Dipoles` score is easier to read, while the energy term is more diagnostic.

## Sampled vs Original

When a sampled dataset also contains `original_*` fields, the visualizer reports:

- headline score deltas for `Validity`, `Compactness`, `Shellness`, and `Dipoles`
- `diffused_shift_mean` and `diffused_shift_max`
- `fixed_shift_mean` and `fixed_shift_max`
- metric deltas such as overlap, matched-face ratio, shell surface ratio, and dipole energy

This is useful for checking whether the sampled assembly preserves the intended scaffold and interaction pattern.

## Badge Colors

The visualizer adds badge colors and `Pass / Warning / Fail` labels as a quick reading aid.

Those badges are heuristic UI thresholds only. They are not part of the metric definitions and should not be treated as scientific quantities.

The raw scores and detailed partials are the authoritative values.

## Important Caveat

These metrics are intentionally continuous and LEGO-specific. They are designed for:

- comparing sampled vs original assemblies
- diagnosing clashes and misplacement
- distinguishing dense packing from shell-like structure
- checking whether dipole complementarity is improving

They are not meant to replace a true physical force field or electrostatics model.
