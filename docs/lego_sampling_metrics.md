# LEGO Sampling Metrics (Deterministic Scaffold Dataset)

This document describes the current evaluation logic for sampled LEGO structures.

Implementation references:

- Scoring core: [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py)
- Report script: [evaluate_lego_samples.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/evaluate_lego_samples.py)
- HTML visualization: [lego_visualizer.py](/home/angiod@usi.ch/GEqDiff/lego/lego_visualizer.py)

Dataset assumption for current generator:

- scaffold topology is non-branching (node degree \(\le 2\))
- `T-shape` bricks are local turn/helix motifs, not junction connectors
- the historical role tag `JUNCTION_T` is kept as a compatibility label for this motif

## Validity philosophy

In the deterministic scaffold dataset, the original structures can contain geometric overlap patterns by construction.

For this reason, metrics are now split into:

- `validity` = **absolute generative quality** (clashes/connectivity of generated structure itself)
- `validity_relative` = **fidelity drift** vs paired original (excess degradation only)

## Score axes

For paired samples, the UI/report uses:

1. `validity` (absolute geometry quality)
2. `validity_relative` (reference-relative geometry quality)
3. `shape` (deterministic shape reconstruction)
4. `dipoles` (vector + energetic consistency)
5. `pose` (anchor-shift quality)

Overall score in HTML:

- `overall = 0.45 * validity + 0.25 * shape + 0.20 * dipoles + 0.10 * pose`

## Validity (absolute)

Uses generated structure geometry only:

- effective overlap volume
- severe overlap pair count
- connected component count

Score:

- `validity = 100 * exp(-w_overlap*effective_overlap - w_severe*severe_pairs) / sqrt(max(num_components, 1))`

This is the main validity axis used in overall score and badges.

## Validity (relative)

Uses sampled vs original metrics from geometry analysis:

- effective overlap volume
- severe overlap pair count
- connected component count
- fixed-node max shift

Relative excess terms:

- `excess_effective_overlap = max(0, sampled_effective - original_effective - eps)`
- `excess_severe_pairs = max(0, sampled_severe - original_severe)`
- `excess_components = max(0, sampled_components - original_components)`

Relative score:

- `validity_relative = 100 * exp(-a*excess_overlap - b*excess_severe - c*excess_components - d*fixed_shift_max)`

with calibrated constants in [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py).

## Shape score (deterministic)

Computed on diffused nodes (`sampled_brick_mask`):

- RMSE between sampled/original `brick_features` (16D)
- decoded brick type accuracy
- rotation similarity (geodesic similarity on rotation matrices, type-matched nodes)

Combined as weighted score in `[0, 100]`:

- RMSE term (Gaussian-like)
- type accuracy term
- rotation similarity term

## Dipole score (deterministic + energetic)

Computed on diffused nodes:

- vector cosine agreement (sampled vs original `brick_dipoles`)
- dipole magnitude RMSE
- dipole energy delta vs original (`weighted_dipole_energy`)

Combined as weighted score in `[0, 100]`:

- cosine-based directional similarity
- magnitude consistency
- energetic consistency

## Pose score

Uses shift metrics from sampled vs original anchors:

- diffused mean shift
- diffused max shift
- fixed max shift

Score decays exponentially with these shifts.

## Raw metrics kept for diagnostics

Even with absolute validity scoring, raw fields are still retained and shown:

- overlap and clash counts/volumes
- connectivity
- matched-face statistics
- dipole contact counts and energies

This keeps debugging transparent while separating:

- generation quality (`validity`)
- reference fidelity (`validity_relative`)
