# LEGO Sampling Metrics (Deterministic Scaffold Dataset)

This document describes the current evaluation logic for sampled LEGO structures.

Implementation references:

- Scoring core: [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py)
- Report script: [evaluate_lego_samples.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/evaluate_lego_samples.py)
- HTML visualization: [lego_visualizer.py](/home/angiod@usi.ch/GEqDiff/lego/lego_visualizer.py)

## Why relative metrics

In the deterministic scaffold dataset, the original structures can contain geometric overlap patterns that are part of the dataset construction itself.

Because of that, evaluation is done **relative to the original paired structure**:

- original = reference baseline
- sampled is penalized for *excess* degradation vs that baseline

This prevents false penalties from dataset-intrinsic overlap.

## Score axes

For paired samples, the UI/report uses:

1. `validity` (relative geometry quality)
2. `shape` (deterministic shape reconstruction)
3. `dipoles` (vector + energetic consistency)
4. `pose` (anchor-shift quality)

Overall score in HTML:

- `overall = 0.45 * validity + 0.25 * shape + 0.20 * dipoles + 0.10 * pose`

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

Score:

- `validity = 100 * exp(-a*excess_overlap - b*excess_severe - c*excess_components - d*fixed_shift_max)`

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

Even with relative scoring, raw fields are still retained and shown:

- overlap and clash counts/volumes
- connectivity
- matched-face statistics
- dipole contact counts and energies

This keeps debugging transparent while making final pass/fail interpretation dataset-aware.
