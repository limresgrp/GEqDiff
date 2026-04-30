# LEGO Deterministic Scaffold Pipeline

This repository now uses a deterministic scaffold generator and direct-tensor flow matching.

## Dataset generation

Main entrypoint: [lattice.py](/home/angiod@usi.ch/GEqDiff/lego/lattice.py)

Pipeline per sample:

1. Sample a scaffold topology and 3D anchors (`beta_sheet`, `alpha_helix`, or `mixed`).
   `chain` is still accepted internally as a legacy alias for `beta_sheet`.
2. Compute local descriptors (`tangent`, curvature/planarity, local branch context).
3. Assign structural roles from descriptors and topology.
4. Map roles to deterministic shape prototypes and LEGO brick placements (`1x1`, `1x2`, `L-shape`, `T-shape`).
5. Assign deterministic color/dipole vectors from local context.
6. Validate topology and geometry metadata.

Important notes:

- The dataset is no longer generated from global SH shell/solid occupancy.
- Junction graph topologies are disabled by design to avoid sequence-order ambiguity on large datasets.
- `T-shape` bricks are now used as local turn/helix motifs (sparse high-curvature sites), not as branch-junction connectors.
- For backward compatibility, this motif currently reuses the historical role label `JUNCTION_T`.
- In `alpha_helix` family mode, helices are sampled with fixed dextrorse handedness and phase-driven periodic brick programs (type + orientation), with collision-safe placement.
- Current brick policy: alpha helices use a simplified periodic `1x1/1x2` motif; beta-sheet straight segments alternate `1x1/1x2`, and turns use `L-shape`.
- `shape_features` are direct 16D coefficients (`1x0e + 1x1o + 1x2e + 1x3o`) from role prototypes.
- `dipole_direction` is a direct 3D vector target (magnitude encoded in vector norm).
- `sequence_position` is stored and used as node input (positional categorical embedding in model configs).
- `branch_kind` is emitted per brick by the raw lattice generator and is preserved through diffusion dataset building and model input configs.

## Training/sampling assumptions

Current LEGO flow-matching setup is **direct-only**:

- Position head predicts `velocity` for `pos`.
- Shape head predicts `shape_features_velocity` for `shape_features`.
- Dipole head predicts `dipole_direction_velocity` for `dipole_direction`.

No decoupled norm+direction coupling path is used anymore in sampling.

Main sampler: [sample_lego.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/sample_lego.py)

Shape decoding during sampling:

- Default is `input_knn`: decode predicted shape coefficients to brick type/rotation using nearest exemplars from the input dataset.
- Optional `keep_original` can keep original type/rotation.

## Evaluation metrics

Scoring logic is in [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py), report script in [evaluate_lego_samples.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/evaluate_lego_samples.py).

For paired sampled/original records, scoring uses mixed semantics:

- `validity` is **absolute** (clashes/connectivity on generated structure itself).
- `validity_relative` measures excess degradation vs original.
- `shape`, `dipoles`, `pose` are reference/fidelity-oriented scores.

This preserves generative validity while still exposing fidelity drift to the paired target.

## Visualization

Visualizer: [lego_visualizer.py](/home/angiod@usi.ch/GEqDiff/lego/lego_visualizer.py)

The HTML score panel now shows:

- Overall (weighted validity/shape/dipoles/pose)
- Validity
- Validity (Relative)
- Shape
- Dipoles
- Pose

and detailed metric cards for sampled/original/compare, including relative-overlap and deterministic shape/dipole diagnostics.

## Useful files

- Generator: [lattice.py](/home/angiod@usi.ch/GEqDiff/lego/lattice.py)
- Scaffold sampler: [scaffold_sampling.py](/home/angiod@usi.ch/GEqDiff/lego/scaffold_sampling.py)
- Roles/descriptors/prototypes:  
  [descriptors.py](/home/angiod@usi.ch/GEqDiff/lego/descriptors.py),  
  [role_assignment.py](/home/angiod@usi.ch/GEqDiff/lego/role_assignment.py),  
  [shape_prototypes.py](/home/angiod@usi.ch/GEqDiff/lego/shape_prototypes.py),  
  [color_rules.py](/home/angiod@usi.ch/GEqDiff/lego/color_rules.py)
- Sampling: [sample_lego.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/sample_lego.py)
- Evaluation: [evaluate_lego_samples.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/evaluate_lego_samples.py)
- Scoring core: [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py)
