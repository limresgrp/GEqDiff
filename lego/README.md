# LEGO Deterministic Scaffold Pipeline

This repository now uses a deterministic scaffold generator and direct-tensor flow matching.

## Dataset generation

Main entrypoint: [lego_engine.py](/home/angiod@usi.ch/GEqDiff/lego/lego_engine.py)

Pipeline per sample:

1. Sample a scaffold topology and 3D anchors (`chain`, `alpha_helix`, `sheet`, `junction`, or `mixed`).
2. Compute local descriptors (`tangent`, curvature/planarity, branch/junction context).
3. Assign structural roles from descriptors and topology.
4. Map roles to deterministic shape prototypes and LEGO brick placements (`1x1`, `1x2`, `L-shape`, `T-shape`).
5. Assign deterministic color/dipole vectors from local context.
6. Validate topology and geometry metadata.

Important notes:

- The dataset is no longer generated from global SH shell/solid occupancy.
- `shape_features` are direct 16D coefficients (`1x0e + 1x1o + 1x2e + 1x3o`) from role prototypes.
- `dipole_direction` is a direct 3D vector target (magnitude encoded in vector norm).
- `sequence_position` is stored and used as node input (positional categorical embedding in model configs).

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

For paired sampled/original records, scores are **relative to original baseline**:

- Original is treated as the reference target (`100` for relative score axes).
- Validity penalizes only *excess* overlap/component issues vs original.
- Shape score checks deterministic reconstruction (`shape_features` RMSE + type accuracy + rotation similarity).
- Dipole score checks vector agreement and dipole-energy consistency vs original.
- Pose score tracks anchor-shift quality for diffused/fixed nodes.

This design avoids over-penalizing overlaps already present in the deterministic dataset itself.

## Visualization

Visualizer: [lego_visualizer.py](/home/angiod@usi.ch/GEqDiff/lego/lego_visualizer.py)

The HTML score panel now shows:

- Overall (weighted validity/shape/dipoles/pose)
- Validity
- Shape
- Dipoles
- Pose

and detailed metric cards for sampled/original/compare, including relative-overlap and deterministic shape/dipole diagnostics.

## Useful files

- Generator: [lego_engine.py](/home/angiod@usi.ch/GEqDiff/lego/lego_engine.py)
- Scaffold sampler: [scaffold_sampling.py](/home/angiod@usi.ch/GEqDiff/lego/scaffold_sampling.py)
- Roles/descriptors/prototypes:  
  [descriptors.py](/home/angiod@usi.ch/GEqDiff/lego/descriptors.py),  
  [role_assignment.py](/home/angiod@usi.ch/GEqDiff/lego/role_assignment.py),  
  [shape_prototypes.py](/home/angiod@usi.ch/GEqDiff/lego/shape_prototypes.py),  
  [color_rules.py](/home/angiod@usi.ch/GEqDiff/lego/color_rules.py)
- Sampling: [sample_lego.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/sample_lego.py)
- Evaluation: [evaluate_lego_samples.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/evaluate_lego_samples.py)
- Scoring core: [score_utils.py](/home/angiod@usi.ch/GEqDiff/lego/score_utils.py)
