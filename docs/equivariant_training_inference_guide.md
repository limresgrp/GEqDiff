# LEGO Flow Matching (Current): Direct-Tensor Training and Sampling

This is the current behavior used by the LEGO pipeline.

Scope:

- Training corruption/targets: [flow_matching.py](/home/angiod@usi.ch/GEqDiff/geqdiff/nn/flow_matching.py)
- Sampling: [sample_lego.py](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/sample_lego.py)
- Losses: [\_loss.py](/home/angiod@usi.ch/GEqDiff/geqdiff/train/_loss.py)

## 1) Fields used (direct only)

LEGO models now use direct targets (no scalar/equivariant coupling path in sampler):

- `pos` (3D)
- `shape_features` (16D, irreps `1x0e + 1x1o + 1x2e + 1x3o`)
- `dipole_direction` (3D vector; magnitude is encoded directly in vector norm)

Deprecated decoupled fields (`shape_scalar_features`, `shape_equiv_features`, `dipole_strength`) are not used by the direct LEGO sampling path.

## 2) Training target definition

For each corrupted field `x`:

- Noisy state at time `tau`:
  - `x_tau = (1 - tau) * x + tau * n`
- Velocity target:
  - `v_target = n - x`

Only masked nodes (typically `ligand_mask`) are updated/supervised for corrupted fields.

## 3) Sampling update

At each reverse step:

1. Build model input from current state.
2. Predict direct velocities:
   - `velocity`
   - `shape_features_velocity`
   - `dipole_direction_velocity`
3. Integrate Euler/Heun in `tau` space.
4. Merge updates only on masked nodes.

Optional clash guidance modifies `velocity` in score-space but does not reintroduce decoupled tensor logic.

## 4) Shape decoding

After integration, continuous `shape_features` are decoded to discrete brick type/rotation via:

- `input_knn` (default): nearest exemplar in shape-feature space built from the input dataset.
- `keep_original`: keep original discrete type/rotation.

`legacy`/automatic legacy decoding is removed from the direct-only sampler.

## 5) Practical checks

If sampling quality is poor, check:

- `corrupt_fields` in model config include exactly the intended direct fields.
- Training/eval masks (especially `ligand_mask`) are correct.
- Decode mode (`input_knn`) and candidate size are appropriate for the dataset.
- Relative evaluation metrics (validity/shape/dipoles/pose) rather than legacy absolute shell/compactness assumptions.
