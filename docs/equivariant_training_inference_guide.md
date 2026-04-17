# Equivariant Tensors in LEGO Flow Matching: Training vs Inference

This document describes what the current code does for equivariant tensors (`shape_equiv_features`, `dipole_direction`) during training and sampling, and what each configuration parameter means.

Scope:
- Training corruption/targets: `geqdiff/nn/flow_matching.py`
- Sampling/inference: `geqdiff/scripts/sample_lego.py`
- Readout normalization: `deps/GEqTrain/geqtrain/nn/readout.py`
- Masked weighted losses: `geqdiff/train/_loss.py`

## 1) Representation used in the LEGO pipeline

- `shape_scalar_features`: `[..., 4]` = `[l0, ||l1||, ||l2||, ||l3||]`
- `shape_equiv_features`: `[..., 15]` = concatenated normalized directions for `l=1,2,3` blocks (`3 + 5 + 7`)
- `dipole_strength`: `[..., 1]`
- `dipole_direction`: `[..., 3]`

Combined physical tensors are reconstructed with:
- `shape = combine_shape_irreps(shape_scalar_features, shape_equiv_features)`
- `dipole = dipole_strength * dipole_direction`

## 2) Training path (ForwardFlowMatchingModule)

For each corrupt field in `corrupt_fields`, training computes:

- Sample `tau ~ U([tau_eps, 1 - tau_eps])`
- Scheduler:
  - `data_scale = 1 - tau`
  - `noise_scale = tau`
  - derivatives: `ddata/dtau = -1`, `dnoise/dtau = +1`
- Build noisy state:
  - `x_tau = (1 - tau) * x + tau * n`
- Build target velocity:
  - `target = d/dtau x_tau = n - x`

Then mask logic:
- If `mask_field == ""`: all nodes are corrupted and supervised.
- Else:
  - masked nodes get `x_tau`, unmasked nodes remain clean (or partially noised if `unmasked_noise_scale > 0`)
  - unmasked nodes get zero target.

Noise sampling for equivariant fields:
- `shape_equiv_features`: random unit vectors per SH block (`3,5,7`) then concatenated.
- `dipole_direction`: random unit vector in `R^3`.

Centering:
- `center` / `center_mask_field` / `center_noise` / `noise_center_mask_field` are applied exactly as configured per field.
- For shape/dipole fields, centering is usually disabled (`center: false`).

## 3) Direction-magnitude coupling in training targets

If `directional_velocity_couplings` is enabled, target tensors are rewritten:

- For each configured `(scalar_index, equiv_slice)`:
  1. Normalize `equiv_target[equiv_slice]` to unit norm (or zero if tiny).
  2. Write its norm into `scalar_target[scalar_index]`.

So with coupling enabled, scalar targets become block-speed magnitudes and equivariant targets become directions.

Important: this modifies `_target` tensors produced by corruption, not only logging.

## 4) Readout behavior (`normalize_equivariant_output`)

For any `ReadoutModule` with equivariant output:

- `normalize_equivariant_output: false`: raw equivariant output (free norm).
- `normalize_equivariant_output: true`: each irrep block is normalized to unit norm (or zero if tiny).

This happens in the head output itself, before loss.

## 5) Loss masking for directional channels

Directional losses usually use weighted masked MSE with `weight_field` such as:
- `shape_l1_weight`, `shape_l2_weight`, `shape_l3_weight`
- `dipole_direction_weight`

Current behavior in `MaskedWeightedMSELoss`:
- applies `mask_field` (e.g. `ligand_mask`)
- and if a matching validity field exists (`*_valid`), it automatically intersects with it:
  - `shape_l1_weight -> shape_l1_valid`
  - `dipole_direction_weight -> dipole_direction_valid`

This prevents invalid directional targets from contributing to loss.

## 6) Sampling path (`sample_lego.py`)

Initial state:
- All fields start from conditioned example.
- For corrupted fields only, masked nodes are replaced with noise (same noise family as training).

At each step:
1. Build graph input with current state and current `tau`.
2. Predict velocities with the model.
3. Optionally apply `directional_velocity_couplings` on predicted outputs (same direction+magnitude composition idea as training).
4. Integrate (Euler or Heun):
   - `x_next = x + (tau_next - tau) * v_pred`
5. Merge updates only on masked nodes (`ligand_mask` usually).
6. If `project_normalized_states: true`, renormalize:
   - shape equiv blocks to unit norm
   - dipole direction to unit norm

At decode, normalization is applied again before exporting brick descriptors.

## 7) Dipole worked example

Assume both dipole fields are corrupted:
- `field: dipole_strength -> out_field: dipole_strength_velocity`
- `field: dipole_direction -> out_field: dipole_direction_velocity`

Without directional coupling:
- target strength velocity: `n_s - s`
- target direction velocity: `n_d - d`

With directional coupling configured on dipole:
- target direction block is normalized to unit direction.
- target strength scalar is replaced by that direction-block norm.

During sampling:
- if `project_normalized_states: true`, `dipole_direction` is projected to unit norm after each step.

## 8) What parameters actually control

- `corrupt_fields[].mask_field`
  - which nodes are diffused/supervised for that field.
- `corrupt_fields[].center`
  - whether that field is mean-centered before noising.
- `corrupt_fields[].center_noise`
  - whether sampled noise for that field is also centered.
- `directional_velocity_couplings`
  - activates direction/magnitude decomposition for both targets (training) and predictions (sampling).
- `normalize_equivariant_output` (per readout head)
  - forces predicted equivariant block norm to 1 (except near-zero blocks).
- `project_normalized_states` (sampling only)
  - projects latent state back to unit-direction manifold after each update.

## 9) Configuration guidance to avoid common mismatches

### A) Pure continuous velocity learning (simpler baseline)
- `normalize_equivariant_output: false`
- `directional_velocity_couplings: []`
- `project_normalized_states: false` during rollout
- Keep final decode normalization only.

Use this when you want model to learn unconstrained continuous velocities.

### B) Explicit direction+magnitude factorization
- `normalize_equivariant_output: true` on equivariant heads
- enable `directional_velocity_couplings`
- use scalar heads/losses as magnitude channels
- be careful with `project_normalized_states`: it changes rollout dynamics vs raw linear FM state updates.

If using this mode, keep train/inference conventions consistent and verify endpoint reconstruction, not only velocity loss.

## 10) Minimal debug checklist

1. Confirm `corrupt_field_map` inferred from checkpoint equals what you intended to diffuse.
2. Print inferred `directional_velocity_couplings` at sampling.
3. Check if `project_normalized_states` is on; test both on/off.
4. Verify loss masks use `ligand_mask` (or intended mask) and directional validity.
5. Compare trajectory MSE from noise to final state, not just train loss.

