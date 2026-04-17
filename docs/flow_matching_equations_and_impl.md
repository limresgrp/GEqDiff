# Flow Matching Equations and Current Implementation

This note summarizes:
- the core Flow Matching (FM) equations,
- how this repository implements them,
- what changes (or not) for equivariant tensors,
- and whether score-to-velocity conversion is currently used.

## 1) Core FM equations (continuous state)

For one field `x` and time `tau`:

- Path:
  - `x_tau = alpha(tau) * x_data + sigma(tau) * x_noise`
- Target velocity:
  - `v*(x_tau, tau) = d/dtau x_tau`
  - `= alpha'(tau) * x_data + sigma'(tau) * x_noise`

With the linear scheduler used here:
- `alpha(tau) = 1 - tau`
- `sigma(tau) = tau`
- `alpha' = -1`, `sigma' = +1`

So:
- `x_tau = (1 - tau) x_data + tau x_noise`
- `v* = x_noise - x_data`

This is exactly what `ForwardFlowMatchingModule` builds.

## 2) What the code does

### Training (forward corruption + target)

In [`geqdiff/nn/flow_matching.py`](/home/angiod@usi.ch/GEqDiff/geqdiff/nn/flow_matching.py):

- Samples `tau`.
- Computes scheduler coefficients (`data_scale`, `noise_scale`) and derivatives.
- Builds:
  - `x_t = data_scale * x + noise_scale * noise`
  - `target = ddata_scale_dtau * x + dnoise_scale_dtau * noise`
- With linear scheduler this is:
  - `x_t = (1-tau)x + tau*noise`
  - `target = noise - x`

Masking:
- Corruption/supervision is applied only where `mask_field` is true (e.g. `ligand_mask`).
- Non-masked nodes keep clean values (or reduced noise if `unmasked_noise_scale > 0`) and get zero target.

### Sampling (reverse integration)

In [`geqdiff/scripts/sample_lego.py`](/home/angiod@usi.ch/GEqDiff/geqdiff/scripts/sample_lego.py):

- Starts from noise on masked nodes for corrupted fields.
- Integrates from high `tau` to low `tau`:
  - Euler: `x_next = x + (tau_next - tau) * v_pred`
  - Heun: trapezoidal correction on velocity.

Because `tau_next < tau`, `(tau_next - tau)` is negative, so this is reverse-time integration (noise -> data).

## 3) Scalars vs equivariant tensors

FM equations are identical for all fields. Differences are in:

- Noise sampling:
  - Scalars: Gaussian noise.
  - Equivariant direction-like fields:
    - `shape_equiv_features`: unit-norm random blocks (`l=1,2,3` blocks).
    - `dipole_direction`: unit random vectors.
- Optional projection at sampling:
  - `project_normalized_states` re-normalizes direction-like states each step.
- Optional direction/magnitude coupling:
  - `directional_velocity_couplings` rewrites targets/predictions to encode direction in equivariant block and magnitude in scalar channels.

So mathematically the FM objective is the same; only the state manifold and post-processing differ.

## 4) Score -> velocity conversion: do we do it?

Short answer: **no**, in current LEGO FM pipeline.

Current FM module explicitly enforces:
- `flow_target_parameterization == "scheduler_velocity"`
- `flow_time_parameterization == "tau"`

So the model is trained to predict velocity directly, not score.

## 5) If we wanted score parameterization

For Gaussian paths, score and velocity are related by the standard affine conversion:

- `u_t(x) = a_t * x + b_t * score_t(x)`

with coefficients determined by `alpha(t), sigma(t)` and their derivatives (as in the FM guide table/equations).

That conversion is **not** used in current training/sampling code. If a model head outputs score, conversion must be added explicitly before the ODE step.

## 6) Important convention note

The repository uses:
- `x_tau = (1 - tau) * data + tau * noise`

So `tau=0` is data, `tau=1` is noise, and sampling integrates `tau: 1 -> 0`.

This is valid and consistent, but opposite to some FM presentations where `t=0` is noise and `t=1` is data.

## 7) Practical correctness check (for your question)

Given the current code and configs:
- FM forward equations are implemented correctly.
- Reverse integration for velocity is implemented correctly.
- There is no missing score->velocity transform, because this pipeline does not use score parameterization.

