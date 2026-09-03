# GEqDiff

GEqDiff is an equivariant flow-matching sandbox built on top of
[GEqTrain](https://github.com/limresgrp/GEqTrain). The current in-repository
tutorial is the LEGO scaffold task: generate deterministic LEGO-like structures,
mask a designable subset of bricks, train an equivariant model to denoise the
masked positions, shapes, and dipoles, then sample/evaluate generated assemblies.

The active LEGO pipeline uses direct tensor prediction:

- `pos -> velocity`
- `shape_features -> shape_features_velocity`
- `dipole_direction -> dipole_direction_velocity`

The old shell/solid and norm-direction-coupled LEGO benchmark paths have been
removed from the active configs.

## Setup

Create the virtual environment and install all dependencies with:

```bash
./venv_setup.sh
```

The setup script creates `.venv-geqdiff`, clones GEqTrain into `deps/GEqTrain`,
installs PyTorch, installs GEqTrain in editable mode, installs GEqDiff in
editable mode, and installs the runtime packages used by the dataset, sampling,
evaluation, and visualization tools.

For a specific PyTorch backend, pass `--torch-backend`:

```bash
./venv_setup.sh --torch-backend cu128
```

Activate the environment before running commands manually:

```bash
source .venv-geqdiff/bin/activate
```

## Repository Layout

- `lego/lattice.py`: deterministic LEGO source dataset generator.
- `geqdiff/scripts/build_diffusion_dataset.py`: converts source datasets into masked flow-matching datasets.
- `config/lego/experiment/flow_mixed.yaml`: mixed alpha-helix/beta-sheet LEGO training config.
- `config/lego/experiment/flow_alpha.yaml`: alpha-helix-only training config.
- `config/lego/experiment/flow_beta.yaml`: beta-sheet-only training config.
- `config/lego/model/flow_interaction.yaml`: current InteractionModule direct-tensor model.
- `geqdiff/scripts/sample_lego.py`: LEGO sampling script.
- `geqdiff/scripts/evaluate_lego_samples.py`: sampled-structure metrics.
- `lego/lego_visualizer.py`: Plotly HTML visualization.
- `lego_menu.sh`: interactive wrapper for the full LEGO workflow.

## Tutorials

- [Alanine dipeptide flow matching](tutorials/alanine_dipeptide.md): inspect the
  training ensemble, train and sample the model, and compare periodic
  Ramachandran density estimates.
- [LEGO flow matching](tutorials/lego.md): generate and inspect datasets, train,
  sample, evaluate, and visualize LEGO structures.

The LEGO workflow is also available through the interactive `./lego_menu.sh`
wrapper.

## Current LEGO Configs

- `flow_mixed.yaml`: mixed alpha-helix-like and beta-sheet-like scaffold task.
- `flow_alpha.yaml`: alpha-helix-only task.
- `flow_beta.yaml`: beta-sheet-only task.
- `flow.yaml`: full position + shape + dipole direct-vector training losses.
- `flow_posonly.yaml`, `flow_shapeonly.yaml`, `flow_dipoleonly.yaml`: diagnostic single-field tasks.

The current model consumes `sequence_position` and `branch_kind` as categorical
node attributes. These fields must be present in datasets and are preserved by
the dataset builder and sampler.
