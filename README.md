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
export PYTHONPATH="$PWD/deps/GEqTrain:${PYTHONPATH:-}"
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

## LEGO Tutorial

The fastest way to run the workflow is:

```bash
./lego_menu.sh
```

The menu exposes dataset generation, diffusion dataset building, training,
sampling, evaluation, and visualization. The manual commands below reproduce the
same workflow.

### 1. Generate A Source Dataset

```bash
python lego/lattice.py \
  --samples 100 \
  --seed 13 \
  --path lego/lego_dataset_scaffold.npz \
  --scaffold-family mixed \
  --min-nodes 18 \
  --max-nodes 40
```

Valid scaffold families are `mixed`, `alpha_helix`, and `beta_sheet`. The source
dataset contains brick anchors, brick types, rotations, direct `shape_features`,
direct `dipole_direction`, `sequence_position`, and `branch_kind`.

### 2. Build A Diffusion Dataset

```bash
python geqdiff/scripts/build_diffusion_dataset.py \
  --input lego/lego_dataset_scaffold.npz \
  --output lego/lego_diffusion_dataset_scaffold.npz \
  --splits-per-frame 4 \
  --split-strategy connected \
  --ligand-size-min 2 \
  --ligand-size-max 8 \
  --seed 17
```

In the LEGO code, `ligand_mask` means the designable/masked bricks and
`pocket_mask` means the fixed context bricks.

### 3. Inspect The Dataset

```bash
python geqdiff/scripts/inspect_diffusion_dataset.py \
  --input lego/lego_diffusion_dataset_scaffold.npz \
  --plot-html lego/lego_diffusion_dataset_scaffold.html \
  --no-open-html
```

The inspector prints a dataset summary and writes an HTML view of selected or all
examples.

### 4. Train

The checked-in LEGO experiment configs default to scratch datasets. For a local
tutorial run, keep the config unchanged and override the dataset path and output
directory:

```bash
PYTHONPATH="$PWD/deps/GEqTrain:${PYTHONPATH:-}" \
.venv-geqdiff/bin/python deps/GEqTrain/geqtrain/scripts/train.py \
  config/lego/experiment/flow_mixed.yaml \
  -d cuda:0 \
  -o train_dataset_list.0.dataset_input="$PWD/lego/lego_diffusion_dataset_scaffold.npz" \
  -o root="$PWD/results/lego" \
  -o run_name=lego_tutorial_mixed \
  -o wandb=false \
  -o max_epochs=100
```

Use `-d cpu` if CUDA is unavailable. For real runs, increase `--samples` during
dataset generation and remove or raise the `max_epochs` override.

### 5. Sample

After training, sample masked LEGO structures from the checkpoint:

```bash
PYTHONPATH="$PWD/deps/GEqTrain:${PYTHONPATH:-}" \
python geqdiff/scripts/sample_lego.py \
  --model results/lego/lego_tutorial_mixed/best_model.pth \
  --input lego/lego_diffusion_dataset_scaffold.npz \
  --output lego/lego_sampled_scaffold_dataset.npz \
  --source-canonical lego/lego_dataset_scaffold.npz \
  --num-samples 8 \
  --steps 100 \
  --sampler heun \
  --device cuda:0 \
  --seed 0 \
  --no-clash-guidance \
  --no-clash-guidance-auto-scale \
  --save-intermediates \
  --save-metrics \
  --metrics-json lego/lego_sampled_scaffold_metrics.json
```

If your run directory differs, locate the checkpoint with:

```bash
find results/lego -name best_model.pth -print
```

Useful sampling options:

- `--skip-diffusion-fields pos`, `shape`, or `dipole` keeps selected predicted fields fixed at ground truth.
- `--start-step N` starts from a partially noised state instead of full noise.
- `--shape-decode-mode input_knn` decodes continuous shape coefficients against input dataset exemplars.

### 6. Evaluate

Sampling with `--save-metrics` already writes metrics. To evaluate an existing
sampled NPZ:

```bash
python geqdiff/scripts/evaluate_lego_samples.py \
  --input lego/lego_sampled_scaffold_dataset.npz \
  --output-json lego/lego_sampled_scaffold_metrics.json
```

The primary metrics are validity, shape fidelity, dipole fidelity, and pose
fidelity. Voxelized and raw variants are reported when available.

### 7. Visualize

```bash
python lego/lego_visualizer.py \
  --path lego/lego_sampled_scaffold_dataset.npz \
  --output-html lego/lego_sampled_scaffold_dataset.html \
  --show-dipoles \
  --trajectory-stride 5
```

The HTML visualizer shows original vs sampled structures, optional trajectory
frames, voxelized final structures when stored, dipole coloring, brick hover
metadata, and metric panels.

## Current LEGO Configs

- `flow_mixed.yaml`: mixed alpha-helix-like and beta-sheet-like scaffold task.
- `flow_alpha.yaml`: alpha-helix-only task.
- `flow_beta.yaml`: beta-sheet-only task.
- `flow.yaml`: full position + shape + dipole direct-vector training losses.
- `flow_posonly.yaml`, `flow_shapeonly.yaml`, `flow_dipoleonly.yaml`: diagnostic single-field tasks.

The current model consumes `sequence_position` and `branch_kind` as categorical
node attributes. These fields must be present in datasets and are preserved by
the dataset builder and sampler.

## Notes

- The active LEGO generator is deterministic by design, to reduce hidden
  ambiguity in the training target.
- The old shell/solid benchmark and decoupled scalar/equivariant shape/dipole
  configs are no longer part of the active workflow.
- Large generated datasets and training results should stay outside git, for
  example under `/scratch/...` or `results/`.
