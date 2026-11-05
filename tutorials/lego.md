# LEGO Flow-Matching Tutorial

This tutorial generates deterministic LEGO-like structures, masks a designable
subset of bricks, trains an equivariant model to denoise their positions,
shapes, and dipoles, and evaluates the sampled assemblies.

Run all commands from the GEqDiff repository root.

## 1. Set Up The Environment

```bash
./venv_setup.sh
source .venv-geqdiff/bin/activate
```

The setup installs GEqTrain from `deps/GEqTrain`, GEqDiff, PyTorch, and the
runtime dependencies. The interactive workflow is available through:

```bash
./lego_menu.sh
```

The following sections show the corresponding commands explicitly.

## 2. Generate A Source Dataset

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
dataset contains brick anchors, brick types, rotations, direct shape and dipole
tensors, sequence positions, and branch kinds.

## 3. Build A Diffusion Dataset

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

In the current implementation, `ligand_mask` denotes designable or masked
bricks, while `pocket_mask` denotes fixed context bricks.

Inspect the resulting dataset and write an interactive HTML visualization:

```bash
python geqdiff/scripts/inspect_diffusion_dataset.py \
  --input lego/lego_diffusion_dataset_scaffold.npz \
  --plot-html lego/lego_diffusion_dataset_scaffold.html \
  --no-open-html
```

## 4. Train

The model predicts direct tensors:

```text
pos              -> velocity
shape_features   -> shape_features_velocity
dipole_direction -> dipole_direction_velocity
```

It also consumes `sequence_position` and `branch_kind` as categorical node
attributes. These fields must remain present throughout dataset construction and
sampling.

Run a local mixed-scaffold tutorial training:

```bash
geqtrain-train config/lego/experiment/flow_mixed.yaml \
  -d cuda:0 \
  -o train_dataset_list.0.dataset_input="$PWD/lego/lego_diffusion_dataset_scaffold.npz" \
  -o root="$PWD/results/lego" \
  -o run_name=lego_tutorial_mixed \
  -o wandb=false \
  -o max_epochs=100
```

Use `-d cpu` if CUDA is unavailable. A production run should use a larger source
dataset and remove or increase the `max_epochs` override.

Monitor the run with:

```bash
tail -f results/lego/lego_tutorial_mixed/log
```

## 5. Sample

```bash
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

Useful diagnostic controls include:

- `--skip-diffusion-fields pos`, `shape`, or `dipole` keeps selected predicted
  fields fixed at their ground-truth values.
- `--start-step N` starts from a partially noised state instead of pure noise.
- `--shape-decode-mode input_knn` decodes shape coefficients against input
  dataset exemplars.

## 6. Evaluate

The sampling command above writes metrics directly. Evaluate an existing sample
file independently with:

```bash
python geqdiff/scripts/evaluate_lego_samples.py \
  --input lego/lego_sampled_scaffold_dataset.npz \
  --output-json lego/lego_sampled_scaffold_metrics.json
```

The primary metrics cover global validity, shape fidelity, dipole fidelity, and
pose fidelity. Raw and voxelized variants are reported when available. See
`docs/lego_sampling_metrics.md` for definitions and interpretation.

## 7. Visualize

```bash
python lego/lego_visualizer.py \
  --path lego/lego_sampled_scaffold_dataset.npz \
  --output-html lego/lego_sampled_scaffold_dataset.html \
  --show-dipoles \
  --trajectory-stride 5
```

The HTML compares original and sampled structures, exposes saved trajectory
frames, shows the voxelized final result when available, and displays per-sample
metrics and brick metadata.

To generate a separate visual primer explaining how the 16 spherical-harmonic
coefficients encode the available LEGO shapes, run:

```bash
python lego/plot_spherical_harmonics.py
```

Open `tutorials/spherical_harmonics/index.html` to navigate the basis-function,
weighted-combination, and smooth-surface-to-brick pages. Pass
`--plotlyjs inline` to make every page work without an internet connection.

## 8. Available Experiments

- `config/lego/experiment/flow_mixed.yaml`: mixed alpha-helix-like and
  beta-sheet-like structures.
- `config/lego/experiment/flow_alpha.yaml`: alpha-helix-only structures.
- `config/lego/experiment/flow_beta.yaml`: beta-sheet-only structures.
- Position-only, shape-only, and dipole-only configurations provide diagnostic
  single-field training tasks.
