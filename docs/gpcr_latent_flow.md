# GPCR latent flow experiment

This experiment learns an invariant, per-residue latent representation from a
single GPCR trajectory with joint flow matching on C-alpha positions and
side-chain directions. The current implementation covers dataset construction,
dataset loading, and the encoder-bottleneck-decoder training configuration.

## 1. Build the dataset

The input trajectory must already be whole and continuous across periodic
boundaries. The builder deliberately does not apply minimum-image or unwrapping
logic.

```bash
python dataset_build_cg.py \
  --topology /storage_common/angiod/MDSimulations/Chiara/beta2.APO.inactive.met/run_protein.gro \
  --trajectory /storage_common/angiod/MDSimulations/Chiara/beta2.APO.inactive.met/run_skip10_protein.xtc \
  --output-dir /storage_common/angiod/CGDiff \
  --cutoff 12.0 \
  --val-fraction 0.1
```

The output is a contiguous temporal split:

- `train.npz`: the first 90% of selected frames;
- `val.npz`: the final 10% of selected frames.

`node_types` and `sequence_position` are fixed per-residue fields. Positions,
side-chain directions, padded edges, edge features, MD time, and source frame
indices vary by frame. Padded edge indices are `-1`; GEqTrain removes them when
constructing an individual graph.

For a quick integrity check:

```bash
python - <<'PY'
import numpy as np

for path in (
    "/storage_common/angiod/CGDiff/train.npz",
    "/storage_common/angiod/CGDiff/val.npz",
):
    with np.load(path) as data:
        print(path)
        for key in data.files:
            print(f"  {key:28s} {data[key].shape} {data[key].dtype}")
PY
```

## 2. Configure the data paths

The checked-in data config uses the portable defaults `data/gpcr1/train.npz`
and `data/gpcr1/val.npz`. Either link/copy the generated files there or override
both paths at launch:

```bash
geqtrain-train config/gpcr/experiment/latent_flow.yaml \
  -d cuda:0 \
  -o train_dataset_list.0.dataset_input=/storage_common/angiod/CGDiff/train.npz \
  -o validation_dataset_list.0.dataset_input=/storage_common/angiod/CGDiff/val.npz
```

The experiment is split into four composable files:

- `config/gpcr/data/cg_flow.yaml`: NPZ mappings and field registration;
- `config/gpcr/model/latent_flow.yaml`: joint flow model and 128-scalar
  per-residue bottleneck;
- `config/gpcr/train/flow.yaml`: losses, optimizer, and batching;
- `config/gpcr/experiment/latent_flow.yaml`: composed experiment defaults.

The edge embedding stage is required: it turns the two
`seq_edge_features` values into `edge_attrs` consumed by both interaction
blocks. The fixed-field declarations are also required so the one-copy
`node_types` and `sequence_position` arrays are expanded correctly for every
trajectory frame.

## 3. Smoke test before a full run

Start with a short dataset slice and conservative memory settings:

```bash
python dataset_build_cg.py \
  --topology /path/to/run_protein.gro \
  --trajectory /path/to/run_protein.xtc \
  --output-dir data/gpcr_smoke \
  --stop 100 \
  --val-fraction 0.1

geqtrain-train config/gpcr/experiment/latent_flow.yaml \
  -d cuda:0 \
  -o train_dataset_list.0.dataset_input=data/gpcr_smoke/train.npz \
  -o validation_dataset_list.0.dataset_input=data/gpcr_smoke/val.npz \
  -o max_epochs=5 \
  -o batch_size=2 \
  -o validation_batch_size=2 \
  -o wandb=false \
  -o run_name=gpcr_latent_flow_smoke
```

Confirm that both velocity losses are finite and decrease, and that a forward
pass exposes `latent_node` with shape `[batch_nodes, 128]`.

## 4. Work still needed for the scientific pipeline

Training is only the representation-learning stage. The following pieces are
not yet implemented and should be added as separate, testable scripts.

1. **Fixed-flow-time latent extraction.** Load a checkpoint, construct the
   same graphs for every original frame, evaluate the encoder at explicitly
   chosen `tau` values, and save `latent`, `time`, and `frame_index`. The code
   must follow the repository convention that `tau -> 1` is the data end of
   the flow; it must not silently use the module's random training-time `tau`.
2. **Offline diffusion maps.** Fit the kernel only after extracting the full
   trajectory. Store kernel bandwidth, normalization convention,
   eigenvalues/eigenvectors, and frame indices so analyses are reproducible.
   For 5,001 frames a dense kernel is feasible but already substantial; larger
   ensembles should use k-nearest-neighbor or Nyström approximations.
3. **Matched baselines.** Compare the latent against aligned raw C-alpha
   coordinates and rotation/translation-invariant internal distances. Raw
   Cartesian coordinates must be aligned before distance calculations.
4. **Stability analysis.** Repeat diffusion maps over temporal blocks and
   bootstrap subsets. Compare eigenfunctions up to sign and account for
   rotations within nearly degenerate eigenspaces.
5. **Trajectory provenance and PBC checks.** Record how the trajectory was
   made whole, atom selections, topology hashes, frame stride, physical time
   units, and replicate identity. A visually plausible wrapped trajectory can
   still create incorrect contact graphs and side-chain vectors.
6. **Split strategy for multiple simulations.** Keep the current temporal
   split for one continuous trajectory. Once independent replicas are used,
   split by replica rather than mixing adjacent correlated frames across
   training and validation.
7. **Transfer beyond one receptor.** Absolute `sequence_position` and flattened
   per-residue latents assume a fixed sequence and length. Multi-GPCR training
   needs an alignment-aware index (for example generic GPCR numbering) and an
   invariant sequence-aware pooling/readout.

Useful engineering additions are unit tests for topology edge cases (missing or
duplicate C-alpha atoms, residue-number gaps, multiple chains), a dataset
inspection command, checkpoint-to-latent extraction tests, and GPU memory
profiling at the full 314-residue graph size.

## 5. Reproducibility checklist

- Save the exact topology and trajectory paths plus hashes.
- Record `--start`, `--stop`, `--stride`, cutoff, and validation fraction.
- Verify all nonzero side-chain vectors have unit norm.
- Verify every non-padded edge is in range and has its reverse edge.
- Record the GEqDiff and GEqTrain revisions used for training.
- Save the resolved Hydra config with each checkpoint.
- Preserve `frame_index` through latent extraction and diffusion-map output.
