# Equivariant Flow-Matching Latent Space for Protein Slow Dynamics

## 1. Project overview

### Goal

Build a structural latent representation of protein conformations using an E(3)-equivariant flow-matching model, then use the learned latent representation to identify slow conformational motions with an **offline diffusion-map analysis**.

The first prototype should be deliberately incremental:

```text
PDB + XTC/TRR
    |
    v
C-alpha + side-chain direction + residue type + sequence position
    |
    v
GEqDiff flow-matching model
    |
    +--> Encoder EGNN --> latent_node
    |                         |
    |                         v
    |                  latent-conditioned
    |                    Decoder EGNN
    |                         |
    |                         v
    +--------------> velocity of coordinates
                              +
                       velocity of side-chain vectors
```

After training:

```text
whole MD trajectory
      |
      v
encoder
      |
      v
latent_node(X_t) for every frame
      |
      v
global latent representation Z(X_t)
      |
      v
offline diffusion map
      |
      v
slow coordinates / slow eigenfunctions
```

The project should **not** initially learn a physical generator, TICA/VAMP objective, or kinetic loss. The first scientific question is:

> Does a generatively learned, equivariant structural latent space provide a better representation for discovering slow protein motions than raw C-alpha coordinates?

The flow-matching model provides a strong structural-learning objective without requiring a physical lag time `tau`. The subsequent diffusion-map calculation is performed on the complete dataset, after the neural model has been trained.

---

## 2. Why flow matching is used

The model is trained to represent the distribution of valid protein conformations.

For each clean configuration

```text
x = {positions, side-chain directions}
```

GEqDiff samples a flow time `tau`, constructs an interpolated/noisy state `x_tau`, and learns its velocity field.

The current GEqDiff implementation already provides:

- random flow-time sampling,
- configurable corruption of multiple fields,
- direct velocity prediction,
- joint handling of Cartesian positions and equivariant vector fields,
- time embedding,
- centering of coordinates,
- unit-normalized noise for directional vector fields.

The existing GEqDiff Alanine tutorial trains coordinate velocity with the same flow-matching infrastructure. The current LEGO configuration additionally demonstrates joint transport of positions and non-scalar node fields.

The new project adds a learned latent bottleneck and makes that latent explicitly condition the velocity-prediction network.

---

# 3. Representation of each protein configuration

Each residue is one graph node.

For residue `i`:

### 3.1 Position

```text
pos[i] = C-alpha Cartesian position
```

Shape:

```text
[N_residues, 3]
```

Transformation:

```text
1x1o
```

because Cartesian positions are polar vectors.

### 3.2 Side-chain direction

For residue `i`, define:

```text
sidechain_direction[i]
    = normalized(
          COM(heavy side-chain atoms)
          - C-alpha position
      )
```

Shape:

```text
[N_residues, 3]
```

Also:

```text
1x1o
```

For glycine, which has no side-chain heavy atom beyond the backbone, use:

```text
[0, 0, 0]
```

A future implementation may choose a different glycine convention, but the first POC should keep the representation simple.

### 3.3 Residue type

Encode the 20 canonical amino acids plus `X` for unknown residues.

Example:

```text
X   = 0
ALA = 1
ARG = 2
...
VAL = 20
```

This is a categorical scalar node field.

### 3.4 Sequence position

Each residue receives its fixed index:

```text
0, 1, 2, ..., N_residues - 1
```

This is extremely important for the first flow-matching experiment.

A protein is not an unordered cloud of residues. The network should know that node 17 is residue 17 and node 18 is residue 18.

Use a learned embedding or categorical/positional embedding of sequence position.

For a first fixed-length single-protein experiment:

```text
num_types = 512
embedding_dimensionality = 32
```

is sufficient for proteins shorter than 512 residues.

For multiple proteins later, this representation must be reconsidered carefully, because absolute residue index is not transferable between proteins.

### 3.5 Flow-time conditioning

GEqDiff internally produces the flow-time embedding and makes it available as the scalar node-level `conditioning` field.

The model therefore has two distinct notions of conditioning:

1. physical/protein identity information:
   - residue type
   - sequence position
2. artificial generative-flow information:
   - flow time `tau`

---

# 4. Graph construction

The graph is constructed from C-alpha distances.

Default cutoff:

```text
12 Å
```

Every pair of residues whose C-alpha distance is <= 12 Å becomes a directed edge.

The two directions are stored:

```text
i -> j
j -> i
```

The edge index is therefore:

```text
edge_index.shape = [2, N_edges]
```

### 4.1 Sequence edge feature

Each edge gets a scalar feature:

```text
seq_edge_features = [is_previous, is_next]
```

Meaning:

```text
[1, 0]   target is previous residue in sequence
[0, 1]   target is next residue in sequence
[0, 0]   spatial contact but not a sequential neighbor
```

This gives the network explicit sequence-connectivity information while preserving the general spatial graph.

The geometric edge direction and distance are still automatically generated from `pos` by GEqTrain.

---

# 5. CG dataset creation script

The implemented script is `dataset_build_cg.py`. See
`docs/gpcr_latent_flow.md` for runnable commands, configuration overrides, and
the remaining implementation roadmap.

It uses MDAnalysis and accepts topology/trajectory formats supported by the
installed MDAnalysis readers, including the tested GRO + XTC combination:

```text
topology = PDB or GRO
trajectory = XTC or TRR
```

Example:

```bash
python dataset_build_cg.py \
    --topology receptor.pdb \
    --trajectory receptor.xtc \
    --output-dir data/gpcr1 \
    --cutoff 12.0 \
    --stride 10 \
    --val-fraction 0.1
```

The script should produce:

```text
data/gpcr1/
    train.npz
    val.npz
```

## 5.1 Dataset fields

### Per-frame fields

```text
pos
    shape: [N_frames, N_residues, 3]
    dtype: float32

sidechain_direction
    shape: [N_frames, N_residues, 3]
    dtype: float32

edge_index
    shape: [N_frames, 2, N_edges_max]
    dtype: int64

seq_edge_features
    shape: [N_frames, N_edges_max, 2]
    dtype: float32

time
    shape: [N_frames]
    dtype: float64

frame_index
    shape: [N_frames]
    dtype: int64
```

### Fixed node fields

```text
node_types
    shape: [N_residues]
    dtype: int64

sequence_position
    shape: [N_residues]
    dtype: int64
```

### Metadata

```text
type_names
cutoff
```

## 5.2 Edge padding

Different trajectory frames can have different numbers of edges because the C-alpha distances change.

The NPZ therefore stores a common:

```text
N_edges_max
```

and pads shorter frames with:

```text
edge_index = -1
```

The corresponding rows in `seq_edge_features` are zero.

GEqTrain's `AtomicData.with_edge_index` explicitly supports this representation and removes padded edges when a frame is instantiated.

## 5.3 Train/validation split

Do not randomly split individual trajectory frames for the first experiment.

Use a contiguous temporal split:

```text
first 90% -> train
last 10%  -> validation
```

This is more conservative because adjacent MD frames are strongly correlated.

If the trajectory contains multiple independent simulations, the preferred future split is by trajectory/replicate rather than by time.

---

# 6. Complete dataset-builder implementation

The checked-in `dataset_build_cg.py` is the authoritative implementation. The
listing below documents the design, but should not be copied independently of
later fixes and validation improvements in that script.

The implementation should follow the script below.

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array


AA_TO_TYPE = {
    "ALA": 1, "ARG": 2, "ASN": 3, "ASP": 4, "CYS": 5,
    "GLN": 6, "GLU": 7, "GLY": 8, "HIS": 9, "ILE": 10,
    "LEU": 11, "LYS": 12, "MET": 13, "PHE": 14, "PRO": 15,
    "SER": 16, "THR": 17, "TRP": 18, "TYR": 19, "VAL": 20,

    "MSE": 13,
    "HID": 9, "HIE": 9, "HIP": 9,
    "HSD": 9, "HSE": 9,
    "CYX": 5,
    "ASH": 4,
    "GLH": 7,
}

TYPE_NAMES = [
    "X",
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

FALLBACK_MASSES = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "SE": 78.971,
}


def is_hydrogen(atom) -> bool:
    try:
        element = str(atom.element).strip().upper()
        if element:
            return element == "H"
    except Exception:
        pass

    return atom.name.strip().upper().startswith("H")


def atom_mass(atom) -> float:
    try:
        mass = float(atom.mass)
        if np.isfinite(mass) and mass > 0:
            return mass
    except Exception:
        pass

    try:
        element = str(atom.element).strip().upper()
    except Exception:
        element = atom.name.strip().upper()

    return FALLBACK_MASSES.get(element, 12.011)


def residue_chain_id(residue) -> str:
    atom = residue.atoms[0]

    for attr in ("chainID", "chainid", "segid"):
        try:
            value = getattr(atom, attr)
            if value is not None and str(value).strip():
                return str(value).strip()
        except Exception:
            pass

    return ""


def sidechain_direction(residue, ca_position) -> np.ndarray:
    atoms = []

    for atom in residue.atoms:
        name = atom.name.strip().upper()

        if name in BACKBONE_ATOMS:
            continue

        if is_hydrogen(atom):
            continue

        atoms.append(atom)

    if not atoms:
        return np.zeros(3, dtype=np.float32)

    coords = np.asarray(
        [atom.position for atom in atoms],
        dtype=np.float64,
    )

    masses = np.asarray(
        [atom_mass(atom) for atom in atoms],
        dtype=np.float64,
    )

    com = np.average(coords, axis=0, weights=masses)

    vector = com - np.asarray(ca_position, dtype=np.float64)
    norm = np.linalg.norm(vector)

    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    return (vector / norm).astype(np.float32)


def build_edges(ca_positions, residues, cutoff):
    distances = distance_array(
        ca_positions,
        ca_positions,
    )

    valid = (
        (distances <= cutoff)
        & (distances > 1e-8)
    )

    src, dst = np.where(valid)

    edge_index = np.stack(
        [src, dst],
        axis=0,
    ).astype(np.int64)

    seq_edge_features = np.zeros(
        (len(src), 2),
        dtype=np.float32,
    )

    chain_ids = np.asarray(
        [residue_chain_id(r) for r in residues],
        dtype=object,
    )

    same_chain = chain_ids[src] == chain_ids[dst]

    previous = (
        same_chain
        & (dst == src - 1)
    )

    next_residue = (
        same_chain
        & (dst == src + 1)
    )

    seq_edge_features[previous, 0] = 1.0
    seq_edge_features[next_residue, 1] = 1.0

    return edge_index, seq_edge_features


def save_dataset(
    output_path,
    positions,
    sidechain_vectors,
    edge_index,
    seq_edge_features,
    node_types,
    sequence_position,
    times,
    frame_indices,
    cutoff,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        pos=positions.astype(np.float32),
        sidechain_direction=sidechain_vectors.astype(np.float32),
        node_types=node_types.astype(np.int64),
        sequence_position=sequence_position.astype(np.int64),
        edge_index=edge_index.astype(np.int64),
        seq_edge_features=seq_edge_features.astype(np.float32),
        time=times.astype(np.float64),
        frame_index=frame_indices.astype(np.int64),
        type_names=np.asarray(TYPE_NAMES),
        cutoff=np.asarray(cutoff, dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)

    parser.add_argument("--cutoff", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)

    parser.add_argument("--val-fraction", type=float, default=0.1)

    args = parser.parse_args()

    universe = mda.Universe(
        args.topology,
        args.trajectory,
    )

    protein = universe.select_atoms("protein")
    residues = list(protein.residues)

    if len(residues) == 0:
        raise RuntimeError("No protein residues found.")

    ca_indices = []

    for residue in residues:
        ca = residue.atoms.select_atoms("name CA")

        if len(ca) != 1:
            raise RuntimeError(
                f"{residue.resname} {residue.resid}: "
                f"expected exactly one CA, found {len(ca)}"
            )

        ca_indices.append(ca[0].index)

    # Keep the vectorized, frame-aware MDAnalysis AtomGroup interface.
    ca_atoms = universe.atoms[
        np.asarray(ca_indices, dtype=np.int64)
    ]

    n_nodes = len(residues)

    node_types = np.asarray(
        [
            AA_TO_TYPE.get(
                residue.resname.upper(),
                0,
            )
            for residue in residues
        ],
        dtype=np.int64,
    )

    sequence_position = np.arange(
        n_nodes,
        dtype=np.int64,
    )

    n_total = len(universe.trajectory)

    stop = (
        n_total
        if args.stop is None
        else min(args.stop, n_total)
    )

    frame_indices = np.arange(
        args.start,
        stop,
        args.stride,
        dtype=np.int64,
    )

    if len(frame_indices) < 2:
        raise RuntimeError("Too few frames selected.")

    n_frames = len(frame_indices)

    print(f"Residues: {n_nodes}")
    print(f"Frames:   {n_frames}")
    print(f"Cutoff:   {args.cutoff:.2f} Å")

    # First pass: determine maximum number of edges.
    max_edges = 0

    for count, frame_id in enumerate(frame_indices):
        universe.trajectory[int(frame_id)]

        ca_positions = np.asarray(
            ca_atoms.positions,
            dtype=np.float32,
        )

        edge_index, _ = build_edges(
            ca_positions,
            residues,
            args.cutoff,
        )

        max_edges = max(
            max_edges,
            edge_index.shape[1],
        )

        if (
            (count + 1) % 100 == 0
            or count == n_frames - 1
        ):
            print(
                f"Graph pass: {count + 1}/{n_frames}; "
                f"max edges = {max_edges}"
            )

    positions = np.zeros(
        (n_frames, n_nodes, 3),
        dtype=np.float32,
    )

    sidechain_vectors = np.zeros(
        (n_frames, n_nodes, 3),
        dtype=np.float32,
    )

    edge_index_all = np.full(
        (n_frames, 2, max_edges),
        -1,
        dtype=np.int64,
    )

    seq_edge_features_all = np.zeros(
        (n_frames, max_edges, 2),
        dtype=np.float32,
    )

    times = np.zeros(
        n_frames,
        dtype=np.float64,
    )

    for count, frame_id in enumerate(frame_indices):
        universe.trajectory[int(frame_id)]

        ca_positions = np.asarray(
            ca_atoms.positions,
            dtype=np.float32,
        ).copy()

        positions[count] = ca_positions

        times[count] = float(
            universe.trajectory.ts.time
        )

        for node_index, residue in enumerate(residues):
            sidechain_vectors[
                count,
                node_index,
            ] = sidechain_direction(
                residue,
                ca_positions[node_index],
            )

        ei, ef = build_edges(
            ca_positions,
            residues,
            args.cutoff,
        )
        num_edges = ei.shape[1]

        edge_index_all[
            count,
            :,
            :num_edges,
        ] = ei

        seq_edge_features_all[
            count,
            :num_edges,
        ] = ef

        if (
            (count + 1) % 100 == 0
            or count == n_frames - 1
        ):
            print(
                f"Extraction pass: {count + 1}/{n_frames}"
            )

    # Temporal validation split.
    n_val = min(
        n_frames - 1,
        max(
            1,
            int(round(args.val_fraction * n_frames)),
        ),
    )

    n_train = n_frames - n_val

    train = slice(0, n_train)
    val = slice(n_train, n_frames)

    save_dataset(
        args.output_dir / "train.npz",
        positions[train],
        sidechain_vectors[train],
        edge_index_all[train],
        seq_edge_features_all[train],
        node_types,
        sequence_position,
        times[train],
        frame_indices[train],
        args.cutoff,
    )

    save_dataset(
        args.output_dir / "val.npz",
        positions[val],
        sidechain_vectors[val],
        edge_index_all[val],
        seq_edge_features_all[val],
        node_types,
        sequence_position,
        times[val],
        frame_indices[val],
        args.cutoff,
    )

    print()
    print("Saved:")
    print(" ", args.output_dir / "train.npz")
    print(" ", args.output_dir / "val.npz")


if __name__ == "__main__":
    main()
```

---

# 7. Important dataset caveat: periodic boundary conditions

For the first prototype, use a protein trajectory that has been made whole/continuous before this script.

Do not silently mix periodic-wrapping behavior into the first experiment.

A future PBC-aware implementation should store `edge_cell_shift` and possibly use `cell`, but this is unnecessary for the first POC and would complicate both dataset validation and flow matching.

---

# 8. GEqDiff data configuration

The GEqTrain/GEqDiff dataset configuration should map the custom NPZ fields into the expected typed fields.

`config/gpcr/data/cg_flow.yaml`:

```yaml
# @package _global_

dataset_num_workers: 4

train_dataset_list:
  - dataset: npz
    dataset_input: data/gpcr1/train.npz

    key_mapping:
      pos: pos
      node_types: node_types
      sequence_position: sequence_position
      sidechain_direction: sidechain_direction
      edge_index: edge_index
      seq_edge_features: seq_edge_features
      time: time
      frame_index: frame_index

validation_dataset_list:
  - dataset: npz
    dataset_input: data/gpcr1/val.npz

    key_mapping:
      pos: pos
      node_types: node_types
      sequence_position: sequence_position
      sidechain_direction: sidechain_direction
      edge_index: edge_index
      seq_edge_features: seq_edge_features
      time: time
      frame_index: frame_index

node_fields:
  - node_types
  - sequence_position
  - sidechain_direction
  - conditioning
  - velocity
  - sidechain_direction_velocity
  - encoder_node_features
  - latent_node
  - decoder_node_features

edge_fields:
  - seq_edge_features
  - encoder_edge_features
  - decoder_edge_features

graph_fields:
  - time
  - frame_index

fixed_fields:
  - node_types
  - sequence_position
```

The important custom fields are registered because GEqTrain distinguishes node/edge/graph fields explicitly.

---

# 9. Model architecture

The model has **two interaction stages**.

## Stage A: structural encoder

Input:

```text
pos
sidechain_direction
node_types
sequence_position
conditioning(tau)
seq_edge_features
geometric edge information
```

The encoder produces a rich equivariant representation.

Conceptually:

```text
current noisy structure
        +
sequence identity
        +
flow time
        |
        v
   InteractionModule
        |
        v
 edge latent features
        |
        v
   EdgewiseReduce
        |
        v
 node features
        |
        v
 scalar projection
        |
        v
 latent_node
```

Use a relatively large hidden space such as:

```text
256 scalar hidden channels
8 copies of l>0 equivariant channels
```

The actual bottleneck is:

```text
latent_node = 128 x 0e
```

That 128-dimensional scalar latent per residue is the first proposed latent space.

### Why scalar latent?

For later diffusion-map analysis we want a configuration-dependent representation that is invariant under global rotation and translation.

Scalar `0e` node features satisfy this requirement.

The encoder may internally use equivariant `l=1` and `l=2` information, but the exposed latent should be invariant/scalar.

---

# 10. Explicit latent bottleneck

The latent projection should be an actual bottleneck, not just an alias to the encoder features.

Example:

```yaml
- _target_: geqtrain.nn.ReadoutModule
  name: latent_projection
  field: encoder_node_features
  out_field: latent_node
  out_irreps: 128x0e
  # The mixed encoder representation contains l=0/l=1/l=2 channels. The
  # invariant bottleneck intentionally reads its geometry-informed l=0 part.
  strict_irreps: false
  readout_latent: geqtrain.nn.ScalarMLPFunction
  readout_latent_kwargs:
    mlp_latent_dimensions:
      - 256
    mlp_nonlinearity: silu
```

This is the field to extract after training.

---

# 11. Stage B: latent-conditioned decoder

The second InteractionModule again sees the current noisy structure.

But it additionally receives:

```text
conditioning_fields:
  - latent_node
```

The latent is therefore not concatenated blindly with Cartesian vectors.

Instead, GEqTrain injects scalar latent information into the interaction network.

Conceptually:

```text
current noisy geometry
       +
node/edge features
       +
latent_node
       +
flow-time embedding
       |
       v
 decoder InteractionModule
       |
       v
 node features
       |
       +----------> velocity of C-alpha position
       |
       +----------> velocity of side-chain vector
```

This is the key architectural idea of the project.

---

# 12. Why use the latent as conditioning instead of only using the second layer?

Suppose the full network were:

```text
Input -> Interaction -> Interaction -> velocity
```

The second block could simply continue processing the input features.

By forcing a bottleneck:

```text
Input
  |
  v
Encoder
  |
  v
latent_node
  |
  v
Decoder
  |
  v
velocity
```

we make the latent representation a distinct object.

The decoder therefore has to use the latent representation to describe how the current noisy structure should evolve toward the data distribution.

This creates a useful candidate representation to extract later.

---

# 13. Complete model configuration

`config/gpcr/model/latent_flow.yaml`

```yaml
# @package _global_

defaults:
  - /common/model/stack_blocks: common
  - _self_

model:
  stack:

    # ------------------------------------------------------------
    # Flow matching: corrupt coordinates and side-chain directions
    # ------------------------------------------------------------

    - _target_: geqdiff.nn.ForwardFlowMatchingModule
      name: flow_matching

      Tmax: 100
      use_aot: false

      flow_target_parameterization: scheduler_velocity
      flow_time_parameterization: tau

      t_embedder_kwargs:
        embedding_dim: 64

      corrupt_fields:

        - field: pos
          out_field: velocity
          center: true
          center_noise: true

        - field: sidechain_direction
          out_field: sidechain_direction_velocity
          center: false
          unit_norm_noise: true

    # ------------------------------------------------------------
    # Input embeddings
    # ------------------------------------------------------------

    - ${stack_blocks.node_input_attrs}
    - ${stack_blocks.edge_input_attrs}
    - ${stack_blocks.edge_radial_attrs}
    - ${stack_blocks.edge_angular_attrs}
    - ${stack_blocks.attrs}

    # ------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------

    - _target_: geqtrain.nn.InteractionModule
      name: encoder

      num_layers: 3

      latent_dim: 256
      eq_latent_multiplicity: 8

      use_attention: true
      attention_mode: edge_self
      attention_head_dim: 32

      out_field: encoder_edge_features

      output_ls:
        - 0
        - 1
        - 2

      use_mace_product: false

      latent_module: geqtrain.nn.ScalarMLPFunction
      latent_module_kwargs:
        mlp_latent_dimensions:
          - 512
          - 512
        mlp_nonlinearity: silu

    - _target_: geqtrain.nn.EdgewiseReduce
      name: encoder_edge_pooling
      field: encoder_edge_features
      out_field: encoder_node_features
      use_attention: true
      attention_head_dim: 32

    # ------------------------------------------------------------
    # Latent bottleneck
    # ------------------------------------------------------------

    - _target_: geqtrain.nn.ReadoutModule
      name: latent_projection
      field: encoder_node_features
      out_field: latent_node
      out_irreps: 128x0e
      strict_irreps: false

      readout_latent: geqtrain.nn.ScalarMLPFunction
      readout_latent_kwargs:
        mlp_latent_dimensions:
          - 256
        mlp_nonlinearity: silu

    # ------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------

    - _target_: geqtrain.nn.InteractionModule
      name: decoder

      num_layers: 3

      latent_dim: 256
      eq_latent_multiplicity: 8

      use_attention: true
      attention_mode: edge_self
      attention_head_dim: 32

      conditioning_fields:
        - latent_node

      out_field: decoder_edge_features

      output_ls:
        - 0
        - 1
        - 2

      use_mace_product: false

      latent_module: geqtrain.nn.ScalarMLPFunction
      latent_module_kwargs:
        mlp_latent_dimensions:
          - 512
          - 512
        mlp_nonlinearity: silu

    - _target_: geqtrain.nn.EdgewiseReduce
      name: decoder_edge_pooling
      field: decoder_edge_features
      out_field: decoder_node_features
      use_attention: true
      attention_head_dim: 32

    # ------------------------------------------------------------
    # Position velocity
    # ------------------------------------------------------------

    - _target_: geqtrain.nn.ReadoutModule
      name: position_velocity_head
      field: decoder_node_features
      out_field: velocity
      out_irreps: 1x1o
      normalize_equivariant_output: false
      strict_irreps: false

      readout_latent: geqtrain.nn.ScalarMLPFunction
      readout_latent_kwargs:
        mlp_latent_dimensions:
          - 256
        mlp_nonlinearity: silu

    # ------------------------------------------------------------
    # Side-chain direction velocity
    # ------------------------------------------------------------

    - _target_: geqtrain.nn.ReadoutModule
      name: sidechain_velocity_head
      field: decoder_node_features
      out_field: sidechain_direction_velocity
      out_irreps: 1x1o
      normalize_equivariant_output: false
      strict_irreps: false

      readout_latent: geqtrain.nn.ScalarMLPFunction
      readout_latent_kwargs:
        mlp_latent_dimensions:
          - 256
        mlp_nonlinearity: silu


# --------------------------------------------------------------
# Geometry
# --------------------------------------------------------------

r_max: 12.0
num_basis: 24
l_max: 2
parity: o3_full


# --------------------------------------------------------------
# Node scalar inputs
# --------------------------------------------------------------

node_attributes:

  node_types:
    attribute_type: categorical
    embedding_mode: embedding
    num_types: 21
    embedding_dimensionality: 32
    fixed: true

  sequence_position:
    attribute_type: categorical
    embedding_mode: embedding
    num_types: 512
    embedding_dimensionality: 32
    fixed: true

  conditioning:
    attribute_type: numerical
    embedding_dimensionality: 64


# --------------------------------------------------------------
# Node equivariant inputs
# --------------------------------------------------------------

eq_node_attributes:

  sidechain_direction:
    attribute_type: numerical
    irreps: 1x1o
    embedding_dimensionality: 3


# --------------------------------------------------------------
# Edge scalar inputs
# --------------------------------------------------------------

edge_attributes:

  seq_edge_features:
    attribute_type: numerical
    embedding_dimensionality: 2
```

The checked-in configuration has been composed and forward-tested against the
installed GEqTrain/GEqDiff revision. Future dependency updates should repeat
that test because field registration and block parameters are API-sensitive.

---

# 14. Loss functions

The first training objective should be pure flow matching.

There are two velocity targets:

```text
velocity
sidechain_direction_velocity
```

The loss is:

```text
L = L_pos + lambda_vector * L_sidechain
```

with:

```text
L_pos =
    MSE(
        predicted position velocity,
        flow target position velocity
    )

L_sidechain =
    MSE(
        predicted side-chain velocity,
        flow target side-chain velocity
    )
```

Recommended first value:

```text
lambda_vector = 2.0
```

Reason: side-chain direction is a weaker/less directly informative target than coordinate transport, so it may otherwise become numerically underweighted.

Do not tune this aggressively at the beginning. First inspect both validation losses.

---

# 15. Training configuration

`config/gpcr/train/flow.yaml`

```yaml
# @package _global_

loss_coeffs:

  - velocity:
      - 1.0
      - MSELoss

  - sidechain_direction_velocity:
      - 2.0
      - MSELoss


metrics_components:

  - velocity:
      - L1Loss

  - sidechain_direction_velocity:
      - L1Loss


metrics_key: validation_loss
metric_criteria: decreasing


max_epochs: 3000

learning_rate: 5.e-4

batch_size: 8
validation_batch_size: 16

validation_freq: 10

dataloader_num_workers: 8


optimizer_name: AdamW

optimizer_params:

  amsgrad: false

  betas:
    - 0.9
    - 0.999

  eps: 1.e-8

  weight_decay: 1.e-5


lr_scheduler_name: ExponentialLR
lr_scheduler_gamma: 0.9995

warmup_epochs: 20

use_ema: true
```

The first smoke test:

```bash
geqtrain-train \
    config/gpcr/experiment/latent_flow.yaml \
    -d cuda:0 \
    -o max_epochs=5 \
    -o run_name=gpcr_latent_flow_smoke
```

---

# 16. Experiment configuration

`config/gpcr/experiment/latent_flow.yaml`

```yaml
# @package _global_

defaults:
  - ../data: cg_flow
  - ../model: latent_flow
  - ../train: flow
  - _self_

root: results/gpcr_latent_flow

run_name: gpcr_latent_flow_poc

experiment_description: >
  E(3)-equivariant flow matching on C-alpha coordinates and
  side-chain direction vectors with an explicit scalar latent
  bottleneck conditioning a second equivariant decoder.

seed: 13
dataset_seed: 17

default_dtype: float32

verbose: info
log_batch_freq: 100

wandb: true
wandb_project: gpcr_latent_flow
```

Run:

```bash
geqtrain-train \
    config/gpcr/experiment/latent_flow.yaml \
    -d cuda:0
```

---

# 17. What exactly is the latent space?

For each configuration `X` and residue `i`:

```text
latent_node[i] = z_i(X)
```

with:

```text
z_i(X) in R^128
```

The complete protein configuration can therefore be represented as:

```text
Z(X) =
[z_1(X),
 z_2(X),
 ...
 z_N(X)]
```

which has shape:

```text
[N_residues, 128]
```

For the first POC, do **not** build a sophisticated global pooling module.

After training, simply flatten:

```text
Z_global(X)
    = flatten(Z(X))
```

giving:

```text
128 * N_residues
```

scalar dimensions for a fixed protein.

This is intentional.

The first experiment is trying to answer whether the representation contains a useful slow manifold, not whether we have already solved variable-length protein pooling.

---

# 18. Extracting the latent representation after training

The implementation must expose the `latent_node` field from the trained model.

For every trajectory frame:

```python
data = model(data)
latent_node = data["latent_node"]
```

Store:

```text
latents.shape =
    [N_frames, N_residues, 128]
```

Do this over the **entire trajectory**, not only batches used during training.

Save:

```text
latent.npy
```

or:

```text
latent.npz
```

containing:

```text
latent
time
frame_index
```

---

# 19. Flow-time dependence

The encoder operates on the flow-corrupted state, so the representation can be considered a function of both configuration and artificial flow time:

```text
z = z(x_tau, tau)
```

The artificial flow time is **not physical time**.

Therefore, do not interpret differences between:

```text
z(tau=0.2)
z(tau=0.5)
z(tau=0.9)
```

as different physical timescales.

Instead, use them as different structural representations.

A useful experiment is:

```text
tau = 0.50
tau = 0.75
tau = 0.90
tau = 0.95
tau = 0.99
```

For each tau:

```text
trajectory
    |
    v
apply encoder at fixed tau
    |
    v
latent representations
    |
    v
diffusion map
```

Then compare whether the leading diffusion coordinates are stable.

Important:

```text
tau -> 1
```

corresponds to the data end of the current GEqDiff convention used by the repository. Near `tau -> 0`, the state is dominated by the noise end of the flow.

---

# 20. Offline diffusion-map analysis

After the flow model is trained, freeze it.

For all trajectory configurations:

```text
X_1, X_2, ..., X_N
```

compute:

```text
Z_1, Z_2, ..., Z_N
```

Then build a diffusion-map kernel on the global latent vectors:

```text
W_ij =
    exp(
        - ||Z_i - Z_j||^2 / epsilon
    )
```

Construct the normalized Markov operator and eigendecompose it.

Obtain:

```text
phi_1(X)
phi_2(X)
phi_3(X)
...
```

These are the candidate slow coordinates.

At this stage, no TICA/VAMP lag time is required.

The diffusion-map analysis is performed **after** model training and does not participate in the training loop.

---

# 21. Main scientific comparison

The first useful comparison should be:

### A. Raw C-alpha geometry

```text
C-alpha coordinates
    -> diffusion map
```

### B. Learned latent

```text
flow-trained latent
    -> diffusion map
```

### C. Simple structural baseline

For example:

```text
C-alpha distance matrix
or
internal C-alpha distances
```

then:

```text
structural descriptor
    -> diffusion map
```

The learned latent is useful only if its diffusion-map coordinates show something better than these simple baselines.

---

# 22. What counts as success?

For a single protein, inspect whether the leading diffusion coordinates:

1. separate known conformational states;
2. vary smoothly along major conformational transitions;
3. correlate with interpretable structural observables;
4. identify slow motions more cleanly than raw C-alpha coordinates;
5. remain stable when the trajectory is split into temporal subsets.

For a controlled POC, also test bootstrap stability:

```text
repeat diffusion-map construction on several
subsets of the trajectory
```

and compare the resulting leading eigenfunctions.

Because eigenfunctions have sign ambiguity, compare:

```text
corr(phi_a, phi_b)
```

up to sign.

---

# 23. Do NOT do these things yet

Do not initially add:

```text
VAMP/TICA losses
kinetic lag-time losses
generator learning
learned committors
free-energy prediction
complex global pooling
variable-length transfer across proteins
```

The first question is simply whether:

```text
equivariant flow-matching representation
```

is a useful coordinate system for:

```text
slow conformational dynamics
```

If it is not, additional kinetic machinery will not rescue the basic representation.

---

# 24. Important future extension: multiple proteins

The initial POC can use one protein.

For a transferable GPCR model, the architecture must eventually handle:

```text
different sequence lengths
different receptor sequences
different conformational ensembles
```

The first major change will likely be global pooling.

Possible future solution:

```text
per-residue latent z_i
       |
       v
sequence-aware attention pooling
       |
       v
global invariant z_global
```

The pooling must preserve enough information about:

- which structural region changed;
- residue identity;
- local topology.

A plain mean can lose this information.

Absolute sequence-position embeddings also need reconsideration when multiple receptor families are combined. A transferable representation may eventually use:

- normalized sequence position,
- residue-number embeddings,
- structural region labels,
- GPCR generic numbering,
- learned positional encodings,
- or relative sequence-distance information.

Do not solve this in the first POC.

---

# 25. Important future extension: latent dimensionality

Start with:

```text
128 scalar dimensions per residue.
```

Do not make the latent tiny.

The diffusion map itself will perform the final dimensionality reduction.

A useful later experiment is:

```text
64
128
256
```

latent dimensions.

The correct criterion is not reconstruction loss alone. Compare the downstream stability and quality of the diffusion-map slow modes.

---

# 26. Why not learn the generator yet?

The final scientific ambition may be to learn an approximation to a continuous-time generator:

```text
L psi_i = -kappa_i psi_i
```

where:

- `psi_i` are slow eigenfunctions,
- `kappa_i` are relaxation rates.

But this is explicitly postponed.

The first POC should establish that the flow-trained latent is good enough that an offline spectral method can recover meaningful slow coordinates.

Only then should we ask whether the expensive offline eigendecomposition can be replaced by a learned generator/operator.

---

# 27. Important distinction: flow time vs physical time

There are two unrelated quantities:

### Flow time

```text
tau
```

controls the artificial generative path used to train GEqDiff.

### Physical MD time

```text
t_MD
```

is the time coordinate of the simulation trajectory.

The model must never be trained as though `tau` were physical MD time.

The desired pipeline is:

```text
physical MD trajectory
        |
        v
many static conformations
        |
        v
flow-matching structural training
        |
        v
structural latent space
        |
        v
diffusion-map kinetic analysis
```

---

# 28. Why this is a good first project

The project cleanly separates three concepts:

### Generative representation learning

```text
Flow matching
```

learns how protein structures vary as a distribution.

### Structural representation

```text
latent_node
```

compresses the geometry into scalar invariant features.

### Kinetic analysis

```text
Diffusion maps
```

finds the slow directions present in the MD ensemble.

This separation avoids putting an arbitrary MD lag time into the neural training objective.

---

# 29. Recommended first POC

Use exactly one protein.

### Step 1

Create:

```text
train.npz
val.npz
```

from a PDB + XTC/TRR.

### Step 2

Run a 5-epoch GEqDiff smoke test.

Check:

```text
dataset loads
edge_index is valid
side-chain vectors are valid
velocity fields have correct irreps
latent_node exists
```

### Step 3

Run a real flow-matching training.

Check:

```text
validation velocity loss decreases
position velocity prediction works
side-chain vector velocity prediction works
```

### Step 4

Extract:

```text
latent_node
```

for every MD frame.

### Step 5

Run diffusion maps on:

```text
flatten(latent_node)
```

### Step 6

Compare against:

```text
raw C-alpha coordinates
```

### Step 7

Inspect:

```text
phi_1
phi_2
```

against interpretable conformational variables.

Only if this works should the project proceed to multi-protein training and more sophisticated global pooling.

---

# 30. Current repository references

The implementation should be based on the actual current GEqTrain/GEqDiff interfaces.

GEqDiff:
- https://github.com/limresgrp/GEqDiff

GEqTrain:
- https://github.com/limresgrp/GEqTrain

Current GEqDiff flow-matching implementation:
- `geqdiff/nn/flow_matching.py`

Current GEqDiff Alanine example:
- `config/alanine/model/flow_model.yaml`
- `config/alanine/experiment/alanine_flow.yaml`
- `config/alanine/data/alanine.yaml`

Current GEqDiff LEGO flow model:
- `config/lego/model/flow_interaction.yaml`

Current GEqTrain interaction implementation:
- `geqtrain/nn/interaction.py`

The current `InteractionModule` supports scalar `conditioning_fields`, including node-level conditioning. This is the mechanism used in the proposed latent-conditioned decoder.

