from typing import Final

### == Define allowed keys as constants == ###

# (n_graphs)
CONDITIONING_KEY: Final[str] = "conditioning"
# (n_graphs)
T_SAMPLED_KEY: Final[str] = "t_sampled"
# (n_graphs, 1)
DIFFUSION_ALPHA_KEY: Final[str] = "diffusion_alpha"
# (n_graphs, 1)
DIFFUSION_SIGMA_KEY: Final[str] = "diffusion_sigma"
# (n_graphs, num_bits)
NUM_ATOMS_BITS_KEY: Final[str] = "num_atoms_bits"

# (n_nodes, 4)
SHAPE_SCALAR_FEATURES_KEY: Final[str] = "shape_scalar_features"
# (n_nodes, 16)
SHAPE_FEATURES_KEY: Final[str] = "shape_features"
# (n_nodes, 15)
SHAPE_EQUIV_FEATURES_KEY: Final[str] = "shape_equiv_features"
# (n_nodes, 1)
DIPOLE_STRENGTH_KEY: Final[str] = "dipole_strength"
# (n_nodes, 3)
DIPOLE_DIRECTION_KEY: Final[str] = "dipole_direction"
# (n_nodes, 1)
LIGAND_MASK_KEY: Final[str] = "ligand_mask"
# (n_nodes, 1)
POCKET_MASK_KEY: Final[str] = "pocket_mask"
# (n_nodes, 3, 3)
ROTATIONS_KEY: Final[str] = "rotations"
# (n_graphs, 1)
SOURCE_FRAME_ID_KEY: Final[str] = "source_frame_id"
# (n_graphs, 1)
SPLIT_ID_KEY: Final[str] = "split_id"
