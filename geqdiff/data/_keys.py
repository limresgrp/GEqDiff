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
