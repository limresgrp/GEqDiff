from typing import Optional
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqdiff.data import AtomicDataDict
from geqdiff.utils.diffusion import center_pos
from geqtrain.nn import GraphModuleMixin

from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.noise_schedulers import NoiseScheduler

# Try to import scipy, which is required for AOT
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@compile_mode("script")
class ForwardDiffusionModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str = AtomicDataDict.NOISE_KEY,
        alpha_field: str = AtomicDataDict.DIFFUSION_ALPHA_KEY,
        sigma_field: str = AtomicDataDict.DIFFUSION_SIGMA_KEY,
        Tmax: int = 1000,
        Tmax_train: Optional[int] = None,
        # modules
        t_embedder = SinusoidalPositionEmbedding,
        t_embedder_kwargs = None,
        noise_scheduler = NoiseScheduler,
        noise_scheduler_kwargs = None,
        # other
        use_aot: bool = False,
        irreps_in=None,
    ):
        super().__init__()
        self.out_field = out_field
        self.out_target_field = self.out_field + "_target"
        self.alpha_field = alpha_field
        self.sigma_field = sigma_field
        # used to register fields for ref_data in geqtrain.trainer.batch_step
        self.ref_data_keys = [self.out_target_field, self.alpha_field, self.sigma_field]
        self.T = Tmax
        if Tmax_train is None: Tmax_train = self.T
        self.T_train = Tmax_train

        self.use_aot = use_aot
        if self.use_aot and linear_sum_assignment is None:
            raise ImportError("scipy is required to use AOT. Please install it: `pip install scipy`")

        if t_embedder_kwargs is None:
            t_embedder_kwargs = {}
        if noise_scheduler_kwargs is None:
            noise_scheduler_kwargs = {}
        self.t_embedder = t_embedder(**t_embedder_kwargs)
        self.noise_scheduler = noise_scheduler(T=self.T, **noise_scheduler_kwargs)
        
        conditioning_dim = self.t_embedder.embedding_dim

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                AtomicDataDict.CONDITIONING_KEY: o3.Irreps(f"{conditioning_dim}x0e"),
                self.alpha_field: o3.Irreps("1x0e"),
                self.sigma_field: o3.Irreps("1x0e"),
            }
        )

    # @torch.no_grad()
    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        t = data[AtomicDataDict.T_SAMPLED_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        alpha, sigma = self.noise_scheduler(t)
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        data[self.alpha_field] = alpha
        data[self.sigma_field] = sigma
        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.T_SAMPLED_KEY in data:
            return self._reverse(data)
        return self._forward(data)

    def _forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        batch = data[AtomicDataDict.BATCH_KEY]
        x = center_pos(data[AtomicDataDict.POSITIONS_KEY], batch=batch)
        num_batches = len(torch.unique(batch))
        device = x.device

        eps_optimal = None

        # TorchScript is used for deployment/inference. Keep AOT (SciPy) only in eager mode.
        if self.use_aot and (not torch.jit.is_scripting()):
            # --- Approximated Optimal Transport (AOT) with Padding & Masking ---
            
            # 1. Get molecule sizes and find the max size for padding
            _, atom_counts = torch.unique(batch, return_counts=True)
            N_max = int(atom_counts.max())
            
            # 2. Create padded tensors for coordinates and a mask
            x_padded = torch.zeros(num_batches, N_max, 3, device=device)
            attention_mask = torch.zeros(num_batches, N_max, device=device, dtype=torch.bool)
            
            for i in range(num_batches):
                atom_indices = (batch == i)
                num_atoms = atom_indices.sum()
                x_padded[i, :num_atoms] = x[atom_indices]
                attention_mask[i, :num_atoms] = True
            
            # 3. Sample a batch of noise, padded to the max size
            eps_padded = torch.randn(num_batches, N_max, 3, device=device)

            # 4. Calculate the masked and normalized BxB cost matrix
            with torch.no_grad():
                # Vectorized distance calculation
                x_expanded = x_padded.unsqueeze(1)  # Shape: (B, 1, N_max, 3)
                eps_expanded = eps_padded.unsqueeze(0) # Shape: (1, B, N_max, 3)
                
                diff_sq = (x_expanded - eps_expanded)**2 # Shape: (B, B, N_max, 3)
                
                # Apply mask to ignore distances for padded atoms
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1) # Shape: (B, 1, N_max, 1)
                masked_diff_sq = diff_sq * mask_expanded
                
                # Sum costs and normalize by the actual number of atoms for a fair comparison
                sum_masked_diff_sq = masked_diff_sq.sum(dim=(-1, -2)) # Shape: (B, B)
                counts_expanded = atom_counts.unsqueeze(1) # Shape: (B, 1)
                
                cost_matrix = torch.sqrt(sum_masked_diff_sq) / counts_expanded

                # 5. Solve the assignment problem
                _, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # 6. Select the optimal noise and "un-pad" it back to a flat tensor
            eps_optimal_padded = eps_padded[col_ind]
            # Use the boolean mask to select only the real (non-padded) atom noise vectors
            eps_optimal = center_pos(eps_optimal_padded[attention_mask], batch=batch)
        else:
            # If not using AOT, the target is just standard random noise
            eps_optimal = center_pos(torch.randn(size=x.shape, device=device), batch=batch)

        # sample t, get alpha(t) and sigma(t)
        t = torch.randint(0, self.T_train, size=(num_batches, 1), device=device)
        alpha, sigma = self.noise_scheduler(t)

        # compute noised coords and set target, both must be float32
        data[AtomicDataDict.POSITIONS_KEY] = alpha[batch] * x  + sigma[batch] * eps_optimal
        data[self.out_target_field] = eps_optimal

        # add t-embedding to data obj
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = t
        data[self.alpha_field] = alpha
        data[self.sigma_field] = sigma

        return data
