from typing import Optional
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqdiff.data import AtomicDataDict
from geqdiff.utils.diffusion import center_pos
from geqtrain.nn import GraphModuleMixin
from geqtrain.data.AtomicData import register_fields

from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.noise_schedulers import NoiseScheduler


@compile_mode("script")
class ForwardDiffusionModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str = AtomicDataDict.NOISE_KEY,
        alpha_field: str = AtomicDataDict.DIFFUSION_ALPHA_KEY,
        sigma_field: str = AtomicDataDict.DIFFUSION_SIGMA_KEY,
        num_atoms_field: str = AtomicDataDict.NUM_ATOMS_BITS_KEY,
        num_atoms_bits: int = 8,
        Tmax: int = 1000,
        Tmax_train: Optional[int] = None,
        # modules
        t_embedder = SinusoidalPositionEmbedding,
        t_embedder_kwargs = None,
        noise_scheduler = NoiseScheduler,
        noise_scheduler_kwargs = None,
        irreps_in=None,
    ):
        super().__init__()
        self.out_field = out_field
        self.out_target_field = self.out_field + "_target"
        self.alpha_field = alpha_field
        self.sigma_field = sigma_field
        self.num_atoms_field = num_atoms_field
        self.num_atoms_bits = int(num_atoms_bits)
        # used to register fields for ref_data in geqtrain.trainer.batch_step
        self.ref_data_keys = [self.out_target_field, self.alpha_field, self.sigma_field]
        self.T = Tmax
        if Tmax_train is None: Tmax_train = self.T
        self.T_train = Tmax_train

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
                self.num_atoms_field: o3.Irreps(f"{self.num_atoms_bits}x0e"),
            }
        )
        register_fields(graph_fields=[self.num_atoms_field])

    @staticmethod
    def _encode_num_atoms(num_atoms: torch.Tensor, num_bits: int) -> torch.Tensor:
        values = num_atoms.to(dtype=torch.long).view(-1, 1)
        bit_shifts = torch.arange(num_bits, device=values.device, dtype=torch.long).view(1, -1)
        bits = (values >> bit_shifts) & 1
        return bits.to(dtype=torch.float32)

    # @torch.no_grad()
    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        t = data[AtomicDataDict.T_SAMPLED_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batches = int(batch.max().item()) + 1
        atom_counts = torch.bincount(batch, minlength=num_batches)
        data[self.num_atoms_field] = self._encode_num_atoms(atom_counts, self.num_atoms_bits)
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
        num_batches = int(batch.max().item()) + 1
        device = x.device
        atom_counts = torch.bincount(batch, minlength=num_batches)
        data[self.num_atoms_field] = self._encode_num_atoms(atom_counts, self.num_atoms_bits)
        eps = center_pos(torch.randn(size=x.shape, device=device), batch=batch)

        # sample t, get alpha(t) and sigma(t)
        t = torch.randint(0, self.T_train, size=(num_batches, 1), device=device)
        alpha, sigma = self.noise_scheduler(t)

        # compute noised coords and set target, both must be float32
        data[AtomicDataDict.POSITIONS_KEY] = alpha[batch] * x  + sigma[batch] * eps
        data[self.out_target_field] = eps

        # add t-embedding to data obj
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = t
        data[self.alpha_field] = alpha
        data[self.sigma_field] = sigma

        return data
