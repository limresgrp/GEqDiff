from typing import Optional
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from geqdiff.data import AtomicDataDict
from geqdiff.utils.diffusion import center_pos
from geqtrain.nn import GraphModuleMixin

from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.noise_schedulers import NoiseScheduler


@compile_mode("script")
class ForwardDiffusionModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str = AtomicDataDict.NOISE_KEY,
        Tmax: int = 1000,
        Tmax_train: Optional[int] = None,
        # modules
        t_embedder = SinusoidalPositionEmbedding,
        t_embedder_kwargs = {},
        noise_scheduler = NoiseScheduler,
        noise_scheduler_kwargs = {},
        # other
        irreps_in=None,
    ):
        super().__init__()
        self.out_field = out_field
        self.out_target_field = self.out_field + "_target"
        self.ref_data_keys = [self.out_target_field] # used to register field for ref_data in geqtrain.trainer.batch_step
        self.T = Tmax
        if Tmax_train is None: Tmax_train = self.T
        self.T_train = Tmax_train
        self.t_embedder = t_embedder(**t_embedder_kwargs)
        self.noise_scheduler = noise_scheduler(T=self.T, **noise_scheduler_kwargs)
        
        conditioning_dim = self.t_embedder.embedding_dim

        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={AtomicDataDict.CONDITIONING_KEY: o3.Irreps(f'{conditioning_dim}x0e')}
        )

    # @torch.no_grad()
    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        t = data[AtomicDataDict.T_SAMPLED_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.T_SAMPLED_KEY in data:
            return self._reverse(data)
        return self._forward(data)

    def _forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        x = data[AtomicDataDict.POSITIONS_KEY]
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batches = len(torch.unique(batch))
        device = x.device

        # sample noise
        eps = center_pos(torch.randn(size=x.shape, device=device))

        # sample t, get alpha(t) and sigma(t)
        t = torch.randint(0, self.T_train, size=(num_batches, 1), device=device)
        alpha, sigma = self.noise_scheduler(t)

        # compute noised coords and set target, both must be float32
        data[AtomicDataDict.POSITIONS_KEY] = alpha[batch] * x  + sigma[batch] * eps
        data[self.out_target_field] = eps

        # add t-embedding to data obj
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = t

        return data