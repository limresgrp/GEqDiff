from typing import Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqdiff.data import AtomicDataDict
from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.diffusion import center_pos
from geqdiff.utils.noise_schedulers import FlowMatchingScheduler
from geqtrain.nn import GraphModuleMixin

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@compile_mode("script")
class ForwardFlowMatchingModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str = "velocity",
        Tmax: int = 100,
        Tmax_train: Optional[int] = None,
        t_embedder=SinusoidalPositionEmbedding,
        t_embedder_kwargs=None,
        flow_scheduler=FlowMatchingScheduler,
        flow_scheduler_kwargs=None,
        use_aot: bool = False,
        irreps_in=None,
    ):
        super().__init__()
        if Tmax < 1:
            raise ValueError(f"Tmax must be >= 1, got {Tmax}")

        self.out_field = out_field
        self.out_target_field = self.out_field + "_target"
        self.ref_data_keys = [self.out_target_field]

        self.T = int(Tmax)
        if Tmax_train is None:
            Tmax_train = self.T
        if Tmax_train < 1:
            raise ValueError(f"Tmax_train must be >= 1, got {Tmax_train}")
        self.T_train = int(Tmax_train)

        self.use_aot = use_aot
        if self.use_aot and linear_sum_assignment is None:
            raise ImportError("scipy is required to use AOT. Please install it: `pip install scipy`")

        if t_embedder_kwargs is None:
            t_embedder_kwargs = {}
        if flow_scheduler_kwargs is None:
            flow_scheduler_kwargs = {}

        self.t_embedder = t_embedder(**t_embedder_kwargs)
        self.flow_scheduler = flow_scheduler(T=self.T, **flow_scheduler_kwargs)

        conditioning_dim = self.t_embedder.embedding_dim
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                AtomicDataDict.CONDITIONING_KEY: o3.Irreps(f"{conditioning_dim}x0e"),
            },
        )

    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        t = data[AtomicDataDict.T_SAMPLED_KEY].to(dtype=torch.float32)
        batch = data[AtomicDataDict.BATCH_KEY]
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
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

        if self.use_aot and (not torch.jit.is_scripting()):
            _, atom_counts = torch.unique(batch, return_counts=True)
            n_max = int(atom_counts.max())

            x_padded = torch.zeros(num_batches, n_max, 3, device=device)
            attention_mask = torch.zeros(num_batches, n_max, device=device, dtype=torch.bool)

            for i in range(num_batches):
                atom_indices = batch == i
                num_atoms = atom_indices.sum()
                x_padded[i, :num_atoms] = x[atom_indices]
                attention_mask[i, :num_atoms] = True

            noise_padded = torch.randn(num_batches, n_max, 3, device=device)

            with torch.no_grad():
                x_expanded = x_padded.unsqueeze(1)
                noise_expanded = noise_padded.unsqueeze(0)
                diff_sq = (x_expanded - noise_expanded) ** 2
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)
                masked_diff_sq = diff_sq * mask_expanded
                sum_masked_diff_sq = masked_diff_sq.sum(dim=(-1, -2))
                counts_expanded = atom_counts.unsqueeze(1)
                cost_matrix = torch.sqrt(sum_masked_diff_sq) / counts_expanded
                _, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

            noise = center_pos(noise_padded[col_ind][attention_mask], batch=batch)
        else:
            noise = center_pos(torch.randn(size=x.shape, device=device), batch=batch)

        if self.T_train == 1:
            t = torch.zeros((num_batches, 1), device=device)
        else:
            t = torch.rand((num_batches, 1), device=device) * float(self.T_train - 1)

        data_scale, noise_scale = self.flow_scheduler(t)
        data[AtomicDataDict.POSITIONS_KEY] = data_scale[batch] * x + noise_scale[batch] * noise
        data[self.out_target_field] = noise - x
        data[AtomicDataDict.CONDITIONING_KEY] = self.t_embedder(t)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = t
        return data
