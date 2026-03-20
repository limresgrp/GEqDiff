from typing import Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqdiff.data import AtomicDataDict
from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.diffusion import center_pos
from geqdiff.utils.noise_schedulers import FlowMatchingScheduler
from geqdiff.utils.optimal_transport import linear_sum_assignment, sample_ot_aligned_noise
from geqtrain.nn import GraphModuleMixin
from geqtrain.data.AtomicData import register_fields


@compile_mode("script")
class ForwardSchrodingerBridgeModule(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        out_field: str = "velocity",
        alpha_field: str = AtomicDataDict.DIFFUSION_ALPHA_KEY,
        sigma_field: str = AtomicDataDict.DIFFUSION_SIGMA_KEY,
        bridge_scale_field: str = "schrodinger_bridge_scale",
        num_atoms_field: str = AtomicDataDict.NUM_ATOMS_BITS_KEY,
        num_atoms_bits: int = 8,
        Tmax: int = 100,
        Tmax_train: Optional[int] = None,
        tau_eps: float = 1e-3,
        bridge_sigma: float = 1.0,
        embed_normalized_time: bool = False,
        flow_time_parameterization: str = "tau",
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
        if bridge_sigma <= 0.0:
            raise ValueError(f"bridge_sigma must be > 0, got {bridge_sigma}")

        self.out_field = out_field
        self.out_target_field = self.out_field + "_target"
        self.alpha_field = alpha_field
        self.sigma_field = sigma_field
        self.bridge_scale_field = bridge_scale_field
        self.num_atoms_field = num_atoms_field
        self.num_atoms_bits = int(num_atoms_bits)
        self.ref_data_keys = [self.out_target_field, self.alpha_field, self.sigma_field]

        self.T = int(Tmax)
        if Tmax_train is None:
            Tmax_train = self.T
        if Tmax_train < 1:
            raise ValueError(f"Tmax_train must be >= 1, got {Tmax_train}")
        self.T_train = int(Tmax_train)
        self.tau_eps = float(tau_eps)
        if not (0.0 < self.tau_eps < 0.5):
            raise ValueError(f"tau_eps must satisfy 0 < tau_eps < 0.5, got {tau_eps}")
        self.bridge_sigma = float(bridge_sigma)
        self.embed_normalized_time = bool(embed_normalized_time)
        if str(flow_time_parameterization) != "tau":
            raise ValueError(
                "ForwardSchrodingerBridgeModule only supports flow_time_parameterization='tau'."
            )

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
                self.alpha_field: o3.Irreps("1x0e"),
                self.sigma_field: o3.Irreps("1x0e"),
                self.bridge_scale_field: o3.Irreps("1x0e"),
                self.num_atoms_field: o3.Irreps(f"{self.num_atoms_bits}x0e"),
            },
        )
        register_fields(graph_fields=[self.num_atoms_field, self.bridge_scale_field])

    @staticmethod
    def _encode_num_atoms(num_atoms: torch.Tensor, num_bits: int) -> torch.Tensor:
        values = num_atoms.to(dtype=torch.long).view(-1, 1)
        bit_shifts = torch.arange(num_bits, device=values.device, dtype=torch.long).view(1, -1)
        bits = (values >> bit_shifts) & 1
        return bits.to(dtype=torch.float32)

    def _time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        return self.t_embedder(tau.to(dtype=torch.float32))

    def _bridge_scale(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.to(dtype=torch.float32)
        return self.bridge_sigma * torch.sqrt((tau * (1.0 - tau)).clamp_min(0.0))

    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        tau = data[AtomicDataDict.T_SAMPLED_KEY].to(dtype=torch.float32)
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batches = int(batch.max().item()) + 1
        atom_counts = torch.bincount(batch, minlength=num_batches)
        data[self.num_atoms_field] = self._encode_num_atoms(atom_counts, self.num_atoms_bits)
        data_scale, noise_scale = self.flow_scheduler(tau)
        data[AtomicDataDict.CONDITIONING_KEY] = self._time_embedding(tau)[batch]
        data[self.alpha_field] = data_scale
        data[self.sigma_field] = noise_scale
        data[self.bridge_scale_field] = self._bridge_scale(tau)
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

        if self.use_aot and (not torch.jit.is_scripting()):
            with torch.no_grad():
                x0 = sample_ot_aligned_noise(x=x, data=data, batch=batch)
        else:
            x0 = center_pos(torch.randn(size=x.shape, device=device), batch=batch)

        bridge_noise = center_pos(torch.randn(size=x.shape, device=device), batch=batch)

        if self.T_train == 1:
            tau = torch.full((num_batches, 1), 0.5, device=device)
        else:
            tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * torch.rand((num_batches, 1), device=device)

        data_scale, noise_scale = self.flow_scheduler(tau)
        bridge_scale = self._bridge_scale(tau)
        tau_batch = tau[batch]

        mu = data_scale[batch] * x + noise_scale[batch] * x0
        xt = mu + bridge_scale[batch] * bridge_noise
        denom = 2.0 * tau_batch * (1.0 - tau_batch) + 1e-8
        bridge_drift = ((1.0 - 2.0 * tau_batch) / denom) * (xt - mu)

        data[AtomicDataDict.POSITIONS_KEY] = xt
        data[self.out_target_field] = bridge_drift + x0 - x
        data[AtomicDataDict.CONDITIONING_KEY] = self._time_embedding(tau)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = tau
        data[self.alpha_field] = data_scale
        data[self.sigma_field] = noise_scale
        data[self.bridge_scale_field] = bridge_scale
        return data
