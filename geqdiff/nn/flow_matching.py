from collections.abc import Mapping
from typing import List, Optional

import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from geqdiff.data import AtomicDataDict
from geqdiff.nn.t_embedders import SinusoidalPositionEmbedding
from geqdiff.utils.diffusion import center_pos
from geqdiff.utils.noise_schedulers import FlowMatchingScheduler
from geqdiff.utils.optimal_transport import linear_sum_assignment, sample_ot_aligned_noise
from geqtrain.data.AtomicData import register_fields
from geqtrain.nn import GraphModuleMixin


@compile_mode("script")
class ForwardFlowMatchingModule(GraphModuleMixin, torch.nn.Module):
    field_names: List[str]
    out_fields: List[str]
    out_target_fields: List[str]
    centered_fields: List[bool]
    center_mask_fields: List[str]
    center_noise_fields: List[bool]
    noise_center_mask_fields: List[str]
    mask_fields: List[str]
    unmasked_noise_scales: List[float]

    def __init__(
        self,
        out_field: str = "velocity",
        corrupt_fields: Optional[List[dict]] = None,
        default_mask_field: Optional[str] = None,
        alpha_field: str = AtomicDataDict.DIFFUSION_ALPHA_KEY,
        sigma_field: str = AtomicDataDict.DIFFUSION_SIGMA_KEY,
        num_atoms_field: str = AtomicDataDict.NUM_ATOMS_BITS_KEY,
        num_atoms_bits: int = 8,
        Tmax: int = 100,
        Tmax_train: Optional[int] = None,
        tau_eps: float = 0.0,
        embed_normalized_time: bool = False,
        flow_target_parameterization: str = "scheduler_velocity",
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

        self.alpha_field = alpha_field
        self.sigma_field = sigma_field
        self.num_atoms_field = num_atoms_field
        self.num_atoms_bits = int(num_atoms_bits)

        parsed_default_mask = "" if default_mask_field is None else str(default_mask_field)
        if corrupt_fields is None:
            corrupt_fields = [
                {
                    "field": AtomicDataDict.POSITIONS_KEY,
                    "out_field": out_field,
                    "center": True,
                    "mask_field": parsed_default_mask,
                    "unmasked_noise_scale": 0.0,
                }
            ]
        if len(corrupt_fields) == 0:
            raise ValueError("ForwardFlowMatchingModule requires at least one corrupt field.")

        self.field_names = []
        self.out_fields = []
        self.out_target_fields = []
        self.centered_fields = []
        self.center_mask_fields = []
        self.center_noise_fields = []
        self.noise_center_mask_fields = []
        self.mask_fields = []
        self.unmasked_noise_scales = []

        for idx, spec in enumerate(corrupt_fields):
            if not isinstance(spec, Mapping):
                raise TypeError(f"`corrupt_fields[{idx}]` must be a mapping, got {type(spec)}")
            field_name = str(spec.get("field"))
            if field_name == "":
                raise ValueError(f"`corrupt_fields[{idx}]` is missing the `field` key.")
            out_field_name = str(spec.get("out_field", out_field if idx == 0 else f"{field_name}_velocity"))
            center_field = bool(spec.get("center", field_name == AtomicDataDict.POSITIONS_KEY))
            center_mask_field = spec.get("center_mask_field", "")
            if center_mask_field in (None, "", "all", "__all__"):
                center_mask_field = ""
            else:
                center_mask_field = str(center_mask_field)
            center_noise = bool(spec.get("center_noise", center_field))
            mask_field = spec.get("mask_field", parsed_default_mask)
            mask_field = "" if mask_field in (None, "") else str(mask_field)
            noise_center_mask_field = spec.get("noise_center_mask_field", "__auto__")
            if noise_center_mask_field == "__auto__":
                if field_name == AtomicDataDict.POSITIONS_KEY and mask_field != "":
                    noise_center_mask_field = mask_field
                else:
                    noise_center_mask_field = center_mask_field
            elif noise_center_mask_field in (None, "", "all", "__all__"):
                noise_center_mask_field = ""
            else:
                noise_center_mask_field = str(noise_center_mask_field)
            unmasked_noise_scale = float(spec.get("unmasked_noise_scale", 0.0))
            if unmasked_noise_scale < 0.0:
                raise ValueError(
                    f"`corrupt_fields[{idx}].unmasked_noise_scale` must be >= 0, got {unmasked_noise_scale}."
                )

            self.field_names.append(field_name)
            self.out_fields.append(out_field_name)
            self.out_target_fields.append(out_field_name + "_target")
            self.centered_fields.append(center_field)
            self.center_mask_fields.append(center_mask_field)
            self.center_noise_fields.append(center_noise)
            self.noise_center_mask_fields.append(noise_center_mask_field)
            self.mask_fields.append(mask_field)
            self.unmasked_noise_scales.append(unmasked_noise_scale)

        self.out_field = self.out_fields[0]
        self.out_target_field = self.out_target_fields[0]
        self.ref_data_keys = list(self.out_target_fields) + [self.alpha_field, self.sigma_field]

        self.T = int(Tmax)
        if Tmax_train is None:
            Tmax_train = self.T
        if Tmax_train < 1:
            raise ValueError(f"Tmax_train must be >= 1, got {Tmax_train}")
        self.T_train = int(Tmax_train)
        self.tau_eps = float(tau_eps)
        if not (0.0 <= self.tau_eps < 0.5):
            raise ValueError(f"tau_eps must satisfy 0 <= tau_eps < 0.5, got {tau_eps}")
        self.embed_normalized_time = bool(embed_normalized_time)
        if str(flow_target_parameterization) != "scheduler_velocity":
            raise ValueError(
                "ForwardFlowMatchingModule only supports flow_target_parameterization='scheduler_velocity'."
            )
        if str(flow_time_parameterization) != "tau":
            raise ValueError(
                "ForwardFlowMatchingModule only supports flow_time_parameterization='tau'."
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
                self.num_atoms_field: o3.Irreps(f"{self.num_atoms_bits}x0e"),
            },
        )
        register_fields(graph_fields=[self.num_atoms_field])

    @staticmethod
    def _encode_num_atoms(num_atoms: torch.Tensor, num_bits: int) -> torch.Tensor:
        values = num_atoms.to(dtype=torch.long).view(-1, 1)
        bit_shifts = torch.arange(num_bits, device=values.device, dtype=torch.long).view(1, -1)
        bits = (values >> bit_shifts) & 1
        return bits.to(dtype=torch.float32)

    def _time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        return self.t_embedder(tau.to(dtype=torch.float32))

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
        return data

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if AtomicDataDict.T_SAMPLED_KEY in data:
            return self._reverse(data)
        return self._forward(data)

    def _expand_graph_values(self, values: torch.Tensor, batch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        expanded = values[batch]
        while expanded.dim() < target.dim():
            expanded = expanded.unsqueeze(-1)
        return expanded.to(dtype=target.dtype)

    def _resolve_mask(self, data: AtomicDataDict.Type, mask_field: str, target: torch.Tensor) -> torch.Tensor:
        if mask_field == "":
            return torch.ones(target.shape[:1], device=target.device, dtype=torch.bool)
        if mask_field not in data:
            raise KeyError(
                f"Mask field '{mask_field}' was requested by ForwardFlowMatchingModule but is missing from the batch."
            )
        mask = data[mask_field]
        if mask.dim() > 1 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        return mask.to(device=target.device) > 0.5

    def _broadcast_mask(self, mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        expanded = mask
        while expanded.dim() < target.dim():
            expanded = expanded.unsqueeze(-1)
        return expanded

    def _sample_noise(
        self,
        field_name: str,
        x: torch.Tensor,
        data: AtomicDataDict.Type,
        batch: torch.Tensor,
        corrupt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if field_name == AtomicDataDict.POSITIONS_KEY and self.use_aot and (not torch.jit.is_scripting()):
            with torch.no_grad():
                return sample_ot_aligned_noise(x=x, data=data, batch=batch, corrupt_mask=corrupt_mask)

        noise = torch.randn(size=x.shape, device=x.device, dtype=x.dtype)
        return noise

    def _resolve_optional_center_mask(
        self,
        data: AtomicDataDict.Type,
        center_mask_field: str,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if center_mask_field == "":
            return None
        return self._resolve_mask(data=data, mask_field=center_mask_field, target=target)

    def _forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        batch = data[AtomicDataDict.BATCH_KEY]
        num_batches = int(batch.max().item()) + 1
        device = batch.device
        atom_counts = torch.bincount(batch, minlength=num_batches)
        data[self.num_atoms_field] = self._encode_num_atoms(atom_counts, self.num_atoms_bits)

        if self.T_train == 1:
            tau = torch.zeros((num_batches, 1), device=device)
        else:
            tau = self.tau_eps + (1.0 - 2.0 * self.tau_eps) * torch.rand((num_batches, 1), device=device)

        data_scale, noise_scale = self.flow_scheduler(tau)
        ddata_scale_dtau, dnoise_scale_dtau = self.flow_scheduler.derivatives(tau)

        for idx, field_name in enumerate(self.field_names):
            if field_name not in data:
                raise KeyError(
                    f"ForwardFlowMatchingModule expected field '{field_name}' in the input batch. "
                    f"Available keys: {list(data.keys())}"
                )
            x = data[field_name]
            center_mask = self._resolve_optional_center_mask(
                data=data,
                center_mask_field=self.center_mask_fields[idx],
                target=x,
            )
            if self.centered_fields[idx]:
                x = center_pos(x, batch=batch, mask=center_mask)

            mask = self._resolve_mask(data=data, mask_field=self.mask_fields[idx], target=x)
            noise_center_mask = self._resolve_optional_center_mask(
                data=data,
                center_mask_field=self.noise_center_mask_fields[idx],
                target=x,
            )
            noise = self._sample_noise(field_name=field_name, x=x, data=data, batch=batch, corrupt_mask=mask)
            if self.centered_fields[idx] and self.center_noise_fields[idx]:
                noise = center_pos(noise, batch=batch, mask=noise_center_mask)

            expanded_data_scale = self._expand_graph_values(data_scale, batch=batch, target=x)
            expanded_noise_scale = self._expand_graph_values(noise_scale, batch=batch, target=x)
            expanded_ddata = self._expand_graph_values(ddata_scale_dtau, batch=batch, target=x)
            expanded_dnoise = self._expand_graph_values(dnoise_scale_dtau, batch=batch, target=x)

            x_t = expanded_data_scale * x + expanded_noise_scale * noise
            target = expanded_ddata * x + expanded_dnoise * noise

            mask_broadcast = self._broadcast_mask(mask=mask, target=x)

            if self.mask_fields[idx] == "":
                corrupted = x_t
                target_out = target
            else:
                unmasked_noise_scale = float(self.unmasked_noise_scales[idx])
                if unmasked_noise_scale > 0.0:
                    # if unmasked_noise_scale > 0, apply a reduced amount of noise to non-masked nodes
                    pocket_corrupted = expanded_data_scale * x + (expanded_noise_scale * unmasked_noise_scale) * noise
                else:
                    # if unmasked_noise_scale == 0, leave non-masked nodes to original clean value
                    pocket_corrupted = x
                corrupted = torch.where(mask_broadcast, x_t, pocket_corrupted)
                target_out = torch.where(mask_broadcast, target, torch.zeros_like(target))

            data[field_name] = corrupted
            data[self.out_target_fields[idx]] = target_out

        data[AtomicDataDict.CONDITIONING_KEY] = self._time_embedding(tau)[batch]
        data[AtomicDataDict.T_SAMPLED_KEY] = tau
        data[self.alpha_field] = data_scale
        data[self.sigma_field] = noise_scale
        return data
