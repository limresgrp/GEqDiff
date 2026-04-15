from collections.abc import Mapping, Sequence
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
    directional_scalar_out_fields: List[str]
    directional_equiv_out_fields: List[str]
    directional_scalar_indices: List[List[int]]
    directional_equiv_starts: List[List[int]]
    directional_equiv_stops: List[List[int]]
    directional_nonnegative_scale: List[bool]

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
        directional_velocity_couplings: Optional[List[dict]] = None,
        directional_velocity_eps: float = 1e-8,
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
        self.directional_velocity_eps = float(directional_velocity_eps)
        if self.directional_velocity_eps <= 0.0:
            raise ValueError("directional_velocity_eps must be > 0.")

        if t_embedder_kwargs is None:
            t_embedder_kwargs = {}
        if flow_scheduler_kwargs is None:
            flow_scheduler_kwargs = {}

        self.t_embedder = t_embedder(**t_embedder_kwargs)
        self.flow_scheduler = flow_scheduler(T=self.T, **flow_scheduler_kwargs)

        self.directional_scalar_out_fields = []
        self.directional_equiv_out_fields = []
        self.directional_scalar_indices = []
        self.directional_equiv_starts = []
        self.directional_equiv_stops = []
        self.directional_nonnegative_scale = []
        self._parse_directional_velocity_couplings(directional_velocity_couplings)

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

    def _parse_directional_velocity_couplings(self, couplings: Optional[List[dict]]) -> None:
        if couplings is None:
            return
        if not isinstance(couplings, Sequence) or isinstance(couplings, (str, bytes)):
            raise TypeError("directional_velocity_couplings must be a list of mappings.")

        known_out_fields = set(self.out_fields)
        for idx, coupling in enumerate(list(couplings)):
            if not isinstance(coupling, Mapping):
                raise TypeError(f"directional_velocity_couplings[{idx}] must be a mapping.")
            scalar_out_field = str(coupling.get("scalar_out_field", ""))
            equiv_out_field = str(coupling.get("equiv_out_field", ""))
            if scalar_out_field == "" or equiv_out_field == "":
                raise ValueError(
                    f"directional_velocity_couplings[{idx}] must define both scalar_out_field and equiv_out_field."
                )
            if scalar_out_field not in known_out_fields:
                raise ValueError(
                    f"directional_velocity_couplings[{idx}] references unknown scalar_out_field '{scalar_out_field}'. "
                    f"Known outputs: {sorted(known_out_fields)}"
                )
            if equiv_out_field not in known_out_fields:
                raise ValueError(
                    f"directional_velocity_couplings[{idx}] references unknown equiv_out_field '{equiv_out_field}'. "
                    f"Known outputs: {sorted(known_out_fields)}"
                )

            block_pairs = coupling.get("block_pairs", [])
            if (
                not isinstance(block_pairs, Sequence)
                or isinstance(block_pairs, (str, bytes))
                or len(block_pairs) == 0
            ):
                raise ValueError(
                    f"directional_velocity_couplings[{idx}] must provide a non-empty block_pairs list."
                )
            scalar_indices: List[int] = []
            equiv_starts: List[int] = []
            equiv_stops: List[int] = []
            for block_idx, block in enumerate(list(block_pairs)):
                if not isinstance(block, Mapping):
                    raise TypeError(
                        f"directional_velocity_couplings[{idx}].block_pairs[{block_idx}] must be a mapping."
                    )
                scalar_index = int(block.get("scalar_index", -1))
                equiv_slice = block.get("equiv_slice", None)
                if (
                    not isinstance(equiv_slice, Sequence)
                    or isinstance(equiv_slice, (str, bytes))
                    or len(equiv_slice) != 2
                ):
                    raise ValueError(
                        f"directional_velocity_couplings[{idx}].block_pairs[{block_idx}] must define "
                        f"equiv_slice: [start, stop]."
                    )
                start = int(equiv_slice[0])
                stop = int(equiv_slice[1])
                if scalar_index < 0:
                    raise ValueError(
                        f"directional_velocity_couplings[{idx}].block_pairs[{block_idx}] has invalid scalar_index "
                        f"{scalar_index}."
                    )
                if stop <= start:
                    raise ValueError(
                        f"directional_velocity_couplings[{idx}].block_pairs[{block_idx}] has invalid equiv_slice "
                        f"[{start}, {stop}]."
                    )
                scalar_indices.append(scalar_index)
                equiv_starts.append(start)
                equiv_stops.append(stop)

            nonnegative_scale = bool(coupling.get("nonnegative_scale", True))
            self.directional_scalar_out_fields.append(scalar_out_field)
            self.directional_equiv_out_fields.append(equiv_out_field)
            self.directional_scalar_indices.append(scalar_indices)
            self.directional_equiv_starts.append(equiv_starts)
            self.directional_equiv_stops.append(equiv_stops)
            self.directional_nonnegative_scale.append(nonnegative_scale)

    def _apply_directional_velocity_target_couplings(self, data: AtomicDataDict.Type) -> None:
        num_couplings = len(self.directional_scalar_out_fields)
        if num_couplings == 0:
            return

        for idx in range(num_couplings):
            scalar_out_field = self.directional_scalar_out_fields[idx]
            equiv_out_field = self.directional_equiv_out_fields[idx]
            scalar_target_field = scalar_out_field + "_target"
            equiv_target_field = equiv_out_field + "_target"
            if scalar_target_field not in data or equiv_target_field not in data:
                continue

            scalar_target = data[scalar_target_field]
            equiv_target = data[equiv_target_field]
            scalar_indices = self.directional_scalar_indices[idx]
            equiv_starts = self.directional_equiv_starts[idx]
            equiv_stops = self.directional_equiv_stops[idx]
            nonnegative = self.directional_nonnegative_scale[idx]

            scalar_target_updated = scalar_target.clone()
            equiv_target_updated = equiv_target.clone()
            for block_idx in range(len(scalar_indices)):
                scalar_index = int(scalar_indices[block_idx])
                start = int(equiv_starts[block_idx])
                stop = int(equiv_stops[block_idx])
                block = equiv_target_updated[..., start:stop]
                norms = torch.linalg.norm(block, dim=-1, keepdim=True)
                normalized = torch.where(
                    norms > self.directional_velocity_eps,
                    block / torch.clamp(norms, min=self.directional_velocity_eps),
                    torch.zeros_like(block),
                )
                equiv_target_updated[..., start:stop] = normalized
                scalar_values = norms
                if nonnegative:
                    scalar_values = torch.clamp(scalar_values, min=0.0)
                scalar_target_updated[..., scalar_index : scalar_index + 1] = scalar_values

            data[scalar_target_field] = scalar_target_updated
            data[equiv_target_field] = equiv_target_updated

    @staticmethod
    def _encode_num_atoms(num_atoms: torch.Tensor, num_bits: int) -> torch.Tensor:
        values = num_atoms.to(dtype=torch.long).view(-1, 1)
        bit_shifts = torch.arange(num_bits, device=values.device, dtype=torch.long).view(1, -1)
        bits = (values >> bit_shifts) & 1
        return bits.to(dtype=torch.float32)

    def _time_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        return self.t_embedder(tau.to(dtype=torch.float32))

    def _resolve_batch(self, data: AtomicDataDict.Type) -> torch.Tensor:
        if AtomicDataDict.BATCH_KEY in data:
            batch = data[AtomicDataDict.BATCH_KEY]
            if batch.dim() > 1 and batch.shape[-1] == 1:
                batch = batch.squeeze(-1)
            return batch.to(dtype=torch.long)

        if AtomicDataDict.POSITIONS_KEY not in data:
            raise KeyError(
                f"ForwardFlowMatchingModule expected '{AtomicDataDict.BATCH_KEY}' or "
                f"'{AtomicDataDict.POSITIONS_KEY}' to infer a single-graph batch."
            )
        pos = data[AtomicDataDict.POSITIONS_KEY]
        return torch.zeros((int(pos.shape[0]),), device=pos.device, dtype=torch.long)

    def _reverse(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        tau = data[AtomicDataDict.T_SAMPLED_KEY].to(dtype=torch.float32)
        batch = self._resolve_batch(data)
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

        if field_name == AtomicDataDict.SHAPE_EQUIV_FEATURES_KEY:
            if x.shape[-1] == 15:
                parts = []
                for width in (3, 5, 7):
                    block = torch.randn((x.shape[0], width), device=x.device, dtype=x.dtype)
                    norms = torch.linalg.norm(block, dim=-1, keepdim=True)
                    block = block / torch.clamp(norms, min=1e-8)
                    parts.append(block)
                return torch.cat(parts, dim=-1)
            return torch.randn(size=x.shape, device=x.device, dtype=x.dtype)

        if field_name == AtomicDataDict.DIPOLE_DIRECTION_KEY:
            if x.shape[-1] == 3:
                noise = torch.randn(size=x.shape, device=x.device, dtype=x.dtype)
                norms = torch.linalg.norm(noise, dim=-1, keepdim=True)
                return noise / torch.clamp(norms, min=1e-8)
            return torch.randn(size=x.shape, device=x.device, dtype=x.dtype)

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
        if center_mask_field not in data:
            return None
        return self._resolve_mask(data=data, mask_field=center_mask_field, target=target)

    def _forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        batch = self._resolve_batch(data)
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
        self._apply_directional_velocity_target_couplings(data)
        return data
