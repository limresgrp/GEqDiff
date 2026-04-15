# geqtrain/train/_loss.py

from typing import Dict, Optional, Sequence, Tuple
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from e3nn import o3
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.train._loss import LossWrapper as GEqTrainLossWrapper
from geqtrain.utils.pytorch_scatter import scatter_sum

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from geqdiff.data import AtomicDataDict as DiffAtomicDataDict
    _DEFAULT_T_SAMPLED_KEY = DiffAtomicDataDict.T_SAMPLED_KEY
    _DEFAULT_DIFFUSION_ALPHA_KEY = DiffAtomicDataDict.DIFFUSION_ALPHA_KEY
    _DEFAULT_DIFFUSION_SIGMA_KEY = DiffAtomicDataDict.DIFFUSION_SIGMA_KEY
except Exception:
    _DEFAULT_T_SAMPLED_KEY = "t_sampled"
    _DEFAULT_DIFFUSION_ALPHA_KEY = "diffusion_alpha"
    _DEFAULT_DIFFUSION_SIGMA_KEY = "diffusion_sigma"

try:
    from geqdiff.utils.dipole_utils import combine_shape_irreps
    from lego.utils import decode_brick_signatures
except Exception:
    combine_shape_irreps = None
    decode_brick_signatures = None


def _normalize_feature_rows_torch(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if values.numel() == 0:
        return values
    norms = torch.linalg.norm(values, dim=-1, keepdim=True)
    safe = torch.clamp(norms, min=eps)
    normalized = values / safe
    return torch.where(norms > eps, normalized, torch.zeros_like(values))


def _combine_shape_irreps_torch(
    shape_scalars: torch.Tensor,
    shape_equivariants: torch.Tensor,
    magnitude_floor: float = 0.0,
) -> torch.Tensor:
    if shape_scalars.shape[-1] != 4:
        raise ValueError(f"Expected shape scalars with dim 4, got {shape_scalars.shape[-1]}.")
    if shape_equivariants.shape[-1] != 15:
        raise ValueError(f"Expected shape equivariants with dim 15, got {shape_equivariants.shape[-1]}.")

    l0 = shape_scalars[..., 0:1]
    l1_mag = torch.clamp(shape_scalars[..., 1:2], min=float(magnitude_floor))
    l2_mag = torch.clamp(shape_scalars[..., 2:3], min=float(magnitude_floor))
    l3_mag = torch.clamp(shape_scalars[..., 3:4], min=float(magnitude_floor))

    l1_dir = _normalize_feature_rows_torch(shape_equivariants[..., 0:3])
    l2_dir = _normalize_feature_rows_torch(shape_equivariants[..., 3:8])
    l3_dir = _normalize_feature_rows_torch(shape_equivariants[..., 8:15])

    l1 = l1_dir * l1_mag
    l2 = l2_dir * l2_mag
    l3 = l3_dir * l3_mag
    return torch.cat([l0, l1, l2, l3], dim=-1)


def _compose_shape_equiv_velocity_from_scalar(
    shape_scalar_velocity: torch.Tensor,
    shape_equiv_velocity: torch.Tensor,
    clamp_nonnegative: bool = True,
) -> torch.Tensor:
    if shape_scalar_velocity.shape[-1] < 4 or shape_equiv_velocity.shape[-1] != 15:
        return shape_equiv_velocity
    composed = shape_equiv_velocity.clone()
    block_specs = ((1, 0, 3), (2, 3, 8), (3, 8, 15))
    for scalar_idx, start, stop in block_specs:
        scale = shape_scalar_velocity[..., scalar_idx : scalar_idx + 1]
        if clamp_nonnegative:
            scale = torch.clamp(scale, min=0.0)
        composed[..., start:stop] = shape_equiv_velocity[..., start:stop] * scale
    return composed


def _compose_dipole_direction_velocity_from_strength(
    dipole_strength_velocity: torch.Tensor,
    dipole_direction_velocity: torch.Tensor,
    clamp_nonnegative: bool = True,
) -> torch.Tensor:
    if dipole_strength_velocity.shape[-1] < 1 or dipole_direction_velocity.shape[-1] != 3:
        return dipole_direction_velocity
    scale = dipole_strength_velocity[..., 0:1]
    if clamp_nonnegative:
        scale = torch.clamp(scale, min=0.0)
    return dipole_direction_velocity * scale


class DiffusionWeightedCrossEntropyLoss:
    """
    Cross-entropy with per-sample weights derived from diffusion or flow-time
    scheduler outputs.

    For flow matching, use the same keys with the interpretation:
      - alpha -> data coefficient, i.e. 1 - tau
      - sigma -> noise coefficient, i.e. tau
    """

    def __init__(
        self,
        weight_source: str = "alpha",
        weight_key: Optional[str] = None,
        weight_power: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        normalize_weights: bool = False,
        detach_weight: bool = True,
        node_level_filter: str = "auto",
        eps: float = 1e-8,
        **ce_kwargs,
    ):
        if weight_key is None:
            if weight_source == "alpha":
                weight_key = _DEFAULT_DIFFUSION_ALPHA_KEY
            elif weight_source == "sigma":
                weight_key = _DEFAULT_DIFFUSION_SIGMA_KEY
            else:
                raise ValueError(
                    f"Unknown weight_source='{weight_source}'. Expected one of: ['alpha', 'sigma'] "
                    "or pass an explicit `weight_key`."
                )
        invert_weight = (weight_source == "sigma")

        self.weight_key = weight_key
        self.invert_weight = bool(invert_weight)
        self.weight_power = float(weight_power)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_weights = bool(normalize_weights)
        self.detach_weight = bool(detach_weight)
        self.node_level_filter = node_level_filter
        self.eps = float(eps)

        self.ignore_index = int(ce_kwargs.pop("ignore_index", -100))
        self.label_smoothing = float(ce_kwargs.pop("label_smoothing", 0.0))
        class_weight = ce_kwargs.pop("weight", None)
        self.class_weight = None
        if class_weight is not None:
            if not torch.is_tensor(class_weight):
                class_weight = torch.tensor(class_weight, dtype=torch.float32)
            self.class_weight = class_weight
        if len(ce_kwargs) > 0:
            unexpected = ", ".join(sorted(ce_kwargs.keys()))
            raise ValueError(f"Unsupported CrossEntropy kwargs for DiffusionWeightedCrossEntropyLoss: {unexpected}")

    def _resolve_weight_source(self, pred: dict, ref: dict) -> torch.Tensor:
        if self.weight_key in pred:
            w = pred[self.weight_key]
        elif self.weight_key in ref:
            w = ref[self.weight_key]
        else:
            raise KeyError(
                f"Missing weight key '{self.weight_key}' in both `pred` and `ref`. "
                "Make sure the first model module stores it in the output dict."
            )
        if not torch.is_tensor(w):
            raise TypeError(f"Weight key '{self.weight_key}' must be a tensor, got {type(w)}.")
        return w

    def _expand_weights_to_nodes(
        self,
        w: torch.Tensor,
        pred: dict,
        ref: dict,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.detach_weight:
            w = w.detach()
        w = w.to(device=device, dtype=dtype)
        if w.dim() > 1 and w.shape[-1] == 1:
            w = w.squeeze(-1)
        if w.dim() > 1:
            w = w.reshape(w.shape[0], -1).mean(dim=-1)

        if w.numel() == 1:
            return w.reshape(1).expand(num_samples)
        if w.shape[0] == num_samples:
            return w.reshape(num_samples)

        batch = pred.get(AtomicDataDict.BATCH_KEY, ref.get(AtomicDataDict.BATCH_KEY, None))
        if batch is None:
            raise KeyError(
                f"Could not broadcast '{self.weight_key}' of shape {tuple(w.shape)} to {num_samples} samples: "
                f"missing '{AtomicDataDict.BATCH_KEY}'."
            )
        batch = batch.to(device=device, dtype=torch.long)
        if batch.dim() > 1:
            batch = batch.squeeze(-1)
        if batch.shape[0] != num_samples:
            raise ValueError(
                f"Batch shape {tuple(batch.shape)} is incompatible with predictions ({num_samples} samples)."
            )
        if w.shape[0] <= int(batch.max().item()):
            raise ValueError(
                f"Weight tensor '{self.weight_key}' has shape {tuple(w.shape)} but batch indices require at least "
                f"{int(batch.max().item()) + 1} graph entries."
            )
        return w[batch]

    def _apply_node_filter(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        data: dict,
        key: str,
    ):
        apply_filter = False
        if self.node_level_filter is True:
            apply_filter = True
        elif self.node_level_filter == "auto" and key in _NODE_FIELDS:
            apply_filter = True

        if not apply_filter:
            return logits, target

        num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
        center_nodes_idx = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
        if logits.shape[0] == num_atoms:
            logits = logits[center_nodes_idx]
        if target.shape[0] == num_atoms:
            target = target[center_nodes_idx]
        return logits, target

    def __call__(self, pred: dict, ref: dict, key: str, mean: bool = True, **kwargs):
        logits = pred.get(key)
        target = ref.get(key)
        assert isinstance(logits, torch.Tensor), f"Prediction for '{key}' not a tensor."
        assert isinstance(target, torch.Tensor), f"Reference for '{key}' not a tensor."

        if logits.dim() != 2:
            raise ValueError(
                f"DiffusionWeightedCrossEntropyLoss expects logits with shape [N, C], got {tuple(logits.shape)}."
            )

        if target.dim() == logits.dim():
            if target.shape[-1] == 1:
                target = target.squeeze(-1)
            elif target.shape[-1] == logits.shape[-1]:
                target = target.argmax(dim=-1)
        if target.dim() != 1:
            target = target.reshape(-1)
        target = target.to(device=logits.device, dtype=torch.long)

        if logits.shape[0] != target.shape[0]:
            raise ValueError(
                f"Logits/target mismatch for '{key}': logits={tuple(logits.shape)} target={tuple(target.shape)}."
            )

        weights_raw = self._resolve_weight_source(pred=pred, ref=ref)
        sample_weights = self._expand_weights_to_nodes(
            w=weights_raw,
            pred=pred,
            ref=ref,
            num_samples=logits.shape[0],
            device=logits.device,
            dtype=logits.dtype,
        )

        if self.invert_weight:
            sample_weights = 1.0 - sample_weights
        sample_weights = sample_weights.clamp(min=self.min_weight, max=self.max_weight)
        if self.weight_power != 1.0:
            sample_weights = sample_weights.pow(self.weight_power)

        valid_mask = target != self.ignore_index
        if self.normalize_weights and torch.any(valid_mask):
            mean_w = sample_weights[valid_mask].mean()
            sample_weights = sample_weights / mean_w.clamp(min=self.eps)

        class_weight = self.class_weight
        if class_weight is not None:
            class_weight = class_weight.to(device=logits.device, dtype=logits.dtype)
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=class_weight,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        weighted_loss = ce_loss * sample_weights

        if mean:
            if torch.any(valid_mask):
                denom = sample_weights[valid_mask].sum().clamp(min=self.eps)
                return weighted_loss[valid_mask].sum() / denom
            return weighted_loss.sum() * 0.0

        weighted_loss = weighted_loss.clone()
        weighted_loss[~valid_mask] = torch.nan
        return weighted_loss

    def __str__(self):
        return "DiffusionWeightedCrossEntropyLoss"


class MaskedLossWrapper(GEqTrainLossWrapper):
    def __init__(
        self,
        func_name: str = "MSELoss",
        mask_field: str = "ligand_mask",
        invert_mask: bool = False,
        label: Optional[str] = None,
        tau_min: Optional[float] = None,
        tau_max: Optional[float] = None,
        ignore_nan: bool = True,
        node_level_filter: str = "auto",
        **loss_kwargs,
    ):
        params = dict(loss_kwargs)
        params.setdefault("ignore_nan", ignore_nan)
        params.setdefault("node_level_filter", node_level_filter)
        super().__init__(func_name=func_name, params=params)

        self.mask_field = str(mask_field)
        self.invert_mask = bool(invert_mask)
        self.label = label
        self.tau_min = None if tau_min is None else float(tau_min)
        self.tau_max = None if tau_max is None else float(tau_max)
        if self.tau_min is not None and self.tau_max is not None and self.tau_max <= self.tau_min:
            raise ValueError("tau_max must be greater than tau_min when both are provided.")

    def _should_apply_node_filter(self, key: str) -> bool:
        if self.node_level_filter is True:
            return True
        if self.node_level_filter == "auto" and key in _NODE_FIELDS:
            return True
        return False

    def _center_nodes_idx(self, data: dict) -> torch.Tensor:
        num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
        center_nodes_idx = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
        if center_nodes_idx.numel() == 0:
            return torch.arange(num_atoms, device=data[AtomicDataDict.POSITIONS_KEY].device)
        return center_nodes_idx

    def _apply_node_filter_and_mask(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
        mask: torch.Tensor,
        data: dict,
        key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._should_apply_node_filter(key):
            return pred_key, ref_key, mask

        num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
        center_nodes_idx = self._center_nodes_idx(data)
        if pred_key.shape[0] == num_atoms:
            pred_key = pred_key[center_nodes_idx]
        if ref_key.shape[0] == num_atoms:
            ref_key = ref_key[center_nodes_idx]
        if mask.shape[0] == num_atoms:
            mask = mask[center_nodes_idx]
        return pred_key, ref_key, mask

    def _apply_node_filter_to_tensor(
        self,
        values: torch.Tensor,
        data: dict,
        key: str,
    ) -> torch.Tensor:
        if not self._should_apply_node_filter(key):
            return values
        if values.dim() == 0:
            return values
        num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
        if values.shape[0] != num_atoms:
            return values
        center_nodes_idx = self._center_nodes_idx(data)
        return values[center_nodes_idx]

    def _apply_node_filter(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
        data: dict,
        key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = self._resolve_mask(pred=data, ref=data, reference=pred_key)
        pred_key, ref_key, mask = self._apply_node_filter_and_mask(
            pred_key=pred_key,
            ref_key=ref_key,
            mask=mask,
            data=data,
            key=key,
        )
        if pred_key.dim() > 0 and pred_key.shape[0] == mask.shape[0]:
            pred_key = pred_key[mask]
        if ref_key.dim() > 0 and ref_key.shape[0] == mask.shape[0]:
            ref_key = ref_key[mask]
        return pred_key, ref_key

    def _resolve_tau_mask(self, pred: dict, ref: dict, reference: torch.Tensor) -> Optional[torch.Tensor]:
        if self.tau_min is None and self.tau_max is None:
            return None

        tau = pred.get(_DEFAULT_T_SAMPLED_KEY, ref.get(_DEFAULT_T_SAMPLED_KEY, None))
        if tau is None:
            raise KeyError(
                f"Masked loss expected flow-time field '{_DEFAULT_T_SAMPLED_KEY}' in prediction or reference dict."
            )
        if not torch.is_tensor(tau):
            raise TypeError(f"Flow-time field '{_DEFAULT_T_SAMPLED_KEY}' must be a tensor, got {type(tau)}.")

        if tau.dim() == 0:
            tau = tau.reshape(1)
        if tau.dim() > 1 and tau.shape[-1] == 1:
            tau = tau.squeeze(-1)
        elif tau.dim() > 1:
            tau = tau.reshape(tau.shape[0], -1).mean(dim=-1)

        tau = tau.to(device=reference.device, dtype=torch.float32)
        num_samples = 1 if reference.dim() == 0 else int(reference.shape[0])

        if tau.numel() == 1:
            expanded_tau = tau.reshape(1).expand(num_samples)
        elif tau.shape[0] == num_samples:
            expanded_tau = tau.reshape(num_samples)
        else:
            batch = pred.get(AtomicDataDict.BATCH_KEY, ref.get(AtomicDataDict.BATCH_KEY, None))
            if batch is None:
                raise KeyError(
                    f"Could not broadcast '{_DEFAULT_T_SAMPLED_KEY}' of shape {tuple(tau.shape)} to "
                    f"{num_samples} samples: missing '{AtomicDataDict.BATCH_KEY}'."
                )
            batch = batch.to(device=reference.device, dtype=torch.long)
            if batch.dim() > 1:
                batch = batch.squeeze(-1)
            if batch.shape[0] != num_samples:
                raise ValueError(
                    f"Batch shape {tuple(batch.shape)} is incompatible with reference shape {tuple(reference.shape)}."
                )
            if tau.shape[0] <= int(batch.max().item()):
                raise ValueError(
                    f"Flow-time tensor '{_DEFAULT_T_SAMPLED_KEY}' has shape {tuple(tau.shape)} but batch indices "
                    f"require at least {int(batch.max().item()) + 1} graph entries."
                )
            expanded_tau = tau[batch]

        tau_mask = torch.ones((num_samples,), device=reference.device, dtype=torch.bool)
        if self.tau_min is not None:
            tau_mask = tau_mask & (expanded_tau >= float(self.tau_min))
        if self.tau_max is not None:
            if self.tau_max >= 1.0:
                tau_mask = tau_mask & (expanded_tau <= float(self.tau_max))
            else:
                tau_mask = tau_mask & (expanded_tau < float(self.tau_max))
        return tau_mask

    def _resolve_mask(self, pred: dict, ref: dict, reference: torch.Tensor) -> torch.Tensor:
        mask = pred.get(self.mask_field, ref.get(self.mask_field, None))
        if mask is None:
            raise KeyError(
                f"Masked loss expected mask field '{self.mask_field}' in prediction or reference dict."
            )
        if not torch.is_tensor(mask):
            raise TypeError(f"Mask field '{self.mask_field}' must be a tensor, got {type(mask)}.")
        if mask.dim() > 1 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        mask = mask.to(device=reference.device) > 0.5
        if self.invert_mask:
            mask = ~mask
        tau_mask = self._resolve_tau_mask(pred=pred, ref=ref, reference=reference)
        if tau_mask is not None:
            if tau_mask.shape[0] != mask.shape[0]:
                raise ValueError(
                    f"Tau-mask shape {tuple(tau_mask.shape)} is incompatible with node mask shape {tuple(mask.shape)}."
                )
            mask = mask & tau_mask
        return mask

    def _broadcast_mask(self, mask: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        expanded = mask
        while expanded.dim() < loss.dim():
            expanded = expanded.unsqueeze(-1)
        return expanded

    def _apply_feature_slice(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return pred_key, ref_key

    def _select_masked_tensors(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool,
        normalization_fields: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_key_name = self._get_pred_key_name(key)
        pred_key, ref_key = self._prepare_tensors(
            pred=pred,
            ref=ref,
            pred_key_name=pred_key_name,
            ref_key_name=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )
        self._initialize_supervision_weights(pred_key.device, pred_key.dtype)
        ref_key = self._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)
        mask = self._resolve_mask(pred=pred, ref=ref, reference=pred_key)
        pred_key, ref_key, mask = self._apply_node_filter_and_mask(
            pred_key=pred_key,
            ref_key=ref_key,
            mask=mask,
            data=pred,
            key=key,
        )
        pred_key, ref_key = self._apply_feature_slice(pred_key=pred_key, ref_key=ref_key)
        if pred_key.dim() > 0 and pred_key.shape[0] == mask.shape[0]:
            pred_key = pred_key[mask]
        if ref_key.dim() > 0 and ref_key.shape[0] == mask.shape[0]:
            ref_key = ref_key[mask]
        return pred_key, ref_key

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        pred_key, ref_key = self._select_masked_tensors(
            pred=pred,
            ref=ref,
            key=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )

        if pred_key.numel() == 0 or ref_key.numel() == 0:
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_empty(pred_key.shape)

        return self._calculate_loss(pred_key, ref_key, mean)

    def __str__(self):
        if self.label:
            return str(self.label)
        return f"Masked{self.func_name}"


class MaskedMSELoss(MaskedLossWrapper):
    def __init__(self, **kwargs):
        super().__init__(func_name="MSELoss", **kwargs)


class MaskedFeatureSliceMSELoss(MaskedLossWrapper):
    def __init__(
        self,
        slice_start: int,
        slice_stop: int,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        self.slice_start = int(slice_start)
        self.slice_stop = int(slice_stop)
        if self.slice_stop <= self.slice_start:
            raise ValueError("slice_stop must be greater than slice_start.")
        self.label = label

    def _apply_feature_slice(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slicer = [slice(None)] * pred_key.dim()
        slicer[-1] = slice(self.slice_start, self.slice_stop)
        return pred_key[tuple(slicer)], ref_key[tuple(slicer)]

    def __str__(self):
        if self.label:
            return self.label
        return f"MaskedSliceMSE_{self.slice_start}_{self.slice_stop}"


class MaskedWeightedMSELoss(MaskedLossWrapper):
    def __init__(
        self,
        weight_field: str,
        weight_source: str = "auto",
        normalize_weights: bool = False,
        eps: float = 1e-8,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        if weight_source not in {"auto", "pred", "ref"}:
            raise ValueError("weight_source must be one of: 'auto', 'pred', 'ref'.")
        self.weight_field = str(weight_field)
        self.weight_source = str(weight_source)
        self.normalize_weights = bool(normalize_weights)
        self.eps = float(eps)
        self.label = label

    def _resolve_weight_tensor(
        self,
        pred: dict,
        ref: dict,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.weight_source == "pred":
            weight = pred.get(self.weight_field, None)
        elif self.weight_source == "ref":
            weight = ref.get(self.weight_field, None)
        else:
            weight = pred.get(self.weight_field, ref.get(self.weight_field, None))
        if weight is None:
            raise KeyError(
                f"Weighted masked loss expected weight field '{self.weight_field}' in prediction or reference dict."
            )
        if not torch.is_tensor(weight):
            raise TypeError(f"Weight field '{self.weight_field}' must be a tensor, got {type(weight)}.")
        if weight.dim() > 1 and weight.shape[-1] == 1:
            weight = weight.squeeze(-1)
        weight = weight.to(device=reference.device, dtype=reference.dtype)
        return weight

    def _select_masked_tensors_and_weights(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool,
        normalization_fields: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_key_name = self._get_pred_key_name(key)
        pred_key, ref_key = self._prepare_tensors(
            pred=pred,
            ref=ref,
            pred_key_name=pred_key_name,
            ref_key_name=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )
        self._initialize_supervision_weights(pred_key.device, pred_key.dtype)
        ref_key = self._handle_supervision_shapes(pred_key, ref_key, pred_key_name, key)
        mask = self._resolve_mask(pred=pred, ref=ref, reference=pred_key)
        weights = self._resolve_weight_tensor(pred=pred, ref=ref, reference=pred_key)

        pred_key, ref_key, mask = self._apply_node_filter_and_mask(
            pred_key=pred_key,
            ref_key=ref_key,
            mask=mask,
            data=pred,
            key=key,
        )
        weights = self._apply_node_filter_to_tensor(weights, data=pred, key=key)

        pred_key, ref_key = self._apply_feature_slice(pred_key=pred_key, ref_key=ref_key)

        if weights.dim() > 1:
            weights = weights.reshape(weights.shape[0], -1).mean(dim=-1)
        if weights.dim() == 0:
            weights = weights.reshape(1)
        if weights.shape[0] == mask.shape[0]:
            weights = weights[mask]

        if pred_key.dim() > 0 and pred_key.shape[0] == mask.shape[0]:
            pred_key = pred_key[mask]
        if ref_key.dim() > 0 and ref_key.shape[0] == mask.shape[0]:
            ref_key = ref_key[mask]
        return pred_key, ref_key, weights

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        pred_key, ref_key, weights = self._select_masked_tensors_and_weights(
            pred=pred,
            ref=ref,
            key=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )

        if pred_key.numel() == 0 or ref_key.numel() == 0 or weights.numel() == 0:
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_empty((0,), dtype=pred_key.dtype)

        per_entry = F.mse_loss(pred_key, ref_key, reduction="none")
        per_sample = per_entry.reshape(per_entry.shape[0], -1).mean(dim=-1)
        weights = torch.clamp(weights.reshape(-1), min=0.0)
        finite = torch.isfinite(per_sample) & torch.isfinite(weights)

        if self.normalize_weights and torch.any(finite & (weights > 0.0)):
            mean_w = weights[finite & (weights > 0.0)].mean()
            weights = weights / torch.clamp(mean_w, min=self.eps)

        valid = finite & (weights > 0.0)
        if not torch.any(valid):
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_full((per_sample.shape[0],), torch.nan, dtype=pred_key.dtype)

        weighted = per_sample[valid] * weights[valid]
        if mean:
            return weighted.sum() / torch.clamp(weights[valid].sum(), min=self.eps)

        out = pred_key.new_full((per_sample.shape[0],), torch.nan, dtype=pred_key.dtype)
        out[valid] = weighted
        return out

    def __str__(self):
        if self.label:
            return self.label
        return f"MaskedWeightedMSE_{self.weight_field}"


class MaskedFeatureSliceWeightedMSELoss(MaskedWeightedMSELoss):
    def __init__(
        self,
        slice_start: int,
        slice_stop: int,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(label=label, **kwargs)
        self.slice_start = int(slice_start)
        self.slice_stop = int(slice_stop)
        if self.slice_stop <= self.slice_start:
            raise ValueError("slice_stop must be greater than slice_start.")
        self.label = label

    def _apply_feature_slice(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        slicer = [slice(None)] * pred_key.dim()
        slicer[-1] = slice(self.slice_start, self.slice_stop)
        return pred_key[tuple(slicer)], ref_key[tuple(slicer)]

    def __str__(self):
        if self.label:
            return self.label
        return f"MaskedWeightedSliceMSE_{self.slice_start}_{self.slice_stop}"


class MaskedFieldMean(MaskedLossWrapper):
    def __init__(
        self,
        field: str,
        field_source: str = "auto",
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        if field_source not in {"auto", "pred", "ref"}:
            raise ValueError("field_source must be one of: 'auto', 'pred', 'ref'.")
        self.field = str(field)
        self.field_source = str(field_source)
        self.label = label

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        reference = pred.get(key, ref.get(key))
        if not torch.is_tensor(reference):
            reference = pred.get(AtomicDataDict.POSITIONS_KEY, ref.get(AtomicDataDict.POSITIONS_KEY))
        if not torch.is_tensor(reference):
            raise KeyError(f"Could not resolve a tensor reference for MaskedFieldMean on key '{key}'.")

        if self.field_source == "pred":
            values = pred.get(self.field, None)
        elif self.field_source == "ref":
            values = ref.get(self.field, None)
        else:
            values = pred.get(self.field, ref.get(self.field, None))
        if values is None:
            raise KeyError(f"MaskedFieldMean expected field '{self.field}' in prediction or reference dict.")
        if not torch.is_tensor(values):
            raise TypeError(f"Field '{self.field}' must be a tensor, got {type(values)}.")

        mask = self._resolve_mask(pred=pred, ref=ref, reference=reference)
        values = values.to(device=reference.device, dtype=reference.dtype)
        if values.dim() > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        values = self._apply_node_filter_to_tensor(values, data=pred, key=key)
        mask = self._apply_node_filter_to_tensor(mask, data=pred, key=key)

        if values.dim() == 0:
            values = values.reshape(1)
        if values.shape[0] == mask.shape[0]:
            values = values[mask]
        values = values.reshape(values.shape[0], -1).mean(dim=-1)
        finite = torch.isfinite(values)
        if mean:
            if torch.any(finite):
                return values[finite].mean()
            return torch.zeros((), device=values.device, dtype=values.dtype)
        out = values.clone()
        out[~finite] = torch.nan
        return out

    def __str__(self):
        if self.label:
            return self.label
        return f"MaskedFieldMean_{self.field}"


class MaskedShapeAwareClashLoss(MaskedLossWrapper):
    def __init__(
        self,
        pair_cutoff: float = 3.0,
        overlap_margin: float = 0.05,
        support_min_extent: float = 0.45,
        support_extent_scale: float = 0.75,
        support_alignment_temperature: float = 8.0,
        clash_sharpness: float = 10.0,
        magnitude_floor: float = 0.0,
        include_ligand_ligand: bool = True,
        include_ligand_pocket: bool = True,
        compose_directional_velocity: bool = False,
        compose_nonnegative_scale: bool = True,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        self.pair_cutoff = float(pair_cutoff)
        self.overlap_margin = float(overlap_margin)
        self.support_min_extent = float(support_min_extent)
        self.support_extent_scale = float(support_extent_scale)
        self.support_alignment_temperature = float(support_alignment_temperature)
        self.clash_sharpness = float(clash_sharpness)
        self.magnitude_floor = float(magnitude_floor)
        self.include_ligand_ligand = bool(include_ligand_ligand)
        self.include_ligand_pocket = bool(include_ligand_pocket)
        self.compose_directional_velocity = bool(compose_directional_velocity)
        self.compose_nonnegative_scale = bool(compose_nonnegative_scale)
        self.label = label
        if self.pair_cutoff <= 0.0:
            raise ValueError("pair_cutoff must be > 0.")
        if self.clash_sharpness <= 0.0:
            raise ValueError("clash_sharpness must be > 0.")
        if self.support_alignment_temperature <= 0.0:
            raise ValueError("support_alignment_temperature must be > 0.")
        if not (self.include_ligand_ligand or self.include_ligand_pocket):
            raise ValueError("At least one of include_ligand_ligand or include_ligand_pocket must be true.")

    @staticmethod
    def _resolve_tensor(pred: dict, ref: dict, field: str) -> Optional[torch.Tensor]:
        tensor = pred.get(field, ref.get(field, None))
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            raise TypeError(f"Field '{field}' must be a tensor, got {type(tensor)}.")
        return tensor

    @staticmethod
    def _expand_graph_tensor(values: torch.Tensor, batch: torch.Tensor, target_dim: int) -> torch.Tensor:
        if values.dim() > 1 and values.shape[-1] == 1 and target_dim == 1:
            values = values.squeeze(-1)
        if values.dim() == 0:
            values = values.reshape(1)
        if values.shape[0] == batch.shape[0]:
            return values
        if values.shape[0] <= int(batch.max().item()):
            raise ValueError(
                f"Graph tensor shape {tuple(values.shape)} is incompatible with batch indices requiring "
                f"{int(batch.max().item()) + 1} entries."
            )
        return values[batch]

    @staticmethod
    def _support_sample_directions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0],
                [0.0, -1.0, 1.0],
                [0.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            device=device,
            dtype=dtype,
        )
        return _normalize_feature_rows_torch(directions)

    def _directional_extent(self, shape_coeffs: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
        basis = o3.spherical_harmonics(
            list(range(4)),
            directions.to(dtype=shape_coeffs.dtype),
            normalize=True,
            normalization="integral",
        )
        support_raw = torch.sum(shape_coeffs * basis, dim=-1)
        return self.support_min_extent + self.support_extent_scale * torch.sigmoid(support_raw)

    def _pair_penalties_for_graph(
        self,
        pos: torch.Tensor,
        shape_coeffs: torch.Tensor,
        active_mask: torch.Tensor,
        ligand_mask: torch.Tensor,
        pocket_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(pos.shape[0])
        if num_nodes < 2 or not torch.any(active_mask):
            return pos.new_empty((0,))

        idx_i, idx_j = torch.triu_indices(num_nodes, num_nodes, offset=1, device=pos.device)
        if idx_i.numel() == 0:
            return pos.new_empty((0,))

        lig_i = ligand_mask[idx_i]
        lig_j = ligand_mask[idx_j]
        poc_i = pocket_mask[idx_i]
        poc_j = pocket_mask[idx_j]

        allowed = torch.zeros_like(lig_i, dtype=torch.bool)
        if self.include_ligand_ligand:
            allowed = allowed | (lig_i & lig_j)
        if self.include_ligand_pocket:
            allowed = allowed | ((lig_i & poc_j) | (poc_i & lig_j))

        active_pair = active_mask[idx_i] | active_mask[idx_j]
        pair_mask = allowed & active_pair
        if not torch.any(pair_mask):
            return pos.new_empty((0,))

        idx_i = idx_i[pair_mask]
        idx_j = idx_j[pair_mask]

        deltas = pos[idx_j] - pos[idx_i]
        distances = torch.linalg.norm(deltas, dim=-1)
        valid = torch.isfinite(distances) & (distances > 1e-6) & (distances < self.pair_cutoff)
        if not torch.any(valid):
            return pos.new_empty((0,))

        idx_i = idx_i[valid]
        idx_j = idx_j[valid]
        deltas = deltas[valid]
        distances = distances[valid]
        directions = deltas / distances.unsqueeze(-1)

        extent_i = self._directional_extent(shape_coeffs[idx_i], directions)
        extent_j = self._directional_extent(shape_coeffs[idx_j], -directions)
        overlap = extent_i + extent_j + self.overlap_margin - distances
        # Harmonic hinge penalty: only overlapping pairs contribute.
        overlap_violation = torch.relu(overlap)
        penalties = 0.5 * self.clash_sharpness * overlap_violation.square()
        finite = torch.isfinite(penalties)
        if not torch.any(finite):
            return pos.new_empty((0,))
        return penalties[finite]

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        pos_t = self._resolve_tensor(pred, ref, AtomicDataDict.POSITIONS_KEY)
        velocity = self._resolve_tensor(pred, ref, "velocity")
        shape_scalar_t = self._resolve_tensor(pred, ref, "shape_scalar_features")
        shape_equiv_t = self._resolve_tensor(pred, ref, "shape_equiv_features")
        shape_scalar_velocity = self._resolve_tensor(pred, ref, "shape_scalar_velocity")
        shape_equiv_velocity = self._resolve_tensor(pred, ref, "shape_equiv_velocity")
        sigma = self._resolve_tensor(pred, ref, _DEFAULT_DIFFUSION_SIGMA_KEY)
        batch = self._resolve_tensor(pred, ref, AtomicDataDict.BATCH_KEY)
        ligand_mask = self._resolve_tensor(pred, ref, "ligand_mask")
        pocket_mask = self._resolve_tensor(pred, ref, "pocket_mask")

        required = {
            "pos": pos_t,
            "velocity": velocity,
            "shape_scalar_features": shape_scalar_t,
            "shape_equiv_features": shape_equiv_t,
            "shape_scalar_velocity": shape_scalar_velocity,
            "shape_equiv_velocity": shape_equiv_velocity,
            "sigma": sigma,
            "batch": batch,
            "ligand_mask": ligand_mask,
            "pocket_mask": pocket_mask,
        }
        missing = [name for name, value in required.items() if value is None]
        if len(missing) > 0:
            raise KeyError(f"MaskedShapeAwareClashLoss is missing required fields: {missing}")

        reference = pos_t
        active_mask = self._resolve_mask(pred=pred, ref=ref, reference=reference)

        pos_t = pos_t.to(dtype=torch.float32)
        velocity = velocity.to(device=pos_t.device, dtype=pos_t.dtype)
        shape_scalar_t = shape_scalar_t.to(device=pos_t.device, dtype=pos_t.dtype)
        shape_equiv_t = shape_equiv_t.to(device=pos_t.device, dtype=pos_t.dtype)
        shape_scalar_velocity = shape_scalar_velocity.to(device=pos_t.device, dtype=pos_t.dtype)
        shape_equiv_velocity = shape_equiv_velocity.to(device=pos_t.device, dtype=pos_t.dtype)
        batch = batch.to(device=pos_t.device, dtype=torch.long)
        if batch.dim() > 1 and batch.shape[-1] == 1:
            batch = batch.squeeze(-1)
        ligand_mask = ligand_mask.to(device=pos_t.device) > 0.5
        pocket_mask = pocket_mask.to(device=pos_t.device) > 0.5
        if ligand_mask.dim() > 1 and ligand_mask.shape[-1] == 1:
            ligand_mask = ligand_mask.squeeze(-1)
        if pocket_mask.dim() > 1 and pocket_mask.shape[-1] == 1:
            pocket_mask = pocket_mask.squeeze(-1)

        sigma = sigma.to(device=pos_t.device, dtype=pos_t.dtype)
        sigma = self._expand_graph_tensor(sigma, batch=batch, target_dim=1).to(dtype=pos_t.dtype).reshape(-1, 1)

        if self.compose_directional_velocity:
            shape_equiv_velocity = _compose_shape_equiv_velocity_from_scalar(
                shape_scalar_velocity,
                shape_equiv_velocity,
                clamp_nonnegative=self.compose_nonnegative_scale,
            )

        pred_pos_0 = pos_t - sigma * velocity
        pred_shape_scalar = shape_scalar_t - sigma * shape_scalar_velocity
        pred_shape_equiv = shape_equiv_t - sigma * shape_equiv_velocity
        pred_shape_coeffs = _combine_shape_irreps_torch(
            pred_shape_scalar,
            pred_shape_equiv,
            magnitude_floor=self.magnitude_floor,
        )

        penalties = []
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for graph_index in range(num_graphs):
            graph_mask = batch == graph_index
            if not torch.any(graph_mask):
                continue
            graph_penalties = self._pair_penalties_for_graph(
                pos=pred_pos_0[graph_mask],
                shape_coeffs=pred_shape_coeffs[graph_mask],
                active_mask=active_mask[graph_mask],
                ligand_mask=ligand_mask[graph_mask],
                pocket_mask=pocket_mask[graph_mask],
            )
            if graph_penalties.numel() > 0:
                penalties.append(graph_penalties)

        if len(penalties) == 0:
            if mean:
                return torch.zeros((), device=pos_t.device, dtype=pos_t.dtype)
            return pos_t.new_empty((0,), dtype=pos_t.dtype)

        values = torch.cat(penalties, dim=0)
        finite = torch.isfinite(values)
        if mean:
            if torch.any(finite):
                return values[finite].mean()
            return torch.zeros((), device=values.device, dtype=values.dtype)
        out = values.clone()
        out[~finite] = torch.nan
        return out

    def __str__(self):
        if self.label:
            return self.label
        return "MaskedShapeAwareClashLoss"


class MaskedBrickLibraryMetric(MaskedLossWrapper):
    def __init__(
        self,
        metric: str = "distance",
        type_names: Optional[Sequence[str]] = None,
        compose_directional_velocity: bool = False,
        compose_nonnegative_scale: bool = True,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        if metric not in {"distance", "type_accuracy"}:
            raise ValueError("metric must be one of: 'distance', 'type_accuracy'.")
        self.metric = str(metric)
        self.type_names = None if type_names is None else [str(name) for name in type_names]
        self.compose_directional_velocity = bool(compose_directional_velocity)
        self.compose_nonnegative_scale = bool(compose_nonnegative_scale)
        self.label = label
        self._warned_missing_decode_inputs = False

    @staticmethod
    def _resolve_tensor(pred: dict, ref: dict, field: str) -> Optional[torch.Tensor]:
        tensor = pred.get(field, ref.get(field, None))
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            raise TypeError(f"Field '{field}' must be a tensor, got {type(tensor)}.")
        return tensor

    @staticmethod
    def _expand_graph_tensor(values: torch.Tensor, batch: torch.Tensor, target_dim: int) -> torch.Tensor:
        if values.dim() > 1 and values.shape[-1] == 1 and target_dim == 1:
            values = values.squeeze(-1)
        if values.dim() == 0:
            values = values.reshape(1)
        if values.shape[0] == batch.shape[0]:
            return values
        if values.shape[0] <= int(batch.max().item()):
            raise ValueError(
                f"Graph tensor shape {tuple(values.shape)} is incompatible with batch indices requiring "
                f"{int(batch.max().item()) + 1} entries."
            )
        return values[batch]

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        if combine_shape_irreps is None or decode_brick_signatures is None:
            raise ImportError("LEGO decode helpers are unavailable for MaskedBrickLibraryMetric.")

        reference = self._resolve_tensor(pred, ref, "shape_equiv_velocity")
        if reference is None:
            raise KeyError("MaskedBrickLibraryMetric expected field 'shape_equiv_velocity' in prediction or reference dict.")
        mask = self._resolve_mask(pred=pred, ref=ref, reference=reference)
        batch = self._resolve_tensor(pred, ref, AtomicDataDict.BATCH_KEY)
        sigma = self._resolve_tensor(pred, ref, _DEFAULT_DIFFUSION_SIGMA_KEY)
        shape_scalar_t = self._resolve_tensor(pred, ref, "shape_scalar_features")
        shape_equiv_t = self._resolve_tensor(pred, ref, "shape_equiv_features")
        shape_scalar_velocity = self._resolve_tensor(pred, ref, "shape_scalar_velocity")
        shape_equiv_velocity = self._resolve_tensor(pred, ref, "shape_equiv_velocity")
        node_types_true = self._resolve_tensor(pred, ref, "node_types_true")
        shape_scalar_norm_means = self._resolve_tensor(pred, ref, "shape_scalar_norm_means")
        shape_scalar_norm_stds = self._resolve_tensor(pred, ref, "shape_scalar_norm_stds")

        required = {
            "batch": batch,
            "sigma": sigma,
            "shape_scalar_features": shape_scalar_t,
            "shape_equiv_features": shape_equiv_t,
            "shape_scalar_velocity": shape_scalar_velocity,
            "shape_equiv_velocity": shape_equiv_velocity,
            "node_types_true": node_types_true,
            "shape_scalar_norm_means": shape_scalar_norm_means,
            "shape_scalar_norm_stds": shape_scalar_norm_stds,
        }
        missing = [name for name, value in required.items() if value is None]
        if len(missing) > 0:
            if not self._warned_missing_decode_inputs:
                print(
                    "MaskedBrickLibraryMetric: skipping decode metric because the batch is missing "
                    f"{missing}. Rebuild diffusion datasets to schema >= 4 to enable this metric."
                )
                self._warned_missing_decode_inputs = True
            if mean:
                return torch.zeros((), device=reference.device, dtype=reference.dtype)
            return reference.new_full((0,), torch.nan, dtype=reference.dtype)

        batch = batch.to(device=reference.device, dtype=torch.long)
        sigma = sigma.to(device=reference.device, dtype=reference.dtype)
        shape_scalar_t = shape_scalar_t.to(device=reference.device, dtype=reference.dtype)
        shape_equiv_t = shape_equiv_t.to(device=reference.device, dtype=reference.dtype)
        shape_scalar_velocity = shape_scalar_velocity.to(device=reference.device, dtype=reference.dtype)
        shape_equiv_velocity = shape_equiv_velocity.to(device=reference.device, dtype=reference.dtype)
        node_types_true = node_types_true.to(device=reference.device, dtype=torch.long)
        shape_scalar_norm_means = shape_scalar_norm_means.to(device=reference.device, dtype=reference.dtype)
        shape_scalar_norm_stds = shape_scalar_norm_stds.to(device=reference.device, dtype=reference.dtype)

        shape_scalar_t, shape_equiv_t, mask = self._apply_node_filter_and_mask(
            pred_key=shape_scalar_t,
            ref_key=shape_equiv_t,
            mask=mask,
            data=pred,
            key=key,
        )
        shape_scalar_velocity = self._apply_node_filter_to_tensor(shape_scalar_velocity, data=pred, key=key)
        shape_equiv_velocity = self._apply_node_filter_to_tensor(shape_equiv_velocity, data=pred, key=key)
        node_types_true = self._apply_node_filter_to_tensor(node_types_true, data=pred, key=key)
        batch = self._apply_node_filter_to_tensor(batch, data=pred, key=key)

        sigma = self._expand_graph_tensor(sigma, batch=batch, target_dim=1).to(dtype=reference.dtype)
        shape_scalar_norm_means = self._expand_graph_tensor(shape_scalar_norm_means, batch=batch, target_dim=4).to(dtype=reference.dtype)
        shape_scalar_norm_stds = self._expand_graph_tensor(shape_scalar_norm_stds, batch=batch, target_dim=4).to(dtype=reference.dtype)

        if shape_scalar_velocity.shape[0] == mask.shape[0]:
            shape_scalar_velocity = shape_scalar_velocity[mask]
        if shape_equiv_velocity.shape[0] == mask.shape[0]:
            shape_equiv_velocity = shape_equiv_velocity[mask]
        if node_types_true.shape[0] == mask.shape[0]:
            node_types_true = node_types_true[mask]
        if sigma.shape[0] == mask.shape[0]:
            sigma = sigma[mask]
        if shape_scalar_norm_means.shape[0] == mask.shape[0]:
            shape_scalar_norm_means = shape_scalar_norm_means[mask]
        if shape_scalar_norm_stds.shape[0] == mask.shape[0]:
            shape_scalar_norm_stds = shape_scalar_norm_stds[mask]
        if shape_scalar_t.shape[0] == mask.shape[0]:
            shape_scalar_t = shape_scalar_t[mask]
        if shape_equiv_t.shape[0] == mask.shape[0]:
            shape_equiv_t = shape_equiv_t[mask]

        if shape_scalar_t.numel() == 0:
            if mean:
                return torch.zeros((), device=reference.device, dtype=reference.dtype)
            return reference.new_empty((0,), dtype=reference.dtype)

        if self.compose_directional_velocity:
            shape_equiv_velocity = _compose_shape_equiv_velocity_from_scalar(
                shape_scalar_velocity,
                shape_equiv_velocity,
                clamp_nonnegative=self.compose_nonnegative_scale,
            )

        sigma = sigma.reshape(-1, 1)
        pred_shape_scalar_norm = shape_scalar_t - sigma * shape_scalar_velocity
        pred_shape_equiv = shape_equiv_t - sigma * shape_equiv_velocity
        pred_shape_scalar_raw = shape_scalar_norm_means + shape_scalar_norm_stds * pred_shape_scalar_norm

        pred_shape = combine_shape_irreps(
            pred_shape_scalar_raw.detach().cpu().numpy().astype(np.float32),
            pred_shape_equiv.detach().cpu().numpy().astype(np.float32),
        )
        decoded = decode_brick_signatures(pred_shape)

        if self.metric == "distance":
            values = torch.as_tensor(decoded["distances"], device=reference.device, dtype=reference.dtype)
        else:
            if self.type_names is None:
                raise ValueError("type_names must be provided for metric='type_accuracy'.")
            true_indices = node_types_true.reshape(-1).detach().cpu().numpy().astype(np.int64)
            true_names = np.asarray([self.type_names[int(index)] for index in true_indices])
            correct = (np.asarray(decoded["brick_types"]) == true_names).astype(np.float32)
            values = torch.as_tensor(correct, device=reference.device, dtype=reference.dtype)

        finite = torch.isfinite(values)
        if mean:
            if torch.any(finite):
                return values[finite].mean()
            return torch.zeros((), device=values.device, dtype=values.dtype)
        out = values.clone()
        out[~finite] = torch.nan
        return out

    def __str__(self):
        if self.label:
            return self.label
        return f"MaskedBrickLibraryMetric_{self.metric}"


class MaskedNormMetric(MaskedLossWrapper):
    def __init__(
        self,
        source: str = "pred",
        slice_start: Optional[int] = None,
        slice_stop: Optional[int] = None,
        label: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        if source not in {"pred", "ref", "error"}:
            raise ValueError("source must be one of: 'pred', 'ref', 'error'.")
        self.source = str(source)
        self.slice_start = None if slice_start is None else int(slice_start)
        self.slice_stop = None if slice_stop is None else int(slice_stop)
        if (self.slice_start is None) != (self.slice_stop is None):
            raise ValueError("slice_start and slice_stop must either both be set or both be None.")
        if self.slice_start is not None and self.slice_stop <= self.slice_start:
            raise ValueError("slice_stop must be greater than slice_start.")
        self.label = label

    def _apply_feature_slice(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.slice_start is None:
            return pred_key, ref_key
        slicer = [slice(None)] * pred_key.dim()
        slicer[-1] = slice(self.slice_start, self.slice_stop)
        return pred_key[tuple(slicer)], ref_key[tuple(slicer)]

    @staticmethod
    def _rowwise_norm(values: torch.Tensor) -> torch.Tensor:
        if values.dim() == 0:
            return values.reshape(1).abs()
        if values.dim() == 1:
            return values.abs()
        return torch.linalg.vector_norm(values.reshape(values.shape[0], -1), dim=-1)

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        pred_key, ref_key = self._select_masked_tensors(
            pred=pred,
            ref=ref,
            key=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )
        if pred_key.numel() == 0 or ref_key.numel() == 0:
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_empty((0,), dtype=pred_key.dtype)

        if self.source == "pred":
            values = pred_key
        elif self.source == "ref":
            values = ref_key
        else:
            values = pred_key - ref_key

        norms = self._rowwise_norm(values)
        finite = torch.isfinite(norms)
        if mean:
            if torch.any(finite):
                return norms[finite].mean()
            return torch.zeros((), device=norms.device, dtype=norms.dtype)
        out = norms.clone()
        out[~finite] = torch.nan
        return out

    def __str__(self):
        if self.label:
            return self.label
        prefix = {"pred": "PredNorm", "ref": "TargetNorm", "error": "ErrorNorm"}[self.source]
        if self.slice_start is None:
            return prefix
        return f"{prefix}_{self.slice_start}_{self.slice_stop}"


class MaskedAngularError(MaskedLossWrapper):
    def __init__(
        self,
        slice_start: Optional[int] = None,
        slice_stop: Optional[int] = None,
        label: Optional[str] = None,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(func_name="MSELoss", **kwargs)
        self.slice_start = None if slice_start is None else int(slice_start)
        self.slice_stop = None if slice_stop is None else int(slice_stop)
        if (self.slice_start is None) != (self.slice_stop is None):
            raise ValueError("slice_start and slice_stop must either both be set or both be None.")
        if self.slice_start is not None and self.slice_stop <= self.slice_start:
            raise ValueError("slice_stop must be greater than slice_start.")
        self.label = label
        self.eps = float(eps)

    def _apply_feature_slice(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.slice_start is None:
            return pred_key, ref_key
        slicer = [slice(None)] * pred_key.dim()
        slicer[-1] = slice(self.slice_start, self.slice_stop)
        return pred_key[tuple(slicer)], ref_key[tuple(slicer)]

    def __call__(
        self,
        pred: dict,
        ref: dict,
        key: str,
        mean: bool = True,
        normalization_fields: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        pred_key, ref_key = self._select_masked_tensors(
            pred=pred,
            ref=ref,
            key=key,
            mean=mean,
            normalization_fields=normalization_fields,
        )
        if pred_key.numel() == 0 or ref_key.numel() == 0:
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_empty((0,), dtype=pred_key.dtype)

        flat_pred = pred_key.reshape(pred_key.shape[0], -1) if pred_key.dim() > 1 else pred_key.unsqueeze(-1)
        flat_ref = ref_key.reshape(ref_key.shape[0], -1) if ref_key.dim() > 1 else ref_key.unsqueeze(-1)
        pred_norm = torch.linalg.vector_norm(flat_pred, dim=-1)
        ref_norm = torch.linalg.vector_norm(flat_ref, dim=-1)
        valid = (
            torch.isfinite(flat_pred).all(dim=-1)
            & torch.isfinite(flat_ref).all(dim=-1)
            & (pred_norm > self.eps)
            & (ref_norm > self.eps)
        )
        if not torch.any(valid):
            if mean:
                return torch.zeros((), device=pred_key.device, dtype=pred_key.dtype)
            return pred_key.new_full((flat_pred.shape[0],), torch.nan, dtype=pred_key.dtype)

        cosine = torch.sum(flat_pred[valid] * flat_ref[valid], dim=-1) / (pred_norm[valid] * ref_norm[valid])
        cosine = cosine.clamp(min=-1.0, max=1.0)
        angles = torch.rad2deg(torch.acos(cosine))

        if mean:
            return angles.mean()
        out = pred_key.new_full((flat_pred.shape[0],), torch.nan, dtype=pred_key.dtype)
        out[valid] = angles
        return out

    def __str__(self):
        if self.label:
            return self.label
        if self.slice_start is None:
            return "AngularError"
        return f"AngularError_{self.slice_start}_{self.slice_stop}"
