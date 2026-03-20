# geqtrain/train/_loss.py

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.train._loss import LossWrapper as GEqTrainLossWrapper
from geqtrain.utils.pytorch_scatter import scatter_sum

try:
    from geqdiff.data import AtomicDataDict as DiffAtomicDataDict
    _DEFAULT_T_SAMPLED_KEY = DiffAtomicDataDict.T_SAMPLED_KEY
    _DEFAULT_DIFFUSION_ALPHA_KEY = DiffAtomicDataDict.DIFFUSION_ALPHA_KEY
    _DEFAULT_DIFFUSION_SIGMA_KEY = DiffAtomicDataDict.DIFFUSION_SIGMA_KEY
except Exception:
    _DEFAULT_T_SAMPLED_KEY = "t_sampled"
    _DEFAULT_DIFFUSION_ALPHA_KEY = "diffusion_alpha"
    _DEFAULT_DIFFUSION_SIGMA_KEY = "diffusion_sigma"


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

    def _apply_node_filter_and_mask(
        self,
        pred_key: torch.Tensor,
        ref_key: torch.Tensor,
        mask: torch.Tensor,
        data: dict,
        key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        apply_filter = False
        if self.node_level_filter is True:
            apply_filter = True
        elif self.node_level_filter == "auto" and key in _NODE_FIELDS:
            apply_filter = True

        if not apply_filter:
            return pred_key, ref_key, mask

        num_atoms = data.get(AtomicDataDict.POSITIONS_KEY).shape[0]
        center_nodes_idx = data[AtomicDataDict.EDGE_INDEX_KEY][0].unique()
        if pred_key.shape[0] == num_atoms:
            pred_key = pred_key[center_nodes_idx]
        if ref_key.shape[0] == num_atoms:
            ref_key = ref_key[center_nodes_idx]
        if mask.shape[0] == num_atoms:
            mask = mask[center_nodes_idx]
        return pred_key, ref_key, mask

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
