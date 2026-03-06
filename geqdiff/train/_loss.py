# geqtrain/train/_loss.py

from typing import Optional
import torch
import torch.nn.functional as F
from geqtrain.data import AtomicDataDict, _NODE_FIELDS
from geqtrain.utils.pytorch_scatter import scatter_sum

try:
    from geqdiff.data import AtomicDataDict as DiffAtomicDataDict
    _DEFAULT_DIFFUSION_ALPHA_KEY = DiffAtomicDataDict.DIFFUSION_ALPHA_KEY
    _DEFAULT_DIFFUSION_SIGMA_KEY = DiffAtomicDataDict.DIFFUSION_SIGMA_KEY
except Exception:
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
