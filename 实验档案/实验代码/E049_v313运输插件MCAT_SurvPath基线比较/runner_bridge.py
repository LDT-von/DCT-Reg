"""Translate the single-patient native adapters to train_runner's batch contract."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class NativeRunnerBridge(nn.Module):
    """Return logits and auxiliary loss as expected by the shared NLL trainer.

    MCAT emits hazards whereas SurvPath emits logits. Both arms of each
    baseline use the same loss, data loader and checkpoint selection protocol.
    """

    def __init__(self, adapter: nn.Module, baseline: str, omic_sizes: list[int]):
        super().__init__()
        if baseline not in {"mcat", "survpath"}:
            raise ValueError(f"Unsupported baseline: {baseline}")
        self.adapter = adapter
        self.baseline = baseline
        self.omic_sizes = tuple(int(size) for size in omic_sizes)
        self.last_training_losses: dict[str, Tensor] = {}

    def forward(self, *, x_wsi: Tensor, cur_epoch: int = 0,
                wsi_missing: bool = False, omic_missing: bool = False,
                event_time=None, y=None, c=None, **omics):
        if wsi_missing or omic_missing:
            raise ValueError("Native MCAT/SurvPath requires both observed modalities")
        if not isinstance(x_wsi, Tensor) or x_wsi.ndim != 3 or x_wsi.shape[0] != 1:
            raise ValueError("Native MCAT/SurvPath requires one WSI bag [1,N,D] per call")
        if x_wsi.shape[1] == 0 or bool((x_wsi == 0).all(dim=-1).any()):
            raise ValueError("Empty or zero-padded WSI tokens cannot enter the native adapter")
        if len(omics) != len(self.omic_sizes):
            raise ValueError(f"Expected {len(self.omic_sizes)} omic groups, got {sorted(omics)}")
        native = {"x_path": x_wsi.squeeze(0) if self.baseline == "mcat" else x_wsi,
                  "cur_epoch": int(cur_epoch), "plugin_compute_aux": bool(self.training)}
        for index, width in enumerate(self.omic_sizes, start=1):
            key = f"x_omic{index}"
            value = omics.get(key)
            if not isinstance(value, Tensor) or value.shape != (1, width):
                raise ValueError(f"{key} must have single-patient shape [1,{width}]")
            native[key] = value.squeeze(0)
        output, auxiliary = self.adapter.forward_with_aux(**native)
        logits = torch.logit(output[0].clamp(1e-7, 1 - 1e-7)) if self.baseline == "mcat" else output
        if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("Native adapter did not return one patient's logits")
        if not torch.isfinite(logits).all() or not torch.isfinite(auxiliary).all():
            raise ValueError("Native adapter returned a non-finite logit or auxiliary loss")
        plugin_output = self.adapter.last_plugin_output
        self.last_training_losses = (plugin_output.aux_losses if self.training and plugin_output else {})
        return logits, auxiliary
