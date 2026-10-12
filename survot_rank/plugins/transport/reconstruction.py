"""Optional training-only reconstruction heads for encoded transport features.

The module has no model, trainer, file-system or survival-label dependencies.
It receives a duck-typed transport state and reuses its factual transported WSI
memory. Targets are detached; gradients continue through slots and transport.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _OmicsQueryDecoder(nn.Module):
    """Learn fixed pathway queries and decode them from slot memory."""

    def __init__(self, dim: int, num_tokens: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim),
        )

    def forward(self, memory: Tensor, num_tokens: int) -> Tensor:
        if num_tokens > self.queries.size(1):
            raise ValueError(
                f"omics input has {num_tokens} tokens, exceeding configured "
                f"num_omics_tokens={self.queries.size(1)}"
            )
        queries = self.queries[:, :num_tokens].expand(memory.size(0), -1, -1)
        keys = self.memory_norm(memory)
        decoded, _ = self.cross_attention(
            self.query_norm(queries), keys, keys, need_weights=False,
        )
        decoded = queries + decoded
        return decoded + self.output_mlp(self.output_norm(decoded))


class _WSIQueryDecoder(nn.Module):
    """Use detached patch queries; slots provide every attention value.

    The random target-query projection is frozen. There is no target/query
    residual into the output, retaining the full-WSI auxiliary branch semantics.
    This reconstructs encoded patch features, rather than pixels.
    """

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.query_projection.requires_grad_(False)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim),
        )

    def forward(self, memory: Tensor, queries: Tensor) -> Tensor:
        keys = self.memory_norm(memory)
        decoded, _ = self.cross_attention(
            self.query_norm(self.query_projection(queries.detach())),
            keys, keys, need_weights=False,
        )
        return decoded + self.output_mlp(self.output_norm(decoded))


def _masked_patient_mean(distances: Tensor, valid: Tensor) -> Tensor:
    """Equal patient weighting after each patient's valid-token mean."""
    counts = valid.sum(dim=1)
    available = counts > 0
    if not bool(available.any()):
        return torch.nan_to_num(distances).sum() * 0.0
    sums = distances.masked_fill(~valid, 0.0).sum(dim=1)
    return (sums[available] / counts[available].to(sums.dtype)).mean()


def _omics_distance(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    target = target.detach().masked_fill(~valid.unsqueeze(-1), 0.0)
    # Clear before normalization, as invalid padded NaNs must never enter it.
    prediction = prediction.masked_fill(~valid.unsqueeze(-1), 0.0)
    prediction = F.layer_norm(prediction, (prediction.size(-1),))
    target = F.layer_norm(target, (target.size(-1),))
    cosine = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    robust_l1 = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    return _masked_patient_mean(0.5 * (cosine + robust_l1), valid)


class TransportReconstructionHeads(nn.Module):
    """Three independent auxiliary branches, or shared omics decoder control.

    ``state`` supplies encoded ``wsi_tokens`` / ``omics_tokens``, boolean
    ``wsi_mask`` / ``omics_mask`` (True means valid), semantic ``wsi_slots`` /
    ``omics_slots``, and ``transported_wsi`` obtained from the same factual
    transport plans as prediction. The stage gate must be detached when that
    memory is formed by the transport core. This module does not recompute it.

    Omics pathways have stable order: a shorter input uses the first P learned
    pathway queries. Missing targets are excluded per branch and each available
    patient contributes equally. Zero ramp/weight skips decoder execution.
    """

    def __init__(
        self, dim: int, num_omics_tokens: int, heads: int = 4,
        mode: str = "separate", self_weight: float = 0.05,
        cross_weight: float = 0.05, wsi_weight: float = 0.05,
        ramp_start: int = 2, ramp_epochs: int = 5, wsi_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim and heads must be positive, with dim divisible by heads")
        if num_omics_tokens <= 0:
            raise ValueError("num_omics_tokens must be positive")
        if mode not in {"separate", "shared"}:
            raise ValueError("reconstruction mode must be 'separate' or 'shared'")
        if ramp_start < 0 or ramp_epochs <= 0 or wsi_chunk_size <= 0:
            raise ValueError("ramp_start must be nonnegative; ramp_epochs and wsi_chunk_size positive")
        weights = (self_weight, cross_weight, wsi_weight)
        if any(not math.isfinite(float(weight)) or weight < 0 for weight in weights):
            raise ValueError("reconstruction branch weights must be finite and nonnegative")
        self.dim = int(dim)
        self.num_omics_tokens = int(num_omics_tokens)
        self.heads = int(heads)
        self.mode = mode
        self.self_weight, self.cross_weight, self.wsi_weight = map(float, weights)
        self.ramp_start = int(ramp_start)
        self.ramp_epochs = int(ramp_epochs)
        self.wsi_chunk_size = int(wsi_chunk_size)
        self.self_decoder = _OmicsQueryDecoder(dim, num_omics_tokens, heads)
        self.cross_decoder = (
            copy.deepcopy(self.self_decoder) if mode == "separate" else self.self_decoder
        )
        self.wsi_decoder = _WSIQueryDecoder(dim, heads)

    def configuration(self) -> dict[str, Any]:
        """Return every construction choice needed to restore this objective."""
        return {
            "dim": self.dim,
            "num_omics_tokens": self.num_omics_tokens,
            "heads": self.heads,
            "mode": self.mode,
            "self_weight": self.self_weight,
            "cross_weight": self.cross_weight,
            "wsi_weight": self.wsi_weight,
            "ramp_start": self.ramp_start,
            "ramp_epochs": self.ramp_epochs,
            "wsi_chunk_size": self.wsi_chunk_size,
            "wsi_head_enabled": self.wsi_decoder is not None,
        }

    @classmethod
    def from_config(cls, configuration: Mapping[str, Any]) -> "TransportReconstructionHeads":
        """Construct from a saved manifest before loading its state dict.

        Native models with disabled WSI reconstruction have no WSI decoder;
        that physical layout is retained instead of creating unused parameters.
        """
        if not isinstance(configuration, Mapping):
            raise ValueError("reconstruction configuration must be a mapping")
        values = dict(configuration)
        if "wsi_head_enabled" not in values:
            raise ValueError("reconstruction configuration is missing wsi_head_enabled")
        enabled = values.pop("wsi_head_enabled")
        if not isinstance(enabled, bool):
            raise ValueError("wsi_head_enabled must be boolean")
        required = {
            "dim", "num_omics_tokens", "heads", "mode", "self_weight", "cross_weight",
            "wsi_weight", "ramp_start", "ramp_epochs", "wsi_chunk_size",
        }
        if set(values) != required:
            missing = sorted(required - set(values))
            extra = sorted(set(values) - required)
            raise ValueError(f"invalid reconstruction configuration: missing={missing}, extra={extra}")
        instance = cls(**values)
        if not enabled:
            if instance.wsi_weight != 0:
                raise ValueError("disabled WSI head requires wsi_weight=0")
            instance.wsi_decoder = None
        return instance

    def get_extra_state(self) -> dict[str, Any]:
        return {"schema_version": 1, "configuration": self.configuration()}

    def set_extra_state(self, state: Any) -> None:
        if not isinstance(state, Mapping) or state.get("schema_version") != 1:
            raise ValueError("unsupported reconstruction checkpoint metadata; use saved configuration")
        saved = state.get("configuration")
        expected = self.configuration()
        if not isinstance(saved, Mapping) or dict(saved) != expected:
            differences = sorted(
                name for name in set(expected) | set(saved if isinstance(saved, Mapping) else {})
                if not isinstance(saved, Mapping) or saved.get(name) != expected.get(name)
            )
            raise ValueError(
                "reconstruction checkpoint configuration mismatch "
                f"({', '.join(differences)}); construct with "
                "TransportReconstructionHeads.from_config(saved_configuration) before loading"
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Even strict=False must not silently load weights under another loss.
        if prefix + "_extra_state" not in state_dict:
            error_msgs.append(
                prefix + "reconstruction checkpoint metadata is missing; "
                "use a checkpoint that includes saved reconstruction configuration"
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @classmethod
    def from_dct(cls, model: Any) -> "TransportReconstructionHeads":
        """Copy existing v3.13 head weights and effective coefficients.

        The host is never retained or registered, and all copied tensors have
        independent storage. This adapts the existing decoder architecture to
        masked per-patient losses, rather than establishing checkpoint metric
        equivalence. A native direct cross-reconstruction control is rejected:
        this plugin's cross memory is the factual transported WSI memory.
        """
        if not callable(getattr(model, "effective_recon_coefficients", None)):
            raise ValueError("from_dct requires a v3.13 model with effective_recon_coefficients()")
        coefficients = model.effective_recon_coefficients()
        weights = {name: float(coefficients.get(name, 0.0)) for name in ("self", "cross", "wsi")}
        if not any(weights.values()):
            raise ValueError("all native reconstruction branches are disabled; use reconstruction=None")
        if coefficients.get("cross_mode", "transport") != "transport" and weights["cross"] > 0:
            raise ValueError("from_dct supports transport cross reconstruction, not the native direct control")
        native_self = getattr(model, "pathway_reconstruction_decoder", None)
        if native_self is None or not hasattr(native_self, "pathway_queries"):
            raise ValueError("native v3.13 pathway_reconstruction_decoder is missing or unsupported")
        mode = str(coefficients.get("omics_decoder_mode", getattr(model, "_omics_decoder_mode", "shared")))
        if mode not in {"shared", "separate"}:
            raise ValueError("native omics_decoder_mode must be shared or separate")
        native_cross = getattr(model, "cross_pathway_reconstruction_decoder", None)
        if mode == "separate" and native_cross is None:
            raise ValueError("native separate cross_pathway_reconstruction_decoder is missing")
        native_wsi = getattr(model, "wsi_reconstruction_decoder", None)
        if weights["wsi"] > 0 and native_wsi is None:
            raise ValueError("native WSI coefficient is active but wsi_reconstruction_decoder is missing")
        query = native_self.pathway_queries
        if query.ndim != 3 or query.size(0) != 1:
            raise ValueError("native pathway queries must have shape [1, P, D]")
        # Initializers do not disturb the caller's CPU random continuation.
        with torch.random.fork_rng(devices=[]):
            instance = cls(
                dim=query.size(-1), num_omics_tokens=query.size(1),
                heads=native_self.cross_attention.num_heads, mode=mode,
                self_weight=weights["self"], cross_weight=weights["cross"],
                wsi_weight=weights["wsi"],
                ramp_start=int(model.dct_v313_reconstruction_ramp_start),
                ramp_epochs=int(model.dct_v313_reconstruction_ramp_epochs),
                wsi_chunk_size=int(getattr(native_wsi, "chunk_size", 256)),
            )
        instance.to(device=query.device, dtype=query.dtype)

        def copy_omics(destination: nn.Module, source: nn.Module) -> None:
            copied = copy.deepcopy(source.state_dict())
            copied["queries"] = copied.pop("pathway_queries")
            destination.load_state_dict(copied, strict=True)
            source_parameters = dict(source.named_parameters())
            for name, parameter in destination.named_parameters():
                source_name = "pathway_queries" if name == "queries" else name
                parameter.requires_grad_(source_parameters[source_name].requires_grad)
            destination.train(source.training)

        copy_omics(instance.self_decoder, native_self)
        if mode == "separate":
            copy_omics(instance.cross_decoder, native_cross)
        if native_wsi is None:
            # An inactive WSI branch carries no randomly initialized decoder.
            instance.wsi_decoder = None
        else:
            instance.wsi_decoder = copy.deepcopy(native_wsi)
        instance.training = model.training
        return instance

    def ramp(self, epoch: int) -> float:
        return min(1.0, max(0.0, (float(epoch) - self.ramp_start) / self.ramp_epochs))

    def _validate_state(self, state: Any) -> None:
        batch = state.wsi_tokens.size(0)
        for name in ("wsi_tokens", "omics_tokens", "wsi_slots", "omics_slots", "transported_wsi"):
            value = getattr(state, name)
            if value.ndim != 3 or value.size(0) != batch or value.size(-1) != self.dim:
                raise ValueError(f"{name} must be [B, N/K, {self.dim}] with shared batch size")
        for name, tokens in (("wsi_mask", state.wsi_tokens), ("omics_mask", state.omics_tokens)):
            mask = getattr(state, name)
            if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean with shape {tuple(tokens.shape[:2])}")
        if state.omics_tokens.size(1) > self.num_omics_tokens:
            raise ValueError(
                f"omics input has {state.omics_tokens.size(1)} tokens, exceeding "
                f"configured num_omics_tokens={self.num_omics_tokens}"
            )

    @staticmethod
    def _zero(state: Any) -> Tensor:
        # Graph-connected even for empty/missing batches, including padding NaNs.
        return sum(
            (torch.nan_to_num(getattr(state, name), posinf=0.0, neginf=0.0) * 0.0).sum()
            for name in ("wsi_slots", "omics_slots", "transported_wsi")
        )

    def _wsi_loss(self, memory: Tensor, target: Tensor, valid: Tensor) -> Tensor:
        target = target.detach().masked_fill(~valid.unsqueeze(-1), 0.0)
        sums = memory.new_zeros(memory.size(0))
        for start in range(0, target.size(1), self.wsi_chunk_size):
            stop = start + self.wsi_chunk_size
            chunk = target[:, start:stop]
            chunk_valid = valid[:, start:stop]
            prediction = self.wsi_decoder(memory, chunk)
            distances = 1.0 - F.cosine_similarity(prediction, chunk, dim=-1)
            sums = sums + distances.masked_fill(~chunk_valid, 0.0).sum(dim=1)
        return (sums / valid.sum(dim=1).to(sums.dtype)).mean()

    def forward(self, state: Any, epoch: int = 0) -> tuple[Tensor, dict[str, Tensor]]:
        self._validate_state(state)
        zero = self._zero(state)
        ramp = zero.new_tensor(self.ramp(epoch))
        losses = {"self": zero, "cross": zero, "wsi": zero}
        if ramp.item() > 0:
            omics_available = state.omics_mask.any(dim=1)
            wsi_available = state.wsi_mask.any(dim=1)
            if self.self_weight > 0 and bool(omics_available.any()):
                memory = state.omics_slots[omics_available]
                prediction = self.self_decoder(memory, state.omics_tokens.size(1))
                losses["self"] = _omics_distance(
                    prediction, state.omics_tokens[omics_available], state.omics_mask[omics_available],
                )
            paired_available = omics_available & wsi_available
            if self.cross_weight > 0 and bool(paired_available.any()):
                memory = state.transported_wsi[paired_available]
                prediction = self.cross_decoder(memory, state.omics_tokens.size(1))
                losses["cross"] = _omics_distance(
                    prediction, state.omics_tokens[paired_available], state.omics_mask[paired_available],
                )
            if self.wsi_weight > 0 and bool(wsi_available.any()):
                losses["wsi"] = self._wsi_loss(
                    state.wsi_slots[wsi_available],
                    state.wsi_tokens[wsi_available], state.wsi_mask[wsi_available],
                )
        total = ramp * (
            self.self_weight * losses["self"]
            + self.cross_weight * losses["cross"]
            + self.wsi_weight * losses["wsi"]
        )
        losses.update(weighted_total=total, ramp=ramp)
        return total, losses
