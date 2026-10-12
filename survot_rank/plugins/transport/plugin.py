"""Shape-preserving residual transport for an existing two-stream host model."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import torch
from torch import nn

from .config import TransportPluginConfig
from .core import TransportCore, TransportState


@dataclass
class TransportPluginOutput:
    wsi: torch.Tensor
    omics: torch.Tensor
    aux_loss: torch.Tensor
    aux_losses: dict[str, torch.Tensor] = field(default_factory=dict)
    transport: TransportState | None = None


class DCTTransportPlugin(nn.Module):
    def __init__(self, config: TransportPluginConfig, reconstruction: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.enabled = True
        self.core = TransportCore(config)
        self.wsi_projection = nn.Identity() if config.wsi_dim == config.dim else nn.Linear(config.wsi_dim, config.dim)
        self.omics_projection = nn.Identity() if config.omics_dim == config.dim else nn.Linear(config.omics_dim, config.dim)
        self.wsi_query_norm = nn.LayerNorm(config.dim)
        self.omics_query_norm = nn.LayerNorm(config.dim)
        self.wsi_writeback = nn.MultiheadAttention(config.dim, config.heads, batch_first=True)
        self.omics_writeback = nn.MultiheadAttention(config.dim, config.heads, batch_first=True)
        self.wsi_output_projection = nn.Linear(config.dim, config.wsi_dim)
        self.omics_output_projection = nn.Linear(config.dim, config.omics_dim)
        self.wsi_gain = nn.Parameter(torch.tensor(float(config.residual_init)))
        self.omics_gain = nn.Parameter(torch.tensor(float(config.residual_init)))
        self.reconstruction = reconstruction
        self.last_output = None

    def set_enabled(self, enabled: bool):
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.enabled = enabled
        return self

    def set_plan_mode(self, mode: str):
        if mode not in {"learned", "independent", "uniform"}:
            raise ValueError("unsupported transport plan mode")
        self.core.plan_mode = mode
        self.config = replace(self.config, plan_mode=mode)
        self.core.config = replace(self.core.config, plan_mode=mode)
        return self

    def get_extra_state(self):
        return {"schema": "dct_transport_plugin_v1", "config": self.config.to_dict(),
                "enabled": self.enabled, "plan_mode": self.core.plan_mode}

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != "dct_transport_plugin_v1":
            raise ValueError("unsupported transport plugin checkpoint metadata")
        saved = TransportPluginConfig.from_dict(state["config"])
        # Arithmetic and evaluation-slot noise are config-dependent. Do not
        # silently load identical-shaped parameters into a different recipe.
        current, expected = self.config.to_dict(), saved.to_dict()
        current.pop("plan_mode"); expected.pop("plan_mode")
        if current != expected:
            raise ValueError("construct the plugin from the checkpoint config before loading")
        self.set_plan_mode(state["plan_mode"])
        self.set_enabled(state["enabled"])

    @staticmethod
    def _mask(tokens, supplied, name):
        if supplied is None:
            result = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if not torch.is_tensor(supplied) or supplied.dtype != torch.bool or supplied.shape != tokens.shape[:2]:
                raise ValueError(f"{name} must be bool [B, N], True for valid tokens")
            result = supplied.to(tokens.device)
        if not bool(torch.isfinite(tokens.masked_fill(~result[..., None], 0.)).all()):
            raise ValueError(f"valid {name} tokens must be finite")
        return result

    def forward(self, wsi, omics, *, wsi_mask=None, omics_mask=None, epoch=0,
                compute_aux=None, enabled=None):
        if wsi.ndim != 3 or omics.ndim != 3 or wsi.size(0) != omics.size(0) or wsi.size(0) < 1:
            raise ValueError("expected paired tensors [B, N, Dh] and [B, P, Dg] with B > 0")
        if wsi.size(-1) != self.config.wsi_dim or omics.size(-1) != self.config.omics_dim:
            raise ValueError("host feature dimensions do not match TransportPluginConfig")
        if wsi.device != omics.device or wsi.dtype != omics.dtype or not wsi.is_floating_point():
            raise ValueError("paired features must have the same floating dtype and device")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        active = self.enabled if enabled is None else enabled
        if not isinstance(active, bool):
            raise TypeError("enabled must be bool")
        zero = wsi.new_zeros(()) + self.wsi_gain.to(wsi.device) * 0.
        if not active:
            result = TransportPluginOutput(wsi, omics, zero)
            self.last_output = result
            return result
        wm, om = self._mask(wsi, wsi_mask, "wsi_mask"), self._mask(omics, omics_mask, "omics_mask")
        safe_wsi, safe_omics = wsi.masked_fill(~wm[..., None], 0.), omics.masked_fill(~om[..., None], 0.)
        h, g = self.wsi_projection(safe_wsi), self.omics_projection(safe_omics)
        h, g = h.masked_fill(~wm[..., None], 0.), g.masked_fill(~om[..., None], 0.)
        state = self.core(h, g, wsi_mask=wm, omics_mask=om, epoch=epoch)
        out_wsi, out_omics = safe_wsi, safe_omics
        if bool(state.pair_available.any()):
            indices = state.pair_available.nonzero(as_tuple=True)[0]
            # The host prediction task trains the stage gate through this path.
            # Only the reconstruction aggregation detaches its weights.
            weighted_events = state.event_tokens[indices] * state.stage_weights[indices, :, None]
            w_context = torch.cat((weighted_events, state.transported_omics[indices]), 1)
            g_context = torch.cat((weighted_events, state.transported_wsi[indices]), 1)
            w_update, _ = self.wsi_writeback(self.wsi_query_norm(h[indices]), w_context, w_context,
                                            need_weights=False)
            g_update, _ = self.omics_writeback(self.omics_query_norm(g[indices]), g_context, g_context,
                                              need_weights=False)
            w_update = self.wsi_output_projection(w_update).to(wsi.dtype).masked_fill(~wm[indices, :, None], 0.)
            g_update = self.omics_output_projection(g_update).to(omics.dtype).masked_fill(~om[indices, :, None], 0.)
            out_wsi = out_wsi.index_add(0, indices, torch.tanh(self.wsi_gain) * w_update)
            out_omics = out_omics.index_add(0, indices, torch.tanh(self.omics_gain) * g_update)
        aux, losses = zero, {}
        # Eval always bypasses all three auxiliary heads, including an explicit
        # compute_aux=True. A training caller can also opt out of auxiliary work.
        if self.training and compute_aux is not False and self.reconstruction is not None:
            aux, losses = self.reconstruction(state, epoch=epoch)
            aux = aux + zero
        result = TransportPluginOutput(out_wsi, out_omics, aux, losses, state)
        self.last_output = result
        return result
