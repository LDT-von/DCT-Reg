"""Thin adapters for the pinned MCAT and SurvPath feature interfaces.

The backbones remain registered modules with their original encoders, fusion,
and prediction heads.  This module imports no third-party ``models`` package:
callers supply a native backbone instance.  These two native forward methods
support a single patient; arbitrary batch sizes are supported by the transport
plugin itself, rather than silently changing either backbone's contract.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .plugin import DCTTransportPlugin, TransportPluginOutput


def _require_modules(backbone: nn.Module, names: tuple[str, ...]) -> None:
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be a torch.nn.Module")
    for name in names:
        if not isinstance(getattr(backbone, name, None), nn.Module):
            raise TypeError(f"native backbone must expose nn.Module {name!r}")


def _first_input_width(module: nn.Module) -> int | None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child.in_features
    return None


def _check_omics(kwargs: dict[str, Any], encoders: nn.ModuleList, count: int) -> list[Tensor]:
    supplied = {int(match.group(1)) for key in kwargs
                if (match := re.fullmatch(r"x_omic(\d+)", key)) is not None}
    expected = set(range(1, count + 1))
    if supplied != expected:
        raise ValueError(f"expected exactly x_omic1..x_omic{count}; got indices {sorted(supplied)}")
    features = [kwargs[f"x_omic{i}"] for i in range(1, count + 1)]
    for index, (feature, encoder) in enumerate(zip(features, encoders), start=1):
        if not isinstance(feature, Tensor) or feature.ndim != 1 or feature.numel() == 0:
            raise ValueError(f"x_omic{index} must be a nonempty single-patient gene vector [G]")
        expected_width = _first_input_width(encoder)
        if expected_width is not None and feature.shape[0] != expected_width:
            raise ValueError(f"x_omic{index} has width {feature.shape[0]}; native encoder expects {expected_width}")
    return features


def _check_path(path: Any, encoder: nn.Module, *, rank: int) -> Tensor:
    if not isinstance(path, Tensor) or path.ndim != rank:
        shape = "[N, D]" if rank == 2 else "[1, N, D]"
        raise ValueError(f"x_path must have native single-patient shape {shape}")
    if rank == 3 and path.shape[0] != 1:
        raise ValueError("this native backbone forward supports batch size 1 only")
    if path.shape[-2] == 0:
        raise ValueError("x_path must contain at least one pathology token")
    expected_width = _first_input_width(encoder)
    if expected_width is not None and path.shape[-1] != expected_width:
        raise ValueError(f"x_path has width {path.shape[-1]}; native encoder expects {expected_width}")
    return path


class _NativeTransportAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, plugin: DCTTransportPlugin) -> None:
        super().__init__()
        if not isinstance(plugin, nn.Module):
            raise TypeError("plugin must be a torch.nn.Module")
        self.backbone = backbone
        self.plugin = plugin
        self.last_plugin_output: TransportPluginOutput | None = None

    def _transport(self, wsi: Tensor, omics: Tensor, kwargs: dict[str, Any]) -> tuple[Tensor, Tensor]:
        # These pinned native hosts do not mask attention/pooling themselves.
        # Reject padded or missing observations rather than silently including
        # their zero tokens in a nominally masked patient prediction.
        for name, tokens in (("plugin_wsi_mask", wsi), ("plugin_omics_mask", omics)):
            mask = kwargs.get(name)
            if mask is not None and (not torch.is_tensor(mask) or mask.dtype != torch.bool
                                     or mask.shape != tokens.shape[:2] or not bool(mask.all())):
                raise ValueError(f"{name}: this native adapter requires fully observed tokens; "
                                 "use a mask-aware host for padded or missing inputs")
        self.last_plugin_output = self.plugin(
            wsi,
            omics,
            wsi_mask=kwargs.get("plugin_wsi_mask"),
            omics_mask=kwargs.get("plugin_omics_mask"),
            epoch=kwargs.get("cur_epoch", 0),
            compute_aux=kwargs.get("plugin_compute_aux"),
            enabled=kwargs.get("plugin_enabled"),
        )
        output = self.last_plugin_output
        if output.wsi.shape != wsi.shape or output.omics.shape != omics.shape:
            raise ValueError("transport plugin must preserve both native encoded feature shapes")
        return output.wsi, output.omics

    def forward_with_aux(self, **kwargs: Any) -> tuple[Any, Tensor]:
        """Return the unchanged native output structure and connected auxiliary loss.

        The training caller must add the returned scalar to its native objective;
        this method never silently changes the backbone's survival criterion.
        ``last_plugin_output`` also exposes individual terms and transport data.
        """
        native_output = self(**kwargs)
        if self.last_plugin_output is None:
            raise RuntimeError("native adapter did not produce a plugin output")
        return native_output, self.last_plugin_output.aux_loss


class MCATTransportAdapter(_NativeTransportAdapter):
    """Insert transport after native MCAT's two encoders, before co-attention.

    Supports the original six-signature ``MCAT_Surv`` contract and both native
    ``concat`` and ``bilinear`` fusion modes.  State dictionaries register the
    supplied original module under ``backbone.*`` and the plugin under
    ``plugin.*``; existing native checkpoints can be loaded into ``backbone``.
    """

    def __init__(self, backbone: nn.Module, plugin: DCTTransportPlugin) -> None:
        _require_modules(backbone, (
            "wsi_net", "sig_networks", "coattn", "path_transformer",
            "path_attention_head", "path_rho", "omic_transformer",
            "omic_attention_head", "omic_rho", "mm", "classifier",
        ))
        if not isinstance(backbone.sig_networks, nn.ModuleList) or len(backbone.sig_networks) != 6:
            raise ValueError("native MCAT adapter requires exactly six signature encoders")
        if getattr(backbone, "fusion", None) not in {"concat", "bilinear"}:
            raise ValueError("native MCAT adapter supports fusion='concat' or 'bilinear'")
        super().__init__(backbone, plugin)

    def forward(self, **kwargs: Any) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        self.last_plugin_output = None
        backbone = self.backbone
        x_path = _check_path(kwargs.get("x_path"), backbone.wsi_net, rank=2)
        x_omic = _check_omics(kwargs, backbone.sig_networks, 6)
        h_path_bag = backbone.wsi_net(x_path).unsqueeze(1)
        h_omic = [backbone.sig_networks[index].forward(feature)
                  for index, feature in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(1)

        wsi, omics = self._transport(
            h_path_bag.transpose(0, 1), h_omic_bag.transpose(0, 1), kwargs,
        )
        h_path_bag, h_omic_bag = wsi.transpose(0, 1), omics.transpose(0, 1)

        h_path_coattn, A_coattn = backbone.coattn(h_omic_bag, h_path_bag, h_path_bag)
        h_path_trans = backbone.path_transformer(h_path_coattn)
        A_path, h_path = backbone.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = backbone.path_rho(h_path).squeeze()

        h_omic_trans = backbone.omic_transformer(h_omic_bag)
        A_omic, h_omic = backbone.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = backbone.omic_rho(h_omic).squeeze()

        if backbone.fusion == "bilinear":
            h = backbone.mm(h_path.unsqueeze(0), h_omic.unsqueeze(0)).squeeze()
        else:
            h = backbone.mm(torch.cat([h_path, h_omic], axis=0))
        logits = backbone.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        return hazards, survival, Y_hat, {"coattn": A_coattn, "path": A_path, "omic": A_omic}


class SurvPathTransportAdapter(_NativeTransportAdapter):
    """Insert transport after native SurvPath encoders, before token concatenation.

    ``x_path`` uses native shape ``[1, N, D]`` and each pathway uses ``[G]``.
    ``return_attn=True`` retains SurvPath's logits plus three attention tensors.
    """

    def __init__(self, backbone: nn.Module, plugin: DCTTransportPlugin) -> None:
        _require_modules(backbone, (
            "wsi_projection_net", "sig_networks", "identity", "cross_attender",
            "feed_forward", "layer_norm", "to_logits",
        ))
        count = getattr(backbone, "num_pathways", None)
        if not isinstance(count, int) or count < 1:
            raise ValueError("native SurvPath must expose positive num_pathways")
        if not isinstance(backbone.sig_networks, nn.ModuleList) or len(backbone.sig_networks) != count:
            raise ValueError("SurvPath num_pathways must match its signature encoders")
        super().__init__(backbone, plugin)

    def forward(self, **kwargs: Any) -> Any:
        self.last_plugin_output = None
        backbone = self.backbone
        wsi = _check_path(kwargs.get("x_path"), backbone.wsi_projection_net, rank=3)
        x_omic = _check_omics(kwargs, backbone.sig_networks, backbone.num_pathways)
        return_attn = kwargs.get("return_attn", False)
        if not isinstance(return_attn, bool):
            raise ValueError("return_attn must be bool")
        h_omic = [backbone.sig_networks[index].forward(feature.float())
                  for index, feature in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)
        wsi_embed = backbone.wsi_projection_net(wsi)
        wsi_embed, h_omic_bag = self._transport(wsi_embed, h_omic_bag, kwargs)

        tokens = backbone.identity(torch.cat([h_omic_bag, wsi_embed], dim=1))
        if return_attn:
            mm_embed, attn_pathways, cross_pathways, cross_histology = backbone.cross_attender(
                x=tokens, mask=None, return_attention=True,
            )
        else:
            mm_embed = backbone.cross_attender(x=tokens, mask=None, return_attention=False)
        mm_embed = backbone.layer_norm(backbone.feed_forward(mm_embed))
        paths_embed = torch.mean(mm_embed[:, :backbone.num_pathways, :], dim=1)
        wsi_embed = torch.mean(mm_embed[:, backbone.num_pathways:, :], dim=1)
        logits = backbone.to_logits(torch.cat([paths_embed, wsi_embed], dim=1))
        if return_attn:
            return logits, attn_pathways, cross_pathways, cross_histology
        return logits
