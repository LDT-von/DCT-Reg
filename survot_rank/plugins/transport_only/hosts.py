"""Attach feature transport to a native host without wrapping its loss/output.

Attach before constructing the optimizer. Recreate this attachment before loading
an augmented state_dict. Official trainers continue to own data, labels, losses,
checkpoint selection and prediction heads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .mechanism import DCTTransportMechanism, TransportOnlyConfig


HOST_ORDER = ("slotspe", "survpath", "pibd", "mmp", "ld_cvae", "tta")


def _name(value):
    name = value.lower().replace("-", "_")
    if name not in HOST_ORDER:
        raise ValueError(f"host must be one of {HOST_ORDER}")
    return name


@dataclass
class TransportAttachment:
    host: nn.Module
    host_name: str
    target_name: str
    handle: object

    @property
    def mechanism(self):
        return self.host.transport_mechanism

    def set_epoch(self, epoch):
        self.mechanism.set_epoch(epoch)
        return self

    def set_enabled(self, enabled):
        self.mechanism.set_enabled(enabled)
        return self

    def remove(self):
        """Remove before building/rebuilding an optimizer, never during a step."""
        self.handle.remove()
        del self.host.transport_mechanism
        del self.host._transport_attachment


def _layout(host, name):
    if name == "slotspe":
        return "cross_attention", "pair", int(host.wsi_projection_dim), None
    if name == "survpath":
        return "identity", "concat", int(host.wsi_projection_dim), int(host.num_pathways)
    if name == "pibd":
        return "PID", "pair", int(host.wsi_projection_dim), None
    if name == "mmp":
        if hasattr(host, "mot"):
            raise ValueError("MMP coattn_mot already contains OT; select the native coattn host arm")
        if getattr(host, "modality", None) not in {"both", "coattn", "survpath"}:
            raise ValueError("MMP transport requires a multimodal coattn host")
        return "identity", "concat", int(host.path_proj_dim), int(host.num_pathways)
    if name == "ld_cvae":
        return "coattn", "qkv_sequence_first", int(host.wsi_projection_dim), None
    if not (host.use_image and host.use_omics and host.label_use_coattn):
        raise ValueError("TTA requires both modalities and fusion_type='coattn'")
    return "label_coattn", "concat", int(host.path_proj_dim), int(host._label_num_path_tokens)


def attach_transport(host: nn.Module, host_name: str, config=None, *, enabled=True):
    """Register only transport parameters and a removable fusion-input hook.

    SlotSPE/PIBD retain every trailing argument. SurvPath/MMP/TTA use their
    original omics-first concatenation. LD-CVAE retains its selected real or
    generated omics tokens and changes only the q/k/v cross-attention path.
    No label, hazard, loss function or auxiliary-loss combiner enters this API.
    """
    name = _name(host_name)
    if not isinstance(host, nn.Module):
        raise TypeError("host must be an nn.Module")
    if hasattr(host, "transport_mechanism") or hasattr(host, "_transport_attachment"):
        raise ValueError("a transport mechanism is already attached")
    target_name, layout, dim, split = _layout(host, name)
    target = getattr(host, target_name, None)
    if not isinstance(target, nn.Module):
        raise ValueError(f"native host must expose {target_name} as an nn.Module")
    native_budget = None
    if name == "slotspe" and hasattr(host, "args"):
        native_budget = max(host.args.slot_num_wsi, host.args.slot_num_omics)
    elif name == "mmp":
        native_budget = max(host.numOfproto, host.num_pathways)
    elif name == "tta":
        native_budget = max(host.shared_proto_num if host.shared_prototypes else host.image_proto_num,
                            host._label_num_path_tokens)
    elif name == "pibd" and hasattr(host, "topk_wsi"):
        native_budget = max(host.topk_wsi, host.topk_omics)
    defaults = {"host_dim": dim}
    if native_budget is not None:
        defaults["slot_budget"] = max(8, native_budget)
    if config is None:
        config = TransportOnlyConfig(**defaults)
    elif isinstance(config, Mapping):
        config = TransportOnlyConfig(**{**defaults, **dict(config)})
    elif not isinstance(config, TransportOnlyConfig):
        raise TypeError("config must be TransportOnlyConfig or a mapping")
    if config.host_dim != dim:
        raise ValueError("transport host_dim must match native fusion tokens")
    if native_budget is not None and config.slot_budget < native_budget:
        raise ValueError("slot_budget must retain all of this host's existing slots/prototypes")
    # Native and augmented hosts keep the same post-construction random stream.
    with torch.random.fork_rng(devices=[]):
        mechanism = DCTTransportMechanism(config)
    reference = next(host.parameters(), None)
    if reference is not None:
        mechanism.to(device=reference.device, dtype=reference.dtype)
    mechanism.train(host.training)
    mechanism.set_enabled(enabled)

    def hook(module, args, kwargs):
        if not mechanism.enabled:
            return None
        values, named = list(args), dict(kwargs)
        if layout == "pair":
            if len(values) < 2:
                raise ValueError(f"{name} fusion requires two positional token tensors")
            values[0], values[1] = mechanism(values[0], values[1])
        elif layout == "concat":
            key = "x" if "x" in named else None
            tokens = named[key] if key else (values[0] if values else None)
            if not torch.is_tensor(tokens) or tokens.ndim != 3 or not 0 < split < tokens.size(1):
                raise ValueError(f"{name} expects nonempty omics-first [B,P+N,D] tokens")
            wsi, omics = mechanism(tokens[:, split:], tokens[:, :split])
            updated = torch.cat((omics, wsi), dim=1)
            if key:
                named[key] = updated
            else:
                values[0] = updated
        else:
            # Robust_MCAT calls sequence-first q=omics, k=v=WSI. Requiring
            # shared k/v catches incompatible model variants instead of silently
            # replacing a distinct value stream.
            keys = ("query", "key", "value")
            qkv = [values[i] if i < len(values) else named.get(key) for i, key in enumerate(keys)]
            q, k, v = qkv
            if not all(torch.is_tensor(t) and t.ndim == 3 for t in qkv):
                raise ValueError("LD-CVAE expects sequence-first query/key/value tensors")
            if k is not v:
                raise ValueError("LD-CVAE transport supports its native shared WSI key/value stream")
            wsi, omics = mechanism(k.transpose(0, 1), q.transpose(0, 1))
            updated = (omics.transpose(0, 1), wsi.transpose(0, 1), wsi.transpose(0, 1))
            for i, (key, value) in enumerate(zip(keys, updated)):
                if i < len(values):
                    values[i] = value
                else:
                    named[key] = value
        return tuple(values), named

    host.add_module("transport_mechanism", mechanism)
    handle = target.register_forward_pre_hook(hook, with_kwargs=True)
    attachment = TransportAttachment(host, name, target_name, handle)
    host._transport_attachment = attachment
    return attachment
