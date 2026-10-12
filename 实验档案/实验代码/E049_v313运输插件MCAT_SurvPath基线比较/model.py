"""E049 native adapter model interface.

This module provides a thin build_model() factory compatible with the E049
worker patching protocol.  For production training, the worker builds adapters
directly using the plugin system in its own execute_task() path.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Arm definitions — must stay in sync with schedule_core.py and worker.py
ARM_DEFINITIONS = {
    "native":                    dict(enabled=False, plan_mode="learned",    reconstruction=False, decoder_mode=None),
    "independent":               dict(enabled=True,  plan_mode="independent", reconstruction=False, decoder_mode=None),
    "learned":                  dict(enabled=True,  plan_mode="learned",     reconstruction=False, decoder_mode=None),
    "independent_three_decoders": dict(enabled=True, plan_mode="independent", reconstruction=True, decoder_mode="separate"),
    "learned_three_decoders":   dict(enabled=True,  plan_mode="learned",   reconstruction=True,  decoder_mode="separate"),
    "learned_shared_decoders":  dict(enabled=True,  plan_mode="learned",   reconstruction=True,  decoder_mode="shared"),
}
ARM_NAMES = tuple(ARM_DEFINITIONS.keys())


def build_model(args, arm, omic_input_dim=None, omic_names=None, pathway_names=None):
    """Build an E049 native adapter (MCAT or SurvPath).

    This factory is used by the E049 worker when patching runner.get_model.
    The actual E049 workflow builds adapters in worker.execute_task() using
    build_native_adapter_for_arm().
    """
    from survot_rank.plugins.transport import build_native_adapter
    from survot_rank.plugins.transport.config import TransportPluginConfig

    arm_def = ARM_DEFINITIONS.get(arm)
    if arm_def is None:
        raise ValueError(f"Unknown E049 arm {arm!r}; choose from {ARM_NAMES}")

    baseline = getattr(args, "e049_baseline", "mcat")
    omic_sizes = getattr(args, "omic_sizes", None)
    if omic_sizes is None:
        raise ValueError("omic_sizes must be set on args for E049")

    config = TransportPluginConfig(
        wsi_dim=256, omics_dim=256, dim=256,
        plan_mode=arm_def["plan_mode"],
        residual_init=0.0,
        eps=0.05, eps_start=0.10, eps_anneal_epochs=12,
        sinkhorn_iters=50, projection_iters=10000, projection_tol=0.0001,
        wsi_slots=8, omics_slots=8,
        stages=4, heads=4, event_layers=2,
        slot_init_mode="deterministic",
        geometries=["cosine", "euclidean", "positive_dot"],
    )

    reconstruction = None
    if arm_def["reconstruction"]:
        reconstruction = {
            "dim": 256,
            "num_omics_tokens": len(omic_sizes),
            "heads": 4,
            "mode": arm_def["decoder_mode"],
            "self_weight": 0.05,
            "cross_weight": 0.05,
            "wsi_weight": 0.05,
            "ramp_start": 2,
            "ramp_epochs": 5,
            "wsi_chunk_size": 256,
        }

    num_classes = getattr(args, "num_classes", 4)
    wsi_input_dim = getattr(args, "wsi_input_dim", 1536)

    return build_native_adapter(
        backbone_name=baseline,
        omic_sizes=omic_sizes,
        wsi_input_dim=wsi_input_dim,
        plugin_config=config,
        reconstruction_config=reconstruction,
        fusion="concat",
        enabled=arm_def["enabled"],
        num_classes=num_classes,
    )


def audit_model(model, args, arm):
    """Return a serializable audit dict for an E049 adapter."""
    arm_def = ARM_DEFINITIONS.get(arm, {})
    return {
        "baseline": getattr(args, "e049_baseline", "unknown"),
        "arm": arm,
        "omic_sizes": getattr(args, "omic_sizes", []),
        "wsi_input_dim": getattr(args, "wsi_input_dim", 1536),
        "enabled": arm_def.get("enabled", True),
        "plan_mode": arm_def.get("plan_mode", "learned"),
        "reconstruction": arm_def.get("reconstruction", False),
        "decoder_mode": arm_def.get("decoder_mode", None),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
