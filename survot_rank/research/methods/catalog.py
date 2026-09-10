"""DCT Research Method Catalog.

DCT (Deep Counterfactual Transformers) is a multi-modal survival prediction framework
with interpretability via intervention audit.

Primary contributions:
1. Survival prediction: Achieves C-index ≥ SlotSPE on multiple TCGA cohorts
2. Interpretability: Cost-space intervention audit explains model predictions
3. Multi-modal fusion: Shared semantic prototypes align WSI ↔ Omics

The OT (Optimal Transport) structure is a technical implementation detail that enables
both accurate prediction and interpretability audit. It is NOT the core contribution claim.

---
METHOD ORGANIZATION (2026-09-09):
================================
- PRIMARY: dct_v310_directional_regularized_transport (paper method)
- CANDIDATE: experimental methods under active development
- LEGACY: archived methods (see survot_rank/research/methods/legacy/)

LEGACY/ABLATION folder contains:
  - legacy/ablation/*: Historical ablation variants (not for paper use)
  - legacy/experimental/*: Experimental methods under exploration

The paper method (v3.10) uses exactly 3 losses:
  1. NLL (from trainer, λ = 1.0)
  2. IPCW Ranking (λ = 0.10)
  3. Direction (λ = 0.05)

All other losses (ETAR, dose, reconfiguration, listwise, MGPTR, etc.) are DISABLED.
They are preserved for ablation studies only.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


CATALOG_UPDATED = "2026-09-09"
PRIMARY_METHOD = "dct_v310_directional_regularized_transport"
METHOD_STATUSES = ("primary", "candidate", "legacy")
STATUS_LABELS = {
    "primary": "frozen paper method",
    "candidate": "experimental method; performance not yet established",
    "legacy": "archived method (see legacy/ folder)",
}


def _method_dir(folder: str) -> str:
    # Handle legacy folder paths (e.g., "legacy/experimental/dct_v311_slot_interpretable")
    if folder.startswith("legacy/"):
        return os.path.join("survot_rank", "research", "methods", folder)
    return os.path.join("survot_rank", "research", "methods", folder)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    family: str
    status: str
    folder: str
    class_name: str
    aliases: tuple[str, ...] = ()
    model_file: str = "model.py"

    @property
    def method_dir(self) -> str:
        return _method_dir(self.folder)


METHOD_SPECS = (
    MethodSpec(
        PRIMARY_METHOD,
        "DCT v3.10 Directionally Regularized Transport (DCT-Reg)",
        "dct",
        "primary",
        "dct_v310_directional_regularized_transport",
        "DCTV310DirectionalRegularizedTransport",
        aliases=("dct_v310", "dct_v3_10", "dct_reg"),
    ),
    # === LEGACY METHODS (see survot_rank/research/methods/legacy/) ===
    # These are archived ablation/experimental variants. NOT for paper use.
    MethodSpec(
        "dct_transport_intervention_consistency",
        "DCT v3.8 Intervention Consistency [LEGACY]",
        "dct",
        "legacy",
        "legacy/ablation/dct_transport_intervention_consistency",
        "DCTTransportInterventionConsistency",
        aliases=("dct_v38",),
    ),
    MethodSpec(
        "dct_risk_ordering_transport",
        "DCT Risk Ordering Transport [LEGACY]",
        "dct", "legacy", "legacy/ablation/dct_risk_ordering_transport",
        "DCTRiskOrderingTransport",
        aliases=("risk_ordering_transport", "dct_rot"),
    ),
    MethodSpec(
        "dct_v310_fixed_anchors",
        "DCT v3.10 Fixed Anchors [LEGACY]",
        "dct",
        "legacy",
        "legacy/ablation/dct_v310_fixed_anchors",
        "DCTV310FixedAnchors",
        aliases=(),
    ),
    MethodSpec(
        "dct_v32_tgsr_objective_study",
        "DCT v3.2 TGSR Objective Study [LEGACY]",
        "dct",
        "legacy",
        "legacy/ablation/dct_v32_tgsr_objective_study",
        "DCTV32TGSRObjectiveStudy",
        aliases=(),
    ),
    MethodSpec(
        "dct_v32_transport_guided_slot_reaggregation",
        "DCT v3.2 Transport-Guided Slot Reaggregation [LEGACY]",
        "dct",
        "legacy",
        "legacy/ablation/dct_v32_transport_guided_slot_reaggregation",
        "DCTV32TransportGuidedSlotReaggregation",
        aliases=(),
    ),
    MethodSpec(
        "dct_v330_closed_loop_prognostic_transport",
        "DCT v3.30 Closed-Loop Prognostic Transport [LEGACY]",
        "dct",
        "legacy",
        "legacy/ablation/dct_v330_closed_loop_prognostic_transport",
        "DCTV330ClosedLoopPrognosticTransport",
        aliases=(),
    ),
    MethodSpec(
        "dct_v311_slot_interpretable",
        "DCT v3.11 Per-Slot Interpretable Transport [EXPERIMENTAL]",
        "dct",
        "legacy",
        "legacy/experimental/dct_v311_slot_interpretable",
        "DCTV311SlotInterpretable",
        aliases=("dct_v311",),
    ),
)

METHOD_CATALOG = {spec.key: spec for spec in METHOD_SPECS}
METHOD_REGISTRY = {
    spec.key: (spec.method_dir, spec.class_name) for spec in METHOD_SPECS
}
METHOD_ALIASES = {
    alias: spec.key for spec in METHOD_SPECS for alias in spec.aliases
}
METHOD_CHOICES = tuple(METHOD_CATALOG) + tuple(METHOD_ALIASES)


def iter_method_specs(status: str | None = None) -> Iterable[MethodSpec]:
    if status is not None and status not in METHOD_STATUSES:
        raise ValueError(f"Unknown method status: {status}")
    return (spec for spec in METHOD_SPECS if status is None or spec.status == status)


def catalog_errors(project_root: str | Path) -> list[str]:
    root = Path(project_root)
    errors: list[str] = []
    for spec in METHOD_SPECS:
        model_path = root / spec.method_dir / spec.model_file
        if not model_path.is_file():
            errors.append(f"{spec.key}: missing {model_path.relative_to(root)}")
    return errors
