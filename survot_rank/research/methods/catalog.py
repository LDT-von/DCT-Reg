"""Frozen DCT-Reg and separately registered research candidates."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


CATALOG_UPDATED = "2026-09-05"
PRIMARY_METHOD = "dct_v310_directional_regularized_transport"
METHOD_STATUSES = ("primary", "ablation", "candidate")
STATUS_LABELS = {
    "primary": "frozen paper method",
    "ablation": "mechanism ablation parent",
    "candidate": "experimental method; performance not yet established",
}


def _method_dir(folder: str) -> str:
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
    MethodSpec(
        "dct_transport_intervention_consistency",
        "DCT intervention-consistency parent",
        "dct",
        "ablation",
        "dct_transport_intervention_consistency",
        "DCTTransportInterventionConsistency",
        aliases=("dct_ablation_parent",),
    ),
    MethodSpec(
        "dct_v32_transport_guided_slot_reaggregation",
        "DCT v3.2 Transport-Guided Slot Reaggregation (TGSR)",
        "dct", "candidate", "dct_v32_transport_guided_slot_reaggregation",
        "DCTV32TransportGuidedSlotReaggregation",
        aliases=("dct_v32", "dct_v3_2", "tgsr"),
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
