#!/usr/bin/env python3
"""Write matched MCAT/SurvPath construction plans; never launch experiments.

The saved arm JSON files are keyword arguments for ``build_native_adapter``.
Without --protocol, every unknown patient/data/training choice stays REQUIRED.
A supplied protocol is checked for completeness, not measured or executed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_SCHEMA = "dct_transport_plugin_protocol_v1"
PLAN_SCHEMA = "dct_transport_plugin_experiment_plan_v1"
REQUIRED = "REQUIRED"
BASELINE_COMMITS = {
    "mcat": "b9cca63be83c67de7f95308d54a58f80b78b0da1",
    "survpath": "3f73ddd6705ec67d643020c5bb04fb13f9f382cc",
}
BASE_ARMS = (
    ("native", False, "learned", None),
    ("independent", True, "independent", None),
    ("learned", True, "learned", None),
    ("independent_three_decoders", True, "independent", "separate"),
    ("learned_three_decoders", True, "learned", "separate"),
)
SHARED_ARM = ("learned_shared_decoders", True, "learned", "shared")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def parse_omic_sizes(value: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in value.split(",")]
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("omic sizes must be comma-separated positive integers") from error
    if not sizes or any(width < 1 for width in sizes):
        raise argparse.ArgumentTypeError("omic sizes must be comma-separated positive integers")
    return sizes


def protocol_template(baselines: list[str], omic_sizes: list[int], *,
                      wsi_input_dim: int = 1536, num_classes: int = 4) -> dict:
    """Describe choices that the experiment owner must fix before training."""
    inventory = lambda: {"path": REQUIRED, "sha256": REQUIRED}
    return {
        "schema": PROTOCOL_SCHEMA,
        "execution_status": "not_run",
        "patient_manifest": inventory(),
        "splits": {**inventory(), "unit": "patient", "folds": REQUIRED},
        "features": {
            "wsi_manifest": inventory(),
            "omics_manifest": inventory(),
            "wsi_encoder": {"name": REQUIRED, "weights_sha256": REQUIRED},
            "wsi_input_dim": wsi_input_dim,
            "omics_groups": {**inventory(), "omic_sizes": list(omic_sizes)},
            "preprocessing_rule": REQUIRED,
        },
        "endpoint": {"time_column": REQUIRED, "censoring_column": REQUIRED,
                     "censoring_convention": REQUIRED},
        "discrete_intervals": {"definition": REQUIRED, "sha256": REQUIRED,
                               "fit_population": REQUIRED, "num_classes": num_classes},
        "training": {
            "seeds": REQUIRED, "epochs": REQUIRED, "batching_rule": REQUIRED,
            "host_initialization": REQUIRED, "plugin_initialization": REQUIRED,
            "random_stream_policy": REQUIRED,
        },
        "checkpoint": {"selection_split": REQUIRED, "metric": REQUIRED,
                       "direction": REQUIRED, "tie_breaking": REQUIRED,
                       "selection_frequency": REQUIRED},
        "evaluation": {"prediction_split": REQUIRED, "c_index_definition": REQUIRED,
                       "aggregation_rule": REQUIRED},
        "baselines": {
            name: {
                "loss": {"name": REQUIRED, "implementation_sha256": REQUIRED,
                         "parameters": REQUIRED},
                "optimizer": {"name": REQUIRED, "parameters": REQUIRED},
                "training_engine": {"name": REQUIRED, "config_sha256": REQUIRED},
            } for name in baselines
        },
    }


def unresolved_fields(value: Any, prefix: str = "") -> list[str]:
    if value == REQUIRED:
        return [prefix]
    if isinstance(value, dict):
        return [path for key, entry in value.items()
                for path in unresolved_fields(entry, f"{prefix}.{key}" if prefix else key)]
    if isinstance(value, list):
        return [path for index, entry in enumerate(value)
                for path in unresolved_fields(entry, f"{prefix}[{index}]")]
    return []


def _validate_shape(value: Any, template: Any, path: str = "protocol") -> None:
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            expected = sorted(template)
            raise ValueError(f"{path} must contain exactly {expected}")
        for key, entry in template.items():
            _validate_shape(value[key], entry, f"{path}.{key}")
    elif template == REQUIRED:
        if value is None or value == REQUIRED or value == "" or value == []:
            raise ValueError(f"{path} must be supplied; REQUIRED is a template placeholder")
    elif value != template:
        raise ValueError(f"{path} must match the construction plan: {template!r}")


def _validate_hashes(value: Any, prefix: str = "protocol") -> None:
    if not isinstance(value, dict):
        return
    for key, entry in value.items():
        path = f"{prefix}.{key}"
        if key.endswith("sha256"):
            if (not isinstance(entry, str) or len(entry) != 64
                    or any(character not in "0123456789abcdef" for character in entry.lower())):
                raise ValueError(f"{path} must be a 64-character SHA-256 hex string")
        elif isinstance(entry, dict):
            _validate_hashes(entry, path)


def _integer_list(value: Any, name: str) -> None:
    if (not isinstance(value, list) or not value
            or any(isinstance(entry, bool) or not isinstance(entry, int) or entry < 0 for entry in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must be a nonempty list of unique nonnegative integers")


def validate_protocol(protocol: dict, template: dict) -> None:
    """Require an explicit, still unexecuted protocol matching all saved arms."""
    _validate_shape(protocol, template)
    _validate_hashes(protocol)
    _integer_list(protocol["splits"]["folds"], "splits.folds")
    _integer_list(protocol["training"]["seeds"], "training.seeds")
    _positive_integer(protocol["training"]["epochs"], "training.epochs")
    text_fields = (
        (protocol["patient_manifest"], "path"), (protocol["splits"], "path"),
        (protocol["features"]["wsi_manifest"], "path"),
        (protocol["features"]["omics_manifest"], "path"),
        (protocol["features"]["omics_groups"], "path"),
        (protocol["features"]["wsi_encoder"], "name"),
        (protocol["features"], "preprocessing_rule"),
        *((protocol["endpoint"], key) for key in protocol["endpoint"]),
        (protocol["discrete_intervals"], "definition"),
        (protocol["discrete_intervals"], "fit_population"),
        *((protocol["training"], key) for key in protocol["training"] if key not in {"seeds", "epochs"}),
        *((protocol["checkpoint"], key) for key in protocol["checkpoint"]),
        *((protocol["evaluation"], key) for key in protocol["evaluation"]),
    )
    for mapping, key in text_fields:
        if not isinstance(mapping[key], str) or not mapping[key].strip():
            raise ValueError(f"{key} must be a nonempty explicit string")
    if protocol["checkpoint"]["selection_split"] not in {"validation", "inner_validation"}:
        raise ValueError("checkpoint.selection_split must be validation or inner_validation")
    if protocol["checkpoint"]["direction"] not in {"maximize", "minimize"}:
        raise ValueError("checkpoint.direction must be maximize or minimize")
    for name, baseline in protocol["baselines"].items():
        for key in ("loss", "optimizer"):
            if not isinstance(baseline[key]["name"], str) or not baseline[key]["name"].strip():
                raise ValueError(f"baselines.{name}.{key}.name must be a nonempty string")
            if not isinstance(baseline[key]["parameters"], dict):
                raise ValueError(f"baselines.{name}.{key}.parameters must be an explicit parameter object")
        if not isinstance(baseline["training_engine"]["name"], str) or not baseline["training_engine"]["name"].strip():
            raise ValueError(f"baselines.{name}.training_engine.name must be a nonempty string")
    # A description cannot establish equality by itself. The run owner must
    # implement these recorded initialization and random-stream choices.


def construction_matrix(baselines: list[str], omic_sizes: list[int], *,
                        include_shared: bool = False, wsi_input_dim: int = 1536,
                        num_classes: int = 4) -> dict[str, dict[str, dict]]:
    """Return factory keyword arguments, with no model/data/job construction."""
    from survot_rank.plugins.transport.config import TransportPluginConfig

    if not baselines or len(set(baselines)) != len(baselines) or not set(baselines) <= set(BASELINE_COMMITS):
        raise ValueError("baselines must be unique supported names: mcat, survpath")
    if not isinstance(omic_sizes, list) or not omic_sizes:
        raise ValueError("omic_sizes must be a nonempty list")
    for width in omic_sizes:
        _positive_integer(width, "omic_sizes entry")
    if "mcat" in baselines and len(omic_sizes) != 6:
        raise ValueError("native MCAT requires exactly six omic sizes")
    _positive_integer(wsi_input_dim, "wsi_input_dim")
    _positive_integer(num_classes, "num_classes")
    if not isinstance(include_shared, bool):
        raise ValueError("include_shared must be bool")
    arms = (*BASE_ARMS, SHARED_ARM) if include_shared else BASE_ARMS
    matrix = {}
    for baseline in baselines:
        baseline_arms = {}
        for arm_id, enabled, mode, decoder_mode in arms:
            config = TransportPluginConfig(wsi_dim=256, omics_dim=256, dim=256,
                                           plan_mode=mode, residual_init=0.0).to_dict()
            reconstruction = None if decoder_mode is None else {
                "dim": 256, "num_omics_tokens": len(omic_sizes), "heads": 4,
                "mode": decoder_mode, "self_weight": 0.05, "cross_weight": 0.05,
                "wsi_weight": 0.05, "ramp_start": 2, "ramp_epochs": 5,
                "wsi_chunk_size": 256,
            }
            baseline_arms[arm_id] = {
                "backbone_name": baseline, "omic_sizes": list(omic_sizes),
                "wsi_input_dim": wsi_input_dim, "plugin_config": config,
                "reconstruction_config": reconstruction, "fusion": "concat",
                "enabled": enabled, "num_classes": num_classes,
            }
        matrix[baseline] = baseline_arms
    return matrix


def _implementation_inventory() -> dict[str, str]:
    paths = [Path("scripts/prepare_transport_plugin_experiments.py")]
    paths.extend(sorted((ROOT / "survot_rank/plugins/transport").glob("*.py")))
    inventory = {}
    for path in paths:
        absolute = path if path.is_absolute() else ROOT / path
        inventory[absolute.relative_to(ROOT).as_posix()] = hashlib.sha256(absolute.read_bytes()).hexdigest()
    return inventory


def write_plan(output: str | Path, baselines: list[str], omic_sizes: list[int], *,
               protocol: dict | None = None, include_shared: bool = False,
               wsi_input_dim: int = 1536, num_classes: int = 4) -> dict:
    """Write only into a new directory; existing outputs are never overwritten."""
    matrix = construction_matrix(baselines, omic_sizes, include_shared=include_shared,
                                 wsi_input_dim=wsi_input_dim, num_classes=num_classes)
    template = protocol_template(baselines, omic_sizes, wsi_input_dim=wsi_input_dim,
                                 num_classes=num_classes)
    if protocol is not None:
        validate_protocol(protocol, template)
    saved_protocol = deepcopy(template if protocol is None else protocol)
    protocol_bytes = _json_bytes(saved_protocol)
    serialized = {f"arms/{baseline}/{arm}.json": _json_bytes(config)
                  for baseline, arms in matrix.items() for arm, config in arms.items()}
    source_manifest = ROOT / "third_party/transport_plugin_baselines/SOURCE_MANIFEST.json"
    source_manifest_hash = (hashlib.sha256(source_manifest.read_bytes()).hexdigest()
                            if source_manifest.is_file() else REQUIRED)
    manifest = {
        "schema": PLAN_SCHEMA, "status": "plan_only_not_run", "executed": False,
        "protocol_file": "protocol.json",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "protocol_status": "pending_required_values" if protocol is None else "recorded_complete_unverified",
        "required_fields": unresolved_fields(saved_protocol),
        "data_hashes_and_patient_assignments_verified": False,
        "baseline_source_commits": {name: BASELINE_COMMITS[name] for name in baselines},
        "baseline_source_manifest_sha256": source_manifest_hash,
        "implementation_sha256": _implementation_inventory(),
        "arms": [{"baseline": baseline, "arm": arm,
                  "construction_file": f"arms/{baseline}/{arm}.json",
                  "construction_sha256": hashlib.sha256(serialized[f"arms/{baseline}/{arm}.json"]).hexdigest()}
                 for baseline, arms in matrix.items() for arm in arms],
        "comparisons": [
            {"arms": ["independent", "learned"],
             "question": "coupling effect without reconstruction; equal module capacity and learned marginals"},
            {"arms": ["independent_three_decoders", "learned_three_decoders"],
             "question": "coupling effect with the same three independent reconstruction heads"},
            {"arms": ["learned", "learned_three_decoders"],
             "question": "effect of auxiliary reconstruction; training head capacity differs"},
            {"arms": ["native", "learned"],
             "question": "whole added feature module; does not isolate optimal transport"},
        ] + ([{"arms": ["learned_three_decoders", "learned_shared_decoders"],
               "question": "self/cross decoder sharing; reconstruction parameter count differs"}]
             if include_shared else []),
        "constraints": [
            "All arms within a baseline use the same host encoder, features, patient splits, censoring and intervals.",
            "Use identical host initial weights; the native factory isolates optional auxiliary initialization from host/plugin RNG.",
            "Record and enforce separate host/plugin/auxiliary initialization and training random streams.",
            "Use each baseline's fixed native survival loss; add only forward_with_aux returned loss for auxiliary arms.",
            "MCAT input Linear uses the chosen WSI width equally in every arm, including native/plugin-off.",
            "Native/plugin-off bypasses the plugin; registered unused plugin parameters are not active host capacity.",
            "Evaluate each native source package in its own process; no training or measurement is performed by this script.",
        ],
    }
    serialized["protocol.json"] = protocol_bytes
    serialized["manifest.json"] = _json_bytes(manifest)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output already exists; choose a fresh directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    for relative, payload in serialized.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=("both", "mcat", "survpath"), default="both")
    parser.add_argument("--omic-sizes", required=True, type=parse_omic_sizes)
    parser.add_argument("--output", required=True, type=Path, help="new output directory; no overwrite")
    parser.add_argument("--protocol", type=Path, help="complete protocol JSON using the generated template schema")
    parser.add_argument("--include-shared", action="store_true", help="add shared self/cross decoder comparison")
    parser.add_argument("--wsi-input-dim", type=int, default=1536)
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol is not None else None
        manifest = write_plan(args.output, ["mcat", "survpath"] if args.baseline == "both" else [args.baseline],
                              args.omic_sizes, protocol=protocol, include_shared=args.include_shared,
                              wsi_input_dim=args.wsi_input_dim, num_classes=args.num_classes)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Saved {len(manifest['arms'])} arm configurations: {args.output.resolve()}")
    print(f"Protocol: {manifest['protocol_status']}; training and measurement were not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
