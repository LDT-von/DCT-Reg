"""YAML experiment configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .project import resolve_project_path


PATH_KEYS = {"data_path", "results_dir"}
ABS_OR_EXTERNAL_PATH_KEYS = {"data_root_dir", "dct_v23_calibration_dir"}
METADATA_KEYS = {"name", "description", "notes", "_dct_user_overrides"}
BOOLEAN_VALUE_KEYS = {
    "dct_v383_center_slots",
    "dct_v39_center_slots",
    "dct_v39_tau_autoscale",
    # DCT v3.13 ablation flags; emitted as explicit "true"/"false" tokens
    # so argparse (type=str) can consume them via --set key=true.
    "dct_v313_disable_self_reconstruction",
    "dct_v313_disable_cross_reconstruction",
}
# Only these new DCT v2 flags use argparse.BooleanOptionalAction. Preserve
# explicit false values and emit their negative flag without changing legacy
# store_true/value-boolean serialization.
BOOLEAN_OPTIONAL_KEYS = {"dct_v2_mass_aware", "dct_v2_local_gating", "dct_v22_evaluate_train"}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten sectioned YAML into argparse-style key/value pairs.

    Reserved metadata sections are ignored. All other nested sections are merged
    into one parameter dictionary because the legacy runner expects flat CLI
    flags.
    """
    flat: dict[str, Any] = {}
    for key, value in config.items():
        if key in METADATA_KEYS:
            continue
        if isinstance(value, dict):
            flat.update({name: item for name, item in value.items() if name not in METADATA_KEYS})
        else:
            flat[key] = value
    return flat


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply explicit CLI values and record constructor override provenance.

    Historical recipes intentionally keep their existing frozen-argument
    behavior: only ``--set`` keys count as explicit.  A repaired v3.13 recipe
    also treats values actually present in its YAML sections as explicit.
    Parser defaults are never added to this list.
    """
    merged = deepcopy(config)
    user_keys = []
    override_values = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format: {item}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        merged[key] = value
        user_keys.append(key)
        override_values[key] = value
    effective = flatten_config(merged)
    # Evaluate the revision after CLI overrides even when a top-level YAML
    # value originally occurs before a section containing the same key.
    effective.update(override_values)
    if effective.get("dct_v313_revision", "legacy") == "repaired":
        # Flattening visits sections in insertion order.  Reinsert explicit
        # top-level overrides last, so a later YAML section cannot shadow them.
        for key, value in override_values.items():
            merged.pop(key, None)
            merged[key] = value
        user_keys = list(flatten_config(merged))
    # Reserved metadata is never a model parameter or a CLI flag.
    user_keys = [key for key in user_keys if key not in METADATA_KEYS]
    merged["_dct_user_overrides"] = user_keys
    return merged


def config_to_argv(config: dict[str, Any]) -> list[str]:
    flat = flatten_config(config)
    argv: list[str] = []
    for key, value in flat.items():
        if value is None or (value is False and key not in BOOLEAN_VALUE_KEYS | BOOLEAN_OPTIONAL_KEYS):
            continue
        if key in PATH_KEYS:
            value = resolve_project_path(value)
        elif key in ABS_OR_EXTERNAL_PATH_KEYS and str(value).startswith("."):
            value = resolve_project_path(value)
        flag = f"--{key}"
        if isinstance(value, bool) and key in BOOLEAN_OPTIONAL_KEYS:
            argv.append(flag if value else f"--no-{key}")
        elif isinstance(value, bool) and key in BOOLEAN_VALUE_KEYS:
            argv.extend([flag, str(value).lower()])
        elif value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            # Special case: clinical_feature_cols is passed as comma-separated string
            if key == "clinical_feature_cols":
                argv.append(flag)
                argv.append(",".join(str(item) for item in value))
            else:
                argv.append(flag)
                argv.extend(str(item) for item in value)
        else:
            argv.extend([flag, str(value)])
    return argv

