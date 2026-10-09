"""Generate the v3.13 experiment-matrix YAML configs for §3.2.

The DCT v3.13 plan §2.5 defines 11 experiment arms × multiple cohorts
× 5 folds × multiple model seeds.  Each arm inherits the same data /
encoder / slot / hyperparameter settings from
``configs/dct_v313_blca_uni2h.yaml`` and differs in which loss terms
or model flags are *enabled*.  Per-arm losses are applied at run time
via ``--set`` overrides emitted by ``survot_rank.training.scheduler``;
this generator only needs to author the *base* YAML — the scheduler
applies the per-arm deltas at launch time.

Usage::

    PYTHONPATH=/data1/DCT-Reg python -m survot_rank.training.generate_v313_configs \
        --base-config configs/dct_v313_blca_uni2h.yaml \
        --output-dir configs \
        --cohorts blca,kirc,hnsc,lusc,skcm

The script is idempotent: re-running with the same arguments produces
identical files.  It will NOT touch any pre-existing files unless
``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import yaml


# Per-arm config templates.  All other fields come from the base config.
ARM_TEMPLATES = {
    "exp0": {
        "name": "dct_v313_exp0",
        "description": (
            "DCT v3.13 §2.1 Exp0 — NLL only.  All auxiliary terms disabled; "
            "per_branch weighting, both reconstruction branches disabled."
        ),
        "specific_simple": "dct_v313_exp0",
    },
    "exp1": {
        "name": "dct_v313_exp1",
        "description": (
            "DCT v3.13 §2.1 Exp1 — NLL + 0.10 IPCW rank.  Per-slot NLL, "
            "diversity, and reconstruction disabled."
        ),
        "specific_simple": "dct_v313_exp1",
    },
    "exp2": {
        "name": "dct_v313_exp2",
        "description": (
            "DCT v3.13 §2.1 Exp2 — Exp1 + 0.05 per-slot NLL.  Diversity and "
            "reconstruction disabled."
        ),
        "specific_simple": "dct_v313_exp2",
    },
    "exp3": {
        "name": "dct_v313_exp3",
        "description": (
            "DCT v3.13 §2.1 Exp3 — Exp2 + 0.10 diversity.  Reconstruction "
            "disabled."
        ),
        "specific_simple": "dct_v313_exp3",
    },
    "exp4": {
        "name": "dct_v313_exp4",
        "description": (
            "DCT v3.13 §2.1 Exp4 — Exp3 + 0.05 self reconstruction only."
        ),
        "specific_simple": "dct_v313_exp4",
    },
    "exp5": {
        "name": "dct_v313_exp5",
        "description": (
            "DCT v3.13 §2.1 Exp5 — Exp3 + 0.05 cross reconstruction only."
        ),
        "specific_simple": "dct_v313_exp5",
    },
    "exp6": {
        "name": "dct_v313_exp6",
        "description": (
            "DCT v3.13 §2.1 Exp6 — full v3.13 (Exp3 + 0.05 self + 0.05 cross)."
        ),
        "specific_simple": "dct_v313_exp6",
    },
    "slotspe_native": {
        "name": "slotspe_native",
        "description": (
            "SlotSPE original-model reference under the v3.13 unified "
            "data protocol.  Native recipe: batch=32, Adam, alpha_surv=0.5."
        ),
        "specific_simple": "slotspe_native",
    },
    "slotspe_matched": {
        "name": "slotspe_matched",
        "description": (
            "SlotSPE original-model reference.  Matched recipe: batch=8, "
            "AdamW, alpha_surv=0.15 (matches DCT v3.13)."
        ),
        "specific_simple": "slotspe_matched",
    },
    "direct": {
        "name": "dct_v313_direct",
        "description": (
            "DCT v3.13 §2.3 — full v3.13 with cross_mode=direct.  "
            "Cross decoder receives un-transported WSI slots."
        ),
        "specific_simple": "dct_v313_direct",
    },
    "independent": {
        "name": "dct_v313_independent",
        "description": (
            "DCT v3.13 §2.4 — full v3.13 with plan_mode=independent.  "
            "Cross-decoder OT plan replaced with the marginal outer product "
            "T = a bᵀ in every consumer."
        ),
        "specific_simple": "dct_v313_independent",
    },
}


def _render_config(base: dict, arm: str, cohort: str) -> dict:
    """Render a per-arm config dict from a base config dict.

    The base config supplies all data / encoder / slot / model fields.
    We override only ``name``, ``description``, ``data.study``,
    ``train.specific_simple``, ``train.results_dir``, and ``train.survot_method``
    (for arms whose ``base_method`` differs).
    """
    spec = ARM_TEMPLATES[arm]
    # Use a deep copy via PyYAML that preserves order.
    import copy
    out = copy.deepcopy(base)
    out["name"] = spec["name"]
    out["description"] = spec["description"]
    # Cohort-specific overrides.
    out["data"]["study"] = cohort
    out["train"]["specific_simple"] = (
        f"{spec['specific_simple']}_{cohort}_uni2h"
    )
    # Per-arm results dir collision-free under the v3.13 matrix root.
    out["train"]["results_dir"] = (
        f"/data1/DCT-Reg/results/v313_paper_v1/base/{arm}/{cohort}"
    )
    # Per-arm method is set by the scheduler via --set, but we still
    # stamp the base config's survot_method for documentation.
    if arm.startswith("slotspe"):
        out["train"]["survot_method"] = "slotspe_reference"
    return out


def _resolve_template_filename(arm: str, cohort: str) -> str:
    """Return the canonical filename for an arm × cohort pair.

    SlotSPE arms keep their existing filename convention so we do not
    collide with the hand-authored configs already in configs/.
    """
    if arm.startswith("slotspe"):
        return f"slotspe_{arm[len('slotspe_'):]}_{cohort}_uni2h.yaml"
    return f"{arm}_{cohort}_uni2h.yaml"


def generate_configs(
    *,
    base_config_path: str,
    output_dir: str,
    cohorts: Iterable[str],
    overwrite: bool = False,
) -> dict[str, list[str]]:
    """Generate per-arm × per-cohort YAML configs.

    Returns a nested mapping ``{cohort: [filenames]}`` for audit logging.
    """
    with open(base_config_path, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)
    outputs: dict[str, list[str]] = {cohort: [] for cohort in cohorts}
    os.makedirs(output_dir, exist_ok=True)
    for cohort in cohorts:
        for arm in ARM_TEMPLATES:
            rendered = _render_config(base, arm, cohort)
            filename = _resolve_template_filename(arm, cohort)
            out_path = os.path.join(output_dir, filename)
            if os.path.exists(out_path) and not overwrite:
                outputs[cohort].append(f"{filename} (skipped)")
                continue
            with open(out_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(rendered, fh, allow_unicode=True, sort_keys=False)
            outputs[cohort].append(filename)
    return outputs


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-arm × per-cohort YAML configs for §3.2."
    )
    parser.add_argument(
        "--base-config",
        default="configs/dct_v313_blca_uni2h.yaml",
        help="Source YAML; per-arm configs inherit every field from this.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs",
        help="Where to write the per-arm × per-cohort configs.",
    )
    parser.add_argument(
        "--cohorts",
        default="blca,kirc",
        help="Comma-separated cohort list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    outputs = generate_configs(
        base_config_path=args.base_config,
        output_dir=args.output_dir,
        cohorts=cohorts,
        overwrite=args.overwrite,
    )
    for cohort, filenames in outputs.items():
        for fname in filenames:
            print(f"[generate] cohort={cohort} -> {fname}")


if __name__ == "__main__":
    main()