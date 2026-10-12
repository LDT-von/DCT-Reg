"""Prepare matched repaired v3.13 configs; never launches training.

Default: print the plan only. --out claims a new config directory exclusively.
This three-reconstruction recipe is separate from historical E048 v1.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full", "shared_decoder", "no_self", "no_cross", "no_wsi", "no_reconstruction", "direct", "independent", "no_diversity", "no_slot_nll")


def build_configs(base, results_root, arms=None):
    model = base.get("model", {})
    if model.get("dct_v313_revision") != "repaired":
        raise ValueError("requires an explicit repaired v3.13 base config")
    if model.get("dct_v313_wsi_reconstruction_budget") != "additive":
        raise ValueError("removal ablations require additive weights; surviving branches must not be rescaled")
    if model.get("dct_v313_recon_weighting") != "per_branch":
        raise ValueError("requires per_branch reconstruction weights")
    if model.get("dct_v313_disable_self_reconstruction") or model.get("dct_v313_disable_cross_reconstruction"):
        raise ValueError("base Full must enable both omics reconstruction branches")
    if model.get("dct_v313_cross_mode") != "transport" or model.get("dct_v313_plan_mode") != "learned":
        raise ValueError("base Full must use transport reconstruction and learned plans")
    for key in ("dct_v313_lambda_reconstruction", "dct_v313_lambda_wsi_reconstruction"):
        if float(model.get(key, 0.0)) <= 0:
            raise ValueError("base Full must enable all three reconstruction branches")
    if float(model.get("dct_v313_reconstruction_self_fraction", 0)) <= 0 or float(model.get("dct_v313_reconstruction_cross_fraction", 0)) <= 0:
        raise ValueError("base Full must enable both omics reconstruction fractions")
    decoder_mode = model.get("dct_v313_omics_decoder_mode", "shared")
    if decoder_mode not in {"shared", "separate"}:
        raise ValueError("unknown omics decoder mode")
    if arms is None:
        arms = ARMS if decoder_mode == "separate" else tuple(arm for arm in ARMS if arm != "shared_decoder")
    if "shared_decoder" in arms and decoder_mode != "separate":
        raise ValueError("shared_decoder control requires a separate-decoder Full base")
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("select unique known arms")
    configs = {}
    for arm in arms:
        item = copy.deepcopy(base)
        item["name"] = "dct_v313_repaired_" + arm
        item["train"]["specific_simple"] = item["name"]
        item["train"]["results_dir"] = str(Path(results_root) / arm)
        settings = item["model"]
        if arm == "shared_decoder":
            settings["dct_v313_omics_decoder_mode"] = "shared"
        if arm in {"no_self", "no_reconstruction"}:
            settings["dct_v313_disable_self_reconstruction"] = True
        if arm in {"no_cross", "no_reconstruction"}:
            settings["dct_v313_disable_cross_reconstruction"] = True
        if arm in {"no_wsi", "no_reconstruction"}:
            settings["dct_v313_lambda_wsi_reconstruction"] = 0.0
        if arm == "direct":
            settings["dct_v313_cross_mode"] = "direct"
        if arm == "independent":
            settings["dct_v313_plan_mode"] = "independent"
        if arm == "no_diversity":
            settings["dct_v311_lambda_slot_diversity"] = 0.0
        if arm == "no_slot_nll":
            settings["dct_v311_lambda_slot_nll"] = 0.0
        configs[arm] = item
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "configs/dct_v313_full_wsi_three_decoders.yaml")
    parser.add_argument("--results-root", required=True, help="A new run root, with a separate child per arm.")
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--out", type=Path, help="Optionally save configs in a new directory; never overwrite.")
    options = parser.parse_args(argv)
    base = yaml.safe_load(options.base.read_text(encoding="utf-8-sig"))
    configs = build_configs(base, options.results_root, options.arm)
    if options.out:
        options.out.mkdir(parents=True, exist_ok=False)
        for arm, config in configs.items():
            with (options.out / f"{arm}.yaml").open("x", encoding="utf-8", newline="\n") as stream:
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    print(json.dumps({"mode": "configs_only_no_training", "arms": list(configs), "results_root": options.results_root}, ensure_ascii=False, indent=2))
    if options.out:
        print("Configs saved. Launch each selected config explicitly with: python -m survot_rank.cli train --config <config.yaml>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
