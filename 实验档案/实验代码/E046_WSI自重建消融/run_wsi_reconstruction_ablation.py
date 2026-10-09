"""E046 v3.13 WSI reconstruction: inspect a plan by default; --execute trains."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
CANCERS = ("brca", "coadread", "kirc", "ucec", "luad", "lusc", "hnsc", "skcm", "blca", "stad")
ARMS = {"full": "additive", "wsi_additive": "additive", "wsi_fixed_total": "fixed_total"}


def sha256(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, default=ROOT / "configs/dct_v313_uni2h.yaml")
    p.add_argument("--results-root", type=Path, required=True, help="Fresh batch; never overwrite prior results")
    p.add_argument("--cancers", nargs="+", default=["blca", "kirc"], choices=[*CANCERS, "all"])
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--seeds", nargs="+", type=int, default=[3])
    p.add_argument("--protocol", choices=["legacy_val", "outer_test"], default="outer_test")
    p.add_argument("--wsi-weight", type=float, default=0.05, help="Pre-specified raw WSI weight; not fold-tuned")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--gpu", default="0", help="One GPU; tasks run sequentially")
    p.add_argument("--data-path", type=Path)
    p.add_argument("--data-root-dir", type=Path)
    p.add_argument("--which-splits")
    p.add_argument("--execute", action="store_true")
    return p


def build_plan(args):
    if not math.isfinite(args.wsi_weight) or args.wsi_weight <= 0 or args.chunk_size <= 0:
        raise ValueError("WSI weight must be finite and positive; chunk size must be positive")
    cancers = list(CANCERS) if args.cancers == ["all"] else args.cancers
    if "all" in cancers or any(c not in CANCERS for c in cancers):
        raise ValueError("Use --cancers all alone, or explicit cohort names")
    for name, values in (("cancers", cancers), ("arms", args.arms), ("folds", args.folds), ("seeds", args.seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
    if any(a not in ARMS for a in args.arms):
        raise ValueError("Unknown arm")
    if any(f < 0 or f > 4 for f in args.folds) or any(s < 0 for s in args.seeds):
        raise ValueError("Folds must be 0..4 and seeds nonnegative")
    config_path = args.base_config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Base config must be a YAML mapping")
    flat = {}
    for key, value in config.items():
        flat.update(value if isinstance(value, dict) else {key: value})
    if flat.get("survot_method") not in ("dct_v313", "dct_v313_transport_reconstruction"):
        raise ValueError("E046 only supports v3.13")
    data_path = args.data_path or Path(flat["data_path"])
    data_root = args.data_root_dir or Path(flat["data_root_dir"])
    which = args.which_splits or (
        "5fold_uni2h_outer_test_seed3" if args.protocol == "outer_test" else flat.get("which_splits", "5fold_uni2h")
    )
    root = args.results_root.resolve()
    source_paths = [
        p.relative_to(ROOT).as_posix()
        for p in sorted((ROOT / "survot_rank").rglob("*.py"))
    ]
    source_paths.append(Path(__file__).resolve().relative_to(ROOT).as_posix())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    plan = dict(
        experiment="E046", status="planned_not_measured", protocol=args.protocol,
        source_revision=revision,
        source_sha256={p: sha256(ROOT / p) for p in source_paths},
        config=str(config_path), config_sha256=sha256(config_path), results_root=str(root),
        target="detached projected WSI patch features; query-conditioned self reconstruction",
        coefficient_note="fixed_total preserves the original active omics budget, not loss magnitude or gradient norm",
        tasks=[],
    )
    for arm, cancer, fold, seed in itertools.product(args.arms, cancers, args.folds, args.seeds):
        output = root / args.protocol / arm / cancer / f"fold{fold}" / f"seed{seed}"
        split = data_path / "splits" / which / cancer / f"fold_{fold}.csv"
        weight = 0.0 if arm == "full" else args.wsi_weight
        options = {
            "survot_method": "dct_v313_transport_reconstruction", "study": cancer,
            "specific_simple": f"v313_e046_{arm}_{cancer}", "seed": seed,
            "k_start": fold, "k_end": fold + 1, "results_dir": str(output),
            "evaluation_protocol": args.protocol, "which_splits": which,
            "data_path": str(data_path), "data_root_dir": str(data_root),
            "gpu": 0, "dct_v313_recon_weighting": "per_branch",
            "dct_v313_lambda_reconstruction_scale": 1.0,
            "dct_v313_disable_self_reconstruction": "false",
            "dct_v313_disable_cross_reconstruction": "false",
            "dct_v313_cross_mode": "transport", "dct_v313_plan_mode": "learned",
            "dct_v313_lambda_wsi_reconstruction": weight,
            "dct_v313_wsi_reconstruction_budget": ARMS[arm],
            "dct_v313_wsi_reconstruction_chunk_size": args.chunk_size,
        }
        if args.protocol == "outer_test":
            options.update(inner_val_fraction=0.20, split_seed=3)
        command = [sys.executable, "-m", "survot_rank.cli", "train", "--config", str(config_path)]
        for key, value in options.items():
            command.extend(["--set", f"{key}={value}"])
        plan["tasks"].append(dict(
            arm=arm, cancer=cancer, fold=fold, seed=seed,
            output=str(output), split=str(split), split_sha256=sha256(split),
            overrides=options, command=command, physical_gpu=str(args.gpu),
        ))
    return plan


def validate_source(plan):
    for relative, digest in plan["source_sha256"].items():
        if not digest or sha256(ROOT / relative) != digest:
            raise ValueError(f"Source changed after planning: {relative}")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != plan["source_revision"]:
        raise ValueError("Source commit changed after planning")


def execute_plan(plan):
    root = Path(plan["results_root"])
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Refuse to overwrite nonempty results root: {root}")
    validate_source(plan)
    if sha256(plan["config"]) != plan["config_sha256"]:
        raise ValueError("Base config changed after planning")
    if any(not task["split_sha256"] for task in plan["tasks"]):
        raise ValueError("All requested split CSVs must exist before training")
    # Preflight the whole batch before creating logs or starting a child.
    for task in plan["tasks"]:
        if sha256(task["split"]) != task["split_sha256"]:
            raise ValueError(f"Split changed: {task['split']}")
        if not Path(task["overrides"]["data_root_dir"]).is_dir():
            raise ValueError("WSI feature root does not exist")
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "base_config.yaml"
    snapshot.write_bytes(Path(plan["config"]).read_bytes())
    for task in plan["tasks"]:
        task["execution_command"] = list(task["command"])
        task["execution_command"][task["execution_command"].index("--config") + 1] = str(snapshot)
    (root / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logs = root / "logs"
    logs.mkdir()
    for task in plan["tasks"]:
        validate_source(plan)
        if sha256(snapshot) != plan["config_sha256"]:
            raise ValueError("Config snapshot changed before launch")
        if sha256(task["split"]) != task["split_sha256"]:
            raise ValueError(f"Split changed before launch: {task['split']}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = task["physical_gpu"]
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # Use the frozen config copy; task overrides remain recorded verbatim.
        command = task["execution_command"]
        log = logs / f"{task['arm']}_{task['cancer']}_f{task['fold']}_s{task['seed']}.log"
        print(f"Training {task['arm']} {task['cancer']} fold{task['fold']} seed{task['seed']}", flush=True)
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.execute:
        execute_plan(plan)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


if __name__ == "__main__":
    main()
