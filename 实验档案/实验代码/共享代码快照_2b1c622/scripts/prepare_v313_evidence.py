#!/usr/bin/env python3
"""Ordered v3.13 evidence workflow; training is dry-run unless --execute."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from survot_rank.evidence.manifest import audit, discover, load_manifest, revision, save_json, sha256


def prepare_plan(args):
    import yaml
    from survot_rank.config import load_config
    from survot_rank.training.scheduler import assign_gpus, build_task_plan, launch_tasks
    if args.epochs < 1 or not args.gpu.split(",") or any(not item.isdigit() for item in args.gpu.split(",")):
        raise ValueError("Require positive epochs and comma-separated physical GPU indices")
    if len(set(args.gpu.split(","))) != len(args.gpu.split(",")):
        raise ValueError("GPU indices must be unique")
    arms = {"controls": ["direct", "independent"], "controls-full": ["exp6", "direct", "independent"],
            "matched-recon": ["exp4", "exp5"]}[args.group]
    tasks = build_task_plan(protocols=["legacy_val"], arms=arms, cancers=args.cancer or ["blca", "kirc"],
        folds=args.fold or range(5), seeds=args.seed or [3], project_root=str(ROOT), data_path=args.data_path,
        outer_split_root="unused", results_root=args.results_root)
    tasks = assign_gpus(tasks, args.gpu.split(","), 1)
    plan_path = Path(args.output).resolve()
    snapshots = plan_path.parent / (plan_path.stem + "_configs")
    snapshots.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        if not Path(task.config_yaml_path).exists() and task.arm in ("exp4", "exp5"):
            task.config_yaml_path = str(ROOT / "configs" / f"exp6_{task.cancer}_uni2h.yaml")
        source_config = Path(task.config_yaml_path)
        config = load_config(source_config)
        snapshot = snapshots / f"{task.arm}_{task.cancer}.yaml"
        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != text:
            raise ValueError(f"Config snapshot already exists with different contents: {snapshot}")
        snapshot.write_text(text, encoding="utf-8")
        task.config_yaml_path = str(snapshot)
        task.extra_set += [f"data_path={args.data_path}", f"data_root_dir={args.feature_root}", f"max_epochs={args.epochs}"]
    plan = dict(schema_version=1, source_commit=revision(ROOT), group=args.group, tasks=[task.to_dict() for task in tasks])
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Existing plan differs; choose a new output filename")
    save_json(plan_path, plan)
    missing = [task.task_id() for task in tasks if task.split_fingerprint == "missing"]
    print(f"[plan] {len(tasks)} tasks; plan={plan_path}; missing splits={len(missing)}")
    if args.execute:
        if missing:
            raise ValueError(f"Cannot launch: missing splits {missing}")
        for task in tasks:
            if Path(task.results_dir).exists() and any(Path(task.results_dir).iterdir()):
                raise ValueError(f"Refuse to overwrite an existing task: {task.results_dir}; use a fresh results root")
        summary = launch_tasks(tasks, project_root=str(ROOT), gpus=args.gpu.split(","), jobs_per_gpu=1, execute=True)
        save_json(plan_path.with_suffix(".launch.json"), summary)
        if summary["failed"]:
            raise ValueError(f"{summary['failed']} training tasks failed; see launch JSON and task logs")
    else:
        print("[plan] dry-run: no training launched. Add --execute on the server to run this exact plan.")


def bind_plan(args):
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    from survot_rank.config import apply_overrides, flatten_config, load_config
    runs, missing = [], []
    for task in plan["tasks"]:
        fold = task["fold"]
        curves = list(Path(task["results_dir"]).rglob(f"epoch_curve_fold{fold}.csv"))
        if len(curves) != 1:
            missing.append(f"{task['task_id']}: expected one curve, found {len(curves)}")
            continue
        directory = curves[0].parent
        flat = flatten_config(apply_overrides(load_config(task["config_yaml_path"]), task["extra_set"]))
        runs.append(dict(id=f"{task['arm']}_{task['cancer']}_f{fold}_s{task['seed']}",
            arm=task["arm"], cancer=task["cancer"], fold=fold, seed=task["seed"], protocol=task["protocol"],
            source_commit=plan["source_commit"], config=task["config_yaml_path"], overrides=task["extra_set"],
            curve=str(curves[0].resolve()), checkpoint=str((directory / f"model_best_s{fold}.pth").resolve()),
            predictions=str((directory / f"split_{fold}_results.pkl").resolve()),
            split_csv=str(Path(flat["data_path"]) / "splits" / flat["which_splits"] / task["cancer"] / f"fold_{fold}.csv")))
    save_json(args.output, dict(schema_version=1, runs=runs, missing=missing))
    if missing:
        raise ValueError("Incomplete plan: " + "; ".join(missing))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Prepare controls/matched recon; dry-run by default")
    plan.add_argument("--group", choices=["controls", "controls-full", "matched-recon"], default="controls")
    plan.add_argument("--results-root", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--data-path", default="/data1/dataset_csv")
    plan.add_argument("--feature-root", default="/data1/TCGA-UNI2-h-features")
    plan.add_argument("--cancer", action="append", choices=["blca", "kirc"])
    plan.add_argument("--fold", action="append", type=int, choices=range(5))
    plan.add_argument("--seed", action="append", type=int)
    plan.add_argument("--epochs", type=int, default=30)
    plan.add_argument("--gpu", default="0,1")
    plan.add_argument("--execute", action="store_true")
    bind = commands.add_parser("bind", help="Resolve actual hash-nested outputs from a training plan")
    bind.add_argument("--plan", required=True)
    bind.add_argument("--output", required=True)
    scan = commands.add_parser("scan", help="Add an explicitly labeled group of historical DCT results")
    for key in ("root", "config", "arm", "cancer", "source-commit", "output"):
        scan.add_argument("--" + key, required=True)
    scan.add_argument("--data-path", default="/data1/dataset_csv")
    scan.add_argument("--seed", type=int, default=3)
    scan.add_argument("--set", action="append", default=[])
    scan.add_argument("--append", action="store_true")
    merge = commands.add_parser("merge", help="Merge manifests explicitly; audit catches duplicate identities")
    merge.add_argument("--manifest", action="append", required=True)
    merge.add_argument("--output", required=True)
    check = commands.add_parser("audit", help="Hash/identity/epoch/metric integrity; exits nonzero on failure")
    check.add_argument("--manifest", required=True)
    check.add_argument("--output", required=True)
    export = commands.add_parser("export", help="User-triggered real-data checkpoint inference")
    export.add_argument("--manifest", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--arm", action="append")
    export.add_argument("--fold", type=int, action="append")
    export.add_argument("--cancer", action="append")
    export.add_argument("--case-id", action="append", default=[])
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    export.add_argument("--km", action="store_true", help="Also infer train patients for fixed train-median KM groups and Brier support")
    export.add_argument("--profile", action="store_true")
    plot = commands.add_parser("plot", help="Generate PDF/PNG figures from audited results/exports")
    plot.add_argument("--manifest", required=True)
    plot.add_argument("--exports")
    plot.add_argument("--coordinates", help="Optional verified feature-to-coordinate manifest")
    plot.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            prepare_plan(args)
        elif args.command == "bind":
            bind_plan(args)
        elif args.command == "scan":
            old = load_manifest(args.output)["runs"] if args.append and Path(args.output).exists() else []
            runs = discover(args.root, args.config, args.arm, args.cancer, args.seed, args.source_commit, args.data_path, args.set)
            save_json(args.output, dict(schema_version=1, runs=old + runs))
            print(f"[scan] added {len(runs)} runs; labels and historical overrides must match the actual experiment")
        elif args.command == "merge":
            runs = [run for path in args.manifest for run in load_manifest(path)["runs"]]
            save_json(args.output, dict(schema_version=1, runs=runs))
        elif args.command == "audit":
            report = audit(load_manifest(args.manifest))
            save_json(args.output, report)
            print(f"[audit] passed={report['passed']} runs={len(report['runs'])} errors={len(report['errors'])}")
            if not report["passed"]:
                raise ValueError("\n".join(report["errors"]))
        elif args.command == "export":
            from survot_rank.evidence.v313 import export_run
            manifest = load_manifest(args.manifest)
            report = audit(manifest)
            if not report["passed"]:
                raise ValueError("\n".join(report["errors"]))
            selected = [run for run in manifest["runs"] if
                (not args.arm or run["arm"] in args.arm) and (not args.fold or run["fold"] in args.fold) and
                (not args.cancer or run["cancer"] in args.cancer)]
            if not selected:
                raise ValueError("No runs match the export selection")
            alphas = tuple(float(value) for value in args.alphas.split(","))
            for run in selected:
                export_run(run, args.output, device=args.device, alphas=alphas, case_ids=args.case_id, km=args.km, profile=args.profile)
        elif args.command == "plot":
            from survot_rank.evidence.plots import make_figures
            make_figures(load_manifest(args.manifest), args.output, exports=args.exports, coordinates=args.coordinates)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"[evidence] {error}\n")


if __name__ == "__main__":
    main()
