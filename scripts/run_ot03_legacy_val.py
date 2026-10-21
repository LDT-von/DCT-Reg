#!/usr/bin/env python3
"""Driver: generate legacy_val configs and optionally launch parallel jobs.

Usage:
    python3 scripts/run_ot03_legacy_val.py --dry-run   # generate configs only
    python3 scripts/run_ot03_legacy_val.py --launch    # launch all jobs
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "survot_rank" / "plugins" / "transport_only" / "slotspe_compare.py"
CONFIG_ROOT = Path("/data1/ot03_v4_legacy_val_configs")
RUN_ROOT = Path("/data1/ot03_v4_legacy_val")
GPUS = ["cuda:0", "cuda:1"]

RUNS = [(f, s) for f in range(5) for s in (3, 5, 7, 9, 11)]


def build_config(fold: int, seed: int) -> dict:
    return {
        "study": "blca",
        "label_col": "survival_months_dss",
        "signature": "combine",
        "folds": [fold],
        "seeds": [seed],
        "epochs": 30,
        "patches": 2048,
        "batch_size": 8,
        "eval_repeats": 20,
        "encoding_dim": 1536,
        "device": "cuda:0",
        "lr": 0.0005,
        "experiment": "reproduction",
        "torch_threads": 2,
        "source_root": "/data1/DCT-Reg/third_party/SlotSPE",
        "metadata_root": "/data1/dataset_csv",
        "features_root": "/data1/DCT-Reg/data/slotspe_pt_features/blca/pt_files",
        "selection": "legacy_val",
    }


def run_name(fold: int, seed: int) -> str:
    return f"OT-03_legacy_fold{fold}_seed{seed}"


def plan_one(fold: int, seed: int, cfg_dir: Path, dry_run: bool = True):
    """Plan a single fold-seed, optionally dry-run."""
    cfg = build_config(fold, seed)
    cfg_path = cfg_dir / f"config_f{fold}_s{seed}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=2))

    output = RUN_ROOT / run_name(fold, seed)
    if output.exists():
        print(f"  SKIP {output.name}: already exists")
        return None

    cmd = [
        sys.executable, "-m", "survot_rank.plugins.transport_only.slotspe_compare",
        "--config", str(cfg_path),
        "--output", str(output),
    ]
    if not dry_run:
        cmd.append("--run")

    print(f"  {'[plan]' if dry_run else '[run]'} fold={fold} seed={seed} → {output.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def launch_jobs(gpu_assignments: dict):
    """Launch jobs, respecting GPU assignments. Returns list of (fold,seed,gpu,popen)."""
    processes = []
    for gpu_idx, runs in gpu_assignments.items():
        for fold, seed in runs:
            cfg = build_config(fold, seed)
            cfg_dir = CONFIG_ROOT / f"gpu{gpu_idx}"
            cfg_path = cfg_dir / f"config_f{fold}_s{seed}.json"
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, indent=2))

            output = RUN_ROOT / run_name(fold, seed)
            # Patch device in config for this GPU
            cfg["device"] = GPUS[gpu_idx]

            cmd = [
                sys.executable, str(SCRIPT),
                "--config", str(cfg_path),
                "--output", str(output),
                "--run",
            ]
            print(f"Launching: fold={fold} seed={seed} gpu={GPUS[gpu_idx]}")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            processes.append((fold, seed, gpu_idx, p))
    return processes


def monitor(processes, poll_interval=30):
    """Monitor running processes, print tail output."""
    import select
    active = {p: (f, s) for f, s, _, p in processes}
    gpu_last_line = {gpu_idx: "" for gpu_idx in range(len(GPUS))}

    while active:
        for p in list(active.keys()):
            f, s = active[p]
            gpu_idx = next(g for ff, ss, ggi, pp in processes if pp is p)
            # Read available stdout
            line = p.stdout.readline()
            if line:
                gpu_last_line[gpu_idx] = line.rstrip()
                if "epoch=" in line or "selected epoch" in line or "error" in line.lower():
                    print(f"  GPU{gpu_idx} f{f}_s{s}: {line.rstrip()}")
            # Check if done
            if p.poll() is not None:
                rc = p.returncode
                status = "✓" if rc == 0 else f"✗ rc={rc}"
                print(f"\n  === GPU{gpu_idx} f{f}_s{s} {status} ===")
                if rc != 0:
                    # Print remaining output
                    remaining = p.stdout.read()
                    if remaining:
                        print(remaining[-2000:])
                del active[p]
        time.sleep(poll_interval)

    print("\nAll jobs finished.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan only (generate configs, no training)")
    parser.add_argument("--launch", action="store_true",
                        help="Launch all jobs in parallel (two GPUs)")
    parser.add_argument("--single", type=int, nargs=2, metavar=("FOLD", "SEED"),
                        help="Run a single fold-seed immediately")
    parser.add_argument("--test-one", action="store_true",
                        help="Run fold=0 seed=3 as quick test")
    args = parser.parse_args()

    if args.single:
        fold, seed = args.single
        print(f"Running single: fold={fold} seed={seed}")
        r = plan_one(fold, seed, CONFIG_ROOT / "single", dry_run=False)
        if r and r.stdout:
            print(r.stdout[-3000:])
        if r and r.stderr:
            print(r.stderr[-2000:])
        return

    if args.test_one:
        print("Test run: fold=0 seed=3")
        r = plan_one(0, 3, CONFIG_ROOT / "test", dry_run=False)
        if r:
            print("STDOUT:", r.stdout[-3000:])
            print("STDERR:", r.stderr[-2000:] if r.stderr else "")
            print("RC:", r.returncode)
        return

    if not args.dry_run and not args.launch:
        parser.error("Specify --dry-run or --launch or --single FOLD SEED or --test-one")

    if args.dry_run:
        print("=== Generating configs (dry run) ===")
        for fold, seed in RUNS:
            r = plan_one(fold, seed, CONFIG_ROOT / "dry_run", dry_run=True)
        print(f"\nGenerated {len(RUNS)} configs in {CONFIG_ROOT / 'dry_run'}")
        return

    if args.launch:
        print("=== Launching all 25 legacy_val runs ===")
        gpu0_runs = [(f, s) for f, s in RUNS if (f * 5 + (3, 5, 7, 9, 11).index(s)) < 13]
        gpu1_runs = [(f, s) for f, s in RUNS if (f, s) not in gpu0_runs]
        print(f"GPU0: {len(gpu0_runs)} runs")
        print(f"GPU1: {len(gpu1_runs)} runs")

        # Generate all configs first
        for gpu_idx, runs in [(0, gpu0_runs), (1, gpu1_runs)]:
            for fold, seed in runs:
                cfg = build_config(fold, seed)
                cfg["device"] = GPUS[gpu_idx]
                cfg_dir = CONFIG_ROOT / f"gpu{gpu_idx}"
                cfg_path = cfg_dir / f"config_f{fold}_s{seed}.json"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                cfg_path.write_text(json.dumps(cfg, indent=2))

        processes = launch_jobs({0: gpu0_runs, 1: gpu1_runs})
        print(f"\n{len(processes)} jobs launched. Monitoring...")
        monitor(processes)


if __name__ == "__main__":
    main()
