"""E046 BLCA wsi_additive folds 1-4 single-GPU driver.

Mirrors run_blca_fold0_2gpu.py but only for arm=wsi_additive, folds 1..4,
seed=3, cancer=blca, on a single GPU (MAX_PARALLEL=1) so the other GPU
stays free.  Sequential launch + wait loop, one task at a time, with a
dedicated results-root that is freshly created.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "实验档案/实验代码/E046_WSI自重建消融/run_wsi_reconstruction_ablation.py"
TRISURV_PY = "/home/ubuntu/.conda/envs/trisurv/bin/python"
GPUS = ["0", "1"]
MAX_PARALLEL = 1   # single-GPU serial; keep GPU 1 free by design
FOLDS = ["1", "2", "3", "4"]
ARM = "wsi_additive"
CANCER = "blca"
SEED = "3"


def gpu_free(gpu: str) -> bool:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
            "-i", gpu,
        ],
        text=True,
    ).strip()
    if not out:
        return True
    for line in out.splitlines():
        if "python" in line.lower():
            return False
    return True


def main():
    results_root = Path(sys.argv[1]).resolve()
    if results_root.exists() and any(results_root.iterdir()):
        sys.exit(f"Refuse to overwrite nonempty results root: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    # Pin to a single GPU for the whole batch.  The user requirement is to
    # leave ONE GPU free; this script uses the requested GPU (default 0) and
    # never touches the other one.
    pinned_gpu = sys.argv[2] if len(sys.argv) > 2 else "0"
    if pinned_gpu not in GPUS:
        sys.exit(f"Pinned GPU must be one of {GPUS}, got {pinned_gpu!r}")

    proc = subprocess.run(
        [
            TRISURV_PY, str(SCRIPT),
            "--results-root", str(results_root),
            "--cancers", CANCER,
            "--arms", ARM,
            "--folds", *FOLDS,
            "--seeds", SEED,
            "--protocol", "outer_test",
            "--gpu", pinned_gpu,
        ],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    plan = json.loads(proc.stdout)

    # Mirror run_wsi_reconstruction_ablation.execute_plan: copy base config
    # into <results-root>/base_config.yaml and rewrite --config in each task
    # command so every child process uses the frozen snapshot.
    snapshot = results_root / "base_config.yaml"
    shutil.copy2(plan["config"], snapshot)
    for task in plan["tasks"]:
        raw = task.get("execution_command") or task["command"]
        cmd = [TRISURV_PY] + raw[1:]
        cfg_idx = cmd.index("--config") + 1
        cmd[cfg_idx] = str(snapshot)
        task["planned_command"] = cmd
    (results_root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logs = results_root / "logs"
    logs.mkdir()

    tasks = plan["tasks"]
    print(f"[plan] {len(tasks)} arm={ARM} cancer={CANCER} folds={FOLDS} seed={SEED}",
          flush=True)
    print(f"[plan] commit={plan['source_revision'][:12]}", flush=True)
    print(f"[plan] pinned_gpu={pinned_gpu} (other GPU left free)", flush=True)

    running: list[tuple[subprocess.Popen, dict, str, Path]] = []
    next_idx = 0
    finished: list[dict] = []

    def try_launch():
        nonlocal next_idx
        if next_idx >= len(tasks) or len(running) >= MAX_PARALLEL:
            return False
        if not gpu_free(pinned_gpu):
            return False
        task = tasks[next_idx]
        next_idx += 1
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = pinned_gpu
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        log_path = logs / f"{task['arm']}_{task['cancer']}_f{task['fold']}_s{task['seed']}.log"
        fh = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            task["planned_command"], cwd=ROOT, env=env,
            stdout=fh, stderr=subprocess.STDOUT,
        )
        running.append((proc, task, pinned_gpu, log_path))
        print(
            f"[launch] arm={task['arm']} fold={task['fold']} seed={task['seed']} "
            f"gpu={pinned_gpu} pid={proc.pid} log={log_path.name}",
            flush=True,
        )
        return True

    while try_launch():
        pass

    while running:
        time.sleep(15)
        still = []
        for proc, task, gpu, log_path in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, task, gpu, log_path))
                continue
            finished.append({"task": task, "gpu": gpu, "log": str(log_path), "returncode": rc})
            print(
                f"[done ] arm={task['arm']} fold={task['fold']} seed={task['seed']} "
                f"gpu={gpu} rc={rc}",
                flush=True,
            )
        running = still
        while try_launch():
            pass

    print("\n===== E046 BLCA wsi_additive folds 1-4 summary =====", flush=True)
    for f in finished:
        print(
            f"  arm={f['task']['arm']:18s} fold={f['task']['fold']} seed={f['task']['seed']} "
            f"gpu={f['gpu']} rc={f['returncode']} log={f['log']}",
            flush=True,
        )
    bad = [f for f in finished if f["returncode"] != 0]
    if bad:
        sys.exit(f"{len(bad)} arm(s) failed; check logs under {logs}")
    print("All arms finished with rc=0.", flush=True)


if __name__ == "__main__":
    main()
