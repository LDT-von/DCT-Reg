"""E046 BLCA fold0 driver: launch 3 arms in parallel on 2 GPUs.

Builds the E046 plan with --cancers blca --folds 0 --seeds 3,
manually copies the frozen base-config snapshot + plan.json into the
results-root (mirroring run_wsi_reconstruction_ablation.execute_plan),
and dispatches the three arms to GPU 0/1, leaving the third job to
take whichever GPU frees first.  Logs go to <results-root>/logs/.
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
MAX_PARALLEL = 2


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
    # any line that includes a python process holding memory => busy
    for line in out.splitlines():
        if "python" in line.lower():
            return False
    return True


def main():
    results_root = Path(sys.argv[1]).resolve()
    if results_root.exists() and any(results_root.iterdir()):
        sys.exit(f"Refuse to overwrite nonempty results root: {results_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    # Build plan via the canonical entry (no --execute, so it only prints JSON).
    proc = subprocess.run(
        [
            TRISURV_PY, str(SCRIPT),
            "--results-root", str(results_root),
            "--cancers", "blca",
            "--arms", "full", "wsi_additive", "wsi_fixed_total",
            "--folds", "0",
            "--seeds", "3",
            "--protocol", "outer_test",
            "--gpu", "0",
        ],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    plan = json.loads(proc.stdout)

    # Mirror run_wsi_reconstruction_ablation.execute_plan: copy base config
    # into <results-root>/base_config.yaml and rewrite the --config path in each
    # task command so every child process uses the frozen snapshot.
    snapshot = results_root / "base_config.yaml"
    shutil.copy2(plan["config"], snapshot)
    for task in plan["tasks"]:
        # build a clean command: same args but --config pointing to snapshot
        raw = task.get("execution_command") or task["command"]
        cmd = [TRISURV_PY] + raw[1:]  # replace python interpreter
        # swap --config argument to use the frozen snapshot
        cfg_idx = cmd.index("--config") + 1
        cmd[cfg_idx] = str(snapshot)
        task["planned_command"] = cmd
    (results_root / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logs = results_root / "logs"
    logs.mkdir()

    tasks = plan["tasks"]
    print(f"[plan] {len(tasks)} arms, results_root={results_root}", flush=True)
    print(f"[plan] commit={plan['source_revision'][:12]}", flush=True)

    # Dispatch: keep at most MAX_PARALLEL jobs running at a time, and pin a
    # different physical GPU to each running job when possible.
    running: list[tuple[subprocess.Popen, dict, str, Path]] = []
    next_idx = 0
    finished: list[dict] = []

    def try_launch():
        nonlocal next_idx
        if next_idx >= len(tasks) or len(running) >= MAX_PARALLEL:
            return False
        # Pick a free GPU, or fall back to round-robin.
        chosen = None
        for g in GPUS:
            if gpu_free(g):
                chosen = g
                break
        if chosen is None:
            chosen = GPUS[len(running) % len(GPUS)]
        task = tasks[next_idx]
        next_idx += 1
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = chosen
        env["PYTHONPATH"] = str(ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        log_path = logs / f"{task['arm']}_{task['cancer']}_f{task['fold']}_s{task['seed']}.log"
        fh = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            task["planned_command"], cwd=ROOT, env=env,
            stdout=fh, stderr=subprocess.STDOUT,
        )
        running.append((proc, task, chosen, log_path))
        print(
            f"[launch] arm={task['arm']} fold={task['fold']} seed={task['seed']} "
            f"gpu={chosen} pid={proc.pid} log={log_path.name}",
            flush=True,
        )
        return True

    # Initial fill
    while try_launch():
        pass

    # Main loop
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

    # Summarize
    print("\n===== E046 BLCA fold0 summary =====", flush=True)
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