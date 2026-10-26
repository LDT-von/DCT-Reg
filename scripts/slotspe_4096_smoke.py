"""Standalone smoke test: SlotSPE official forward/backward at 4096 patches, UNI2-H.

Does not touch the proof harness output tree, does not write patient results.
Measures peak CUDA memory and wall time per training step for a real batch
drawn from the actual UNI2-H feature files, at the requested patch budget.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import numpy as np
import torch

from survot_rank.research.slotspe_proof import model as proof_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-root", type=Path,
                   default=Path("/data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files"))
    p.add_argument("--patches", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--encoding-dim", type=int, default=1536)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--source-root", type=Path, default=REPO / "third_party/SlotSPE")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_bags(features_root, patches, encoding_dim, count, seed=1234):
    """Load real slide tensors and sample exactly `patches` rows per patient."""
    files = sorted(Path(features_root).glob("*.h5")) + sorted(Path(features_root).glob("*.pt"))
    if not files:
        raise SystemExit(f"no feature files under {features_root}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    bags = []
    for idx in order:
        if len(bags) >= count:
            break
        path = Path(files[idx])
        if path.suffix == ".h5":
            import h5py
            with h5py.File(path, "r") as handle:
                # Official layout: features has shape (1, N, C).
                raw = handle["features"][0]
            tensor = torch.from_numpy(raw)
        else:
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        tensor = tensor.float()
        if tensor.shape[0] < patches:
            continue
        sel = np.sort(rng.choice(tensor.shape[0], size=patches, replace=False))
        bags.append(tensor[sel])
    if not bags:
        raise SystemExit("no patient reached the requested patch budget")
    return torch.stack(bags)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(3)
    np.random.seed(3)

    print(f"[smoke] loading real bags: patches={args.patches} bs={args.batch_size}", flush=True)
    t0 = time.perf_counter()
    bags = load_bags(args.features_root, args.patches, args.encoding_dim, args.batch_size)
    load_seconds = time.perf_counter() - t0
    print(f"[smoke] loaded {tuple(bags.shape)} in {load_seconds:.1f}s", flush=True)

    omic_sizes = [20] * 10
    args_ns = proof_model.make_args(omic_sizes, encoding_dim=args.encoding_dim)
    source = proof_model.load_source(args.source_root)
    model = source.SlotSPE(args_ns).to(device)
    model.train()

    x_wsi = bags.to(device)
    # Pathways mode expects x_omic1..x_omicN, one tensor per signature.
    x_omics = torch.randn(args.batch_size, len(omic_sizes), 20, device=device)
    kwargs = {"x_wsi": x_wsi, "omic_missing": False}
    for i in range(len(omic_sizes)):
        kwargs[f"x_omic{i + 1}"] = x_omics[:, i]
    # Synthetic but shape-correct survival targets for the official loss.
    kwargs["y"] = torch.randint(0, args.n_classes, (args.batch_size,), device=device)
    kwargs["c"] = torch.randint(0, 2, (args.batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    step_times = []
    for step in range(args.steps):
        t = time.perf_counter()
        out = model(**kwargs)
        logits, auxiliary = out[0], out[1]
        main = model.loss_fn(logits, y=kwargs["y"], c=kwargs["c"], t=None) / args.batch_size
        total = main + (auxiliary if isinstance(auxiliary, torch.Tensor) else torch.zeros((), device=device))
        total.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t)
        print(f"[smoke] step {step}: {step_times[-1]:.2f}s total={float(total):.4f}", flush=True)

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    report = {
        "status": "smoke_ok",
        "patches": args.patches,
        "batch_size": args.batch_size,
        "encoding_dim": args.encoding_dim,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "bag_load_seconds": load_seconds,
        "step_seconds": step_times,
        "mean_step_seconds": float(np.mean(step_times)),
        "peak_allocated_cuda_bytes": int(peak),
        "peak_allocated_cuda_gib": round(peak / 1024**3, 3),
        "free_gpu_gib_at_start": None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report["free_gpu_gib_after"] = round(free_bytes / 1024**3, 3)
        report["total_gpu_gib"] = round(total_bytes / 1024**3, 3)
    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
