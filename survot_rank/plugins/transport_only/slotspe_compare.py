"""Paired SlotSPE transport experiment; execution requires an explicit --run.

The source SlotSPE is never repaired. Its main NLL and complete auxiliary
objective own all gradients. Data preparation reuses the independently audited
complete-case/cache utility; this is a matched protocol, not a reproduction of
the upstream benchmark trainer. Imports alone never read patient data.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
ARMS = ("native", "learned", "independent")
SCHEMA = "slotspe_transport_comparison_v1"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    paths = [*sorted((ROOT / "survot_rank/plugins/transport_only").rglob("*.py")),
             *sorted((ROOT / "survot_rank/plugins/transport").glob("*.py")),
             ROOT / "survot_rank/research/components/slot_attention.py"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def make_plan(config, output):
    """No patient reads or model construction; validate a single-fold queue."""
    from .slotspe_support.data import _config, _source_hashes
    cfg = dict(config)
    mechanism = cfg.pop("mechanism", {})
    if set(cfg) - {"source_root", "metadata_root", "features_root", "study", "label_col",
                   "signature", "folds", "seeds", "epochs", "patches", "batch_size",
                   "eval_repeats", "encoding_dim", "device", "lr", "torch_threads", "selection",
                   "experiment", "split_dir", "pad_short_bags"}:
        raise ValueError("Unknown comparison field; losses and host repairs cannot be configured here")
    for key in ("source_root", "metadata_root", "features_root"):
        cfg[key] = os.path.expandvars(str(cfg[key]))
        if "$" in cfg[key] or "%" in cfg[key]:
            raise ValueError(f"Unresolved environment variable in {key}")
    if not Path(cfg["source_root"]).is_absolute():
        cfg["source_root"] = str(ROOT / cfg["source_root"])
    cfg = _config(cfg)
    if len(cfg["folds"]) != 1 or len(cfg["seeds"]) != 1:
        raise ValueError("First-stage delivery accepts one fold and one seed only; audit before expanding")
    if cfg["batch_size"] < 2:
        raise ValueError("batch_size must be at least 2; upstream batch=1 uses a different accumulation recipe")
    if int(cfg.get("torch_threads", 2)) < 1:
        raise ValueError("torch_threads must be positive")
    # Import validates the feature-only recipe, without creating a model.
    from .mechanism import TransportOnlyConfig
    values = {"transport_mode": "balanced", "eps": .05, "eps_start": .05, **mechanism}
    recipe = TransportOnlyConfig(**values)
    if recipe.transport_mode != "balanced" or "plan_mode" in mechanism:
        raise ValueError("This first comparison locks balanced OT; arm owns plan_mode")
    if recipe.host_dim != 256 or recipe.slot_budget != 8 or recipe.eps != recipe.eps_start:
        raise ValueError("SlotSPE uses dim=256, eight native slots, fixed epsilon")
    selection = cfg.get("selection", "fixed_final_epoch")
    if selection not in ("fixed_final_epoch", "legacy_val"):
        raise ValueError("selection must be 'fixed_final_epoch' or 'legacy_val'")
    split_dir = str(cfg.get("split_dir", "5fold"))
    if not split_dir or "/" in split_dir or split_dir.startswith("."):
        raise ValueError("split_dir must be a plain directory name under splits/")
    return {"schema": SCHEMA, "config": cfg, "mechanism": values,
            "output_root": str(Path(output).resolve()), "arms": list(ARMS),
            "host_version": "unrepaired_local_SlotSPE_snapshot",
            "host_source_hashes": _source_hashes(cfg), "code_sha256": code_hashes(),
            "objective": "native_main_NLL_per_patient + native_auxiliary_total",
            "dct_added_losses": [], "selection": selection,
            "metric": "mean_of_repeat_Cindices_on_validation",
            "independent_test": False, "patient_execution_started": False,
            "matching": "same inputs, train-only bins, initialization, order, step/eval RNG, optimizer and budget"}


class TransportObserver:
    """Detached per-patient telemetry; never changes an output or objective."""
    def __init__(self, mechanism):
        self.rows = []
        self.handle = mechanism.register_forward_hook(self.capture)

    def capture(self, module, inputs, outputs):
        import torch
        self.rows = []
        if module.last_plans is None:
            return
        with torch.no_grad():
            p = module.last_plans
            rows, cols = p.sum(-1), p.sum(-2)
            mass = p.sum((-2, -1))
            independent = rows[..., :, None] * cols[..., None, :] / mass[..., None, None].clamp_min(1e-20)
            q = p / mass[..., None, None].clamp_min(1e-20)
            entropy = -(q * q.clamp_min(1e-20).log()).sum((-2, -1))
            error = torch.maximum(
                (rows - module.last_reference_rows[:, :, None]).abs().flatten(1).amax(-1),
                (cols - module.last_reference_cols[:, :, None]).abs().flatten(1).amax(-1))
            for i in range(len(inputs[0])):
                entry = {"transport_mass": float(mass[i].mean()),
                         "marginal_error_max": float(error[i]),
                         "plan_entropy": float(entropy[i].mean()),
                         "plan_independent_l1": float((p[i] - independent[i]).abs().sum((-2, -1)).mean())}
                for side, original, updated in zip(("wsi", "omics"), inputs, outputs):
                    delta = (updated[i] - original[i]).float()
                    entry[side + "_delta_l2"] = float(delta.norm())
                    entry[side + "_delta_relative_l2"] = float(delta.norm() / original[i].float().norm().clamp_min(1e-12))
                    entry[side + "_gain"] = float(torch.tanh(getattr(module, side + "_gain")))
                self.rows.append(entry)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot export an empty evidence table")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _score(events, times, risks):
    import numpy as np
    from sksurv.metrics import concordance_index_censored
    try:
        value = float(concordance_index_censored(np.asarray(events, bool), np.asarray(times, float),
                                                np.asarray(risks, float))[0])
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def evaluate(model, root, manifest, fold_data, config, seed, observer, telemetry, phase):
    import numpy as np
    import torch
    from .slotspe_support.data import _rng, _batch
    model.eval()
    ids = fold_data["val"]
    predictions, scores, losses = [], [], []
    for repeat in range(config["eval_repeats"]):
        _rng(seed * 1000000 + 80000 + repeat, torch, np)
        logits_all, loss = [], 0.
        with torch.no_grad():
            for start in range(0, len(ids), config["batch_size"]):
                selected = ids[start:start + config["batch_size"]]
                batch = _batch(root, manifest, fold_data, selected, config["device"], torch)
                logits, _ = model(**batch)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("Nonfinite validation prediction")
                loss += float(model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None))
                logits_all.append(logits.cpu())
                if observer is not None:
                    telemetry.extend({"phase": phase, "repeat": repeat, "patient_id": pid, **row}
                                     for pid, row in zip(selected, observer.rows))
        logits = torch.cat(logits_all)
        risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
        events = [manifest["patients"][pid]["censor"] < 1 for pid in ids]
        times = [manifest["patients"][pid]["time"] for pid in ids]
        score = _score(events, times, risks.tolist())
        if score is None or not math.isfinite(loss):
            raise RuntimeError("Undefined validation metric or loss; cannot accept this pair")
        scores.append(score)
        losses.append(loss / len(ids))
        for index, pid in enumerate(ids):
            predictions.append({"patient_id": pid, "repeat": repeat, "time": times[index],
                "censor": manifest["patients"][pid]["censor"], "label": fold_data["labels"][pid],
                "risk": float(risks[index]), **{f"logit_{k}": float(v) for k, v in enumerate(logits[index])}})
    return {"cindex_mean": float(np.mean(scores)), "cindex_by_repeat": scores,
            "cindex_repeat_std": float(np.std(np.asarray(scores) - scores[0])),
            "nll_mean": float(np.mean(losses)), "patient_count": len(ids)}, predictions


def execute(plan):
    import numpy as np
    import torch
    from .slotspe_support import data
    from .slotspe_support.model import build_model, make_args
    from .hosts import attach_transport
    root, cfg = Path(plan["output_root"]), plan["config"]
    selection = plan.get("selection", "fixed_final_epoch")
    # A prepared cache is reusable: the preparation is deterministic and is
    # re-verified against the current source, metadata and cached tensors.  Any
    # other content in the output directory means a partial or historical run
    # and is refused so nothing is overwritten.
    prepared = root / "prepared/manifest.json"
    if prepared.is_file():
        stray = [p for p in root.iterdir() if p.name not in ("prepared",)]
        if stray:
            raise FileExistsError(
                f"Output directory holds results besides the prepared cache: "
                f"{[p.name for p in stray]}; use a new output directory")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new output directory; partial and historical results are retained")
    if code_hashes() != plan["code_sha256"] or data._source_hashes(cfg) != plan["host_source_hashes"]:
        raise RuntimeError("Code changed after planning")
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no silent CPU or budget fallback")
    root.mkdir(parents=True, exist_ok=True)
    dump(root / "plan.json", plan)
    dump(root / "status.json", {"status": "running", "patient_execution_started": True})
    try:
        manifest = data.prepare(cfg, root)
        fold, seed = cfg["folds"][0], cfg["seeds"][0]
        fold_data = manifest["folds"][str(fold)]
        torch.set_num_threads(int(cfg.get("torch_threads", 2)))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        data._rng(seed, torch, np)
        args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
        base = build_model(cfg["source_root"], args, "original")
        initial = copy.deepcopy(base.state_dict())
        baseline_parameters = sum(p.numel() for p in base.parameters())
        shared_plugin = attach_transport(base, "slotspe", plan["mechanism"])
        plugin_initial = copy.deepcopy(shared_plugin.mechanism.state_dict())
        shared_plugin.remove()
        del base, shared_plugin
        torch.save({"host": initial, "transport": plugin_initial}, root / "shared_initial.pt")
        protocol = {**plan, "manifest_sha256": sha(root / "prepared/manifest.json"),
            "shared_initial_sha256": sha(root / "shared_initial.pt"),
            "optimizer": "Adam_lr_only_weight_decay_0", "scheduler": "cosine_eta_min_1e-6",
            "host_args": vars(args), "torch": str(torch.__version__), "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "timing_scope": "synchronized train epoch wall time including per-batch telemetry; excludes validation/data preparation",
            "data_protocol": ("complete cases; fixed real patches with zero-padding for short bags when "
                "pad_short_bags is enabled; train-only uncensored qcut; exploratory validation"),
            "feature_hash_scope": manifest["feature_provenance"], "counts": fold_data["counts"]}
        dump(root / "protocol.json", protocol)
        summaries = {}
        for arm in ARMS:
            out = root / arm
            out.mkdir()
            data._rng(seed, torch, np)
            model = build_model(cfg["source_root"], args, "original")
            model.load_state_dict(initial, strict=True)
            attachment = observer = None
            if arm != "native":
                attachment = attach_transport(model, "slotspe", plan["mechanism"])
                attachment.mechanism.load_state_dict(plugin_initial, strict=True)
                attachment.mechanism.set_plan_mode(arm if arm == "independent" else "learned")
                observer = TransportObserver(attachment.mechanism)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            if attachment is not None:
                optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
                if any(id(p) not in optimized for p in attachment.mechanism.parameters()):
                    raise RuntimeError("Optimizer omitted transport parameters")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"], eta_min=1e-6)
            curve, telemetry, steps = [], [], 0
            peak_allocated, peak_reserved = 0, 0
            best_val_cindex = None
            best_val_epoch = None
            for epoch in range(cfg["epochs"]):
                model.train()
                order = torch.randperm(len(fold_data["train"]), generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()
                losses = np.zeros(3)
                grad_sums = {}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for index, start in enumerate(range(0, len(order), cfg["batch_size"])):
                    ids = [fold_data["train"][i] for i in order[start:start + cfg["batch_size"]]]
                    data._rng(seed * 100000 + epoch * 1000 + index, torch, np)
                    batch = data._batch(root, manifest, fold_data, ids, device, torch)
                    optimizer.zero_grad(set_to_none=True)
                    logits, auxiliary = model(**batch)
                    main = model.loss_fn(logits, y=batch["y"], c=batch["c"], t=None) / len(ids)
                    total = main + auxiliary
                    if not bool(torch.isfinite(total)):
                        raise RuntimeError(f"Nonfinite objective {arm}/{epoch}/{index}")
                    total.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if not bool(torch.isfinite(parameter.grad).all()):
                                raise RuntimeError(f"Nonfinite gradient {arm}/{name}")
                            if name.startswith("transport_mechanism."):
                                grad_sums[name] = grad_sums.get(name, 0.) + float(parameter.grad.norm())
                    optimizer.step()
                    if observer is not None:
                        telemetry.extend({"phase": f"train_epoch_{epoch + 1}", "repeat": -1,
                                          "patient_id": pid, **row} for pid, row in zip(ids, observer.rows))
                    losses += np.asarray([float(main.detach()), float(auxiliary.detach()), float(total.detach())]) * len(ids)
                    steps += 1
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
                    peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
                curve.append({"epoch": epoch + 1, "train_main_nll": float(losses[0] / len(order)),
                    "train_auxiliary_total": float(losses[1] / len(order)), "train_total": float(losses[2] / len(order)),
                    "train_seconds": time.perf_counter() - started, "optimizer_steps": steps,
                    "transport_gradient_norm_sums": grad_sums})
                if selection == "legacy_val":
                    val_metrics, _ = evaluate(model, root, manifest, fold_data, cfg, seed, observer,
                                              [], "train_epoch_val")
                    curve[-1]["val_cindex_mean"] = val_metrics["cindex_mean"]
                    if best_val_cindex is None or val_metrics["cindex_mean"] > best_val_cindex:
                        best_val_cindex = val_metrics["cindex_mean"]
                        best_val_epoch = epoch + 1
                        torch.save({"state_dict": copy.deepcopy(model.state_dict()), "arm": arm,
                                    "protocol": protocol, "epochs": cfg["epochs"],
                                    "optimizer_steps": steps, "selected_epoch": best_val_epoch,
                                    "selected_val_cindex": best_val_cindex},
                                   out / "best.pt")
                dump(out / "curve.json", curve)
                print(f"fold={fold} seed={seed} arm={arm} epoch={epoch + 1}/{cfg['epochs']} loss={curve[-1]['train_total']:.6f}"
                      + (f" val_cindex={val_metrics['cindex_mean']:.4f}" if selection == "legacy_val" else ""), flush=True)
                scheduler.step()
            checkpoint = {"state_dict": model.state_dict(), "arm": arm, "protocol": protocol,
                          "epochs": cfg["epochs"], "optimizer_steps": steps}
            torch.save(checkpoint, out / "last.pt")
            if selection == "legacy_val" and best_val_epoch is not None:
                best_ckpt = torch.load(out / "best.pt", map_location=device)
                model.load_state_dict(best_ckpt["state_dict"], strict=True)
                print(f"  {arm}: selected epoch={best_val_epoch} (val_cindex={best_val_cindex:.4f})", flush=True)
            metrics, predictions = evaluate(model, root, manifest, fold_data, cfg, seed, observer, telemetry, "validation")
            _write_csv(out / "predictions.csv", predictions)
            summary = {"arm": arm, **metrics, "checkpoint_sha256": sha(out / "last.pt"),
                "epochs": cfg["epochs"], "optimizer_steps": steps,
                "parameters": sum(p.numel() for p in model.parameters()),
                "extra_parameters": sum(p.numel() for p in model.parameters()) - baseline_parameters,
                "train_seconds": sum(e["train_seconds"] for e in curve),
                "peak_allocated_bytes": peak_allocated if device.type == "cuda" else None,
                "peak_reserved_bytes": peak_reserved if device.type == "cuda" else None,
                "selection": selection}
            if selection == "legacy_val" and best_val_epoch is not None:
                summary["selected_epoch"] = best_val_epoch
                summary["selected_val_cindex"] = best_val_cindex
            if arm == "learned":
                # Same weights and RNG: isolates dependence on topology and on the branch.
                controls = {}
                for mode in ("independent", "bypass"):
                    attachment.set_enabled(mode != "bypass")
                    attachment.mechanism.set_plan_mode("independent")
                    if observer is not None:
                        observer.rows = []
                    controls[mode], rows = evaluate(model, root, manifest, fold_data, cfg, seed, observer,
                                                   telemetry, "same_weights_" + mode)
                    _write_csv(out / ("same_weights_" + mode + "_predictions.csv"), rows)
                dump(out / "same_weights_controls.json", controls)
            if telemetry:
                _write_csv(out / "transport.csv", telemetry)
            dump(out / "summary.json", summary)
            summaries[arm] = summary
            if observer is not None:
                observer.handle.remove()
            del optimizer, scheduler, checkpoint, model, attachment, observer
            del logits, auxiliary, total, main, batch, parameter
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if code_hashes() != plan["code_sha256"] or data._source_hashes(cfg) != plan["host_source_hashes"]:
            raise RuntimeError("Code changed during experiment")
        dump(root / "paired_summary.json", {"summaries": summaries,
            "learned_minus_native": summaries["learned"]["cindex_mean"] - summaries["native"]["cindex_mean"],
            "learned_minus_independent": summaries["learned"]["cindex_mean"] - summaries["independent"]["cindex_mean"],
            "scope": "one fold/seed exploratory validation; no generalization or superiority claim"})
        data._manifest(cfg, root, verify_cache=True)
        dump(root / "status.json", {"status": "completed_pending_independent_audit", "patient_execution_started": True})
    except BaseException as error:
        dump(root / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    plan = make_plan(cfg, args.output)
    if args.run:
        execute(plan)
    else:
        print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
