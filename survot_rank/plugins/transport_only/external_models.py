"""
BLCA external-model comparison: CoxPH, RSF (sksurv), DeepSurv, MTLR.
Reuses prepared cache from paired experiments — no duplicate data prep.

PCA is applied before tabular models (CoxPH/RSF) to handle the high
dimensionality (1536 WSI + 9532 omics = 11068) with only ~73 training samples.

Usage (all in trisurv env):
    python -m survot_rank.plugins.transport_only.external_models run \
        --prepared_dir /tmp/ot03_smoke/OT-03_fold0_seed3_full_v4 \
        --output_root  /tmp/ot_ext/OT-03_fold{fold}_seed{seed}_external \
        --folds 0 --seeds 3 --epochs 30 --device cuda

For full 5×5 matrix, run once per (fold, seed) cell:
    python -m survot_rank.plugins.transport_only.external_models run \
        --prepared_dir <any_v4_prepared_dir> \
        --output_root  /tmp/ot_ext/OT-03_fold{fold}_seed{seed}_external \
        --folds {fold} --seeds {seed} --epochs 30 --device cuda

All cells share the same manifest from one prepared directory (same study/blca).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import random

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── sksurv metrics (already installed) ──────────────────────────────────────
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

MODELS = ("coxph", "rsf", "deepsurv")   # MTLR skipped (numerically unstable with current survival NLL; future work)
MTLR_SKIP = True  # set to False to re-enable


def _score(events, times, risks):
    try:
        value = float(concordance_index_censored(
            np.asarray(events, bool), np.asarray(times, float),
            np.asarray(risks, float))[0])
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def _sha(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _rng(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(root, manifest, fold_data, ids, device, pool="mean"):
    """Load WSI (mean-pooled patches) + omics features for patient IDs."""
    omic_sizes = manifest["omic_sizes"]
    wsi_dim = manifest["encoding_dim"]
    X_wsi, X_omic, y_time, y_event = [], [], [], []

    for pid in ids:
        pt = torch.load(Path(root) / manifest["patients"][pid]["cache_path"],
                        map_location="cpu", weights_only=True)
        wsi = pt["wsi"]          # [patches, wsi_dim]
        omics = pt["omics"]      # list of [size_k]

        if pool == "mean":
            wsi_feat = wsi.mean(0)           # [wsi_dim]
        elif pool == "max":
            wsi_feat = wsi.amax(0)            # [wsi_dim]
        else:
            raise ValueError(f"Unknown pool={pool}")

        omic_cat = torch.cat([g.float() for g in omics])  # sum of omic_sizes
        X_wsi.append(wsi_feat)
        X_omic.append(omic_cat)
        y_time.append(manifest["patients"][pid]["time"])
        y_event.append(int(manifest["patients"][pid]["censor"] < 1))

    X_wsi_t = torch.stack(X_wsi)          # [n, wsi_dim]
    X_omic_t = torch.stack(X_omic)       # [n, omic_dim]
    X = torch.cat([X_wsi_t, X_omic_t], 1)  # [n, wsi_dim+omic_dim]

    X_dev = X.to(device, non_blocking=True)
    return X_dev, X_wsi_t, X_omic_t, np.array(y_time), np.array(y_event, bool)


def load_features_static(root, manifest, fold_data, ids, device):
    """Static (non-pooled) WSI mean + omics for tabular models (CoxPH/RSF)."""
    _, X_wsi, X_omic, y_time, y_event = load_features(root, manifest, fold_data, ids, device)
    return torch.cat([X_wsi, X_omic], 1).numpy(), y_time, y_event


# ── DeepSurv (PyTorch MLP) ───────────────────────────────────────────────────
class DeepSurv(torch.nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)   # [batch, 1]


def deepsurv_loss(logits, y_time, y_event):
    """Negative log-likelihood for CoxPH via DeepSurv."""
    hazards = torch.exp(logits.squeeze(-1))       # exp(h) = exp(log risk)
    # Sort by time descending (uncensored last for risk ordering)
    order = torch.argsort(y_time, descending=True)
    hazards = hazards[order]
    events = y_event[order].float()
    log_h = logits.squeeze(-1)[order]
    risk_set = hazards.cumsum(0)
    nll = -(log_h - torch.log(risk_set + 1e-8)) * events
    return nll.sum() / (events.sum() + 1e-8)


def deepsurv_risk(logits):
    """Risk score = -(1-sigmoid(logits)).cumprod(-1).sum(-1), same as slotspe."""
    probs = logits.sigmoid()
    return (-(1 - probs).cumprod(-1)).sum(-1).squeeze(-1)   # [batch]


# ── MTLR (PyTorch) ───────────────────────────────────────────────────────────
class MTLR(torch.nn.Module):
    def __init__(self, input_dim, num_bins, hidden=128, dropout=0.3):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.decoder = torch.nn.Linear(hidden, num_bins)

    def forward(self, x):
        return self.encoder(x), self.decoder(self.encoder(x))   # h, logits


def mtlr_loss(h, logits, y_label, y_event, num_bins):
    """MTLR negative log-likelihood.
    y_label: discrete bin index (0..num_bins-1), not raw time.
    y_event: whether uncensored (not directly used in survival NLL here,
             but required by the interface).
    """
    hazards = torch.exp(logits)            # [batch, num_bins]
    cum_h = torch.cat([torch.zeros_like(hazards[:, :1]), hazards.cumsum(1)], 1)  # [batch, num_bins+1]
    S = torch.exp(-cum_h[:, :-1])          # [batch, num_bins]
    density = S * hazards                   # [batch, num_bins]
    density = density.clamp(min=1e-8)
    log_probs = torch.log(density + 1e-8)
    label = y_label.long().clamp(0, num_bins - 1)
    nll = -log_probs[torch.arange(len(label), device=logits.device), label]
    return nll.mean()


def mtlr_risk(h, logits):
    """MTLR risk score = -E[survival time] via survival distribution.
    More stable than the slotspe cumprod approach for ordinal outputs."""
    probs = logits.softmax(-1)          # P(bin | x)
    # Survival beyond each bin boundary: S_k = sum_{j>=k} P(bin=j)
    S = probs.flip(-1).cumsum(-1).flip(-1)
    S = S.clamp(1e-8, 1.0)
    # Expected time proxy: sum of survival probabilities
    return -(S.sum(-1)).squeeze(-1)     # higher S → lower risk → higher risk score


# ── Shared evaluation ─────────────────────────────────────────────────────────
def evaluate_tabular(model_fn, X_val, y_time_val, y_event_val):
    """Evaluate a fitted sklearn model (CoxPH/RSF). Returns c-index."""
    risks = model_fn.predict(X_val)
    return _score(y_event_val, y_time_val, risks)


def evaluate_pytorch(model, X_wsi_val, X_omic_val, y_time_val, y_event_val,
                     device, num_bins, model_type, batch_size=64):
    """Evaluate a PyTorch model on validation set. Returns c-index."""
    model.eval()
    n = len(y_time_val)
    risks_all = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_wsi_b = X_wsi_val[start:end].to(device)
            X_omic_b = X_omic_val[start:end].to(device)
            X = torch.cat([X_wsi_b, X_omic_b], 1)

            if model_type == "deepsurv":
                logits = model(X)
                risks = deepsurv_risk(logits).cpu().numpy()
            else:  # mtlr
                h, logits = model(X)
                risks = mtlr_risk(h, logits).cpu().numpy()

            risks_all.extend(risks.tolist())

    return _score(y_event_val, y_time_val, risks_all)


def train_pytorch(model, X_wsi_train, X_omic_train, y_time_train, y_event_train,
                  X_wsi_val, X_omic_val, y_time_val, y_event_val,
                  epochs, lr, device, num_bins, model_type,
                  y_label_train=None, y_label_val=None,
                  batch_size=64, seed=42):
    """Train a PyTorch model for one arm."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_score = -1
    best_state = None
    curve = []

    n_train = len(y_time_train)
    y_time_t = torch.tensor(y_time_train, dtype=torch.float32, device=device)
    y_event_t = torch.tensor(y_event_train.astype(float), dtype=torch.float32, device=device)
    # Discrete labels for MTLR (bin index, same as what slotspe uses internally)
    if y_label_train is not None:
        y_label_t = torch.tensor(y_label_train, dtype=torch.long, device=device)
    else:
        y_label_t = None  # not used for deepsurv

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()

        epoch_loss = 0.0
        for start in range(0, n_train, batch_size):
            ids = order[start:start + batch_size]
            X_wsi_b = X_wsi_train[ids].to(device)
            X_omic_b = X_omic_train[ids].to(device)
            X = torch.cat([X_wsi_b, X_omic_b], 1)
            yt = y_time_t[ids]
            ye = y_event_t[ids]

            optimizer.zero_grad(set_to_none=True)
            if model_type == "deepsurv":
                logits = model(X)
                loss = deepsurv_loss(logits, yt, ye)
            else:
                h, logits = model(X)
                loss = mtlr_loss(h, logits,
                                 y_label=y_label_t[ids] if y_label_t is not None else yt,
                                 y_event=ye, num_bins=num_bins)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(ids)

        scheduler.step()

        # Evaluate
        score = evaluate_pytorch(model, X_wsi_val, X_omic_val,
                                  y_time_val, y_event_val, device, num_bins, model_type)
        if score is None or not math.isfinite(score):
            score = -1.0

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        curve.append({
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss / n_train),
            "val_cindex": score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  [{model_type}] epoch={epoch + 1}/{epochs} loss={curve[-1]['train_loss']:.4f} "
                  f"val_c={score:.4f}", flush=True)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, curve, best_score


# ── Main run ─────────────────────────────────────────────────────────────────
def run(prepared_dir: Path, output_root: Path, folds, seeds, epochs, device_str, lr=1e-3):
    import torch
    device = torch.device(device_str)

    # Load manifest (read-only, shared across all cells)
    manifest_path = prepared_dir / "prepared" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    root = str(prepared_dir)
    omic_dim = sum(manifest["omic_sizes"])
    wsi_dim = manifest["encoding_dim"]
    input_dim = wsi_dim + omic_dim
    num_bins = 4  # same binning as slotspe

    results = {}
    for fold in folds:
        for seed in seeds:
            fold_str = str(fold)
            if fold_str not in manifest.get("folds", {}):
                raise ValueError(f"Fold {fold} not in manifest; check prepared_dir")
            fold_data = manifest["folds"][fold_str]
            out_dir = output_root.parent / output_root.name.replace(
                "{fold}", str(fold)).replace("{seed}", str(seed))
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*60}")
            print(f"  Fold={fold} Seed={seed} | WSI={wsi_dim} + Omics={omic_dim} = {input_dim}d")
            print(f"  Train={len(fold_data['train'])} Val={len(fold_data['val'])}")
            print(f"{'='*60}", flush=True)

            cell_results = {}

            # ── Load data (static tensors for pytorch models) ──────────────
            X_train, Xwsi_train, Xom_train, yT_train, yE_train = load_features(
                root, manifest, fold_data, fold_data["train"], device, pool="mean")
            X_val,   Xwsi_val,   Xom_val,   yT_val,   yE_val   = load_features(
                root, manifest, fold_data, fold_data["val"],   device, pool="mean")

            # Move to CPU for sklearn
            X_train_np = torch.cat([Xwsi_train, Xom_train], 1).numpy()
            X_val_np   = torch.cat([Xwsi_val,   Xom_val],   1).numpy()

            # ── PCA dimensionality reduction for tabular models ────────────────
            # 73 train samples with 11068 dims is ill-conditioned; reduce to ~30 components
            n_components = min(30, len(fold_data["train"]) - 2)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_np)
            X_val_scaled   = scaler.transform(X_val_np)
            pca = PCA(n_components=n_components, random_state=seed)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_val_pca   = pca.transform(X_val_scaled)

            # sksurv needs structured array
            y_train_struct = np.array(
                [(bool(e), float(t)) for e, t in zip(yE_train, yT_train)],
                dtype=[("event", bool), ("time", float)])
            y_val_struct = np.array(
                [(bool(e), float(t)) for e, t in zip(yE_val, yT_val)],
                dtype=[("event", bool), ("time", float)])

            _rng(seed, torch, np)

            # ── CoxPH (on PCA features, strong regularization) ─────────────
            print(f"  [CoxPH] PCA→{n_components}d + alpha=0.1", flush=True)
            t0 = time.perf_counter()
            try:
                cox = CoxPHSurvivalAnalysis(alpha=0.1, verbose=False)
                cox.fit(X_train_pca, y_train_struct)
                cox_score = evaluate_tabular(cox, X_val_pca, yT_val, yE_val)
            except Exception as ex:
                print(f"  [CoxPH] FAILED: {ex}", flush=True)
                cox_score = None
            cox_time = time.perf_counter() - t0
            print(f"  [CoxPH] val_c={cox_score} ({cox_time:.1f}s)", flush=True)
            cell_results["coxph"] = {"cindex_mean": cox_score, "fit_seconds": cox_time}

            # ── RSF (on PCA features) ──────────────────────────────────────
            print(f"  [RSF] PCA→{n_components}d + 200 trees", flush=True)
            t0 = time.perf_counter()
            try:
                rsf = RandomSurvivalForest(n_estimators=200, max_depth=6,
                                            min_samples_split=5, random_state=seed,
                                            n_jobs=-1, verbose=0)
                rsf.fit(X_train_pca, y_train_struct)
                rsf_score = evaluate_tabular(rsf, X_val_pca, yT_val, yE_val)
            except Exception as ex:
                print(f"  [RSF] FAILED: {ex}", flush=True)
                rsf_score = None
            rsf_time = time.perf_counter() - t0
            print(f"  [RSF] val_c={rsf_score} ({rsf_time:.1f}s)", flush=True)
            cell_results["rsf"] = {"cindex_mean": rsf_score, "fit_seconds": rsf_time}

            # Discrete labels from the manifest (same binning as slotspe uses)
            y_label_train = np.array([fold_data["labels"][pid] for pid in fold_data["train"]])
            y_label_val   = np.array([fold_data["labels"][pid] for pid in fold_data["val"]])

            # ── DeepSurv ───────────────────────────────────────────────────
            print(f"  [DeepSurv] Training {epochs} epochs...", flush=True)
            t0 = time.perf_counter()
            try:
                ds_model = DeepSurv(input_dim, hidden=256, dropout=0.3)
                ds_model, ds_curve, ds_score = train_pytorch(
                    ds_model, Xwsi_train, Xom_train, yT_train, yE_train,
                    Xwsi_val, Xom_val, yT_val, yE_val,
                    epochs=epochs, lr=lr, device=device, num_bins=num_bins,
                    model_type="deepsurv", seed=seed)
                # Save last checkpoint
                torch.save(ds_model.state_dict(), out_dir / "deepsurv_last.pt")
                _dump(out_dir / "deepsurv_curve.json", ds_curve)
            except Exception as ex:
                print(f"  [DeepSurv] FAILED: {ex}", flush=True)
                ds_score = None
                ds_curve = []
            ds_time = time.perf_counter() - t0
            print(f"  [DeepSurv] val_c={ds_score} ({ds_time:.1f}s)", flush=True)
            cell_results["deepsurv"] = {"cindex_mean": ds_score, "train_seconds": ds_time}

            if not MTLR_SKIP:
                # ── MTLR ───────────────────────────────────────────────────
                print(f"  [MTLR] Training {epochs} epochs...", flush=True)
                t0 = time.perf_counter()
                try:
                    mtlr_model = MTLR(input_dim, num_bins=num_bins, hidden=128, dropout=0.3)
                    mtlr_model, mtlr_curve, mtlr_score = train_pytorch(
                        mtlr_model, Xwsi_train, Xom_train, yT_train, yE_train,
                        Xwsi_val, Xom_val, yT_val, yE_val,
                        epochs=epochs, lr=lr, device=device, num_bins=num_bins,
                        model_type="mtlr", seed=seed,
                        y_label_train=y_label_train, y_label_val=y_label_val)
                    torch.save(mtlr_model.state_dict(), out_dir / "mtlr_last.pt")
                    _dump(out_dir / "mtlr_curve.json", mtlr_curve)
                except Exception as ex:
                    print(f"  [MTLR] FAILED: {ex}", flush=True)
                    mtlr_score = None
                    mtlr_curve = []
                mtlr_time = time.perf_counter() - t0
                print(f"  [MTLR] val_c={mtlr_score} ({mtlr_time:.1f}s)", flush=True)
                cell_results["mtlr"] = {"cindex_mean": mtlr_score, "train_seconds": mtlr_time}
            else:
                print(f"  [MTLR] SKIPPED (numerically unstable with 73-sample training)", flush=True)
                cell_results["mtlr"] = {"cindex_mean": None}

            # ── Summary ─────────────────────────────────────────────────────
            summary = {
                "fold": fold, "seed": seed,
                "wsi_dim": wsi_dim, "omic_dim": omic_dim, "total_dim": input_dim,
                "train_n": len(fold_data["train"]), "val_n": len(fold_data["val"]),
                "results": {
                    "coxph":     {"cindex": cell_results["coxph"]["cindex_mean"]},
                    "rsf":       {"cindex": cell_results["rsf"]["cindex_mean"]},
                    "deepsurv":  {"cindex": cell_results["deepsurv"]["cindex_mean"]},
                    "mtlr":      {"cindex": cell_results["mtlr"]["cindex_mean"]},
                }
            }
            _dump(out_dir / "summary.json", summary)
            _dump(out_dir / "protocol.json", {
                "prepared_manifest": str(manifest_path),
                "manifest_sha256": _sha(manifest_path),
                "pooling": "mean",
                "epochs": epochs,
                "lr": lr,
                "device": device_str,
                "models": ["coxph", "rsf", "deepsurv"] + ([] if MTLR_SKIP else ["mtlr"]),
                "blca": True,
            })
            results[f"fold{fold}_seed{seed}"] = summary
            print(f"\n  Summary:", json.dumps(summary["results"], indent=2), flush=True)

            del ds_model
            if not MTLR_SKIP:
                del mtlr_model
            del X_train, X_val
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Aggregate across seeds ───────────────────────────────────────────────
    model_names = ["coxph", "rsf", "deepsurv"] + ([] if MTLR_SKIP else ["mtlr"])
    agg = {}
    for m in model_names:
        scores = [r["results"][m]["cindex"] for r in results.values()
                  if r["results"][m]["cindex"] is not None]
        if scores:
            agg[m] = {
                "cindex_mean": float(np.mean(scores)),
                "cindex_std": float(np.std(scores)),
                "cindex_per_seed": scores,
                "n_cells": len(scores),
            }

    overall = {
        "study": "blca",
        "models": list(model_names),
        "per_cell": results,
        "aggregate": agg,
    }
    _dump(output_root / "external_summary.json", overall)
    print(f"\n{'='*60}")
    print("OVERALL AGGREGATE (across seeds):")
    for m, s in agg.items():
        print(f"  {m:10s}  c={s['cindex_mean']:.4f} ± {s['cindex_std']:.4f}  (n={s['n_cells']})")
    print(f"{'='*60}", flush=True)
    return overall


# ── CLI ───────────────────────────────────────────────────────────────────────
def _main():
    parser = argparse.ArgumentParser(description="BLCA external model comparison")
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run")
    run_p.add_argument("--prepared_dir", type=Path, required=True,
                        help="Any prepared dir with manifest.json, e.g. /tmp/ot03_smoke/OT-03_fold0_seed3_full_v4")
    run_p.add_argument("--output_root", type=Path, required=True,
                        help="Output root; supports {fold} and {seed} placeholders")
    run_p.add_argument("--folds", type=int, nargs="+", default=[0],
                        help="Fold indices (0-4)")
    run_p.add_argument("--seeds", type=int, nargs="+", default=[3],
                        help="Seed values")
    run_p.add_argument("--epochs", type=int, default=30)
    run_p.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    run_p.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    if args.cmd == "run":
        run(args.prepared_dir, args.output_root,
            args.folds, args.seeds, args.epochs, args.device, args.lr)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
