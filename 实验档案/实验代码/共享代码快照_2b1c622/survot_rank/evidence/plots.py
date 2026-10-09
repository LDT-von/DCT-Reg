"""Publication-format figures from audited artifacts, never invented results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .manifest import audit, predictions, save_json, sha256


def km_curve(times, censor):
    times, censor = np.asarray(times), np.asarray(censor)
    x, survival, low, high = [0.0], [1.0], [1.0], [1.0]
    s, greenwood = 1.0, 0.0
    for t in np.unique(times[censor == 0]):
        n, d = int((times >= t).sum()), int(((times == t) & (censor == 0)).sum())
        s *= 1 - d / n
        if n > d:
            greenwood += d / (n * (n - d))
        delta = 1.96 * s * np.sqrt(greenwood)
        x.append(float(t)); survival.append(float(s))
        low.append(max(0, float(s - delta))); high.append(min(1, float(s + delta)))
    x.append(float(times.max()))
    survival.append(survival[-1]); low.append(low[-1]); high.append(high[-1])
    return tuple(np.asarray(value) for value in (x, survival, low, high))


def pairable(left, right, run_left, run_right):
    """Same patients, outcomes, split and declared common training conditions."""
    from survot_rank.config import apply_overrides, flatten_config, load_config
    if left["case_ids"] != right["case_ids"] or left["time"] != right["time"] or left["censor"] != right["censor"]:
        return False, "validation patients/outcomes differ"
    if left["hashes"]["split_csv"] != right["hashes"]["split_csv"]:
        return False, "split files differ"
    keys = ("wsi_encoder", "encoding_dim", "num_patches", "which_splits", "signature", "rna_format", "label_col",
            "max_epochs", "batch_size", "lr", "opt", "reg", "scheduler", "fit_bins_on_train", "binning_mode",
            "slot_num_wsi", "slot_num_omics", "slot_iters", "wsi_projection_dim", "dct_slot_init_mode")
    flats = [flatten_config(apply_overrides(load_config(run["config"]), run.get("overrides", []))) for run in (run_left, run_right)]
    differences = [key for key in keys if flats[0].get(key) != flats[1].get(key)]
    return not differences, "common settings differ: " + ", ".join(differences) if differences else "matched declared common settings; inspect source/objective provenance"


def make_figures(manifest, output, *, exports=None, coordinates=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 300, "axes.titleweight": "bold"})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Figure output directory must be empty to prevent mixing stale and current figures")
    report = audit(manifest)
    save_json(out / "audit.json", report)
    if not report["passed"]:
        raise ValueError("Integrity audit failed: " + "; ".join(report["errors"]))
    index = dict(figures=[], skipped=[], note="All scores are best-validation legacy_val results, not outer-test scores. Fold dots are descriptive; no independent-fold significance claim.")
    runs = {run["id"]: run for run in manifest["runs"]}
    rows = {row["id"]: row for row in report["runs"]}

    def save(fig, name, note):
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{name}.png", bbox_inches="tight")
        plt.close(fig)
        index["figures"].append(dict(name=name, note=note))

    groups = {}
    for row in rows.values():
        groups.setdefault((row["cancer"], row["seed"]), {}).setdefault(row["arm"], {})[row["fold"]] = row
    scores = []
    for (cancer, seed), arms in groups.items():
        complete = {arm: folds for arm, folds in arms.items() if set(folds) == set(range(5))}
        incomplete = sorted(set(arms) - set(complete))
        if incomplete:
            index["skipped"].append(f"{cancer}/seed{seed}: no five-fold summary for {incomplete}")
        ordered = [arm for arm in [*(f"exp{i}" for i in range(7)), "direct", "independent", "slotspe_matched", "slotspe_native"] if arm in complete]
        if ordered:
            fig, ax = plt.subplots(figsize=(max(5, len(ordered) * .8), 3.8))
            for j, arm in enumerate(ordered):
                values = np.array([complete[arm][f]["cindex"] for f in range(5)])
                color = "#BE4451" if arm == "exp6" else "#3B7795"
                ax.bar(j, values.mean(), color=color, alpha=.25, width=.62)
                ax.scatter(j + np.linspace(-.16, .16, 5), values, s=27, color=color, zorder=3)
                ax.errorbar(j, values.mean(), yerr=values.std(ddof=1), fmt="_", color=color, capsize=5)
                scores.append(dict(cancer=cancer, seed=seed, arm=arm, mean=values.mean(), sd=values.std(ddof=1)))
            ax.set(xticks=range(len(ordered)), xticklabels=ordered, ylabel="C-index (best validation)",
                   title=f"{cancer.upper()} · seed {seed} · five folds")
            ax.set_ylim(max(0, min(row["cindex"] for folds in complete.values() for row in folds.values()) - .03), 1)
            ax.tick_params(axis="x", rotation=30)
            save(fig, f"scores_{cancer}_s{seed}", "Mean ± sample SD; five fold dots. Historical reconstruction coefficients and native baseline recipes may differ; this overview is descriptive.")
        if "exp6" in complete:
            for arm in ordered:
                if arm == "exp6":
                    continue
                pairs = [(complete["exp6"][fold], complete[arm][fold]) for fold in range(5)]
                checks = [pairable(a, b, runs[a["id"]], runs[b["id"]]) for a, b in pairs]
                if not all(check[0] for check in checks):
                    index["skipped"].append(f"paired {cancer}/exp6 vs {arm}: {[x[1] for x in checks if not x[0]]}")
                    continue
                delta = np.array([a["cindex"] - b["cindex"] for a, b in pairs])
                fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
                for fold, (a, b) in enumerate(pairs):
                    axes[0].plot([0, 1], [b["cindex"], a["cindex"]], "o-", alpha=.75, label=f"Fold {fold}")
                axes[0].set(xticks=[0, 1], xticklabels=[arm, "Full"], ylabel="C-index")
                axes[0].legend(fontsize=7)
                axes[1].axhline(0, color=".6", linewidth=.8)
                axes[1].bar(range(5), delta * 100, color=["#BE4451" if x >= 0 else "#3B7795" for x in delta])
                axes[1].set(xlabel="Fold", ylabel="Full gain (percentage points)", title=f"Mean gain {delta.mean() * 100:+.2f} pp")
                fig.suptitle(f"{cancer.upper()} · Full vs {arm} · seed {seed}")
                fig.tight_layout()
                save(fig, f"paired_{cancer}_{arm}_s{seed}", "Paired patient/split/common-setting checks passed. No statistical significance asserted; source revisions and objective coefficients are in the manifests.")
    if scores:
        with (out / "scores.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader(); writer.writerows(scores)

    coord = json.loads(Path(coordinates).read_text(encoding="utf-8")) if coordinates else None
    exported, diagnostic_rows, profiles = [], [], []
    for path in sorted(Path(exports).glob("*/export.json")) if exports else []:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        run_id = metadata["run"]["id"]
        if run_id not in rows:
            continue
        if metadata["hashes"] != rows[run_id]["hashes"]:
            raise ValueError(f"Export source hashes differ from current manifest: {run_id}")
        arrays = np.load(path.parent / "patients.npz", allow_pickle=False)
        pred = predictions(runs[run_id]["predictions"])
        mapping = {cid: i for i, cid in enumerate(arrays["case_ids"].tolist())}
        if set(mapping) != set(pred["case_ids"]) or len(mapping) != len(arrays["case_ids"]):
            raise ValueError(f"Export patient IDs differ: {run_id}")
        order = [mapping[cid] for cid in pred["case_ids"]]
        if not np.allclose(arrays["risk"][order], pred["risk"], atol=2e-5, rtol=1e-5):
            raise ValueError(f"Export factual risks differ: {run_id}")
        exported.append(metadata)
        diagnostic_rows.append(dict(run_id=run_id, arm=runs[run_id]["arm"], cancer=runs[run_id]["cancer"],
            seed=runs[run_id]["seed"], fold=runs[run_id]["fold"],
            **{f"{modality}_{key}": float(arrays[f"{modality}_{key}"].mean())
               for modality in ("wsi", "omic") for key in ("mean_cosine", "mean_distance", "hazard_variance")}))
        if metadata.get("profile"):
            profiles.append(dict(run_id=run_id, **metadata["profile"]))
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        alpha = [item["alpha"] for item in metadata["sweep"]]
        for ax, key, label in zip(axes, ("cindex", "mean_abs_risk_change", "mean_cross_error"),
                                  ("C-index", "Mean absolute risk change", "Δ latent cross-reconstruction error")):
            values = np.asarray([item[key] for item in metadata["sweep"]])
            if key == "mean_cross_error":
                ax.set_title(f"Factual error: {values[0]:.5f}", fontsize=9)
                values = values - values[0]
            ax.plot(alpha, values, "o-", color="#3B7795")
            ax.set(xlabel="Plan replacement α", ylabel=label)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        fig.suptitle(run_id); fig.tight_layout()
        save(fig, f"plan_sweep_{run_id}", "Intervention in a fixed trained model; not equivalent to training Independent. No monotonicity assumed. Cross error is within-model latent error, not gene recovery.")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, key, label in zip(axes, ("mean_cosine", "mean_distance", "hazard_variance"),
                                  ("Pairwise cosine", "RMS slot distance", "Within-patient hazard variance")):
            ax.boxplot([arrays[f"wsi_{key}"], arrays[f"omic_{key}"]], showfliers=False)
            ax.set(xticks=[1, 2], xticklabels=["WSI", "Omics"], ylabel=label)
        fig.suptitle(run_id); fig.tight_layout()
        save(fig, f"slot_diagnostics_{run_id}", "Validation-patient distributions; collapse diagnosis requires matched Exp2/Exp3/Full comparisons, not a favorable single plot.")
        if "km_high" in arrays:
            fig, ax = plt.subplots(figsize=(5, 4))
            for high, label, color in ((False, "Low risk", "#3B7795"), (True, "High risk", "#BE4451")):
                mask = arrays["km_high"] == high
                if mask.any():
                    x, s, low, upper = km_curve(arrays["time"][mask], arrays["censor"][mask])
                    ax.step(x, s, where="post", color=color, label=f"{label} (n={mask.sum()})")
                    ax.fill_between(x, low, upper, step="post", color=color, alpha=.12)
            ax.set(xlabel="Follow-up time (dataset units)", ylabel="Survival probability", ylim=(0, 1.03), title=run_id)
            ax.legend()
            ticks = np.linspace(0, arrays["time"].max(), 5)
            text = []
            for high, label in ((False, "Low"), (True, "High")):
                mask = arrays["km_high"] == high
                text.append(f"{label}: " + ", ".join(f"{t:.1f}: {(arrays['time'][mask] >= t).sum()}" for t in ticks))
            fig.text(.12, -.04, "At risk (time: count)\n" + "\n".join(text), fontsize=7)
            save(fig, f"km_{run_id}", "Groups fixed by this fold's training-patient median risk; fold models are not pooled by raw risk. Bands use Greenwood normal approximation. Best-validation descriptive evidence.")
            clinical_figures(arrays, run_id, plt, save, index)
        else:
            index["skipped"].append(f"KM/Brier/calibration {run_id}: export --km to infer training-reference patients")
        for case in metadata["cases"]:
            bag = np.load(path.parent / case["file"], allow_pickle=False)
            prefix = f"case_{run_id}_{Path(case['file']).stem}"
            plans = bag["plans"][0]  # [stage, geometry, WSI coordinate, omics coordinate]
            fig, axes = plt.subplots(plans.shape[0], plans.shape[1], figsize=(plans.shape[1] * 2.8, plans.shape[0] * 2.2), squeeze=False, constrained_layout=True)
            for stage in range(plans.shape[0]):
                for geometry in range(plans.shape[1]):
                    ax = axes[stage, geometry]
                    im = ax.imshow(plans[stage, geometry], cmap="magma", vmin=0, vmax=plans.max())
                    ax.set(title=f"Geometry {geometry + 1}" if stage == 0 else "",
                           xlabel="Omics slot" if stage == plans.shape[0] - 1 else "",
                           ylabel=f"Stage {stage + 1}\nWSI slot" if geometry == 0 else "",
                           xticks=range(plans.shape[-1]), xticklabels=range(1, plans.shape[-1] + 1),
                           yticks=range(plans.shape[-2]), yticklabels=range(1, plans.shape[-2] + 1))
                    ax.tick_params(labelsize=8)
            fig.suptitle(case["case_id"], fontsize=11)
            fig.colorbar(im, ax=list(axes.flat), shrink=.6, label="Transport mass")
            save(fig, prefix + "_plans", "Every stage and geometry, common color scale; stages are latent coordinates, not established clinical stages.")
            names = arrays["pathway_names"]
            attention = bag["attention_omic"][0]
            top = np.argsort(attention.max(axis=0))[-min(20, len(names)):]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            im = axes[0].imshow(attention[:, top].T, aspect="auto", cmap="Blues")
            axes[0].set(yticks=range(len(top)), yticklabels=[str(names[i])[:55] for i in top], xlabel="Omics slot", title="Top pathway pooling weights",
                        xticks=range(attention.shape[0]), xticklabels=range(1, attention.shape[0] + 1))
            fig.colorbar(im, ax=axes[0], shrink=.7)
            axes[1].barh(range(len(top)), bag["cross_pathway_error"][0, 0, top], color="#BE4451", alpha=.7)
            axes[1].set(yticks=range(len(top)), yticklabels=[], xlabel="Latent reconstruction error", title="Factual cross branch")
            axes[1].set_ylim(len(top) - .5, -.5)
            fig.suptitle(case["case_id"]); fig.tight_layout()
            save(fig, prefix + "_pathways", "Pathways selected by pooling weights, not survival outcomes. Encoded-token reconstruction error; no enrichment or biological causality claim.")
            if coord:
                spatial_figures(case, bag, coord, Path(coordinates).resolve().parent, prefix, plt, save, index)
            else:
                index["skipped"].append(f"Spatial map {prefix}: verified feature coordinates were not supplied")
    if exports and not exported:
        index["skipped"].append("No completed exports matched the audited manifest")
    for cancer, seed in groups:
        candidates = {arm: {row["fold"]: row for row in diagnostic_rows if
                      row["arm"] == arm and row["cancer"] == cancer and row["seed"] == seed}
                      for arm in ("exp2", "exp3", "exp6")}
        complete = [arm for arm, fold_rows in candidates.items() if set(fold_rows) == set(range(5))]
        if len(complete) < 2:
            continue
        checks = [pairable(rows[candidates[complete[0]][fold]["run_id"]], rows[candidates[arm][fold]["run_id"]],
                          runs[candidates[complete[0]][fold]["run_id"]], runs[candidates[arm][fold]["run_id"]])
                  for arm in complete[1:] for fold in range(5)]
        if not all(item[0] for item in checks):
            index["skipped"].append(f"Collapse comparison {cancer}/seed{seed}: unmatched patient/split/common settings")
            continue
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        for i, modality in enumerate(("wsi", "omic")):
            for j, (key, label) in enumerate(zip(("mean_cosine", "mean_distance", "hazard_variance"), ("Pairwise cosine", "RMS slot distance", "Hazard variance"))):
                for fold in range(5):
                    axes[i, j].plot(range(len(complete)), [candidates[arm][fold][f"{modality}_{key}"] for arm in complete], "o-", alpha=.5)
                axes[i, j].set(xticks=range(len(complete)), xticklabels=complete, ylabel=f"{modality.upper()} · {label}")
        fig.suptitle(f"{cancer.upper()} · Anti-collapse diagnostics · seed {seed}"); fig.tight_layout()
        save(fig, f"collapse_comparison_{cancer}_s{seed}", "Five paired fold means; Exp2→Exp3 isolates the diversity addition when source/objective provenance agrees. Full also adds reconstruction, so attribution is not automatic.")
    if profiles:
        save_json(out / "profiles.json", profiles)
        profile_groups = {}
        for profile in profiles:
            key = tuple(profile.get(field) for field in ("hardware", "torch_version", "batch_size", "patches", "forward"))
            profile_groups.setdefault(key, []).append(profile)
        for j, members in enumerate(profile_groups.values()):
            fig, axes = plt.subplots(1, 2, figsize=(max(7, len(members) * .8), 3.5))
            for ax, key, label in zip(axes, ("parameters", "latency_ms_median"), ("Parameters", "Median latency (ms)")):
                ax.bar(range(len(members)), [row[key] for row in members], color="#3B7795", alpha=.75)
                ax.set(xticks=range(len(members)), xticklabels=[row["run_id"] for row in members], ylabel=label)
                ax.tick_params(axis="x", rotation=75, labelsize=7)
            fig.tight_layout()
            save(fig, f"efficiency_group{j}", "Same recorded hardware, torch version, batch size, patch count and forward definition. CUDA timings synchronize and exclude warmup; not end-to-end slide extraction time.")
    save_json(out / "figure_index.json", index)
    print(f"[plot] {len(index['figures'])} figures; {len(index['skipped'])} skipped; {out / 'figure_index.json'}")
    return index


def clinical_figures(data, run_id, plt, save, index):
    """Discrete-bin horizons only, with training censoring reference and support."""
    try:
        from sksurv.metrics import brier_score
        from sksurv.util import Surv
        train = Surv.from_arrays(event=data["train_censor"] == 0, time=data["train_time"])
        val = Surv.from_arrays(event=data["censor"] == 0, time=data["time"])
        horizons = data["bins"][1:-1]
        valid = (horizons >= max(data["train_time"].min(), data["time"].min())) & (horizons < min(data["train_time"].max(), data["time"].max()))
        columns = np.flatnonzero(valid)
        if not len(columns):
            raise ValueError("No discrete-bin edge within train/validation follow-up support")
        horizons = horizons[valid]
        prediction = data["survival"][:, columns]
        _, scores = brier_score(train, val, prediction, horizons)
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        ax.plot(horizons, scores, "o-", color="#3B7795")
        ax.set(xlabel="Bin-edge follow-up time", ylabel="IPCW Brier score", title=run_id)
        save(fig, f"brier_{run_id}", "Training-fold censoring reference; only actual finite bin edges within train/validation support. Unavailable IPCW is reported, not replaced by zero.")
        # Fixed rule: median supported horizon, three prediction groups, >=10 patients/group.
        middle = len(horizons) // 2
        horizon, p = horizons[middle], prediction[:, middle]
        boundaries = np.quantile(p, [1 / 3, 2 / 3])
        group = np.searchsorted(boundaries, p, side="right")
        points = []
        for j in range(3):
            mask = group == j
            if mask.sum() < 10 or data["time"][mask].max() < horizon:
                continue
            x, s, low, high = km_curve(data["time"][mask], data["censor"][mask])
            at = np.searchsorted(x, horizon, side="right") - 1
            points.append((p[mask].mean(), s[at], low[at], high[at], mask.sum()))
        if len(points) < 2:
            raise ValueError("Calibration needs at least two supported prediction groups with >=10 patients each")
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot([0, 1], [0, 1], "--", color=".6")
        for predicted, observed, low, high, n in points:
            ax.errorbar(predicted, observed, yerr=[[observed - low], [high - observed]], fmt="o", color="#BE4451", capsize=4)
            ax.annotate(f"n={n}", (predicted, observed), xytext=(4, 5), textcoords="offset points", fontsize=8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted survival", ylabel="Kaplan–Meier survival", title=f"{run_id}\nt={horizon:.2f}")
        save(fig, f"calibration_{run_id}", "Median supported bin-edge horizon; three survival-prediction groups, >=10 patients each. Within-group KM corrects censoring; Greenwood bands are approximate. No outcome-tuned horizon.")
    except (ImportError, ValueError) as error:
        index["skipped"].append(f"Brier/calibration {run_id}: {error}")


def spatial_figures(case, bag, coordinates, base, prefix, plt, save, index):
    for j, feature in enumerate(case["features"]):
        matches = [item for item in coordinates["features"] if item["feature_sha256"] == feature["sha256"]]
        if len(matches) != 1:
            index["skipped"].append(f"Spatial {prefix}/feature{j}: need one matching feature hash")
            continue
        item = matches[0]
        coord_path = base / item["coords"]
        if sha256(coord_path) != item["coords_sha256"]:
            raise ValueError("Coordinate file hash mismatch")
        xy = np.load(coord_path, allow_pickle=False)
        if xy.shape != (feature["patch_count"], 2) or not np.isfinite(xy).all():
            raise ValueError("Coordinate rows must exactly match original feature rows")
        mask = bag["slide_index"] == j
        if not mask.any():
            index["skipped"].append(f"Spatial {prefix}/feature{j}: no sampled patches on this slide")
            continue
        xy = xy[bag["patch_index"][mask]]
        attention = bag["attention_wsi"][0][:, mask]
        slots = attention.shape[0]
        columns = min(4, slots)
        fig, axes = plt.subplots(int(np.ceil(slots / columns)), columns, figsize=(3 * columns, 3 * int(np.ceil(slots / columns))), squeeze=False)
        thumbnail = plt.imread(base / item["thumbnail"]) if item.get("thumbnail") else None
        scale = np.asarray(item.get("thumbnail_scale", [1, 1]), dtype=float)
        if scale.shape != (2,) or np.any(scale <= 0):
            raise ValueError("thumbnail_scale must be two positive factors")
        plot_xy = xy * scale if thumbnail is not None else xy
        if thumbnail is not None and ((plot_xy < 0).any() or (plot_xy[:, 0] >= thumbnail.shape[1]).any() or (plot_xy[:, 1] >= thumbnail.shape[0]).any()):
            raise ValueError("Scaled patch coordinates fall outside the thumbnail")
        for slot, ax in enumerate(axes.flat):
            if slot >= slots:
                ax.axis("off"); continue
            if thumbnail is not None:
                ax.imshow(thumbnail)
            ax.scatter(plot_xy[:, 0], plot_xy[:, 1], c=attention[slot], cmap="inferno", s=7, vmin=0, vmax=max(float(attention.max()), 1e-10), alpha=.7)
            ax.set_title(f"WSI slot {slot + 1}")
            if thumbnail is None:
                ax.invert_yaxis(); ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(case["case_id"]); fig.tight_layout()
        save(fig, f"{prefix}_spatial_{j}", "Verified feature/coordinate row hashes; exact deterministic sampled patches, excluding padding. Attention rollout is descriptive; missing WSI thumbnails produce coordinate-only maps.")
