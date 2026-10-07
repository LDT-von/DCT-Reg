"""Paper figures from completed, hashed additional v3.13 diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .manifest import cindex, save_json, sha256
from .additional import RECON_MODES, summarize_patches


def read_exports(root):
    paths = sorted(Path(root).rglob("additional.json"))
    if not paths:
        raise ValueError("No completed additional exports")
    exports, identities = [], set()
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        run = record["run"]
        identity = tuple(run[k] for k in ("cancer", "arm", "fold", "seed", "protocol"))
        if record.get("schema_version") != 1 or run["protocol"] != "legacy_val" or identity in identities:
            raise ValueError("Unsupported or duplicated experiment identity")
        identities.add(identity)
        array_path = path.parent / "additional.npz"
        if sha256(array_path) != record["array_sha256"]:
            raise ValueError(f"Diagnostic array hash mismatch: {array_path}")
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        ids = arrays["case_ids"].tolist()
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("Invalid patient IDs")
        if any(v.dtype.kind in "fc" and not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Nonfinite diagnostic array")
        if any(arrays[k].shape != (len(ids),) for k in ("risk", "time", "censor")):
            raise ValueError("Patient/outcome vector length mismatch")
        if "patches" in record and record["patches"] != summarize_patches(arrays):
            raise ValueError("Patch summary disagrees with raw patient arrays")
        if "reconstruction" in record:
            for mode in RECON_MODES:
                errors = arrays[f"reconstruction_{mode}_error"] if mode != "shuffled" else arrays["reconstruction_shuffled_all_error"]
                ranks = arrays[f"retrieval_{mode}_rank"] if mode != "shuffled" else arrays["retrieval_shuffled_all_rank"]
                if errors.shape[0] != len(ids) or ranks.shape[0] != len(ids) or np.any((ranks < 1) | (ranks > len(ids))):
                    raise ValueError("Invalid reconstruction shape or retrieval ranks")
                stats = record["reconstruction"][mode]
                if not np.allclose([stats["mean_error"], stats["top1"], stats["mrr"]],
                                   [errors.mean(), (ranks == 1).mean(), (1 / ranks).mean()], atol=1e-7, rtol=1e-6):
                    raise ValueError("Reconstruction summary disagrees with raw patient arrays")
            donors = arrays["donor_case_ids"]
            if donors.shape != (record["repeats"], len(ids)) or any(set(row.tolist()) != set(ids) for row in donors) or np.any(donors == arrays["case_ids"][None]):
                raise ValueError("Invalid shuffled donor identities")
            scores = [cindex(arrays["time"], arrays["censor"], arrays["pairing_shuffled_risk"][:, r]) for r in range(record["repeats"])]
            if not np.allclose(scores, record["pairing"]["shuffled_cindex"], atol=1e-10):
                raise ValueError("Pairing summary disagrees with raw patient arrays")
        for previous, previous_arrays, _ in exports:
            old = previous["run"]
            if tuple(old[k] for k in ("cancer", "arm", "seed", "protocol")) == tuple(run[k] for k in ("cancer", "arm", "seed", "protocol")):
                if set(ids) & set(previous_arrays["case_ids"].tolist()):
                    raise ValueError("Validation patients overlap across folds")
        exports.append((record, arrays, path))
    return exports


def make_additional_figures(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exports = read_exports(root)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Figure directory must be empty; use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    index, table = [], []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                        "pdf.fonttype": 42, "savefig.dpi": 220})
    for record, arrays, source in exports:
        run = record["run"]
        label = f"{run['cancer'].upper()} / {run['arm']} / fold {run['fold']} / seed {run['seed']}"
        prefix = f"{run['cancer']}_{run['arm']}_f{run['fold']}_s{run['seed']}"

        def save(fig, name):
            fig.suptitle(label + " — frozen best-validation diagnostic", fontsize=10)
            fig.tight_layout(rect=(0, .06, 1, .95))
            fig.text(.5, .015, "Conditional on selected checkpoint; random repeats are not independent patients.",
                     ha="center", fontsize=7)
            for extension in ("pdf", "png"):
                fig.savefig(out / f"{prefix}_{name}.{extension}", bbox_inches="tight")
            plt.close(fig)
            index.append(dict(figure=f"{prefix}_{name}", source=str(source.resolve()),
                              source_sha256=sha256(source), array_sha256=record["array_sha256"]))

        if "reconstruction" in record:
            stats = record["reconstruction"]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            labels = ["Native", "Product", "Direct", "Shuffled", "Train mean", "Self"]
            errors = [arrays[f"reconstruction_{k}_error"].mean(-1) if k != "shuffled" else
                      arrays["reconstruction_shuffled_all_error"].mean((1, 2)) for k in RECON_MODES]
            axes[0].boxplot(errors, tick_labels=labels, showfliers=False)
            axes[0].set_ylabel("Normalized latent reconstruction error ↓")
            axes[0].tick_params(axis="x", rotation=35)
            axes[1].bar(labels, [stats[k]["top1"] for k in RECON_MODES], color="#4382b4")
            axes[1].axhline(1 / len(arrays["case_ids"]), color="#b54a48", ls="--", label="1/N chance")
            axes[1].set_ylabel("Centered patient retrieval Top-1 ↑")
            axes[1].tick_params(axis="x", rotation=35)
            axes[1].legend(fontsize=8)
            save(fig, "reconstruction_retrieval")
            # Positive means native has lower error. Pathways are never selected by outcome.
            native = arrays["reconstruction_native_error"]
            differences = np.stack([arrays[f"reconstruction_{k}_error"].mean(0) - native.mean(0)
                                    for k in ("product", "direct", "train_mean")])
            shuffle_difference = arrays["reconstruction_shuffled_all_error"].mean((0, 1)) - native.mean(0)
            differences = np.concatenate((differences, shuffle_difference[None]))
            fig, ax = plt.subplots(figsize=(12, 3.5))
            scale = max(float(np.abs(differences).max()), 1e-8)
            heat = ax.imshow(differences, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
            ax.set_yticks(range(4), ["Product − Native", "Direct − Native", "Train mean − Native", "Shuffled − Native"])
            ax.set_xlabel("All pathway indices in the recorded pathway_names order")
            fig.colorbar(heat, ax=ax, label="Control error − native error (positive favors native)")
            save(fig, "pathway_reconstruction_advantage")
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            shuffled_scores = record["pairing"]["shuffled_cindex"]
            axes[0].scatter(np.zeros(len(shuffled_scores)), shuffled_scores, color="#b54a48", label="shuffled repeats")
            axes[0].axhline(record["pairing"]["native_cindex"], color="#4382b4", label="native matched pairing")
            axes[0].set_xticks([0], ["Mismatch WSI, keep recipient omics"])
            axes[0].set_ylabel("C-index")
            axes[0].legend(fontsize=7)
            axes[1].hist(np.abs(arrays["pairing_shuffled_risk"] - arrays["risk"][:, None]).mean(1), bins=15, color="#4382b4")
            axes[1].set_xlabel("Per-patient mean |Δ risk| over shuffles")
            axes[1].set_ylabel("Patients")
            save(fig, "patient_pairing")
            for k in RECON_MODES:
                table.append(dict(cancer=run["cancer"], arm=run["arm"], fold=run["fold"], seed=run["seed"],
                                  experiment="reconstruction", condition=k, value=stats[k]["mean_error"],
                                  top1=stats[k]["top1"], mrr=stats[k]["mrr"]))
        if "patches" in record:
            stats = record["patches"]
            fractions = arrays["fractions"]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for mode, color in (("top", "#b54a48"), ("bottom", "#4382b4"), ("random", "#666666")):
                values = np.asarray([row[mode + "_cindex"] for row in stats["deletion"]])
                axes[0].plot(fractions, values.mean(1) - stats["unpadded_cindex"], "o-", color=color, label=mode)
                axes[1].plot(fractions, [row[mode + "_abs_delta"] for row in stats["deletion"]], "o-", color=color, label=mode)
                if mode == "random":
                    axes[0].fill_between(fractions, values.min(1) - stats["unpadded_cindex"],
                                         values.max(1) - stats["unpadded_cindex"], color=color, alpha=.15, label="random replicate range")
            axes[0].set_ylabel("Δ C-index from unpadded reference")
            axes[1].set_ylabel("Mean |Δ risk| from unpadded reference")
            for ax in axes:
                ax.set_xlabel("Deleted fraction of verified real patches")
                ax.legend(fontsize=7)
            save(fig, "patch_deletion")
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            retained = 1 - fractions
            values = np.asarray([row["cindex"] for row in stats["budget"]])
            axes[0].plot(retained, values.mean(1), "o-", color="#4382b4")
            axes[0].fill_between(retained, values.min(1), values.max(1), color="#4382b4", alpha=.2)
            axes[0].set_ylabel("C-index (mean and range over subset replicates)")
            axes[1].plot(retained, [row["abs_delta"] for row in stats["budget"]], "o-", color="#4382b4")
            axes[1].set_ylabel("Mean |Δ risk|")
            for ax in axes:
                ax.set_xlabel("Retained fraction of verified real patches")
            save(fig, "patch_budget")
            for row in stats["deletion"]:
                for mode in ("top", "bottom", "random"):
                    table.append(dict(cancer=run["cancer"], arm=run["arm"], fold=run["fold"], seed=run["seed"],
                        experiment="deletion", condition=f"{mode}@{row['removed_fraction']}",
                        value=float(np.mean(row[mode + "_cindex"])), top1="", mrr=""))
    with (out / "fold_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("cancer", "arm", "fold", "seed", "experiment", "condition", "value", "top1", "mrr")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    save_json(out / "figure_index.json", dict(schema_version=1, figures=index,
        note="Per-fold metrics only; never pool incompatible checkpoint risk scales. No significance claims."))
    return index
