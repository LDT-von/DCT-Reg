#!/usr/bin/env python3
"""Checked five-fold v3.13 summaries; never train or load a checkpoint.

Legacy exports remain usable for error/pairing/patch plots. Retrieval requires
metric v2 because v1 normalized the arithmetic training center a second time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from survot_rank.evidence.additional import RECON_MODES, RETRIEVAL_VERSION, RETRIEVAL_CENTER
from survot_rank.evidence.additional_plots import read_exports
from survot_rank.evidence.manifest import save_json, sha256

CLASSES = ("reconstruction", "pathway_advantage", "pairing", "patch_deletion", "patch_budget")


def collect_groups(roots, *, arm="exp6", cancers=None):
    groups, identities = {}, set()
    for root in roots:
        for record, arrays, source in read_exports(root):
            run = record["run"]
            if run["arm"] != arm or cancers and run["cancer"] not in cancers:
                continue
            key = tuple(run[k] for k in ("cancer", "arm", "seed", "protocol"))
            identity = (*key, run["fold"])
            if identity in identities:
                raise ValueError("Duplicate experiment across export roots")
            identities.add(identity)
            groups.setdefault(key, []).append((record, arrays, source))
    if not groups:
        raise ValueError("No matching completed exports")
    for group in groups.values():
        group.sort(key=lambda x: x[0]["run"]["fold"])
        if [r["run"]["fold"] for r, _, _ in group] != list(range(5)):
            raise ValueError("Every summary requires exactly folds 0 through 4")
        seen = set()
        first = group[0][0]
        for record, arrays, _ in group:
            ids = set(arrays["case_ids"].tolist())
            if seen & ids:
                raise ValueError("Validation patients overlap across summary folds")
            seen.update(ids)
            if record["coefficients"] != first["coefficients"]:
                raise ValueError("Different effective training coefficients across folds")
    return groups


def summarize_group(group, classes=CLASSES):
    records, arrays = [x[0] for x in group], [x[1] for x in group]
    run = records[0]["run"]
    result = dict(cancer=run["cancer"], arm=run["arm"], seed=run["seed"], protocol=run["protocol"],
                  folds=[r["run"]["fold"] for r in records], patients=[len(a["case_ids"]) for a in arrays],
                  classes=list(classes), note="Equal-weight fold summaries; repeats are nested within fold.")
    needs_reconstruction = any(c in classes for c in ("reconstruction", "pathway_advantage", "pairing"))
    if needs_reconstruction:
        if not all("reconstruction" in r for r in records):
            raise ValueError("Requested reconstruction/pairing class is absent")
        names = arrays[0]["pathway_names"]
        if any(not np.array_equal(a["pathway_names"], names) for a in arrays):
            raise ValueError("Pathway order differs across folds; cannot average indices")
        if "reconstruction" in classes and any(r.get("retrieval_metric_version") != RETRIEVAL_VERSION or
                r.get("retrieval_center_definition") != RETRIEVAL_CENTER for r in records):
            raise ValueError("Retrieval v1 center is invalid for the corrected plot; re-export reconstruction only")
        native = np.stack([a["reconstruction_native_error"].mean(0) for a in arrays])
        result["reconstruction"] = {
            k: dict(error_by_fold=[float(a["reconstruction_"+k+"_error"].mean()) if k != "shuffled" else
                                   float(a["reconstruction_shuffled_all_error"].mean()) for a in arrays],
                    **({"top1_by_fold":[r["reconstruction"][k]["top1"] for r in records]} if "reconstruction" in classes else {}))
            for k in RECON_MODES}
        result["retrieval_metric_versions"] = [r.get("retrieval_metric_version", 1) for r in records]
        result["chance_by_fold"] = [1/n for n in result["patients"]]
        if "pathway_advantage" in classes:
            differences = [np.stack([a["reconstruction_"+k+"_error"].mean(0) for a in arrays])-native
                           for k in ("product", "direct", "train_mean")]
            differences.append(np.stack([a["reconstruction_shuffled_all_error"].mean((0, 1)) for a in arrays])-native)
            result["pathway_names"] = names.tolist()
            result["pathway_advantage_by_fold"] = np.stack(differences).tolist()
        result["pairing"] = dict(native_cindex_by_fold=[r["pairing"]["native_cindex"] for r in records],
            shuffled_cindex_by_fold=[r["pairing"]["shuffled_cindex"] for r in records],
            mean_abs_delta_by_fold=[r["pairing"]["mean_abs_delta"] for r in records],
            repeats_by_fold=[r["repeats"] for r in records])
    if any(c in classes for c in ("patch_deletion", "patch_budget")):
        if not all("patches" in r for r in records):
            raise ValueError("Requested patch class is absent")
        fractions = [d["removed_fraction"] for d in records[0]["patches"]["deletion"]]
        retained = [b["retained_fraction"] for b in records[0]["patches"]["budget"]]
        for r in records:
            if fractions != [d["removed_fraction"] for d in r["patches"]["deletion"]] or retained != [b["retained_fraction"] for b in r["patches"]["budget"]]:
                raise ValueError("Patch fractions/order differ across folds")
        result["patches"] = dict(deleted_fractions=fractions, retained_fractions=retained,
            unpadded_cindex_by_fold=[r["patches"]["unpadded_cindex"] for r in records],
            attention_relative_range_by_fold=[r["patches"]["attention_relative_range_median"] for r in records],
            padding_abs_delta_by_fold=[r["patches"]["padding_abs_delta_mean"] for r in records])
        for k in ("top", "bottom", "random"):
            result["patches"][k+"_delta_cindex"] = [[float(np.mean(d[k+"_cindex"]))-r["patches"]["unpadded_cindex"] for d in r["patches"]["deletion"]] for r in records]
            result["patches"][k+"_abs_delta"] = [[d[k+"_abs_delta"] for d in r["patches"]["deletion"]] for r in records]
        result["patches"]["budget_cindex"] = [[float(np.mean(b["cindex"])) for b in r["patches"]["budget"]] for r in records]
        result["patches"]["budget_abs_delta"] = [[b["abs_delta"] for b in r["patches"]["budget"]] for r in records]
    return result


def check_paper_reference(group, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = next(m["folds"] for m in payload["models"] if m["id"] == "dct_v313")
    for record, arrays, _ in group:
        run = record["run"]
        if run["arm"] != "exp6":
            continue
        cancer = run["cancer"].upper()
        from survot_rank.evidence.manifest import cindex
        actual = cindex(arrays["time"], arrays["censor"], arrays["risk"])
        expected = folds[cancer][run["fold"]]
        if abs(actual-expected) > 5.1e-5:
            raise ValueError(f"Paper Full mismatch: {cancer} fold {run['fold']} diagnostic C={actual:.8f}, paper C={expected:.8f}. Check checkpoint, patients and split; do not replace scores.")


def make_summary_figures(roots, output, *, classes=CLASSES, arm="exp6", cancers=None, paper_input=None):
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes) or any(c not in CLASSES for c in classes):
        raise ValueError("Select distinct known summary classes")
    groups = collect_groups(roots, arm=arm, cancers=cancers)
    summaries = []
    # Finish all identity/metric checks before writing any output.
    for key, group in sorted(groups.items()):
        if paper_input is not None:
            check_paper_reference(group, paper_input)
        summary = summarize_group(group, classes)
        summary["sources"] = [dict(json=str(p.resolve()), json_sha256=sha256(p),
                                   array_sha256=r["array_sha256"], training_hashes=r["hashes"],
                                   source_commit=r["run"]["source_commit"], export_commit=r.get("export_commit")) for r, _, p in group]
        summaries.append(summary)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Summary output must be empty; use a fresh directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9, "axes.spines.top":False, "axes.spines.right":False,
                         "pdf.fonttype":42, "savefig.dpi":220})
    out.mkdir(parents=True, exist_ok=True)
    figures = []
    for s in summaries:
        prefix = f"{s['cancer']}_{s['arm']}_s{s['seed']}_summary"
        title = f"{s['cancer'].upper()} / {s['arm']} / five folds"
        def save(fig, kind):
            fig.suptitle(title+" — "+kind.replace("_", " "), fontsize=11)
            fig.tight_layout(rect=(0,.06,1,.94))
            fig.text(.5,.015,"Frozen best-validation checkpoints. Folds shown separately; random repeats are not independent patients.",ha="center",fontsize=7)
            for ext in ("pdf","png"):
                path = out/f"{prefix}_{kind}.{ext}"
                fig.savefig(path,bbox_inches="tight")
                figures.append(dict(path=path.name,sha256=sha256(path),cancer=s["cancer"],kind=kind))
            plt.close(fig)
        def curves(ax, xs, ys, label, color):
            vals = np.asarray(ys)
            ax.plot(xs,vals.mean(0),"o-",label=label,color=color)
            ax.fill_between(xs,vals.min(0),vals.max(0),color=color,alpha=.13)
        if "reconstruction" in classes:
            fig, axes = plt.subplots(1,2,figsize=(11,4))
            x = np.arange(len(RECON_MODES))
            for axis, field in zip(axes,("error_by_fold","top1_by_fold")):
                values = np.asarray([s["reconstruction"][k][field] for k in RECON_MODES])
                axis.bar(x,values.mean(1),yerr=values.std(1,ddof=1),color="#4382b4",capsize=3)
                for fold in range(5):
                    axis.scatter(x+(fold-2)*.055,values[:,fold],s=10,color="#333333",alpha=.65)
                axis.set_xticks(x,RECON_MODES,rotation=30)
            axes[0].set_ylabel("Normalized latent error (fold mean ± sample SD) ↓")
            chance = float(np.mean(s["chance_by_fold"]))
            axes[1].axhline(chance,color="#b54a48",ls="--",label=f"mean fold 1/N = {chance:.4f}")
            axes[1].set_ylabel("Patient retrieval Top-1 v2 ↑")
            axes[1].legend(fontsize=7)
            save(fig,"reconstruction")
        if "pathway_advantage" in classes:
            values = np.asarray(s["pathway_advantage_by_fold"]).mean(1)
            fig, ax = plt.subplots(figsize=(12,3.5))
            scale = max(float(np.abs(values).max()),1e-8)
            heat = ax.imshow(values,aspect="auto",cmap="RdBu_r",vmin=-scale,vmax=scale)
            ax.set_yticks(range(4),["Product − Native","Direct − Native","Train mean − Native","Shuffled − Native"])
            ax.set_xlabel(f"All {len(s['pathway_names'])} pathways; verified identical order across folds")
            fig.colorbar(heat,ax=ax,label="Control error − native error (positive favors native)")
            save(fig,"pathway_advantage")
        if "pairing" in classes:
            p=s["pairing"]
            fig, axes=plt.subplots(1,2,figsize=(10,4))
            colors=plt.get_cmap("tab10")(np.arange(5))
            for fold,(native,shuffled) in enumerate(zip(p["native_cindex_by_fold"],p["shuffled_cindex_by_fold"])):
                shift=(fold-2)*.04
                axes[0].plot([shift,1+shift],[native,np.mean(shuffled)],"o-",color=colors[fold],label=f"fold {fold}: {len(shuffled)} shuffles")
                axes[0].scatter(np.full(len(shuffled),1+shift),shuffled,color=colors[fold],s=9,alpha=.35)
            axes[0].set_xticks([0,1],["Native pairing","Shuffled WSI"])
            axes[0].set_ylabel("C-index (paired within each fold)")
            axes[0].legend(fontsize=7)
            axes[1].bar(range(5),p["mean_abs_delta_by_fold"],color=colors)
            axes[1].set_xticks(range(5),[f"f{i}" for i in range(5)])
            axes[1].set_ylabel("Within-fold mean |Δ risk| (raw risk units)")
            save(fig,"pairing")
        if "patch_deletion" in classes:
            p=s["patches"]
            fig, axes=plt.subplots(1,2,figsize=(10,4))
            for mode,color in (("top","#b54a48"),("bottom","#4382b4"),("random","#666666")):
                curves(axes[0],p["deleted_fractions"],p[mode+"_delta_cindex"],mode,color)
                curves(axes[1],p["deleted_fractions"],p[mode+"_abs_delta"],mode,color)
            axes[0].set_ylabel("Δ C-index from unpadded reference")
            axes[1].set_ylabel("Within-fold mean |Δ risk|")
            for ax in axes:
                ax.set_xlabel("Requested deleted fraction of verified real patches")
                ax.legend(fontsize=8)
            save(fig,"patch_deletion")
        if "patch_budget" in classes:
            p=s["patches"]
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            order=np.argsort(p["retained_fractions"])
            xs=np.asarray(p["retained_fractions"])[order]
            curves(axes[0],xs,np.asarray(p["budget_cindex"])[:,order],"fold mean and range","#4382b4")
            axes[0].axhline(np.mean(p["unpadded_cindex_by_fold"]),color="#b54a48",ls="--",label="unpadded reference")
            curves(axes[1],xs,np.asarray(p["budget_abs_delta"])[:,order],"fold mean and range","#4382b4")
            axes[0].set_ylabel("C-index")
            axes[1].set_ylabel("Within-fold mean |Δ risk|")
            for ax in axes:
                ax.set_xlabel("Requested retained fraction (from budget records)")
                ax.legend(fontsize=8)
            save(fig,"patch_budget")
    index=dict(schema_version=2,groups=summaries,figures=figures,
               paper_reference_checked=paper_input is not None,
               note="Validated source hashes and five-fold identities. No pooled checkpoint risk ranking; no significance claims.")
    save_json(out/"summary_index.json",index)
    return index


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--classes",nargs="+",choices=CLASSES,default=list(CLASSES))
    p.add_argument("--arm",default="exp6")
    p.add_argument("--cancer",action="append")
    p.add_argument("--paper-input",type=Path,help="Check factual C-index against the manuscript's 50 Full folds")
    a=p.parse_args(argv)
    try:
        index=make_summary_figures(a.exports,a.output,classes=a.classes,arm=a.arm,cancers=a.cancer,paper_input=a.paper_input)
    except (ValueError,OSError,KeyError) as error:
        p.exit(1,f"[summary] {error}\nNo model was run.\n")
    print(f"[summary] wrote {len(index['figures'])} figure files; {len(index['groups'])} validated five-fold groups")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
