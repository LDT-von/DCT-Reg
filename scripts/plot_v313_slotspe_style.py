#!/usr/bin/env python3
"""SlotSPE-inspired publication figures from existing DCT exports; no inference.

case: real slide, spatial slot maps, Top-5 tissue patches, pathway links.
pathways: slot-specific Top-3/Bottom-3 and pathway-to-slot assignment map.
coupling: learned OT, factual-marginal independent product, and their residual.
cohort: four risk groups, Top-10 pathways per group, raw and centered weights.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from survot_rank.evidence.slotspe_style import (
    collect_attention, group_pathways, load_case, load_spatial_assets,
    patch_reader, top_patch_rows, transport_maps,
)

COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#A6761D", "#666666"]


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300})
    return plt


def save(fig, output, name, metadata):
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{name}.{suffix}" for suffix in ("png", "pdf", "json")]
    if any(p.exists() for p in targets):
        raise ValueError(f"Figure already exists: {name}; choose a fresh output directory")
    fig.savefig(targets[0], bbox_inches="tight", facecolor="white")
    fig.savefig(targets[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    targets[2].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[figure] {targets[0]}")


def case_meta(case):
    return dict(case_id=case["case_id"], run=case["meta"]["run"], source=case["source"],
                time_months=case["time"], censor=case["censor"],
                risk=float(case["arrays"]["risk"][0]), risk_direction="higher_is_worse",
                attention="final pooling plus shared-prototype rollout",
                note="Descriptive slot associations; attention/OT is not causal risk attribution")


def wrap(name, width=40):
    return textwrap.fill(str(name).replace("_", " "), width=width)


def pathway_panel(ax, weights, names, title, label_ax=None):
    n = min(3, len(weights) // 2)
    if n < 1:
        raise ValueError("At least two pathways are required")
    order = np.argsort(weights, kind="stable")
    chosen = np.concatenate([order[-n:][::-1], order[:n]])
    ax.barh(range(2*n), weights[chosen], color=["#0072B2"]*n + ["#AAB7C4"]*n)
    ax.set_yticks(range(2*n))
    if label_ax is None:
        ax.set_yticklabels([wrap(names[i]) for i in chosen], fontsize=7)
    else:
        # Reserve a real column for names so long pathways cannot cover tissue.
        ax.set_yticklabels([])
        label_ax.set_xlim(0, 1)
        label_ax.set_ylim(2*n-.5, -.5)
        label_ax.axis("off")
        for y, index in enumerate(chosen):
            label_ax.text(.98, y, wrap(names[index], 32), ha="right", va="center",
                          fontsize=8, clip_on=True)
    ax.invert_yaxis()
    (label_ax if label_ax is not None else ax).set_title(
        title, loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("Raw pathway pooling attention", fontsize=8)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    return [dict(pathway=str(names[i]), attention=float(weights[i]),
                 selection="top" if j < n else "bottom") for j, i in enumerate(chosen)]


def plot_pathways(case, output):
    plt = plotting()
    ao = case["arrays"]["attention_omic"][0]
    rows = int(np.ceil(ao.shape[0] / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(15, 3.6*rows), squeeze=False)
    panels = []
    for slot, ax in enumerate(axes.flat):
        if slot >= len(ao):
            ax.axis("off"); continue
        panels.append(dict(slot=slot, pathways=pathway_panel(ax, ao[slot], case["names"],
                                                           f"Omics slot {slot}: Top-3 / Bottom-3")))
    fig.suptitle(f"{case['case_id']} | slot-specific pathway attention", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97], w_pad=4)
    save(fig, output, "slot_pathways", dict(**case_meta(case), panels=panels))
    # All pathways x slots, sorted by preferred slot. Actual values remain raw.
    winner = ao.argmax(axis=0)
    order = np.lexsort((-ao.max(axis=0), winner))
    fig, ax = plt.subplots(figsize=(6, 9))
    im = ax.imshow(ao[:, order].T, aspect="auto", cmap="magma", vmin=0)
    ax.set(xticks=range(len(ao)), xticklabels=range(len(ao)), xlabel="Omics slot",
           ylabel=f"Pathways (N={len(order)}), grouped by strongest slot",
           title=f"{case['case_id']} | pathway-to-slot pooling")
    fig.colorbar(im, ax=ax, label="Raw attention", shrink=.7)
    save(fig, output, "pathway_slot_map", dict(**case_meta(case),
         pathway_order=[str(case["names"][i]) for i in order],
         assignment="argmax of final pooling attention; a visual grouping, not a biological label"))


def projections(case):
    a = case["arrays"]
    return transport_maps(a["plans"][0], a["stage_gate"][0],
                          a["attention_wsi"][0], a["attention_omic"][0])


def plot_coupling(case, output):
    plt = plotting()
    values = projections(case)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    limit = max(float(values[k].max()) for k in ("transport", "independent"))
    for ax, key, title in zip(axes, ("transport", "independent", "residual"),
                              ("A  Learned transport", "B  Same-marginal independent plan", "C  Learned minus independent")):
        matrix = values[key]
        vmax = max(float(np.abs(matrix).max()), 1e-12) if key == "residual" else limit
        im = ax.imshow(matrix, aspect="equal", cmap="RdBu_r" if key == "residual" else "YlOrRd",
                       vmin=-vmax if key == "residual" else 0, vmax=vmax)
        ax.set(xticks=range(matrix.shape[1]), yticks=range(matrix.shape[0]),
               xlabel="Omics slot", ylabel="WSI slot", title=title)
        fig.colorbar(im, ax=ax, shrink=.65)
    save(fig, output, "transport_association", dict(**case_meta(case),
         stage_gate=case["arrays"]["stage_gate"][0].tolist(),
         learned=values["transport"].tolist(), independent=values["independent"].tolist(),
         residual=values["residual"].tolist(),
         note_projection="Geometry mean then factual-stage-gate fusion; product formed per stage/geometry before fusion"))


def spatial_axis(ax, spatial, coords, footprint, weights=None, labels=None, vmax=None):
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    ax.imshow(spatial["thumbnail"])
    patches = [Rectangle(xy*spatial["scale"], *(footprint*spatial["scale"])) for xy in coords]
    collection = PatchCollection(patches, linewidths=0, alpha=.70)
    if labels is not None:
        collection.set_facecolor([COLORS[int(i) % len(COLORS)] for i in labels])
    else:
        collection.set_cmap("inferno")
        collection.set_norm(Normalize(0, vmax))
        collection.set_array(np.asarray(weights))
    ax.add_collection(collection)
    ax.set_xlim(0, spatial["thumbnail"].width)
    ax.set_ylim(spatial["thumbnail"].height, 0)
    ax.axis("off")
    return collection


def plot_case(case, spatial, *, slide, slots, output):
    """Main-paper morphology/pathway panels plus all requested Top-5 patches."""
    plt = plotting()
    a, maps = case["arrays"], projections(case)
    aw, om = a["attention_wsi"][0], maps["omics_spatial"]
    si, pi = a["slide_index"], a["patch_index"]
    valid = np.flatnonzero((si == slide) & (pi >= 0))
    if not len(valid):
        raise ValueError("No actual sampled patches for this slide")
    if any(s < 0 or s >= min(len(aw), len(om)) for s in slots):
        raise ValueError("Requested slots must exist in both modalities")
    if pi[valid].max() >= len(spatial["coords"]):
        raise ValueError("Sampled patch index exceeds original coordinate rows")
    # One spatial rectangle per original row; no padding and no fabricated tissue.
    _, unique = np.unique(pi[valid], return_index=True)
    valid = valid[np.sort(unique)]
    coords = spatial["coords"][pi[valid]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(spatial["thumbnail"]); axes[0].axis("off")
    axes[0].set_title("A  Original WSI", loc="left", fontweight="bold")
    spatial_axis(axes[1], spatial, coords, spatial["footprint"], labels=aw[:, valid].argmax(0))
    axes[1].set_title("B  WSI-slot assignment", loc="left", fontweight="bold")
    spatial_axis(axes[2], spatial, coords, spatial["footprint"], labels=om[:, valid].argmax(0))
    axes[2].set_title("C  OT-projected omics-slot assignment", loc="left", fontweight="bold")
    from matplotlib.patches import Patch
    slot_count = max(len(aw), len(om))
    fig.legend(handles=[Patch(color=COLORS[s % len(COLORS)], label=f"Slot {s}") for s in range(slot_count)],
               loc="lower center", ncol=min(8, slot_count), frameon=False)
    fig.suptitle(f"{case['case_id']} | sampled tissue only | slide {slide}", fontweight="bold")
    fig.tight_layout(rect=[0, .07, 1, .92])
    info = dict(**case_meta(case), slide_index=slide, sampled_unique_patches=len(valid),
                coords_sha256=spatial["coords_sha256"], thumbnail_sha256=spatial["thumbnail_sha256"])
    save(fig, output, "wsi_slot_assignments", info)
    read_patch, close = patch_reader(spatial)
    tile_records = []
    try:
        fig = plt.figure(figsize=(17, 4.0*len(slots)))
        grid = fig.add_gridspec(len(slots), 3, width_ratios=[1.1, 1.1, 2.4], hspace=.50, wspace=.30)
        for row, slot in enumerate(slots):
            for col, (weights, label) in enumerate(((aw[slot], "WSI"), (om[slot], "OT-projected omics"))):
                sub = grid[row, col].subgridspec(2, 5, height_ratios=[5, 1.2], hspace=.02, wspace=.04)
                ax = fig.add_subplot(sub[0, :])
                image = spatial_axis(ax, spatial, coords, spatial["footprint"],
                                     weights=weights[valid], vmax=max(float(weights[valid].max()), 1e-12))
                ax.set_title(f"{label} slot {slot}", fontsize=10, loc="left", fontweight="bold")
                fig.colorbar(image, ax=ax, fraction=.04, pad=.01).ax.tick_params(labelsize=6)
                chosen = top_patch_rows(weights, si, pi, slide=slide, count=5)
                for j in range(5):
                    tile = fig.add_subplot(sub[1, j]); tile.axis("off")
                    if j < len(chosen):
                        sample = int(chosen[j]); original = int(pi[sample])
                        tile.imshow(read_patch(original))
                        tile.set_title(f"#{original}", fontsize=6)
                        tile_records.append(dict(slot=slot, modality=label, rank=j+1,
                            sampled_index=sample, original_patch_index=original,
                            attention=float(weights[sample]), coords=spatial["coords"][original].tolist()))
            pathway_grid = grid[row, 2].subgridspec(1, 2, width_ratios=[1.6, 1], wspace=.04)
            labels = fig.add_subplot(pathway_grid[0, 0])
            ax = fig.add_subplot(pathway_grid[0, 1])
            pathway_panel(ax, a["attention_omic"][0, slot], case["names"],
                          f"Omics slot {slot}: Top-3 / Bottom-3", label_ax=labels)
        fig.suptitle(f"{case['case_id']} | tissue regions, Top-5 patches and pathway associations",
                     fontweight="bold", y=.995)
        save(fig, output, "tissue_pathway_case", dict(**info, selected_slots=slots, patches=tile_records,
             row_pairing="Equal indices used for display only; WSI/omics slots are not assumed to match one-to-one",
             per_map_color_scale="0 to each displayed slot's maximum; raw weights recorded in JSON",
             patch_labels="Original feature-row IDs; tissue types need independent pathology annotation"))
    finally:
        close()


def plot_cohort(cohort, output, top=10):
    plt = plotting()
    table = group_pathways(cohort, top)
    height = max(5, .24*len(table["names"]) + 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), gridspec_kw={"wspace": .25})
    for j, (ax, matrix, title) in enumerate(zip(axes, (table["mean"], table["delta"]),
                           ("A  Raw mean pathway attention", "B  Difference from each pathway's four-group mean"))):
        lim = max(float(np.abs(matrix).max()), 1e-12)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r" if j else "YlOrRd",
                       vmin=-lim if j else 0, vmax=lim)
        ax.set_xticks(range(4), [f"Q{i+1}\nn={n}" for i, n in enumerate(table["counts"])])
        ax.set_yticks(range(len(table["names"])),
                      [wrap(n, 50) for n in table["names"]] if j == 0 else [])
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Within-fold predicted-risk percentile: low to high")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=.6, label="Attention" if j == 0 else "Attention difference")
    label = "PARTIAL EXPLORATORY COHORT | " if cohort["partial"] else ""
    fig.suptitle(f"{label}{cohort['cancer'].upper()} | {cohort['arm']} | N={len(cohort['ids'])}",
                 fontweight="bold", y=.99)
    fig.subplots_adjust(left=.40, right=.94, top=.88, bottom=.14)
    metadata = dict(cancer=cohort["cancer"], arm=cohort["arm"], seed=cohort["seed"],
        n=len(cohort["ids"]), folds=cohort["folds"], partial=cohort["partial"],
        sources=cohort["sources"], aggregation="equal mean of real omics-slot attention; no hazard weights",
        grouping="within-fold validation risk midrank quartiles, higher score means worse",
        top_per_group=top, names=table["names"].tolist(), mean=table["mean"].tolist(),
        difference=table["delta"].tolist(), group_counts=table["counts"],
        note="Exploratory pathway/risk association on validation-selected development predictions; no enrichment/causality claim")
    save(fig, output, "cohort_top_pathways", metadata)
    output = Path(output)
    with (output / "patient_groups.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(["case_id", "fold", "raw_risk", "within_fold_percentile", "risk_group"])
        writer.writerows(zip(cohort["ids"], cohort["fold_ids"], cohort["risk"], cohort["percentile"], cohort["groups"]+1))
    with (output / "pathway_group_means.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(["pathway", "Q1", "Q2", "Q3", "Q4"])
        writer.writerows((str(n), *values) for n, values in zip(table["names"], table["mean"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("case", "pathways", "coupling"):
        p = sub.add_parser(name)
        p.add_argument("--export", type=Path, required=True, help="Exact directory containing export.json")
        p.add_argument("--case-id", required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--check-only", action="store_true")
        if name == "case":
            p.add_argument("--assets", type=Path, required=True)
            p.add_argument("--slide-index", type=int, default=0)
            p.add_argument("--slots", default="0,1,2", help="Explicit fixed slot indices; no score-based cherry-picking")
    p = sub.add_parser("cohort")
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--cancer", required=True)
    p.add_argument("--arm", default="exp6")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--allow-partial", action="store_true", help="Watermark incomplete cohorts as exploratory")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.command == "cohort":
            cohort = collect_attention(args.exports, cancer=args.cancer, arm=args.arm,
                                       seed=args.seed, allow_partial=args.allow_partial)
            group_pathways(cohort, args.top)
            print(f"[checked] {args.cancer}: folds={cohort['folds']}, N={len(cohort['ids'])}, partial={cohort['partial']}")
            if not args.check_only:
                plot_cohort(cohort, args.output, args.top)
        else:
            case = load_case(args.export, args.case_id)
            projections(case)
            if args.command == "case":
                spatial = load_spatial_assets(case, args.assets, args.slide_index)
                slots = [int(s) for s in args.slots.split(",")]
                if any(s < 0 or s >= min(case['arrays']['attention_wsi'].shape[1], case['arrays']['attention_omic'].shape[1]) for s in slots):
                    raise ValueError("Requested slot does not exist")
                read, close = patch_reader(spatial)
                try:
                    # Validate selected real tiles before creating any output.
                    maps = projections(case)
                    for weights in (case['arrays']['attention_wsi'][0][slots], maps['omics_spatial'][slots]):
                        for row in weights:
                            chosen = top_patch_rows(row, case['arrays']['slide_index'], case['arrays']['patch_index'], slide=args.slide_index)
                            if not len(chosen):
                                raise ValueError("Requested slide has no real sampled patches")
                            for index in chosen:
                                read(int(case['arrays']['patch_index'][index]))
                finally:
                    close()
                if not args.check_only:
                    plot_case(case, spatial, slide=args.slide_index, slots=slots, output=args.output)
            elif not args.check_only:
                (plot_pathways if args.command == "pathways" else plot_coupling)(case, args.output)
            print(f"[checked] {args.case_id}: patient-ID lookup and pathway/slot dimensions match")
    except (ValueError, OSError, KeyError, ImportError) as error:
        ap.exit(1, f"[figure] {error}\nNo model was executed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
