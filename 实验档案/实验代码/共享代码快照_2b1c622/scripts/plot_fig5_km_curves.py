#!/usr/bin/env python3
"""Ten-cohort Full KM figures, using five-fold validation groups.

Reads existing audited --km exports only. Does not run models or replace missing
training thresholds with validation medians. Full is the default model.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAPER_CANCERS = ("blca", "brca", "coadread", "hnsc", "kirc",
                "luad", "lusc", "skcm", "stad", "ucec")


def plot_cohort(cohort, output, *, show_logrank=False):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.0, 5.9))
    fig.subplots_adjust(left=.20, right=.97, top=.91, bottom=.32)
    groups = {}
    for group, label, color in [("low", "Low risk", "#2878A8"), ("high", "High risk", "#C64B4B")]:
        rows = [r for r in cohort["patients"] if r["group"] == group]
        times = np.asarray([r["time"] for r in rows])
        events = np.asarray([r["event"] for r in rows])
        groups[group] = (times, events)
        km = KaplanMeierFitter(alpha=.05)
        km.fit(times, events, label=f"{label} (n={len(rows)})")
        km.plot_survival_function(ax=ax, color=color, linewidth=2.2, ci_show=True,
            ci_alpha=.12, show_censors=True, censor_styles={"marker": "+", "ms": 5, "mew": 1})
    max_time = max(r["time"] for r in cohort["patients"])
    if max_time <= 0:
        raise ValueError("At least one positive follow-up time is required")
    ticks = np.linspace(0, max_time, 5)
    ax.set(title=f"{cohort['cancer'].upper()} · DCT Full" if cohort["arm"] == "exp6"
        else f"{cohort['cancer'].upper()} · {cohort['arm']}",
        xlabel="Time (months)", ylabel="Survival probability", ylim=(0, 1.02),
        xlim=(0, max_time), xticks=ticks)
    ax.grid(axis="y", alpha=.15, linewidth=.6)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    # Y(t-) = patients whose observed follow-up is >= t. Compute directly so
    # last-event risk sets cannot be carried forward beyond final follow-up.
    counts = {group: [int((times >= tick).sum()) for tick in ticks]
              for group, (times, _) in groups.items()}
    cohort["number_at_risk"] = dict(times=ticks.tolist(), counts=counts,
        definition="Observed follow-up >= displayed time; immediately before removals at that time")
    ax.text(0, -.19, "Number at risk", transform=ax.transAxes, fontsize=9, clip_on=False)
    for group, label, ypos in [("low", "Low risk", -.245), ("high", "High risk", -.30)]:
        ax.text(-.04, ypos, label, transform=ax.transAxes, ha="right", fontsize=8, clip_on=False)
        for tick, count in zip(ticks, counts[group]):
            ax.text(tick / max_time, ypos, str(count), transform=ax.transAxes,
                    ha="center", fontsize=8, clip_on=False)
    logrank = None
    if show_logrank:
        from lifelines.statistics import logrank_test
        result = logrank_test(groups["low"][0], groups["high"][0],
            event_observed_A=groups["low"][1], event_observed_B=groups["high"][1])
        p, chi2 = float(result.p_value), float(result.test_statistic)
        valid = bool(np.isfinite(p) and np.isfinite(chi2))
        logrank = {"p_value": p if valid else None, "chi2": chi2 if valid else None,
            "interpretation": "exploratory; ignores cross-validation model dependence",
            "status": "available" if valid else "not_estimable"}
        label = f"Exploratory log-rank p = {p:.2g}" if valid else "Log-rank not estimable"
        ax.text(.04, .08, label, transform=ax.transAxes, fontsize=8)
    fig.text(.20, .035, "Groups assigned with each fold's training-risk median, then pooled.\n"
             "Five-fold development predictions. Shading: nominal 95% CI.\n"
             "CI and optional log-rank do not adjust for CV model dependence.", fontsize=6.7)
    stem = f"km_{cohort['cancer']}_{cohort['arm']}_oof"
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    return logrank


def collect_requested_cohorts(export_root, cancers, *, arm="exp6", seed=3):
    """Report every incomplete cohort before permitting any figure output."""
    from survot_rank.evidence.km_oof import collect_cohorts

    cohorts, errors = {}, []
    for cancer in cancers:
        try:
            cohorts.update(collect_cohorts(export_root, [cancer], arm=arm, seed=seed))
        except (ValueError, OSError, KeyError) as error:
            errors.append(f"{cancer.upper()}: {error}")
    if errors:
        raise ValueError("Requested KM coverage incomplete:\n" + "\n".join(errors))
    return cohorts


def plot_ten_cohort_overview(cohorts, output, *, show_logrank=False):
    """Two readable multi-panel pages; all ten cohorts must already be validated."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines import KaplanMeierFitter

    if set(cohorts) != set(PAPER_CANCERS):
        raise ValueError("The paper overview requires all ten cancer cohorts")
    images = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False}):
        for part, names in enumerate((PAPER_CANCERS[:6], PAPER_CANCERS[6:]), 1):
            rows = (len(names) + 1) // 2
            fig, axes = plt.subplots(rows, 2, figsize=(10.5, 4.2 * rows + .4),
                                     squeeze=False)
            fig.subplots_adjust(left=.12, right=.97, top=.93, bottom=.13,
                                hspace=.86, wspace=.52)
            for ax, cancer in zip(axes.flat, names):
                cohort = cohorts[cancer]
                max_time = max(r["time"] for r in cohort["patients"])
                ticks = np.linspace(0, max_time, 5)
                for group, label, color, ypos in [
                    ("low", "Low", "#2878A8", -.26),
                    ("high", "High", "#C64B4B", -.37)]:
                    patients = [r for r in cohort["patients"] if r["group"] == group]
                    times = np.asarray([r["time"] for r in patients])
                    events = np.asarray([r["event"] for r in patients])
                    km = KaplanMeierFitter(alpha=.05).fit(
                        times, events, label=f"{label} risk (n={len(patients)})")
                    km.plot_survival_function(ax=ax, color=color, linewidth=1.8,
                        ci_show=True, ci_alpha=.12, show_censors=True,
                        censor_styles={"marker": "+", "ms": 4, "mew": .8})
                    ax.text(-.04, ypos, label, transform=ax.transAxes,
                            ha="right", fontsize=10, color=color, clip_on=False)
                    for tick in ticks:
                        ax.text(tick/max_time, ypos, str(int((times >= tick).sum())),
                                transform=ax.transAxes, ha="center",
                                fontsize=10, clip_on=False)
                index = PAPER_CANCERS.index(cancer)
                ax.set(title=f"({chr(97+index)}) {cancer.upper()}",
                       xlabel="Time (months)", ylabel="Survival probability",
                       ylim=(0, 1.02), xlim=(0, max_time), xticks=ticks)
                ax.tick_params(labelsize=10)
                ax.legend(frameon=False, loc="upper right", fontsize=10.5)
                ax.grid(axis="y", alpha=.15, linewidth=.6)
                ax.text(0, -.16, "Number at risk", transform=ax.transAxes,
                        fontsize=10, clip_on=False)
                if show_logrank:
                    result = cohort.get("logrank") or {}
                    p_value = result.get("p_value")
                    label = f"Exploratory p={p_value:.2g}" if p_value is not None else "p not estimable"
                    ax.text(.04, .06, label, transform=ax.transAxes, fontsize=9)
            for ax in list(axes.flat)[len(names):]:
                ax.set_visible(False)
            fig.suptitle(f"DCT v3.13 Full: ten-cohort risk stratification ({part}/2)",
                         y=.985, fontsize=16)
            fig.text(.12, .02, "Fold-specific training-risk medians; pooled validation group labels.\n"
                     "Validation-selected checkpoints; shading = nominal 95% CI.",
                     fontsize=10)
            stem = f"figure5_ten_cancers_part{part}"
            for suffix in ("png", "pdf"):
                fig.savefig(output/f"{stem}.{suffix}", dpi=300, facecolor="white")
            images.append(stem+".png")
            plt.close(fig)
    return dict(expected_cancers=list(PAPER_CANCERS), completed_cancers=list(PAPER_CANCERS),
                complete=True, overview_images=images,
                patients={c:cohorts[c]["n"] for c in PAPER_CANCERS},
                panel_order=list(PAPER_CANCERS),
                grouping="per-fold training-risk median, then pool validation labels")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", type=Path, default=ROOT / "results/v313_ten_cancer_km/exports")
    parser.add_argument("--output", type=Path, default=ROOT / "paper/figures/fig5_ten_cancers")
    parser.add_argument("--cancer", action="append", help="Repeat for any cancer with five complete exports; default all ten manuscript cohorts")
    parser.add_argument("--arm", default="exp6", help="Default Full; 'full' is an alias for exp6")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--show-logrank", action="store_true", help="Add an exploratory, unadjusted log-rank p-value")
    parser.add_argument("--check-only", action="store_true", help="Validate existing data without drawing or writing")
    args = parser.parse_args(argv)
    cancers = args.cancer or list(PAPER_CANCERS)
    if len(cancers) != len(set(cancers)) or any(not re.fullmatch(r"[a-z][a-z0-9_]*", c) for c in cancers):
        parser.error("Cancer names must be unique lowercase identifiers")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.arm):
        parser.error("Invalid model arm")
    try:
        from survot_rank.evidence.manifest import save_json, sha256
        # Validate ALL requested cohorts before generating the first plot.
        cohorts = collect_requested_cohorts(args.exports, cancers, arm=args.arm, seed=args.seed)
        if args.check_only:
            for cancer, cohort in cohorts.items():
                print(f"[KM check] {cancer}: five folds, {cohort['n']} unique patients, train thresholds present")
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        for cancer, cohort in cohorts.items():
            result = plot_cohort(cohort, args.output, show_logrank=args.show_logrank)
            stem = f"km_{cancer}_{cohort['arm']}_oof"
            cohort["logrank"] = result
            save_json(args.output / f"{stem}.json", cohort)
            with (args.output / f"{stem}_patients.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(cohort["patients"][0]))
                writer.writeheader()
                writer.writerows(cohort["patients"])
            print(f"[KM] {cancer}: {args.output / (stem + '.png')}")
        if set(cancers) == set(PAPER_CANCERS):
            overview = plot_ten_cohort_overview(cohorts, args.output,
                                               show_logrank=args.show_logrank)
            overview["source_jsons"] = {
                c: {"path": str(args.output / f"km_{c}_{cohorts[c]['arm']}_oof.json"),
                    "sha256": sha256(args.output / f"km_{c}_{cohorts[c]['arm']}_oof.json")}
                for c in PAPER_CANCERS}
            save_json(args.output / "figure5_ten_cancers.json", overview)
            print("[KM] Full manuscript coverage: 10/10; two overview pages written")
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(1, f"[KM] {error}\nNo model was run. Resolve missing exports/thresholds, then retry.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
