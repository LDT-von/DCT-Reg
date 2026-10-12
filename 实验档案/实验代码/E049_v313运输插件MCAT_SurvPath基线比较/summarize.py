"""Summarize E049 transport-plugin experiment results.

Reads completion.json from every task and produces:
  1. Raw score table (all arms, all folds, all seeds)
  2. Mean ± std per arm per baseline
  3. Paired difference table (learned - independent, learned_three_decoders - independent_three_decoders, etc.)
  4. C-statistic significance via Wilcoxon signed-rank (paired folds, same seed)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Arms and comparison groups
ALL_ARMS = (
    "native", "independent", "learned",
    "independent_three_decoders", "learned_three_decoders", "learned_shared_decoders",
)

COMPARISONS = [
    # (label, arm_a, arm_b, question)
    ("native_vs_learned",           "native",                    "learned",                   "Whole plugin effect: Learned OT vs native (no reconstruction)"),
    ("native_vs_learned_three",     "native",                    "learned_three_decoders",     "Whole plugin effect: Full plugin vs native"),
    ("ind_vs_learned",              "independent",               "learned",                   "Transport coupling: Learned OT vs Independent (same capacity, same marginals)"),
    ("ind3_vs_learned3",           "independent_three_decoders", "learned_three_decoders",    "Transport coupling with reconstruction: Learned OT vs Independent"),
    ("learned_vs_learned3",         "learned",                   "learned_three_decoders",     "Auxiliary reconstruction: full vs no reconstruction"),
    ("ind3_vs_ind",                "independent",               "independent_three_decoders",  "Auxiliary reconstruction (Independent): with vs without three decoders"),
    ("shared3_vs_sep3",             "learned_three_decoders",    "learned_shared_decoders",     "Decoder sharing: separate vs shared (self/cross)"),
]

STATISTICAL_H0_MESSAGE = (
    "Statistical comparisons use two-sided Wilcoxon signed-rank test on paired "
    "C-indices (same fold, same seed, same baseline).  "
    "Results are exploratory: N ≤ 10 folds per comparison; "
    "report actual p-values and sample sizes alongside any claim."
)


def read_batch(batch_dir: Path):
    """Yield (baseline, arm, fold, seed, protocol, score) from all completion.json files."""
    completions = list(batch_dir.rglob("completion.json"))
    if not completions:
        raise FileNotFoundError(f"No completion.json found under {batch_dir}")
    rows = []
    completed_ids = set()
    for path in completions:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
            continue
        if data.get("status") != "complete":
            print(f"WARNING: task {path.parent.name} not complete: {data.get('error', '?')}",
                  file=sys.stderr)
            continue
        # Parse task_id: {protocol}/{baseline}/{arm}/fold{fold}/seed{seed}
        task_id = data.get("task_id", "")
        if task_id in completed_ids:
            raise ValueError(f"Duplicate completed task_id: {task_id}")
        completed_ids.add(task_id)
        parts = task_id.split("/")
        if len(parts) < 4:
            parts = [data.get("protocol", ""), data.get("baseline", ""),
                     data.get("arm", ""), str(data.get("fold", ""))]
        protocol = data.get("protocol") or parts[0]
        study = data.get("study") or (parts[1] if len(parts) == 6 else "unknown")
        baseline = data.get("baseline") or parts[-4]
        arm = data.get("arm") or parts[-3]
        fold = int(data.get("fold", parts[-2].replace("fold", "")))
        seed = int(data.get("seed", parts[-1].replace("seed", "")))
        score = float(data.get("score", float("nan")))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid C-index in {path}: {score}")
        rows.append(dict(study=study, baseline=baseline, arm=arm, fold=fold, seed=seed,
                         protocol=protocol, score=score))
    plan_path = batch_dir / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        planned_ids = {task["task_id"] for task in plan["tasks"]}
        missing = planned_ids - completed_ids
        unexpected = completed_ids - planned_ids
        if missing or unexpected:
            raise ValueError(f"Batch is incomplete or mismatched: missing={len(missing)}, unexpected={len(unexpected)}")
    return rows


def group_by(data, *keys):
    result = defaultdict(list)
    for row in data:
        result[tuple(row[k] for k in keys)].append(row["score"])
    return result


def mean_std(scores):
    import statistics
    if not scores:
        return float("nan"), float("nan")
    return statistics.mean(scores), statistics.stdev(scores) if len(scores) > 1 else 0.0


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank test, paired samples."""
    if len(a) != len(b) or len(a) < 5:
        return float("nan"), None
    diffs = [ai - bi for ai, bi in zip(a, b)]
    if not any(d != 0 for d in diffs):
        return 1.0, len(diffs)
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
        return float(res.pvalue), len(diffs)
    except ImportError:
        return float("nan"), len(diffs)


def fmt_row(baseline, arm, scores, indent="  "):
    mu, sd = mean_std(scores)
    return f"{indent}{baseline:<12}{arm:<35} {mu:.4f} ± {sd:.4f}  (n={len(scores)})"


def fmt_comparison_table(comparisons, rows, baselines):
    lines = ["", "=" * 90, "PAIRED COMPARISONS (Wilcoxon signed-rank, same seed, same folds)", "=" * 90]
    lines.append(STATISTICAL_H0_MESSAGE)
    lines.append("")
    header = f"{'Comparison':<30} {'Baseline':<12} {'N folds':<8} {'Delta B-A':>12} {'p-value':>12} {'Sig':<8} {'Question'}"
    lines.append(header)
    lines.append("-" * 100)
    indexed = {}
    for row in rows:
        key = (row["study"], row["protocol"], row["baseline"],
               row["arm"], row["fold"], row["seed"])
        if key in indexed:
            raise ValueError(f"Duplicate completion for {key}")
        indexed[key] = row["score"]
    strata = sorted({(row["study"], row["protocol"], row["baseline"]) for row in rows})
    for label, arm_a, arm_b, question in comparisons:
        for study, protocol, baseline in strata:
            left = {(fold, seed): score for (s, p, b, arm, fold, seed), score in indexed.items()
                    if (s, p, b, arm) == (study, protocol, baseline, arm_a)}
            right = {(fold, seed): score for (s, p, b, arm, fold, seed), score in indexed.items()
                     if (s, p, b, arm) == (study, protocol, baseline, arm_b)}
            if not left or not right:
                continue
            if set(left) != set(right):
                lines.append(f"INCOMPLETE {study}/{protocol}/{baseline}/{label}: unmatched fold/seed")
                continue
            pairs = sorted(left)
            scores_a = [left[pair] for pair in pairs]
            scores_b = [right[pair] for pair in pairs]
            p, n = wilcoxon_paired(scores_a, scores_b)
            delta = mean_std([bi - ai for ai, bi in zip(scores_a, scores_b)])[0]
            sig = ("*" * sum(1 for thr in [0.05, 0.01, 0.001] if p < thr)) if p == p and p < 0.05 else ("ns" if p == p else "?")
            lines.append(f"{label:<30} {baseline:<12} {n:<8} {delta:>+12.4f} {p:>12.4f} {sig:<8} {study}/{protocol}: {question}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="root of the experiment batch")
    parser.add_argument("--output", type=Path, help="write summary to file (default: stdout)")
    parser.add_argument("--protocol", help="filter by protocol (default: all)")
    parser.add_argument("--seed", type=int, help="filter by seed (default: all)")
    parser.add_argument("--baseline", help="filter by baseline (default: all)")
    parser.add_argument("--no-statistics", action="store_true", help="skip statistical tests")
    args = parser.parse_args(argv)

    rows = read_batch(args.batch_dir)

    # Apply filters
    if args.protocol:
        rows = [r for r in rows if r["protocol"] == args.protocol]
    if args.seed is not None:
        rows = [r for r in rows if r["seed"] == args.seed]
    if args.baseline:
        rows = [r for r in rows if r["baseline"] == args.baseline]

    if not rows:
        raise ValueError("No rows after filtering")

    baselines = sorted(set(r["baseline"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
    folds = sorted(set(r["fold"] for r in rows))
    protocols = sorted(set(r["protocol"] for r in rows))
    arms = sorted(set(r["arm"] for r in rows))

    rows.sort(key=lambda r: (r["study"], r["protocol"], r["baseline"],
                             r["arm"], r["seed"], r["fold"]))
    grouped = group_by(rows, "study", "protocol", "baseline", "arm")
    strata = sorted({(r["study"], r["protocol"], r["baseline"]) for r in rows})

    # ---- Build summary ----
    sections = []

    # Section 1: Raw scores
    sections.append("\n".join([
        "=" * 90,
        "E049 TRANSPORT PLUGIN EXPERIMENT RESULTS",
        "=" * 90,
        f"Batch: {args.batch_dir}",
        f"Protocols: {protocols}   Seeds: {seeds}   Folds: {folds}",
        f"Baselines: {baselines}",
        "",
        "RAW SCORES (individual fold C-indices)",
        "-" * 90,
    ]))
    for study, protocol, baseline in strata:
        sections.append(f"\n--- {study}/{protocol}/{baseline} ---")
        for arm in ALL_ARMS:
            key = (study, protocol, baseline, arm)
            scores = grouped.get(key, [])
            if scores:
                fold_str = ", ".join(f"{s:.4f}" for s in scores)
                mu, sd = mean_std(scores)
                sections.append(f"  {arm:<35} {mu:.4f} ± {sd:.4f}  [{fold_str}]")

    # Section 2: Summary table
    sections.append("\n" + "=" * 90)
    sections.append("SUMMARY: mean ± sample std, n = folds × seeds")
    sections.append("=" * 90)
    sections.append(f"{'Baseline':<12} {'Arm':<35} {'C-index':>12}  {'n':<6}")
    sections.append("-" * 70)
    for study, protocol, baseline in strata:
        for arm in ALL_ARMS:
            key = (study, protocol, baseline, arm)
            scores = grouped.get(key, [])
            if scores:
                mu, sd = mean_std(scores)
                sections.append(f"{study}/{protocol}/{baseline:<12} {arm:<35} {mu:>8.4f} ± {sd:.4f}  {len(scores):<6}")

    # Section 3: Comparison table
    if not args.no_statistics:
        sections.append(fmt_comparison_table(COMPARISONS, rows, baselines))

    # Section 4: Key findings template
    sections.append("\n" + "=" * 90)
    sections.append("KEY FINDINGS (fill in after reviewing results above)")
    sections.append("=" * 90)
    sections.append(textwrap.dedent("""\
        1. Whole-plugin effect (native vs learned_three_decoders):
           MCAT  Δ = ? (? / ? folds improved, p = ?)
           SurvPath Δ = ? (? / ? folds improved, p = ?)

        2. Transport coupling effect (independent vs learned):
           MCAT  Δ = ? (? / ? folds improved, p = ?)
           SurvPath Δ = ? (? / ? folds improved, p = ?)

        3. Reconstruction effect (learned vs learned_three_decoders):
           MCAT  Δ = ? (? / ? folds improved, p = ?)
           SurvPath Δ = ? (? / ? folds improved, p = ?)

        4. Decoder sharing (learned_three_decoders vs learned_shared_decoders):
           MCAT  Δ = ? (? / ? folds improved, p = ?)
           SurvPath Δ = ? (? / ? folds improved, p = ?)

        Conclusion:
        - Learned OT outperforms Independent (transport coupling):
        - Full plugin outperforms native baseline:
        - Auxiliary reconstruction contributes:
        - Decoder sharing effect:
    """))

    text = "\n".join(sections)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Summary written to {args.output}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
