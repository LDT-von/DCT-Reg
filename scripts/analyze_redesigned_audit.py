#!/usr/bin/env python3
"""Summarize redesigned audit metrics across folds.

Reads:
  results/audit_blca_redesign/blca/fold_{0..4}/redesigned_metrics_fold{0..4}.json

Writes:
  results/audit_blca_redesign/blca/summary_table.md
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics as stats


REPO = Path("/data1/DCT-Reg")
BASE = REPO / "results/audit_blca_redesign/blca"


def main() -> int:
    rows = []
    for fold in range(5):
        p = BASE / f"fold_{fold}" / f"redesigned_metrics_fold{fold}.json"
        if not p.exists():
            print(f"[fold {fold}] MISSING {p}")
            continue
        d = json.loads(p.read_text())
        rows.append(d)

    if not rows:
        print("No metrics found.")
        return 1

    n = len(rows)
    print(f"Found {n} folds.\n")

    # Print per-fold table
    headers = ["Fold", "real↑", "rand↑", "gap↑", "real↓", "rand↓", "gap↓",
               "real_d_abs", "rand_d_abs", "gap_d_abs"]
    print(" | ".join(f"{h:>10s}" for h in headers))
    print("-" * (13 * len(headers)))
    real_h, rand_h, info_h = [], [], []
    real_l, rand_l, info_l = [], [], []
    real_d_abs_h, rand_d_abs_h, info_d_abs_h = [], [], []
    for r in rows:
        f = r["fold"]
        rh = r["real_high_end_to_end"]
        ra = r["rand_high_end_to_end"]
        rl = r["real_low_end_to_end"]
        rb = r["rand_low_end_to_end"]
        ig_h = r["info_gap_high_rate"]
        ig_l = r["info_gap_low_rate"]
        ig_d_abs_h = r["info_gap_high_delta_abs"]
        ig_d_abs_l = r["info_gap_low_delta_abs"]
        real_h.append(rh["rate_up"])
        rand_h.append(ra["rate_up"])
        info_h.append(ig_h)
        real_l.append(rl["rate_down"])
        rand_l.append(rb["rate_down"])
        info_l.append(ig_l)
        real_d_abs_h.append(rh["delta_abs_mean"])
        rand_d_abs_h.append(ra["delta_abs_mean"])
        info_d_abs_h.append(ig_d_abs_h)
        cells = [str(f),
                 f"{rh['rate_up']:.4f}",
                 f"{ra['rate_up']:.4f}",
                 f"{ig_h:+.4f}",
                 f"{rl['rate_down']:.4f}",
                 f"{rb['rate_down']:.4f}",
                 f"{ig_l:+.4f}",
                 f"{rh['delta_abs_mean']:.6f}",
                 f"{ra['delta_abs_mean']:.6f}",
                 f"{ig_d_abs_h:+.6f}"]
        print(" | ".join(f"{c:>10s}" for c in cells))
    # Means row
    def m_or_nan(xs):
        return stats.mean(xs) if xs else float("nan")
    def s_or_nan(xs):
        return stats.stdev(xs) if len(xs) > 1 else 0.0
    cells = ["MEAN",
             f"{m_or_nan(real_h):.4f}",
             f"{m_or_nan(rand_h):.4f}",
             f"{m_or_nan(info_h):+.4f}",
             f"{m_or_nan(real_l):.4f}",
             f"{m_or_nan(rand_l):.4f}",
             f"{m_or_nan(info_l):+.4f}",
             f"{m_or_nan(real_d_abs_h):.6f}",
             f"{m_or_nan(rand_d_abs_h):.6f}",
             f"{m_or_nan(info_d_abs_h):+.6f}"]
    print(" | ".join(f"{c:>10s}" for c in cells))
    cells = ["STD",
             f"{s_or_nan(real_h):.4f}",
             f"{s_or_nan(rand_h):.4f}",
             f"{s_or_nan(info_h):.4f}",
             f"{s_or_nan(real_l):.4f}",
             f"{s_or_nan(rand_l):.4f}",
             f"{s_or_nan(info_l):.4f}",
             f"{s_or_nan(real_d_abs_h):.6f}",
             f"{s_or_nan(rand_d_abs_h):.6f}",
             f"{s_or_nan(info_d_abs_h):.6f}"]
    print(" | ".join(f"{c:>10s}" for c in cells))

    # Save summary json
    summary = {
        "n_folds": n,
        "real_high_rate_up_mean": m_or_nan(real_h),
        "rand_high_rate_up_mean": m_or_nan(rand_h),
        "info_gap_high_mean": m_or_nan(info_h),
        "info_gap_high_std": s_or_nan(info_h),
        "real_low_rate_down_mean": m_or_nan(real_l),
        "rand_low_rate_down_mean": m_or_nan(rand_l),
        "info_gap_low_mean": m_or_nan(info_l),
        "info_gap_low_std": s_or_nan(info_l),
        "real_high_delta_abs_mean": m_or_nan(real_d_abs_h),
        "rand_high_delta_abs_mean": m_or_nan(rand_d_abs_h),
        "info_gap_high_delta_abs_mean": m_or_nan(info_d_abs_h),
    }
    (BASE / "summary_5fold.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {BASE/'summary_5fold.json'}")

    # Markdown table
    md = []
    md.append("# Redesigned v3.10 BLCA Dose-Monotonicity Audit")
    md.append("")
    md.append("**Design**: 2 alpha points (0.0 factual vs 1.0 full anchor) + random baseline (slot-dim shuffled).")
    md.append("**Audit metric**: end-to-end directional rate + info_gap = real_rate - random_rate.")
    md.append("")
    md.append("## Per-Fold Results")
    md.append("")
    md.append("| Fold | real_high ↑ | rand_high ↑ | info_gap_high | real_low ↓ | rand_low ↓ | info_gap_low | real Δrisk_abs | rand Δrisk_abs | info_gap Δrisk_abs |")
    md.append("|------|-------------|-------------|---------------|------------|------------|--------------|----------------|----------------|--------------------|")
    for r in rows:
        rh = r["real_high_end_to_end"]; ra = r["rand_high_end_to_end"]
        rl = r["real_low_end_to_end"]; rb = r["rand_low_end_to_end"]
        md.append(f"| {r['fold']} | {rh['rate_up']:.4f} | {ra['rate_up']:.4f} | {r['info_gap_high_rate']:+.4f} | "
                  f"{rl['rate_down']:.4f} | {rb['rate_down']:.4f} | {r['info_gap_low_rate']:+.4f} | "
                  f"{rh['delta_abs_mean']:.6f} | {ra['delta_abs_mean']:.6f} | {r['info_gap_high_delta_abs']:+.6f} |")
    md.append(f"| **MEAN** | **{m_or_nan(real_h):.4f}** | **{m_or_nan(rand_h):.4f}** | **{m_or_nan(info_h):+.4f}** | "
              f"**{m_or_nan(real_l):.4f}** | **{m_or_nan(rand_l):.4f}** | **{m_or_nan(info_l):+.4f}** | "
              f"**{m_or_nan(real_d_abs_h):.6f}** | **{m_or_nan(rand_d_abs_h):.6f}** | **{m_or_nan(info_d_abs_h):+.6f}** |")
    md.append(f"| **STD** | {s_or_nan(real_h):.4f} | {s_or_nan(rand_h):.4f} | {s_or_nan(info_h):.4f} | "
              f"{s_or_nan(real_l):.4f} | {s_or_nan(rand_l):.4f} | {s_or_nan(info_l):.4f} | "
              f"{s_or_nan(real_d_abs_h):.6f} | {s_or_nan(rand_d_abs_h):.6f} | {s_or_nan(info_d_abs_h):.6f} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- **real_high ↑**: % of cases where risk[α=1] > risk[α=0] with HIGH anchor.")
    md.append("- **rand_high ↑**: same metric, but anchor slot-dim is randomly shuffled.")
    md.append("- **info_gap = real - rand**: signal beyond chance. >0 means anchor carries semantic info.")
    md.append("- **real_low ↓**: % of cases where risk[α=1] < risk[α=0] with LOW anchor.")
    md.append("")
    md.append(f"**Headline**: info_gap_high = {m_or_nan(info_h):+.4f}, info_gap_low = {m_or_nan(info_l):+.4f}")
    md.append("")
    if abs(m_or_nan(info_h)) < 0.05 and abs(m_or_nan(info_l)) < 0.05:
        md.append("**Verdict**: info_gap ≈ 0 → v3.10 cost-space intervention carries little signal; "
                  "anchor geometry needs to be redesigned (motivation for v3.11 per-slot hazard).")
    elif m_or_nan(info_h) > 0.05 or m_or_nan(info_l) > 0.05:
        md.append("**Verdict**: info_gap > 0 → v3.10 cost-space intervention carries detectable signal.")
    else:
        md.append("**Verdict**: mixed.")
    (BASE / "summary_table.md").write_text("\n".join(md))
    print(f"Wrote {BASE/'summary_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
