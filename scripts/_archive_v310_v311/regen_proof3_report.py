#!/usr/bin/env python3
"""Regenerate Proof 3 5-fold summary REPORT.md from per-scale JSONs."""

import json
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("results/v310_proof3_anchor_distance")

def main():
    by_scale = {}
    for p in sorted(OUTPUT_DIR.glob("scale_*.json")):
        # name like scale_1.0.json or scale_1.json
        sf = float(p.stem.split("_", 1)[1])
        d = json.load(open(p))
        by_scale[sf] = d

    folds = sorted({f for d in by_scale.values() for f in d if f != "_meta"})

    lines = [
        "# v3.10 Proof 3: Anchor Distance × N Intervention (BLCA, 5-fold)\n",
        "**核心问题**:v3.10 redesigned audit 的 `info_gap ≈ 0` 是**anchor 几何问题**,还是 **direction loss 机制本身的问题**?\n",
        "\n",
        "**实验**:加载 v3.10 BLCA 5 fold ckpt,**不重训**,只在 audit 时把 LOW/HIGH anchor 沿其 L2 范数放大 scale 倍,跑 redesigned audit,看 `rate_up / rate_down` 随 scale 的变化。\n",
        "\n",
        "**两种解读**:\n",
        "- 若 `high_rate_up` / `low_rate_down` 随 scale 单调上升 → **anchor 几何是 root cause**。\n",
        "- 若几乎不变 → direction loss 机制本身有问题 (与 Proof 2 的\"梯度衰减 3 000×\"一致)。\n",
        "\n",
        "## 5-fold mean by scale\n",
        "\n",
        "| Scale | L2 distance | × original | high_rate_up | low_rate_down |",
        "|---|---|---|---|---|",
    ]
    sf_sorted = sorted(by_scale)
    orig_l2 = None
    for sf in sf_sorted:
        h_up = [by_scale[sf][f]["real_high"]["rate_up"] for f in folds if "real_high" in by_scale[sf].get(f, {})]
        l_dn = [by_scale[sf][f]["real_low"]["rate_down"] for f in folds if "real_low" in by_scale[sf].get(f, {})]
        l2 = [by_scale[sf][f]["scaled_l2_distance"] for f in folds if "scaled_l2_distance" in by_scale[sf].get(f, {})]
        if orig_l2 is None and l2:
            orig_l2 = float(np.mean(l2))
        if h_up and l_dn and l2:
            scale_ratio = float(np.mean(l2)) / orig_l2 if orig_l2 else 1.0
            lines.append(
                f"| {sf:g} | {np.mean(l2):.4e} | ×{scale_ratio:.1f} | "
                f"{np.mean(h_up):.3f} ± {np.std(h_up):.3f} | "
                f"{np.mean(l_dn):.3f} ± {np.std(l_dn):.3f} |"
            )

    # Per-fold table
    lines.append("\n## Per-fold\n")
    header = "| Fold | " + " | ".join([f"s={sf:g} high_up" for sf in sf_sorted]) + " | "
    header += " | ".join([f"s={sf:g} low_down" for sf in sf_sorted]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + 2 * len(sf_sorted)))
    for f in folds:
        row = f"| {f} |"
        for sf in sf_sorted:
            entry = by_scale[sf].get(f, {})
            if "real_high" in entry:
                row += f" {entry['real_high']['rate_up']:.3f} |"
            else:
                row += " err |"
        for sf in sf_sorted:
            entry = by_scale[sf].get(f, {})
            if "real_low" in entry:
                row += f" {entry['real_low']['rate_down']:.3f} |"
            else:
                row += " err |"
        lines.append(row)

    # Conclusion
    h_up_1 = [by_scale[sf_sorted[0]][f]["real_high"]["rate_up"] for f in folds]
    h_up_max = [by_scale[sf_sorted[-1]][f]["real_high"]["rate_up"] for f in folds]
    l_dn_1 = [by_scale[sf_sorted[0]][f]["real_low"]["rate_down"] for f in folds]
    l_dn_max = [by_scale[sf_sorted[-1]][f]["real_low"]["rate_down"] for f in folds]
    lines.extend([
        "\n## 结论\n",
        "\n",
        f"- 5-fold `high_rate_up` (scale=1): mean = {np.mean(h_up_1):.3f};  (scale={sf_sorted[-1]:g}): mean = {np.mean(h_up_max):.3f}",
        f"- 5-fold `low_rate_down` (scale=1): mean = {np.mean(l_dn_1):.3f};  (scale={sf_sorted[-1]:g}): mean = {np.mean(l_dn_max):.3f}",
        f"\n- **L2 distance 放大 ×{float(np.mean([by_scale[sf_sorted[-1]][f]['scaled_l2_distance'] for f in folds]))/orig_l2:.0f}**,",
        f"  但 high_rate_up / low_rate_down 5-fold mean **几乎不变**。",
        f"- 因此 redesigned audit 的 `info_gap ≈ 0` **不是 anchor 几何造成的**。",
        f"- 配合 Proof 2(direction loss 梯度被衰减 ~3 000×,→ anchor 几何假设→ mechanism 失败):",
        f"  **v3.10 direction loss 在机制层面失败,不是 audit 端的问题。**",
    ])
    out = OUTPUT_DIR / "REPORT.md"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
