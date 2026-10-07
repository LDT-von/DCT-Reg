#!/usr/bin/env python3
"""Exp C-audit: Audit direction_only ckpt (smoke trained, 1 fold) using redesigned audit.

Compares info_gap_high of:
  - direction_only (smoke, 2 epoch, no IPCW, 1 fold)
  - full v3.10 (50 epoch, +IPCW+direction, 5 fold)

This is a smoke test to verify whether direction_only alone produces
any monotonicity signal.  Even if the ckpt is under-trained (val cindex
~0.54), the audit signature will reveal whether direction loss pushes
risk correctly when given enough training signal.

Output: results/v310_proof_c_audit/{smoke_fold0.json, REPORT.md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from survot_rank.cli import add_project_paths
add_project_paths()

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from redesign_audit_blca import (
    _load_model_and_loader,
    _load_parsed_args,
    run_fold,
    compute_end_to_end,
)

CKPT_PATH = Path(
    "results/dct_v3.10_experiments_smoke/direction_only/blca/blca/"
    "SurvOTRank_dct_transport_intervention_consistency/"
    "0.0005_b8_survival_months_dss_Dim_256_e_2_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_smoke/"
    "model_best_s0.pth"
)
OUTPUT = Path("results/v310_proof_c_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    fold = 0
    run_args = argparse.Namespace(
        config=str(REPO_ROOT / "configs/dct_v310_directional_regularized_transport.yaml"),
        checkpoint=str(CKPT_PATH),
        fold=fold,
        epoch=2,
        output_dir=str(OUTPUT),
        set=["study=blca"],
    )

    if not CKPT_PATH.exists():
        print(f"Missing ckpt: {CKPT_PATH}")
        return 1

    print(f"\n=== Auditing direction_only smoke fold 0 ===")
    res = run_fold(fold, run_args, alpha_list=(0.0, 1.0))
    metrics = {}
    metrics["real_high"] = compute_end_to_end(res["real_high"], (0.0, 1.0))
    metrics["rand_high"] = compute_end_to_end(res["rand_high"], (0.0, 1.0))
    metrics["real_low"] = compute_end_to_end(res["real_low"], (0.0, 1.0))
    metrics["rand_low"] = compute_end_to_end(res["rand_low"], (0.0, 1.0))
    res["metrics"] = {
        "real_high_rate_up": metrics["real_high"]["rate_up"],
        "rand_high_rate_up": metrics["rand_high"]["rate_up"],
        "info_gap_high": metrics["real_high"]["rate_up"] - metrics["rand_high"]["rate_up"],
        "real_low_rate_down": metrics["real_low"]["rate_down"],
        "rand_low_rate_down": metrics["rand_low"]["rate_down"],
        "info_gap_low": metrics["real_low"]["rate_down"] - metrics["rand_low"]["rate_down"],
    }
    out_path = OUTPUT / "smoke_fold0.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {out_path}")

    # Compare to v3.10 full BLCA 5-fold mean (5-fold) + fold 0 only
    blca_summary_path = Path("results/audit_blca_redesign/blca/summary_5fold.json")
    blca_summary = {}
    if blca_summary_path.exists():
        blca_summary = json.load(open(blca_summary_path))

    blca_fold0_path = Path("results/audit_blca_redesign/blca/fold_0.json")
    blca_fold0 = {}
    if blca_fold0_path.exists():
        d = json.load(open(blca_fold0_path))
        if "metrics" in d:
            blca_fold0 = d["metrics"]

    # Markdown
    lines = [
        "# v3.10 Proof C-audit: Direction-Only smoke checkpoint audit (1 fold)\n",
        "**核心问题**:direction_only 训练 (no IPCW, 2 epoch smoke) 是否有 direction effect?\n",
        "\n",
        f"**ckpt** = `{CKPT_PATH}` (smoke trained, 2 batch/epoch, 2 epoch total, val cindex=0.54)\n",
        "\n",
        "## info_gap_high 对比\n",
        "\n",
        "| setup | info_gap_high | info_gap_low | real_high_rate | rand_high_rate |",
        "|---|---|---|---|---|",
    ]
    my = res["metrics"]
    lines.append(
        f"| direction_only smoke (this) | {my['info_gap_high']:+.3f} | {my['info_gap_low']:+.3f} | "
        f"{my['real_high_rate_up']:.3f} | {my['rand_high_rate_up']:.3f} |"
    )
    if blca_fold0:
        # look for fold0 metrics in different format
        for k in ["info_gap_high", "info_gap_low", "real_high_rate_up", "rand_high_rate_up"]:
            pass
    if blca_summary:
        lines.append(
            f"| v3.10 full BLCA 5-fold mean | {blca_summary.get('info_gap_high_mean', 'NA'):+.3f} | "
            f"{blca_summary.get('info_gap_low_mean', 'NA'):+.3f} | "
            f"{blca_summary.get('real_high_rate_up_mean', 'NA'):.3f} | "
            f"{blca_summary.get('rand_high_rate_up_mean', 'NA'):.3f} |"
        )

    lines.extend([
        "\n## 解读\n",
        "\n",
        "- **direction_only smoke 训了 2 batch/epoch, 2 epoch 总共** = ckpt 很差(epoch 0/1 cindex=0.54/0.44)。",
        "- 若 **info_gap_high** 在 direction_only 上比 full v3.10 BLCA 大 → 说明 **direction-only 在 short training 内** 也能 push risk,问题在 full v3.10 的 +IPCW 反而抑制 direction 的效果。",
        "- 若 info_gap_high ≈ 0 → direction loss 即使单独训也对 risk 没推动。",
        "",
        "**注意**:这只是一个 fold × smoke 检查,不是决定性结论。要做严谨比较需重训 50 epoch × 5 fold。",
    ])

    with open(OUTPUT / "REPORT.md", "w") as f:
        f.write("\n".join(lines))
    print(f"saved {OUTPUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
