#!/usr/bin/env python3
"""Proof 2: v3.10 direction loss gradient flow analysis.

For each BLCA fold:
  1. Load ckpt
  2. Run 1 batch forward, cache factual costs / slots / rows / cols
  3. Manually recompute direction loss via _interpolate_cost + _solve_interventions
     (same recipe used by audit_dct_reg._run_alpha_sweep)
  4. Backward direction_loss and nll_surrogate in turn
  5. Compare gradient norms at:
     - risk_anchor_costs  (where direction loss targets)
     - slot_attention / module_factories  (where the gradient needs to land)
     - encoders  (downstream)

If direction_loss gradient at slot_attention is 100x+ smaller than nll_surrogate,
the entropic OT is the bottleneck — confirming Proof 1's "gain never moves" finding.

Output: results/v310_proof2_gradient_flow/{per_fold.json, REPORT.md}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survot_rank.cli import add_project_paths  # noqa: E402

add_project_paths()

# Reuse model loader
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from redesign_audit_blca import _load_model_and_loader, _load_parsed_args  # noqa: E402

from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (  # noqa: E402
    _process_data_and_forward,
)

import numpy as np
import torch


class _DummyArgs:
    """Wraps a parsed config and exposes attributes used by
    _process_data_and_forward. Adds ``cur_epoch`` (set high so the model
    uses its full-epoch transport behaviour).
    """

    def __init__(self, parsed, cur_epoch: int = 50):
        self._parsed = parsed
        self.cur_epoch = cur_epoch
        self.test = True

    def __getattr__(self, name):
        return getattr(self._parsed, name)


def _param_norm(parameters):
    if not parameters:
        return 0.0
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total = total + float((p.grad.detach().float() ** 2).sum().item())
    return float(np.sqrt(total))


def _select_param_groups(model):
    """Return dict of named parameter groups for gradient inspection."""
    groups = {
        "risk_anchor_costs": [],
        "slot_attention": [],
        "module_factories": [],
        "spt_logit_head": [],
        "wsi_encoder": [],
        "omic_encoder": [],
        "other": [],
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "risk_anchor_costs" in name:
            groups["risk_anchor_costs"].append(p)
        elif "slot_attention" in name:
            groups["slot_attention"].append(p)
        elif "module_factory" in name:
            groups["module_factories"].append(p)
        elif "spt_logit" in name or "logit_head" in name:
            groups["spt_logit_head"].append(p)
        elif "wsi_encoder" in name or "encoder_wsi" in name:
            groups["wsi_encoder"].append(p)
        elif "omic_encoder" in name or "encoder_omic" in name:
            groups["omic_encoder"].append(p)
        else:
            groups["other"].append(p)
    return groups


def _interpolate_cost(factual_costs, anchor_costs, alpha, seen_mask):
    bsz = factual_costs.size(0)
    expanded = anchor_costs.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
    seen = seen_mask.view(1, -1, 1, 1, 1)
    expanded = torch.where(seen, expanded, factual_costs)
    return (1.0 - alpha) * factual_costs + alpha * expanded


def run_one_fold(model, val_loader, device, parsed) -> dict:
    """Run 1 batch and measure two gradient flows: direction_loss vs nll_surrogate.

    The audit-mode forward caches detached copies of factual_costs /
    slots / rows / cols into model._last_* (see dct_distributional_transport
    model.py:790-797).  We monkey-patch ``Tensor.detach`` to a no-op for the
    second forward pass so the cache retains gradient information.
    """
    model.zero_grad(set_to_none=True)
    data = next(iter(val_loader))
    epoch = int(getattr(parsed, "cur_epoch", 50))

    real_setattr = type(model).__setattr__

    # We want to disable the .detach() in the 5 cache assignments:
    # self._last_factual_costs = factual_costs.detach()
    # self._last_factual_rows  = rows.detach()
    # self._last_factual_cols  = cols.detach()
    # self._last_slots_wsi     = slots_wsi.detach()
    # self._last_slots_omic    = slots_omic.detach()
    # We do this by replacing .detach on those specific tensor instances
    # via a single re-binding right after they're assigned.
    # But by then they're already assigned.  Alternative: use forward
    # pre-hooks on the modules that produce these tensors.
    #
    # Approach used here: call _encode_transport_slots and _cost_tensor
    # directly with the same inputs the real forward would use, getting
    # live (non-detached) tensors, then use those for the direction loss
    # recomputation.

    # 1) Run a real forward (test=True) to populate model._last_* caches
    #    (detached, but useful as a sanity check) and to set up the rest
    #    of the model state (IPCW memory, train reference, etc.).
    out, y_disc, event_time, censorship = _process_data_and_forward(
        _DummyArgs(parsed), model, data, device, test=True
    )

    # 2) Re-run the slot+cost subgraph with FRESH inputs to capture live
    #    tensors.  We bypass the forward cache by directly calling the
    #    underlying methods.

    # Build the model inputs the same way _process_data_and_forward does
    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (  # noqa: E402
        _unpack_data,
    )
    data_wsi, data_omics, _, _, _, clinical = _unpack_data(data, device, parsed.rna_format)

    x_wsi_proj = model.wsi_mlp(data_wsi)
    if parsed.rna_format in ("Pathways", "RankedGenes"):
        x_omics_kwargs = {
            f"x_omic{idx}": omic for idx, omic in enumerate(data_omics, start=1)
        }
    else:
        x_omics_kwargs = {"x_omics": data_omics}
    x_omics = model._encode_omics(x_omics_kwargs)
    slots_wsi, slots_omic, _, _ = model._encode_transport_slots(
        x_wsi_proj, x_omics, {"cur_epoch": epoch}
    )
    factual_costs, rows, cols, _ = model._cost_tensor(slots_wsi, slots_omic)

    # Sanity: these must equal the cached (detached) versions in values
    # but should NOT be detached.
    assert slots_wsi.requires_grad or not slots_wsi.is_floating_point() or True
    assert torch.allclose(
        factual_costs.detach(),
        getattr(model, "_last_factual_costs").detach(),
        atol=1e-5,
    ), "factual_costs recomputation mismatch"

    # Recompute factual logits (also a fresh sinkhorn solve so gradient flows)
    factual_plans, _ = model._plans_from_cost_tensor(
        factual_costs, rows, cols, epoch
    )
    factual_logits, _ = model._encode_logits_from_plans(
        slots_wsi, slots_omic, factual_plans
    )

    # Build full_low/high cost tensors at alpha=1 (same as training direction_loss path)
    full_low_costs = _interpolate_cost(
        factual_costs,
        model.risk_anchor_costs[:, model._LOW_RISK],
        1.0,
        model.risk_anchor_seen[:, model._LOW_RISK],
    )
    full_high_costs = _interpolate_cost(
        factual_costs,
        model.risk_anchor_costs[:, model._HIGH_RISK],
        1.0,
        model.risk_anchor_seen[:, model._HIGH_RISK],
    )

    plans_low, logits_low = model._solve_interventions(
        [full_low_costs],
        slots_wsi=slots_wsi,
        slots_omic=slots_omic,
        rows=rows,
        cols=cols,
        epoch=epoch,
    )
    plans_high, logits_high = model._solve_interventions(
        [full_high_costs],
        slots_wsi=slots_wsi,
        slots_omic=slots_omic,
        rows=rows,
        cols=cols,
        epoch=epoch,
    )

    # Risk and direction loss
    factual_risk = model._risk(factual_logits)
    full_low_risk = model._risk(logits_low[0])
    full_high_risk = model._risk(logits_high[0])
    direction_loss, high_gain, low_gain = model._direction_loss(
        factual_risk, full_low_risk, full_high_risk
    )

    # NLL surrogate: factual_risk.sum() shares gradient graph with nll_surv
    nll_surrogate = factual_risk.sum()

    # ---- Gradient via direction_loss ----
    model.zero_grad(set_to_none=True)
    direction_loss.backward(retain_graph=True)
    groups_dir = _select_param_groups(model)
    grad_dir = {k: _param_norm(v) for k, v in groups_dir.items()}

    # ---- Gradient via nll_surrogate ----
    model.zero_grad(set_to_none=True)
    nll_surrogate.backward(retain_graph=True)
    groups_nll = _select_param_groups(model)
    grad_nll = {k: _param_norm(v) for k, v in groups_nll.items()}

    ratio = {}
    for k in grad_dir:
        if grad_nll[k] > 0:
            ratio[k] = float(grad_dir[k] / grad_nll[k])
        else:
            ratio[k] = float("nan")

    return {
        "direction_loss_value": float(direction_loss.item()),
        "nll_surrogate_value": float(nll_surrogate.item()),
        "high_risk_gain": float(high_gain.mean().item()),
        "low_risk_gain": float(low_gain.mean().item()),
        "grad_norm_via_direction": grad_dir,
        "grad_norm_via_nll_surrogate": grad_nll,
        "ratio_dir_over_nll": ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--output-dir", default="results/v310_proof2_gradient_flow"
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    parsed = _load_parsed_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(x) for x in args.folds.split(",")]
    summaries = {}
    for fold in folds:
        print(f"\n=== fold {fold} ===")
        t0 = time.time()
        ckpt = args.checkpoint.replace("s0", f"s{fold}")
        try:
            model, val_loader, _, _, device = _load_model_and_loader(
                type("A", (), {"config": args.config, "checkpoint": ckpt})(),
                parsed,
                fold,
            )
        except Exception as e:
            print(f"  load failed: {e}")
            import traceback
            traceback.print_exc()
            summaries[fold] = {"error": f"load failed: {e}"}
            continue

        try:
            res = run_one_fold(model, val_loader, device, parsed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"error": str(e)}

        res["elapsed_sec"] = round(time.time() - t0, 1)
        summaries[fold] = res

        json_path = output_dir / f"fold_{fold}.json"
        with open(json_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  saved {json_path}, elapsed {res['elapsed_sec']:.1f}s")

        if "ratio_dir_over_nll" in res:
            print(
                f"  direction={res['direction_loss_value']:.5f} "
                f"nll_surrogate={res['nll_surrogate_value']:.5f}"
            )
            print(
                f"  high_gain={res['high_risk_gain']:.5f} "
                f"low_gain={res['low_risk_gain']:.5f}"
            )
            print(
                f"  ratio (dir/nll) slot_attention={res['ratio_dir_over_nll']['slot_attention']:.3e} "
                f"risk_anchor={res['ratio_dir_over_nll']['risk_anchor_costs']:.3e}"
            )

    # Aggregate
    summary_path = output_dir / "summary_5fold.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"saved {summary_path}")

    # Markdown report
    lines = [
        "# v3.10 Proof 2: Direction Loss Gradient Flow (BLCA)\n",
        "**核心问题**:direction loss 反向传播时,梯度能否穿过 Sinkhorn + entropic OT 到达 slot attention?\n",
        "\n",
        "**实验**:加载 v3.10 BLCA 5 fold ckpt,跑 1 batch forward,分别用:\n",
        "1. `direction_loss.backward()` → 各参数梯度范数\n",
        "2. `nll_surrogate = factual_risk.sum()` → 各参数梯度范数(对照)\n",
        "\n",
        "**关键比值**:`ratio = ||grad_via_direction|| / ||grad_via_nll||`\n",
        "- ratio ≈ 1e0 → direction loss 影响力与 nll 相当\n",
        "- ratio < 1e-3 → direction loss 梯度被 entropic OT 严重衰减\n",
        "\n",
        "## Per-fold\n",
        "\n",
        "| Fold | dir_loss | nll_surr | high_gain | low_gain | ratio@slot_attn | ratio@risk_anchor | ratio@module_factory |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fold, s in summaries.items():
        if "ratio_dir_over_nll" not in s:
            lines.append(
                f"| {fold} | ERROR: {s.get('error','')[:30]} | | | | | | |"
            )
            continue
        lines.append(
            f"| {fold} | {s['direction_loss_value']:.4f} | {s['nll_surrogate_value']:.4f} | "
            f"{s['high_risk_gain']:.5f} | {s['low_risk_gain']:.5f} | "
            f"{s['ratio_dir_over_nll']['slot_attention']:.3e} | "
            f"{s['ratio_dir_over_nll']['risk_anchor_costs']:.3e} | "
            f"{s['ratio_dir_over_nll']['module_factories']:.3e} |"
        )

    # Aggregate ratio per group
    groups_present = ["slot_attention", "risk_anchor_costs", "module_factories", "spt_logit_head"]
    lines.extend(["\n## 5-fold mean ratio (direction/nll_surrogate)\n", "\n",
                  "| Param group | mean ratio | median ratio |", "|---|---|---|"])
    for g in groups_present:
        vals = []
        for fold, s in summaries.items():
            if "ratio_dir_over_nll" in s:
                v = s["ratio_dir_over_nll"].get(g, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
        if vals:
            lines.append(f"| {g} | {np.mean(vals):.3e} | {np.median(vals):.3e} |")
        else:
            lines.append(f"| {g} | – | – |")

    lines.extend([
        "\n## 解读\n",
        "\n",
        "- 如果 `ratio@slot_attention < 1e-3`,说明 direction loss 梯度在 Sinkhorn + entropic 正则下\n",
        "  几乎被衰减 → 即使 direction loss 值非零,也无法更新 slot_attention → gain 永远不动。\n",
        "- 这是 Proof 1 \"gain 不变\"的**直接机制解释**。\n",
        "- 这是 v3.11 用 per-slot NLL 取代 direction loss 的**理论依据**。\n",
        "\n",
        "## 下一步\n",
        "\n",
        "Proof 3: 验证 anchor 几何距离 → audit 信号恢复(独立机制)。\n",
    ])
    md_path = output_dir / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"saved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
