#!/usr/bin/env python3
"""Patch: Fix v3.11 slot_diversity_loss to constrain WSI and Omics SEPARATELY.

Issue: 原版代码合并计算 WSI+Omics 的 variance，omics 的 hazard 范围 [0,1] 比 WSI 大，
合并时被 omics "撑"起来，掩盖了 WSI 槽的坍缩。

Fix: 分别计算 WSI 和 Omics 的 variance，每个模态独立 hinge。

Usage:
    python scripts/fix_v311_diversity_per_modality.py
"""
from __future__ import annotations
from pathlib import Path

TARGET = Path("/data1/DCT-Reg/survot_rank/research/methods/legacy/experimental/dct_v311_slot_interpretable/model.py")

OLD_LOSS_BODY = '''        hazard_wsi = torch.sigmoid(self.per_slot_hazard_wsi(slots_wsi))   # [B, K_w, C]
        hazard_omic = torch.sigmoid(self.per_slot_hazard_omic(slots_omic))  # [B, K_o, C]

        # Concatenate all slot predictions per sample.
        all_preds = torch.cat([hazard_wsi, hazard_omic], dim=1)   # [B, K_w+K_o, C]

        # Mean hazard across slots per sample.
        mean_pred = all_preds.mean(dim=1, keepdim=True)             # [B, 1, C]

        # Per-sample variance across slots, averaged over hazard bins.
        variance = ((all_preds - mean_pred) ** 2).mean(dim=(1, 2))   # [B]

        # Per-sample diversity loss: two-sided hinge on variance.
        # Each sample's slots must have variance in [min, max].
        margin_min = float(getattr(self.args, "dct_v311_variance_min", self.VARIANCE_MIN))
        margin_max = float(getattr(self.args, "dct_v311_variance_max", self.VARIANCE_MAX))

        per_sample_loss = F.relu(margin_min - variance) + F.relu(variance - margin_max)  # [B]
        # Return mean over batch to get scalar loss.
        batch_loss = per_sample_loss.mean()
        self._last_slot_variance = variance.mean().item()
        self._last_slot_diversity = batch_loss.item()
        return batch_loss'''

NEW_LOSS_BODY = '''        hazard_wsi = torch.sigmoid(self.per_slot_hazard_wsi(slots_wsi))   # [B, K_w, C]
        hazard_omic = torch.sigmoid(self.per_slot_hazard_omic(slots_omic))  # [B, K_o, C]

        margin_min = float(getattr(self.args, "dct_v311_variance_min", self.VARIANCE_MIN))
        margin_max = float(getattr(self.args, "dct_v311_variance_max", self.VARIANCE_MAX))

        # PER-MODALITY diversity: each modality must independently satisfy the variance band.
        # This prevents "the healthy modality carries the unhealthy one" (bug in merged version).
        def _per_sample_variance(hazard):
            mean = hazard.mean(dim=1, keepdim=True)
            return ((hazard - mean) ** 2).mean(dim=(1, 2))  # [B]

        def _hinge(var):
            return F.relu(margin_min - var) + F.relu(var - margin_max)

        var_wsi = _per_sample_variance(hazard_wsi)
        var_omic = _per_sample_variance(hazard_omic)

        # Track metrics for monitoring
        self._last_slot_variance_wsi = var_wsi.mean().item()
        self._last_slot_variance_omic = var_omic.mean().item()
        # Backward-compat (combined metric for monitoring only)
        self._last_slot_variance = 0.5 * (var_wsi.mean().item() + var_omic.mean().item())

        # Each modality contributes equally to the diversity loss.
        loss_wsi = _hinge(var_wsi).mean()
        loss_omic = _hinge(var_omic).mean()
        batch_loss = 0.5 * (loss_wsi + loss_omic)

        self._last_slot_diversity = batch_loss.item()
        return batch_loss'''


def main():
    text = TARGET.read_text()
    if "_per_sample_variance(hazard_wsi)" in text:
        print(f"Already patched: {TARGET}")
        return 0
    if OLD_LOSS_BODY not in text:
        print(f"ERROR: could not find expected loss body in {TARGET}")
        return 1
    new_text = text.replace(OLD_LOSS_BODY, NEW_LOSS_BODY)
    TARGET.write_text(new_text)
    print(f"✅ Patched: {TARGET}")
    print("   - WSI and Omics slots now constrained independently")
    print("   - Combined metric kept for monitoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
