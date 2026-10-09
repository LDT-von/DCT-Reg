#!/usr/bin/env python3
"""Extract v38 direction loss metrics from v3.10 training logs.

Proves whether v3.10's direction loss was actually learning anything
by looking at high_risk_gain / low_risk_gain / plan_shift trends.

A working direction loss should show:
  - high_risk_gain > 0  (moving toward high-risk → risk should INCREASE)
  - low_risk_gain < 0   (moving toward low-risk → risk should DECREASE)
  - plan_shift should be non-trivial (coupling actually changes)

If high_risk_gain ≈ low_risk_gain ≈ 0, direction loss is dead.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path("/data1/DCT-Reg")
LOG_GLOB = "logs/20260909_v310_blca_uni2h.log"

PATTERN = re.compile(
    r"v38_direction=(?P<direction>[\d.]+)"
    r"\s+v38_dose=(?P<dose>[\d.]+)"
    r"\s+v38_reconfiguration=(?P<reconf>[\d.]+)"
    r"\s+v38_total=(?P<total>[\d.]+)"
    r"\s+v38_high_risk_gain=(?P<high_gain>[\d.\-e]+)"
    r"\s+v38_low_risk_gain=(?P<low_gain>[\d.\-e]+)"
    r"\s+v38_high_plan_shift=(?P<high_shift>[\d.\-e]+)"
    r"\s+v38_low_plan_shift=(?P<low_shift>[\d.\-e]+)"
    r"\s+v38_finite=(?P<finite>[\d.]+)"
)


def parse_log(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    text = path.read_text(errors="ignore")
    for m in PATTERN.finditer(text):
        try:
            d = {
                "v38_direction": float(m["direction"]),
                "v38_high_risk_gain": float(m["high_gain"]),
                "v38_low_risk_gain": float(m["low_gain"]),
                "v38_high_plan_shift": float(m["high_shift"]),
                "v38_low_plan_shift": float(m["low_shift"]),
                "v38_dose": float(m["dose"]),
                "v38_reconfiguration": float(m["reconf"]),
                "v38_total": float(m["total"]),
            }
            rows.append(d)
        except Exception:
            continue
    return rows


def main():
    log_path = REPO / LOG_GLOB
    rows = parse_log(log_path)
    print(f"Parsed {len(rows)} epochs from {log_path.name}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(f"\n=== v3.10 Direction Loss Internal Metrics ===\n")
    print("Column descriptions:")
    print("  v38_direction         = direction loss value (training)")
    print("  v38_high_risk_gain    = Δ risk when interpolating toward HIGH-risk anchor")
    print("  v38_low_risk_gain     = Δ risk when interpolating toward LOW-risk anchor")
    print("  v38_high_plan_shift   = L1 distance moved by OT plan for high-risk interp")
    print("  v38_low_plan_shift    = L1 distance moved by OT plan for low-risk interp")
    print("  v38_total             = combined v38 loss (direction + dose + reconfiguration)\n")
    print("If direction loss worked, we'd see:")
    print("  - v38_high_risk_gain trending UP   (positive)")
    print("  - v38_low_risk_gain trending DOWN  (negative)")
    print("  - plan_shifts non-trivial (≠ 0)\n")

    # First vs last
    print("=== First vs Last Epoch ===")
    first = df.iloc[0]
    last = df.iloc[-1]
    for col in ["v38_direction", "v38_high_risk_gain", "v38_low_risk_gain",
                "v38_high_plan_shift", "v38_low_plan_shift"]:
        print(f"  {col:30s}  first={first[col]:+.5f}   last={last[col]:+.5f}   "
              f"Δ={(last[col] - first[col]):+.5f}")

    # Stats
    print(f"\n=== Aggregate stats over {len(df)} epochs ===")
    for col in ["v38_direction", "v38_high_risk_gain", "v38_low_risk_gain",
                "v38_high_plan_shift", "v38_low_plan_shift"]:
        vals = df[col]
        print(f"  {col:30s}  mean={vals.mean():+.5f}   std={vals.std():.5f}   "
              f"min={vals.min():+.5f}   max={vals.max():+.5f}")

    # Diagnostic: does high_risk_gain > 0 and low_risk_gain < 0 most of the time?
    correct_signs = ((df["v38_high_risk_gain"] > 0) & (df["v38_low_risk_gain"] < 0)).mean()
    print(f"\n=== Direction Loss Learning Diagnostic ===")
    print(f"  Fraction of epochs with correct sign (high_gain>0 AND low_gain<0): {correct_signs:.2%}")
    print(f"  → If near 50% or below, direction loss is NOT teaching monotonic risk response")
    print(f"  → Random would be ~50%, working should be >70%")

    # Save
    out_csv = REPO / "results/audit_comparison/v310_direction_loss_internal.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics to {out_csv}")


if __name__ == "__main__":
    main()
