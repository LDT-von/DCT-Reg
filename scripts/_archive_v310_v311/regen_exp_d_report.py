#!/usr/bin/env python3
"""Regenerate REPORT.md for Exp D by reading per_fold.pkl across all folds."""

import json
import pickle
from pathlib import Path

import numpy as np

OUTPUT = Path("results/v310_proof_d_per_slot_hazard")

pkl = OUTPUT / "per_fold.pkl"
if not pkl.exists():
    print("no per_fold.pkl")
    raise SystemExit(1)
per_fold = pickle.load(open(pkl, "rb"))
print(f"loaded {len(per_fold)} folds from pkl: {sorted(per_fold.keys())}")
