#!/usr/bin/env python3
"""Inspect pkl files from v3.11 training to understand available data."""
import pickle
from pathlib import Path

pkl_dir = Path("/data1/DCT-Reg/results/dct_v311_blca_uni/blca/SurvOTRank_dct_v311/0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_")

for pkl_path in sorted(pkl_dir.glob("split_*_results_final.pkl")):
    print(f"\n=== {pkl_path.name} ===")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"  {k}: len={len(v)}, type={type(v)}")
                if len(v) > 0 and hasattr(v[0], 'shape'):
                    print(f"    [0] shape={v[0].shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {repr(v)[:100]}")
    elif isinstance(data, (list, tuple)):
        print(f"Len: {len(data)}")
        for i, item in enumerate(data):
            if hasattr(item, 'shape'):
                print(f"  [{i}]: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  [{i}]: {type(item).__name__} = {repr(item)[:100]}")
