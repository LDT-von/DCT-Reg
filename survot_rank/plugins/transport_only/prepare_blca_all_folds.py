"""
One-time BLCA 5-fold data preparation.
Creates a manifest with all 5 folds (not just fold 0).
Subsequent experiments reuse this prepared directory.
"""
import json
import sys
import time
from pathlib import Path

# Add project root to path so 'survot_rank' can be imported as a package
sys.path.insert(0, "/data1/DCT-Reg")

from survot_rank.plugins.transport_only.slotspe_support import data as slotspe_data

def prepare_blca_all_folds(output_root: Path):
    cfg = {
        "study": "blca",
        "label_col": "survival_months_dss",
        "signature": "combine",
        "folds": [0, 1, 2, 3, 4],
        "patches": 2048,
        "batch_size": 8,
        "eval_repeats": 20,
        "epochs": 30,
        "encoding_dim": 1536,
        "device": "cuda:0",
        "lr": 0.0005,
        "source_root": "/data1/DCT-Reg/third_party/SlotSPE",
        "metadata_root": "/data1/dataset_csv",
        "features_root": "/data1/DCT-Reg/data/slotspe_pt_features/blca/pt_files",
        "torch_threads": 2,
    }

    print(f"Preparing BLCA all 5 folds → {output_root}", flush=True)
    t0 = time.perf_counter()
    manifest = slotspe_data.prepare(cfg, output_root)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)
    # Quick summary
    for fold_info in manifest.get("folds", {}).values():
        print(f"  fold {fold_info['fold']}: train={fold_info['counts']['train']} "
              f"val={fold_info['counts']['val']} excluded={fold_info['counts']['excluded']}", flush=True)
    return manifest

if __name__ == "__main__":
    root = Path("/tmp/ot03_prep_all5")
    root.mkdir(parents=True, exist_ok=True)
    prepare_blca_all_folds(root)
