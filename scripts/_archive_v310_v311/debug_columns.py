#!/usr/bin/env python3
"""Debug column names in survival dataset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import SurvivalDatasetFactory

factory = SurvivalDatasetFactory(
    study="blca",
    data_path="/data1/dataset_csv",
    rna_format="Pathways",
    signature="combine",
    n_bins=4,
    label_col="survival_months_dss",
    num_patches=4096,
    which_splits="5fold",
)

print(f"Label df columns: {list(factory.label_df.columns)}")
print(f"First few rows:\n{factory.label_df.head(2)}")
