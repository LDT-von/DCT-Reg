"""Save the repaired implementation and effective recipe without replacing evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_implementation_audit(model, args):
    """Create one immutable per-fold audit; Git HEAD is not a source identity.

    The normal trainer calls this after construction, before any training.
    Existing evidence is never replaced, even when its values appear identical.
    """
    root = Path(__file__).resolve().parents[6]
    source_hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "survot_rank").rglob("*.py"))
    }
    fields = (
        "seed", "study", "survot_method", "evaluation_protocol", "which_splits",
        "batch_size", "max_epochs", "lr", "reg", "opt", "scheduler",
        "rna_format", "encoding_dim", "wsi_encoder", "num_patches",
        "slot_num_wsi", "slot_num_omics", "slot_iters", "dct_slot_init_mode",
        "fit_bins_on_train", "binning_mode", "label_col", "signature",
        "data_path", "data_root_dir", "wsi_projection_dim", "n_classes",
        "omic_sizes", "omic_names", "pathway_names",
    )
    fold = int(getattr(args, "cur_fold", 0))
    audit = {
        "schema": "dct_v313_repaired_implementation_v1",
        "fold": fold,
        "implementation": model.implementation_audit(),
        "recipe": {key: getattr(args, key, None) for key in fields},
        "explicit_config_keys": sorted(getattr(model, "_v313_explicit_config_keys", getattr(args, "_dct_user_overrides", ()))),
        "source_sha256": source_hashes,
        "evidence_level": "implementation_configuration",
        "creation_phase": "before_training",
    }
    output = Path(args.results_dir) / f"v313_implementation_fold{fold}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def claim_repaired_results_root(args):
    """Claim a new run root before the legacy runner can overwrite any log."""
    if getattr(args, "only_test", False):
        raise ValueError("repaired training entry requires a fresh run; use its outer-test protocol for evaluation")
    output = Path(args.results_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError("repaired v3.13 requires a new results_dir: " + str(output)) from error
    return output
