"""Build a small self-contained task ZIP, excluding data/results/checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ["models/SlotSPE.py", "models/slot_attention.py", "models/transformer.py",
    "models/omics_encoder.py", "utils/loss_func.py", "utils/core_utils.py",
    "utils/process_args.py", "dataset/dataset_survival.py", "README.md", "requirements.txt"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new delivery filename; existing archives are retained")
    paths = [ROOT / "survot_rank/__init__.py", ROOT / "survot_rank/research/__init__.py",
        *sorted((ROOT / "survot_rank/research/slotspe_proof").glob("*.py")),
        ROOT / "scripts/run_slotspe_proof.py", ROOT / "scripts/build_slotspe_proof_delivery.py",
        ROOT / "run_slotspe_proof.sh",
        ROOT / "configs/slotspe_proof_server.json", ROOT / "docs/SLOTSPE_PROOF_TASK.json",
        ROOT / "configs/slotspe_attribution_server.json",
        ROOT / "scripts/audit_slotspe_return.py", ROOT / "scripts/pack_slotspe_return.py",
        ROOT / "docs/SlotSPE证明任务_服务器一键运行.txt", ROOT / "tests/test_slotspe_proof_queue.py",
        ROOT / "tests/test_slotspe_attribution.py"]
    paths += [ROOT / "third_party/SlotSPE" / rel for rel in REFERENCE]
    prefix = "slotspe_proof_task_20261019/"
    source = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {"schema": "slotspe_proof_delivery_v1", "prefix": prefix, "files_sha256": source,
        "contains_patient_data": False, "contains_checkpoints": False,
        "baseline_source": "Local repository SlotSPE snapshot; original attribution retained in its README",
        "extract": "Extract to a new task directory; no need to overlay an existing checkout",
        "run": "bash run_slotspe_proof.sh --run --folds 0 --seeds 3 --epochs 3 --patches 256 (after supplying actual server input directories)",
        "requested_server_reproduction": {"folds": [0], "seeds": [3], "epochs_per_arm": 3,
            "patches": 256, "training_runs": 2, "eval_repeats": 20,
            "executor": "server-side AI", "status": "awaiting_server_artifacts",
            "is_abc_attribution": False},
        "available_attribution": {"config": "configs/slotspe_attribution_server.json", "arms": ["A", "B", "C"],
            "training_runs": 3, "epochs": 3, "patches": 256, "default_mode": "plan",
            "patient_execution": "not_automatically_authorized_by_packaging"},
        "patient_slot_intervention": "Implemented for final checkpoints; no server observations until raw outputs return",
        "default_without_overrides": "30 training runs; not the requested one-pair short reproduction"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(prefix + "DELIVERY_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": len(paths)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
