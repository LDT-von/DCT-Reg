"""Build an isolated transport comparison task; no data/results/checkpoints."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "slotspe_transport_task_20261019/"


def build(output):
    output = Path(output)
    paths = [ROOT / "survot_rank/__init__.py", ROOT / "survot_rank/plugins/__init__.py",
        ROOT / "survot_rank/research/__init__.py", ROOT / "survot_rank/research/components/__init__.py",
        ROOT / "survot_rank/research/components/slot_attention.py",
        ROOT / "scripts/run_slotspe_transport_compare.py", ROOT / "scripts/audit_slotspe_transport_compare.py",
        ROOT / "scripts/build_slotspe_transport_delivery.py", ROOT / "configs/slotspe_transport_compare.json",
        ROOT / "docs/运输插拔_SlotSPE服务器交接_20261019.txt", ROOT / "tests/test_slotspe_transport_compare.py"]
    for rel in ("survot_rank/plugins/transport_only", "survot_rank/plugins/transport",
                "third_party/SlotSPE/models", "third_party/SlotSPE/utils", "third_party/SlotSPE/dataset"):
        paths.extend(sorted((ROOT / rel).rglob("*.py")))
    paths.extend(ROOT / "third_party/SlotSPE" / rel for rel in ("README.md", "requirements.txt"))
    paths = sorted(set(paths))
    files = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {"schema": "slotspe_transport_delivery_v1", "files_sha256": files,
        "contains_patient_data": False, "contains_checkpoints": False,
        "default": "plan only; one fold/seed; native/learned/independent; original unrepaired SlotSPE",
        "execution": "Server AI sets OT_METADATA_ROOT and OT_FEATURES_ROOT; inspect plan then explicitly use --run",
        "claim": "Interfaces and synthetic execution only until real patient artifacts pass audit"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, PREFIX + path.relative_to(ROOT).as_posix())
        archive.writestr(PREFIX + "DELIVERY_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"path": str(output.resolve()), "files": len(paths), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))
