"""Create an isolated V2 server source snapshot; never overwrite server files."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "survot_rank").rglob("*.py"))
    files += sorted((ROOT / "common").rglob("*.py"))
    files += [ROOT / "scripts" / name for name in (
        "prepare_dct_v23_calibration.py", "run_dct_v23_workflow.py", "build_dct_v23_workflow_delivery.py")]
    files += [ROOT / "tests" / name for name in (
        "test_dct_v23_workflow.py", "test_dct_v23_model.py",
        "test_dct_v23_calibration.py", "test_dct_v23_config_integration.py",
        "test_dct_v23_loss_pruning.py", "test_dct_v23_simplified_core.py")]
    files += [ROOT / "tests/test_training_failure_propagation.py", ROOT / "requirements.txt", ROOT / "pyproject.toml",
        ROOT / "docs/V2_方法归因与服务器交接_20261019.txt", ROOT / "docs/DCT_V23_METHOD_CONTRACT.json"]
    files += sorted((ROOT / "configs").glob("dct_v23_blca_legacy_val_*.yaml"))
    files += [ROOT / "configs/dct_rg_blca_legacy_val.yaml"]
    contents = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(files))}
    manifest = dict(schema="dct_v23_isolated_source_delivery_v1", as_of="2026-10-19",
        local_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        origin="Current local source snapshot including existing shared-runtime changes; not a remote release claim",
        scope="V2 workflow and controls; legacy and shared Python are included only as executable dependencies",
        patient_data_included=False, files={name: hashlib.sha256(data).hexdigest() for name, data in contents.items()})
    contents["SOURCE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr("dct_v23_server_20261019/" + name, data)
    return dict(path=str(destination), sha256=hashlib.sha256(destination.read_bytes()).hexdigest(), files=len(contents))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(build(parser.parse_args().output), ensure_ascii=False))
