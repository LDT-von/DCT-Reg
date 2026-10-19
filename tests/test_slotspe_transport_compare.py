"""Invented tensors/metadata only; no real patient training or inference."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from survot_rank.plugins.transport_only.mechanism import DCTTransportMechanism, TransportOnlyConfig
from survot_rank.plugins.transport_only.slotspe_compare import make_plan, TransportObserver

ROOT = Path(__file__).resolve().parents[1]


def config(tmp_path, **updates):
    return {"source_root": str(ROOT / "third_party/SlotSPE"), "metadata_root": str(tmp_path / "metadata"),
        "features_root": str(tmp_path / "features"), "folds": [0], "seeds": [3], "device": "cpu",
        "epochs": 1, "patches": 8, "batch_size": 2, "eval_repeats": 2, "encoding_dim": 16,
        "mechanism": {"stages": 1, "geometries": ["cosine"], "eps": .2, "eps_start": .2}, **updates}


def fixture(tmp_path):
    import pandas as pd
    m, f = tmp_path / "metadata", tmp_path / "features"
    for directory in (m / "clinical/all", m / "splits/5fold/blca", m / "raw_rna_data_inter", m / "signatures", f):
        directory.mkdir(parents=True)
    ids = [f"SYNTHETIC_{i}" for i in range(7)]
    pd.DataFrame({"case id": ids, "survival_months_dss": [1., 2., 3., 4., 5., 6., 12.],
        "censorship_dss": [0] * 7, "wsi": [pid + ".svs" for pid in ids]}).to_csv(m / "clinical/all/blca.csv", index=False)
    pd.DataFrame({"train": pd.Series(ids[:5]), "val": pd.Series(ids[5:])}).to_csv(m / "splits/5fold/blca/fold_0.csv", index=False)
    pd.DataFrame({pid: [float(i+1), float(i+2), float(i+3)] for i,pid in enumerate(ids)},
                 index=["G1", "G2", "G3"]).to_csv(m / "raw_rna_data_inter/blca_rna_inter.csv")
    pd.DataFrame({"PATH_1": ["G1", "G2"], "PATH_2": ["G2", "G3"]}).to_csv(m / "signatures/combine_signatures.csv", index=False)
    generator = torch.Generator().manual_seed(909)
    for pid in ids:
        torch.save(torch.randn(10, 16, generator=generator), f / (pid + ".pt"))


def test_plan_never_reads_patient_inputs_or_constructs_model(tmp_path):
    plan = make_plan(config(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "metadata").exists()
    assert plan["arms"] == ["native", "learned", "independent"] and plan["dct_added_losses"] == []
    assert plan["host_version"] == "unrepaired_local_SlotSPE_snapshot"


@pytest.mark.parametrize("updates", [{"folds": [0,1]}, {"batch_size": 1},
    {"host_version": "readout_fix"}, {"mechanism": {"plan_mode": "independent"}},
    {"mechanism": {"transport_mode": "uot"}}, {"mechanism": {"lambda_recon": 1.}},
    {"metadata_root": "${UNSET_TRANSPORT_TEST_VAR}"}])
def test_rejects_protocol_drift(tmp_path, updates):
    with pytest.raises((ValueError, TypeError)):
        make_plan(config(tmp_path, **updates), tmp_path / "out")


def test_observer_measures_actual_writeback_and_independent_topology():
    torch.set_num_threads(1)
    module = DCTTransportMechanism(TransportOnlyConfig(host_dim=8, slot_budget=4, stages=1,
        geometries=("cosine",), eps=.2, eps_start=.2))
    observer = TransportObserver(module)
    w, o = torch.randn(2,3,8), torch.randn(2,4,8)
    updated = module(w,o)
    assert observer.rows[0]["wsi_delta_l2"] == pytest.approx(float((updated[0][0]-w[0]).norm()))
    assert observer.rows[0]["transport_mass"] == pytest.approx(1., abs=1e-5)
    module.set_plan_mode("independent")(w,o)
    assert all(row["plan_independent_l1"] < 1e-6 for row in observer.rows)
    observer.handle.remove()


def test_standalone_three_arm_execution_audit_and_corruption_rejection(tmp_path):
    """Run the shipped archive in isolation using invented patients only."""
    archive = tmp_path / "delivery.zip"
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_slotspe_transport_delivery.py"),
        "--output", str(archive)], cwd=ROOT, capture_output=True, check=True)
    with zipfile.ZipFile(archive) as z:
        assert not any(".review/" in p or "/results/" in p or "/transport_v2/" in p or "/slotspe_proof/" in p for p in z.namelist())
        z.extractall(tmp_path / "unpacked")
    task = tmp_path / "unpacked/slotspe_transport_task_20261019"
    fixture(tmp_path)
    profile = tmp_path / "config.json"
    profile.write_text(json.dumps(config(tmp_path, source_root=str(task / "third_party/SlotSPE"))), encoding="utf-8")
    output = tmp_path / "output"
    command = [sys.executable, "-B", str(task / "scripts/run_slotspe_transport_compare.py"),
        "--config", str(profile), "--output", str(output), "--run"]
    environment = {**os.environ, "PYTHONPATH": "", "PYTHONIOENCODING": "utf-8"}
    run = subprocess.run(command, cwd=task, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    audit = [sys.executable, "-B", str(task / "scripts/audit_slotspe_transport_compare.py"), str(output)]
    run = subprocess.run(audit, cwd=task, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["status"] == "artifact_consistency_passed"
    summary = json.loads((output / "paired_summary.json").read_text())
    assert summary["summaries"]["learned"]["extra_parameters"] == summary["summaries"]["independent"]["extra_parameters"] > 0
    assert summary["summaries"]["native"]["extra_parameters"] == 0
    # Full-source parameter counts differ only by transport; every epoch kept host aux.
    for arm in ("native", "learned", "independent"):
        curve = json.loads((output / arm / "curve.json").read_text())
        assert curve[0]["train_auxiliary_total"] > 0
    run = subprocess.run(command, cwd=task, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode != 0 and "new output" in run.stderr
    path = output / "learned/predictions.csv"
    original = path.read_text()
    path.write_text(original.replace("SYNTHETIC_5", "WRONG_PATIENT"))
    run = subprocess.run(audit, cwd=task, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode != 0 and "identities" in run.stderr
    path.write_text(original)
    curve_path = output / "independent/curve.json"
    curve = json.loads(curve_path.read_text())
    curve[0]["train_total"] += 1
    curve_path.write_text(json.dumps(curve))
    run = subprocess.run(audit, cwd=task, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode != 0 and "host original" in run.stderr
