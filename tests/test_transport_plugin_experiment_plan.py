"""Matched construction/protocol checks; no patient data or training jobs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_transport_plugin_experiments", ROOT / "scripts/prepare_transport_plugin_experiments.py",
)
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)
SIZES = [10, 20, 30, 40, 50, 60]


def complete_protocol(baselines=("mcat", "survpath"), sizes=SIZES):
    protocol = PLAN.protocol_template(list(baselines), sizes)

    def fill(value):
        if isinstance(value, dict):
            return {key: ("a" * 64 if key.endswith("sha256") else fill(entry))
                    for key, entry in value.items()}
        return "explicit test protocol value" if value == PLAN.REQUIRED else deepcopy(value)

    protocol = fill(protocol)
    protocol["splits"]["folds"] = [0, 1, 2, 3, 4]
    protocol["training"]["seeds"] = [3, 7]
    protocol["training"]["epochs"] = 20
    protocol["checkpoint"]["selection_split"] = "validation"
    protocol["checkpoint"]["direction"] = "maximize"
    for baseline in protocol["baselines"].values():
        baseline["loss"]["parameters"] = {"alpha": 0.15}
        baseline["optimizer"]["parameters"] = {"lr": 0.0001}
    return protocol


def test_matrix_and_factory_constructor_contract():
    from survot_rank.plugins.transport.native import build_native_adapter

    matrix = PLAN.construction_matrix(["mcat", "survpath"], SIZES, include_shared=True)
    signature = inspect.signature(build_native_adapter)
    for baseline, arms in matrix.items():
        assert len(arms) == 6
        assert arms["native"]["enabled"] is False
        assert arms["native"]["reconstruction_config"] is None
        for name, config in arms.items():
            signature.bind(**config)
            assert config["backbone_name"] == baseline
            assert config["wsi_input_dim"] == 1536
            assert config["plugin_config"]["dim"] == 256
            assert config["plugin_config"]["wsi_dim"] == config["plugin_config"]["omics_dim"] == 256
            assert config["omic_sizes"] == SIZES
            if config["reconstruction_config"] is not None:
                recon = config["reconstruction_config"]
                assert [recon[key] for key in ("self_weight", "cross_weight", "wsi_weight")] == [0.05] * 3
                assert recon["num_omics_tokens"] == 6
                assert recon["ramp_start"] == 2 and recon["ramp_epochs"] == 5
                assert recon["mode"] == ("shared" if name.endswith("shared_decoders") else "separate")


@pytest.mark.parametrize("with_aux", [False, True])
def test_only_plan_mode_differs_in_transport_comparison(with_aux):
    matrix = PLAN.construction_matrix(["mcat"], SIZES)["mcat"]
    suffix = "_three_decoders" if with_aux else ""
    independent = deepcopy(matrix["independent" + suffix])
    learned = deepcopy(matrix["learned" + suffix])
    assert independent["plugin_config"].pop("plan_mode") == "independent"
    assert learned["plugin_config"].pop("plan_mode") == "learned"
    assert independent == learned


def test_default_plan_keeps_required_fields_and_hashes(tmp_path):
    output = tmp_path / "new"
    manifest = PLAN.write_plan(output, ["mcat", "survpath"], SIZES)
    assert len(manifest["arms"]) == 10
    assert manifest["status"] == "plan_only_not_run" and manifest["executed"] is False
    assert manifest["protocol_status"] == "pending_required_values"
    assert manifest["data_hashes_and_patient_assignments_verified"] is False
    assert "training.seeds" in manifest["required_fields"]
    protocol_bytes = (output / "protocol.json").read_bytes()
    assert manifest["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    for arm in manifest["arms"]:
        payload = (output / arm["construction_file"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == arm["construction_sha256"]
        assert json.loads(payload)["backbone_name"] == arm["baseline"]
    assert json.loads((output / "manifest.json").read_text()) == manifest


def test_complete_recorded_protocol_still_does_not_claim_execution(tmp_path):
    protocol = complete_protocol()
    manifest = PLAN.write_plan(tmp_path / "plan", ["mcat", "survpath"], SIZES, protocol=protocol)
    assert manifest["required_fields"] == []
    assert manifest["protocol_status"] == "recorded_complete_unverified"
    assert manifest["executed"] is False
    assert manifest["data_hashes_and_patient_assignments_verified"] is False


def test_no_overwrite_even_of_an_empty_existing_directory(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    with pytest.raises(FileExistsError, match="fresh directory"):
        PLAN.write_plan(output, ["mcat"], SIZES)
    sentinel = output / "existing.json"
    sentinel.write_bytes(b"preserve existing results")
    with pytest.raises(FileExistsError):
        PLAN.write_plan(output, ["mcat"], SIZES)
    assert sentinel.read_bytes() == b"preserve existing results"


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p.update(execution_status="completed"), "execution_status"),
    (lambda p: p.update(results={"c_index": 0.99}), "exactly"),
    (lambda p: p["patient_manifest"].update(sha256="not-a-hash"), "SHA-256"),
    (lambda p: p["training"].update(seeds=[3, 3]), "unique"),
    (lambda p: p["training"].update(epochs=0), "positive"),
    (lambda p: p["training"].update(epochs=True), "positive"),
    (lambda p: p["checkpoint"].update(selection_split="test"), "validation"),
    (lambda p: p["features"].update(wsi_input_dim=1024), "construction"),
    (lambda p: p["features"]["omics_groups"].update(omic_sizes=[1, 2, 3]), "construction"),
    (lambda p: p["discrete_intervals"].update(num_classes=3), "construction"),
    (lambda p: p["endpoint"].update(censoring_convention="REQUIRED"), "placeholder"),
    (lambda p: p["endpoint"].update(censoring_convention={}), "string"),
    (lambda p: p["training"].update(random_stream_policy="   "), "string"),
    (lambda p: p["baselines"]["mcat"]["optimizer"]["parameters"].update(lr=float("nan")), "JSON"),
])
def test_invalid_protocol_fails_before_creating_output(tmp_path, mutation, message):
    protocol = complete_protocol()
    mutation(protocol)
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match=message):
        PLAN.write_plan(output, ["mcat", "survpath"], SIZES, protocol=protocol)
    assert not output.exists()


def test_independent_configuration_objects():
    matrix = PLAN.construction_matrix(["mcat", "survpath"], SIZES)
    matrix["mcat"]["learned_three_decoders"]["reconstruction_config"]["self_weight"] = 1.0
    matrix["mcat"]["learned"]["omic_sizes"][0] = 1
    assert matrix["mcat"]["independent_three_decoders"]["reconstruction_config"]["self_weight"] == 0.05
    assert matrix["survpath"]["learned"]["omic_sizes"] == SIZES
    assert SIZES == [10, 20, 30, 40, 50, 60]


def test_variable_survpath_group_count_and_mcat_validation():
    matrix = PLAN.construction_matrix(["survpath"], [10, 20, 30])
    assert matrix["survpath"]["learned_three_decoders"]["reconstruction_config"]["num_omics_tokens"] == 3
    with pytest.raises(ValueError, match="six"):
        PLAN.construction_matrix(["mcat"], [10, 20, 30])
    with pytest.raises(ValueError, match="positive"):
        PLAN.construction_matrix(["survpath"], [True])
    with pytest.raises(ValueError, match="unique"):
        PLAN.construction_matrix(["mcat", "mcat"], SIZES)


def test_cli_parses_real_widths_and_writes_plan_only(tmp_path, capsys):
    output = tmp_path / "cli"
    assert PLAN.main(["--baseline", "survpath", "--omic-sizes", "10,20,30",
                      "--output", str(output), "--include-shared"]) == 0
    assert len(json.loads((output / "manifest.json").read_text())["arms"]) == 6
    assert "were not started" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        PLAN.main(["--omic-sizes", "10,0", "--output", str(tmp_path / "invalid")])
    assert error.value.code == 2


def test_reproducible_bytes_in_fresh_outputs(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    PLAN.write_plan(first, ["mcat"], SIZES)
    PLAN.write_plan(second, ["mcat"], SIZES)
    first_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*.json")}
    second_files = {path.relative_to(second): path.read_bytes() for path in second.rglob("*.json")}
    assert first_files == second_files
