"""Configuration preparation only; no model jobs or patient data."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.prepare_v313_repaired_ablation import build_configs, main


def base_config():
    return yaml.safe_load(Path("configs/dct_v313_full_wsi_repaired.yaml").read_text(encoding="utf-8"))


def test_removal_arms_preserve_other_branch_coefficients_and_common_recipe():
    base = base_config()
    saved = copy.deepcopy(base)
    configs = build_configs(base, "new_runs")
    assert base == saved
    for arm, item in configs.items():
        m = item["model"]
        assert m["dct_v313_lambda_reconstruction"] == 0.10
        assert m["dct_v313_reconstruction_self_fraction"] == 0.50
        assert m["dct_v313_reconstruction_cross_fraction"] == 0.50
        assert m["dct_v313_wsi_reconstruction_budget"] == "additive"
        assert item["data"] == base["data"] and item["split"] == base["split"]
        assert item["slot"] == base["slot"]
    assert configs["no_self"]["model"]["dct_v313_lambda_wsi_reconstruction"] == 0.05
    assert configs["no_cross"]["model"]["dct_v313_lambda_wsi_reconstruction"] == 0.05
    assert configs["no_wsi"]["model"]["dct_v313_disable_self_reconstruction"] is False
    assert configs["no_reconstruction"]["model"]["dct_v313_lambda_wsi_reconstruction"] == 0.0


def test_plan_only_writes_nothing_and_saving_refuses_existing_directory(tmp_path, capsys):
    out = tmp_path / "configs"
    assert main(["--results-root", str(tmp_path / "runs"), "--arm", "full"]) == 0
    assert not out.exists() and not (tmp_path / "runs").exists()
    assert "configs_only_no_training" in capsys.readouterr().out
    args = ["--results-root", str(tmp_path / "runs"), "--out", str(out)]
    assert main(args) == 0
    before = (out / "full.yaml").read_bytes()
    with pytest.raises(FileExistsError):
        main(args)
    assert (out / "full.yaml").read_bytes() == before


def test_fixed_total_and_legacy_cannot_be_mislabeled_as_matched_removal():
    base = base_config()
    base["model"]["dct_v313_wsi_reconstruction_budget"] = "fixed_total"
    with pytest.raises(ValueError, match="surviving branches"):
        build_configs(base, "new_runs")
    base["model"]["dct_v313_revision"] = "legacy"
    with pytest.raises(ValueError, match="explicit repaired"):
        build_configs(base, "new_runs")
