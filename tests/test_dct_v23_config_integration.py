"""Matched recipes and real constructor/trainer wiring for the new candidate."""
from pathlib import Path

import pytest

from survot_rank.config import config_to_argv, load_config
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.training.extended_args import process_args_extended


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,mode", [("control", "off"),
                                     ("masked_plain", "masked_plain"),
                                     ("gain_calibrated", "gain_calibrated")])
def test_matched_candidate_arms_keep_protocol_capacity_and_parent_objectives(arm, mode):
    args = process_args_extended(config_to_argv(load_config(
        ROOT / f"configs/dct_v23_blca_legacy_val_{arm}.yaml")))
    assert args.survot_method == "dct_v23"
    assert METHOD_ALIASES["dct_v2_3"] == "dct_v23"
    assert METHOD_CATALOG["dct_v23"].class_name == "DCTV23"
    assert args.dct_v23_reconstruction_mode == mode
    assert args.which_splits == "5fold_uni2h" and args.evaluation_protocol == "legacy_val"
    assert (args.seed, args.max_epochs, args.early_stop_patience) == (3, 30, 0)
    assert (args.wsi_projection_dim, args.dct_v22_wsi_hidden_dim) == (128, 256)
    assert (args.dct_v22_wsi_dropout, args.otehv2_dropout) == (.30, .30)
    assert (args.lr, args.reg) == (.0001, .005)
    assert [args.dct_v2_lambda_ipcw_rank, args.dct_v2_lambda_slot_nll,
            args.dct_v2_lambda_diversity, args.dct_v2_lambda_relation,
            args.dct_v2_lambda_native] == [.10, .05, .10, .05, .05]
    assert args.dct_v23_mask_ratio == .30 and args.dct_v23_recon_hidden_dim == 64
    assert args.dct_v23_lambda_reconstruction == .05
    assert args.dct_v23_lambda_calibration == .02
    assert f"/E052_v23_20261014/{arm}" in args.results_dir.replace("\\", "/")
    assert args.dct_v23_calibration_dir.endswith("E052_v23_20261014/calibration")


def test_real_factory_builds_candidate_and_stable_nll():
    from survot_rank.research.methods.dct_v2.losses import StableNLLSurvLoss
    from survot_rank.training.train_runner import get_model, init_loss_function
    args = process_args_extended(config_to_argv(load_config(
        ROOT / "configs/dct_v23_blca_legacy_val_control.yaml")))
    args.omic_sizes = (2, 3)
    model = get_model(method="dct_v23", args=args, omic_input_dim=None,
                      omic_names=(("g0", "shared"), ("shared", "g2", "g3")),
                      pathway_names=("p0", "p1"))
    assert type(model).__name__ == "DCTV23"
    assert model.get_extra_state()["schema"] == "dct_v23_standalone_v1"
    assert model.config.reconstruction_mode == "off"
    assert isinstance(init_loss_function(args), StableNLLSurvLoss)


def test_existing_candidates_keep_their_factory_registration():
    assert METHOD_CATALOG["dct_v2"].class_name == "DCTV2"
    assert METHOD_CATALOG["dct_v21"].class_name == "DCTV21"
    assert METHOD_CATALOG["dct_v22"].class_name == "DCTV22"
