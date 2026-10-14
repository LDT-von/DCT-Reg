"""Standalone v2 integration checks without constructing/reading patient data."""

from pathlib import Path

import pytest

from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.training.extended_args import process_args_extended


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key", ["dct_v2_mass_aware", "dct_v2_local_gating"])
@pytest.mark.parametrize("value", [False, True])
def test_v2_boolean_config_preserves_explicit_value(key, value):
    config = apply_overrides(
        load_config(ROOT / "configs/dct_v2_blca_uni2h.yaml"),
        [f"{key}={str(value).lower()}"],
    )
    argv = config_to_argv(config)
    assert (f"--{key}" if value else f"--no-{key}") in argv
    assert getattr(process_args_extended(argv), key) is value


def test_legacy_boolean_serialization_is_unchanged():
    config = {
        "fit_bins_on_train": False,
        "event_stratified_batches": True,
        "dct_v313_disable_self_reconstruction": False,
        "dct_v313_disable_cross_reconstruction": True,
    }
    assert config_to_argv(config) == [
        "--event_stratified_batches",
        "--dct_v313_disable_self_reconstruction", "false",
        "--dct_v313_disable_cross_reconstruction", "true",
    ]


def test_standalone_default_yaml_is_registered_and_parses():
    args = process_args_extended(config_to_argv(load_config(ROOT / "configs/dct_v2_blca_uni2h.yaml")))
    spec = METHOD_CATALOG[args.survot_method]
    assert args.survot_method == "dct_v2"
    assert spec.folder == "dct_v2" and spec.class_name == "DCTV2"
    assert spec.status == "candidate"
    assert METHOD_ALIASES["dct_v2_standalone"] == "dct_v2"
    assert args.alpha_surv == 0
    assert args.encoding_dim == 1536
    assert args.slot_num_wsi == args.slot_num_omics == 8
    assert args.dct_num_stages == 4
    assert args.evaluation_protocol == "outer_test"
    assert args.which_splits == "5fold_uni2h_dct_v2_outer_test_seed3"
    assert args.fit_bins_on_train and args.event_stratified_batches
    assert args.dct_v2_lambda_ipcw_rank == .10
    assert args.dct_v2_lambda_slot_nll == .05
    assert args.dct_v2_lambda_diversity == .10
    assert args.dct_v2_lambda_relation == args.dct_v2_lambda_native == .05


def test_v2_alias_and_direct_negative_cli_flag():
    args = process_args_extended(["--survot_method", "dct_v2_standalone", "--no-dct_v2_mass_aware"])
    assert args.survot_method == "dct_v2_standalone"
    assert args.dct_v2_mass_aware is False


def test_old_method_parser_defaults_are_preserved():
    args = process_args_extended(["--survot_method", "dct_v313"])
    assert args.alpha_surv == .5
    assert args.dct_v313_revision == "legacy"
    assert args.dct_v313_plan_mode == "learned"
    assert args.dct_v313_lambda_wsi_reconstruction == 0
