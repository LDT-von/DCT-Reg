"""CLI and protocol guarantees for the separate overfitting candidate."""
from pathlib import Path
import pytest
from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.research.methods.catalog import METHOD_ALIASES, METHOD_CATALOG
from survot_rank.training.extended_args import process_args_extended

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arm,dim,hidden,dropout,lr,reg", [
    ("control", 256, 1536, 0., .0005, .0005),
    ("capacity", 128, 256, 0., .0005, .0005),
    ("regularized", 128, 256, .30, .0001, .005),
])
def test_candidate_recipes_retain_e050_protocol_and_all_six_terms(arm, dim, hidden, dropout, lr, reg):
    config = load_config(ROOT / f"configs/dct_v22_blca_legacy_val_{arm}.yaml")
    args = process_args_extended(config_to_argv(config))
    assert args.survot_method == "dct_v22"
    assert METHOD_CATALOG[args.survot_method].class_name == "DCTV22"
    assert METHOD_ALIASES["dct_v2_2"] == "dct_v22"
    assert args.evaluation_protocol == "legacy_val" and args.which_splits == "5fold_uni2h"
    assert args.seed == 3 and args.max_epochs == 30 and args.early_stop_patience == 0
    assert args.wsi_projection_dim == dim and args.dct_v22_wsi_hidden_dim == hidden
    assert args.dct_v22_wsi_dropout == dropout and args.lr == lr and args.reg == reg
    assert args.dct_v22_evaluate_train is True
    assert args.dct_v22_relation_gradient_interval == (0 if arm == "control" else 10)
    assert args.dct_v21_local_readout == "centered_pair"
    assert args.dct_v21_relation_objective == "contrast_probe"
    assert args.dct_v21_diversity_objective == "collision_distance"
    assert [args.dct_v2_lambda_ipcw_rank, args.dct_v2_lambda_slot_nll,
            args.dct_v2_lambda_diversity, args.dct_v2_lambda_relation,
            args.dct_v2_lambda_native] == [.10, .05, .10, .05, .05]
    assert f"/E051_v22_20261014/{arm}" in args.results_dir.replace("\\", "/")


def test_explicit_train_snapshot_disable_survives_yaml_serialization():
    config = apply_overrides(load_config(ROOT / "configs/dct_v22_blca_legacy_val_regularized.yaml"),
                             ["dct_v22_evaluate_train=false"])
    argv = config_to_argv(config)
    assert "--no-dct_v22_evaluate_train" in argv
    assert process_args_extended(argv).dct_v22_evaluate_train is False


def test_real_factory_and_stable_training_loss_accept_new_candidate():
    from survot_rank.training.train_runner import get_model, init_loss_function
    from survot_rank.research.methods.dct_v2.losses import StableNLLSurvLoss
    args = process_args_extended(config_to_argv(load_config(
        ROOT / "configs/dct_v22_blca_legacy_val_regularized.yaml")))
    # Synthetic constructor widths, without loading patients or features.
    args.omic_sizes = (5, 7)
    model = get_model(method=args.survot_method, args=args, omic_input_dim=None,
                      omic_names=("test_a", "test_b"), pathway_names=("test_a", "test_b"))
    assert type(model).__name__ == "DCTV22"
    assert model.config.dim == 128 and model.config.wsi_hidden_dim == 256
    assert model.config.relation_gradient_interval == 10
    assert model.get_extra_state()["schema"] == "dct_v22_standalone_v1"
    assert isinstance(init_loss_function(args), StableNLLSurvLoss)
