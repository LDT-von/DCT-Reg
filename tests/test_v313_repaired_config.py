"""Pure configuration checks for the opt-in v3.13 repaired recipe."""

from __future__ import annotations

import argparse
from copy import deepcopy

import pytest

from survot_rank.config import apply_overrides, config_to_argv, flatten_config, load_config


def _parse_config(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dct_v313_revision", choices=("legacy", "repaired"), default="legacy")
    parser.add_argument("--dct_v311_lambda_slot_diversity", type=float, default=0.10)
    parser.add_argument("--dct_v313_lambda_wsi_reconstruction", type=float, default=0.0)
    parser.add_argument("--dct_v313_disable_self_reconstruction", default="false")
    parser.add_argument("--dct_v313_disable_cross_reconstruction", default="false")
    return parser.parse_args(config_to_argv(config))


def test_repaired_yaml_parameters_are_explicit_without_cli_overrides(tmp_path):
    path = tmp_path / "repaired.yaml"
    path.write_text(
        "name: repaired Full\n"
        "description: reproducible explicit coefficients\n"
        "model:\n"
        "  dct_v313_revision: repaired\n"
        "  dct_v311_lambda_slot_diversity: 0.10\n"
        "  dct_v313_lambda_wsi_reconstruction: 0.05\n",
        encoding="utf-8",
    )
    config = load_config(path)
    before = deepcopy(config)
    merged = apply_overrides(config, [])
    parsed = _parse_config(merged)

    assert config == before
    assert parsed.dct_v311_lambda_slot_diversity == pytest.approx(0.10)
    assert parsed.dct_v313_lambda_wsi_reconstruction == pytest.approx(0.05)
    assert set(merged["_dct_user_overrides"]) == {
        "dct_v313_revision",
        "dct_v311_lambda_slot_diversity",
        "dct_v313_lambda_wsi_reconstruction",
    }
    # Constructor freezing only applies to keys absent from this metadata.
    frozen = {"dct_v311_lambda_slot_diversity": 0.02}
    for key, value in frozen.items():
        if key not in merged["_dct_user_overrides"]:
            setattr(parsed, key, value)
    assert parsed.dct_v311_lambda_slot_diversity == pytest.approx(0.10)
    assert "dct_v313_disable_self_reconstruction" not in merged["_dct_user_overrides"]


def test_repaired_set_wins_over_section_and_existing_top_level_key():
    # Assignment alone retains the first key's insertion position; without
    # reinsertion the later model section would overwrite the CLI value.
    config = {
        "dct_v311_lambda_slot_diversity": 0.01,
        "model": {
            "dct_v313_revision": "repaired",
            "dct_v311_lambda_slot_diversity": 0.10,
        },
    }
    merged = apply_overrides(config, ["dct_v311_lambda_slot_diversity=0.07"])
    assert _parse_config(merged).dct_v311_lambda_slot_diversity == pytest.approx(0.07)
    assert merged["_dct_user_overrides"].count("dct_v311_lambda_slot_diversity") == 1


def test_set_can_select_repaired_revision_and_last_override_wins():
    config = {"model": {"dct_v313_revision": "legacy", "dct_v311_lambda_slot_diversity": 0.10}}
    merged = apply_overrides(
        config,
        [
            "dct_v313_revision=repaired",
            "dct_v311_lambda_slot_diversity=0.04",
            "dct_v311_lambda_slot_diversity=0.08",
        ],
    )
    parsed = _parse_config(merged)
    assert parsed.dct_v313_revision == "repaired"
    assert parsed.dct_v311_lambda_slot_diversity == pytest.approx(0.08)
    assert "dct_v311_lambda_slot_diversity" in merged["_dct_user_overrides"]


@pytest.mark.parametrize("value", [False, True])
def test_repaired_boolean_values_are_preserved_as_explicit_tokens(value):
    config = {
        "model": {
            "dct_v313_revision": "repaired",
            "dct_v313_disable_self_reconstruction": value,
            "dct_v313_disable_cross_reconstruction": value,
        },
    }
    merged = apply_overrides(config, [])
    argv = config_to_argv(merged)
    parsed = _parse_config(merged)
    expected = str(value).lower()
    for key in ("dct_v313_disable_self_reconstruction", "dct_v313_disable_cross_reconstruction"):
        assert argv[argv.index(f"--{key}") + 1] == expected
        assert getattr(parsed, key) == expected
        assert key in merged["_dct_user_overrides"]


def test_repaired_set_false_overrides_yaml_true():
    config = {
        "model": {
            "dct_v313_revision": "repaired",
            "dct_v313_disable_self_reconstruction": True,
        },
    }
    merged = apply_overrides(config, ["dct_v313_disable_self_reconstruction=false"])
    assert _parse_config(merged).dct_v313_disable_self_reconstruction == "false"
    assert "dct_v313_disable_self_reconstruction" in merged["_dct_user_overrides"]


@pytest.mark.parametrize("revision", [None, "legacy"])
def test_legacy_recipes_track_only_set_keys(revision):
    model = {"dct_v311_lambda_slot_diversity": 0.10}
    if revision is not None:
        model["dct_v313_revision"] = revision
    config = {"model": model, "_dct_user_overrides": ["stale_key"]}
    assert apply_overrides(config, [])["_dct_user_overrides"] == []
    merged = apply_overrides(config, ["dct_v313_lambda_wsi_reconstruction=0.05"])
    assert merged["_dct_user_overrides"] == ["dct_v313_lambda_wsi_reconstruction"]


def test_set_can_select_legacy_revision_without_marking_yaml_parameters_explicit():
    config = {"model": {"dct_v313_revision": "repaired", "dct_v311_lambda_slot_diversity": 0.10}}
    merged = apply_overrides(config, ["dct_v313_revision=legacy"])
    assert _parse_config(merged).dct_v313_revision == "legacy"
    assert merged["_dct_user_overrides"] == ["dct_v313_revision"]


def test_legacy_flatten_order_is_unchanged():
    config = {
        "dct_v311_lambda_slot_diversity": 0.01,
        "model": {"dct_v311_lambda_slot_diversity": 0.10},
    }
    merged = apply_overrides(config, ["dct_v311_lambda_slot_diversity=0.07"])
    assert _parse_config(merged).dct_v311_lambda_slot_diversity == pytest.approx(0.10)


def test_reserved_metadata_is_not_emitted_or_tracked_even_inside_a_section():
    config = {
        "notes": {"dct_v311_lambda_slot_diversity": 99},
        "model": {
            "dct_v313_revision": "repaired",
            "description": "section description",
            "_dct_user_overrides": ["stale_key"],
        },
    }
    merged = apply_overrides(config, [])
    assert flatten_config(merged) == {"dct_v313_revision": "repaired"}
    assert merged["_dct_user_overrides"] == ["dct_v313_revision"]
    assert config_to_argv(merged) == ["--dct_v313_revision", "repaired"]


def test_invalid_override_retains_actionable_error():
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides({"model": {"dct_v313_revision": "repaired"}}, ["invalid"])
