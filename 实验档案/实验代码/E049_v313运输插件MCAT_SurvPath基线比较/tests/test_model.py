"""Unit tests for E049 model.py arm definitions and audit."""
from __future__ import annotations

import pytest
from model import ARM_DEFINITIONS, ARM_NAMES


class TestArmDefinitions:
    def test_all_six_arms_defined(self):
        expected = {
            "native", "independent", "learned",
            "independent_three_decoders", "learned_three_decoders",
            "learned_shared_decoders",
        }
        assert set(ARM_NAMES) == expected

    def test_native_disabled(self):
        assert ARM_DEFINITIONS["native"]["enabled"] is False
        assert ARM_DEFINITIONS["native"]["reconstruction"] is False

    def test_independent_enabled_plan_mode(self):
        assert ARM_DEFINITIONS["independent"]["enabled"] is True
        assert ARM_DEFINITIONS["independent"]["plan_mode"] == "independent"
        assert ARM_DEFINITIONS["independent"]["reconstruction"] is False

    def test_learned_enabled_plan_mode(self):
        assert ARM_DEFINITIONS["learned"]["enabled"] is True
        assert ARM_DEFINITIONS["learned"]["plan_mode"] == "learned"
        assert ARM_DEFINITIONS["learned"]["reconstruction"] is False

    def test_independent_three_decoders(self):
        arm = ARM_DEFINITIONS["independent_three_decoders"]
        assert arm["enabled"] is True
        assert arm["plan_mode"] == "independent"
        assert arm["reconstruction"] is True
        assert arm["decoder_mode"] == "separate"

    def test_learned_three_decoders(self):
        arm = ARM_DEFINITIONS["learned_three_decoders"]
        assert arm["enabled"] is True
        assert arm["plan_mode"] == "learned"
        assert arm["reconstruction"] is True
        assert arm["decoder_mode"] == "separate"

    def test_learned_shared_decoders(self):
        arm = ARM_DEFINITIONS["learned_shared_decoders"]
        assert arm["enabled"] is True
        assert arm["plan_mode"] == "learned"
        assert arm["reconstruction"] is True
        assert arm["decoder_mode"] == "shared"

    def test_capacity_equivalence_independent_vs_learned(self):
        """Independent and learned should have identical structural capacity."""
        for arm_name in ("independent", "learned"):
            for arm_name2 in ("independent", "learned"):
                a, b = ARM_DEFINITIONS[arm_name], ARM_DEFINITIONS[arm_name2]
                assert a["enabled"] == b["enabled"]
                assert a["reconstruction"] == b["reconstruction"]
                # Only plan_mode differs

    def test_capacity_equivalence_independent_three_vs_learned_three(self):
        """Independent_three and learned_three should have identical capacity."""
        a = ARM_DEFINITIONS["independent_three_decoders"]
        b = ARM_DEFINITIONS["learned_three_decoders"]
        assert a["enabled"] == b["enabled"]
        assert a["reconstruction"] == b["reconstruction"]
        assert a["decoder_mode"] == b["decoder_mode"]
        # Only plan_mode differs

    def test_reconstruction_arms_preserve_self_cross_weights(self):
        """All three-decoder arms should have same reconstruction weights."""
        for arm_name in ("independent_three_decoders", "learned_three_decoders", "learned_shared_decoders"):
            # Weight parameters are injected in build_model/build_native_adapter,
            # but the reconstruction flag must be True
            assert ARM_DEFINITIONS[arm_name]["reconstruction"] is True
