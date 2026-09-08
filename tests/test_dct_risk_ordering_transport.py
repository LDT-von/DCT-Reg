from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts.e4_audit_adapted import analyze_direction_consistency
from tests.test_dct_v310_directional_regularized_transport import batch, make_args, reference
from survot_rank.research.methods.dct_risk_ordering_transport.model import (
    DCTRiskOrderingTransport,
)
from survot_rank.training.model_factory import get_model


def test_zero_response_is_not_directional_monotonicity():
    rows = []
    for direction in ("low_risk", "high_risk"):
        for alpha in (0.0, 0.5, 1.0):
            rows.append({
                "patient_id": "p0",
                "alpha": alpha,
                "direction": direction,
                "risk_pred": 0.25,
            })
    metrics = analyze_direction_consistency(pd.DataFrame(rows))
    assert metrics["responsive_rate_low"] == 0.0
    assert metrics["responsive_rate_high"] == 0.0
    assert metrics["monotonic_decrease_rate"] == 0.0
    assert metrics["monotonic_increase_rate"] == 0.0


@pytest.mark.parametrize("dose_weight", [0.0, 0.05])
def test_risk_ordering_candidate_uses_configured_objective(dose_weight):
    torch.manual_seed(7)
    args = make_args(
        dct_rot_lambda_direction=0.10,
        dct_rot_lambda_dose=dose_weight,
    )
    model = DCTRiskOrderingTransport(args, omic_input_dim=20)
    model.configure_train_reference(*reference())
    model.train()

    logits, auxiliary_loss = model(**batch())
    diagnostics = model.last_training_losses
    expected = (
        0.10 * diagnostics["ipcw_rank"]
        + 0.10 * diagnostics["v38_direction"]
        + dose_weight * diagnostics["v38_dose"]
    )

    assert logits.shape == (8, 4)
    assert torch.isfinite(auxiliary_loss)
    torch.testing.assert_close(auxiliary_loss, expected)
    assert model.objective_weights()["dose_ordering"] == dose_weight


def test_risk_ordering_candidate_is_registered():
    model = get_model(
        "dct_rot",
        make_args(dct_rot_lambda_direction=0.10, dct_rot_lambda_dose=0.05),
        omic_input_dim=20,
    )
    assert type(model).__name__ == "DCTRiskOrderingTransport"
