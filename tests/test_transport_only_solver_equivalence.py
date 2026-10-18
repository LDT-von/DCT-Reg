"""Optional historical-kernel equivalence on synthetic tensors only.

Set DCT_TRANSPORT_ONLY_REFERENCE_ROOT to an existing checkout containing the
historical transport_v2 package. A normal clone needs no historical v2 files;
these comparisons skip explicitly when the optional reference is unavailable.
"""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from survot_rank.plugins.transport_only.mechanism import DCTTransportMechanism, TransportOnlyConfig


@pytest.fixture(scope="module")
def historical_reference():
    supplied = os.environ.get("DCT_TRANSPORT_ONLY_REFERENCE_ROOT")
    if not supplied:
        pytest.skip("optional historical reference: set DCT_TRANSPORT_ONLY_REFERENCE_ROOT")
    reference_root = Path(supplied).resolve()
    plugins_path = reference_root / "survot_rank/plugins"
    if not (plugins_path / "transport_v2/core.py").is_file():
        pytest.skip("optional historical reference has no transport_v2/core.py: " + str(reference_root))
    import survot_rank.plugins as plugins
    previous_path = plugins.__path__
    prefix = "survot_rank.plugins.transport_v2"
    previous_modules = {name: module for name, module in sys.modules.items()
                        if name == prefix or name.startswith(prefix + ".")}
    for name in previous_modules:
        sys.modules.pop(name)
    plugins.__path__ = [*previous_path, str(plugins_path)]
    try:
        config_module = importlib.import_module(prefix + ".config")
        core_module = importlib.import_module(prefix + ".core")
        actual = Path(inspect.getfile(core_module)).resolve()
        assert actual == (plugins_path / "transport_v2/core.py").resolve()
        yield config_module.TransportPluginV2Config, core_module.TransportCoreV2
    finally:
        plugins.__path__ = previous_path
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                sys.modules.pop(name)
        sys.modules.update(previous_modules)


@pytest.mark.parametrize("transport_mode,cost_mode", [("balanced", "legacy"), ("uot", "bounded")])
@pytest.mark.parametrize("plan_mode", ["learned", "independent"])
def test_extracted_solver_matches_historical_values_and_parameter_gradients(
        historical_reference, transport_mode, cost_mode, plan_mode):
    ReferenceConfig, ReferenceCore = historical_reference
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(913)
        common = dict(stages=2, geometries=("cosine", "euclidean", "positive_dot"),
            coordinate_temperature=.3, stage_cost_weight=.2, evidence_cost_weight=.15,
            evidence_mass_floor=.05, marginal_strength=.7, eps=.2, eps_start=.2,
            eps_anneal_epochs=12, sinkhorn_iters=30, projection_iters=3000,
            projection_tol=1e-9, tau_wsi=.5, tau_omics=.6, cost_scale=1.3,
            euclidean_scale=2.7, uot_iters=400, uot_tolerance=1e-10,
            transport_mode=transport_mode, cost_mode=cost_mode, plan_mode=plan_mode)
        reference = ReferenceCore(ReferenceConfig(wsi_dim=8, omics_dim=8, dim=8,
            wsi_slots=3, omics_slots=4, heads=2, event_layers=1, slot_iters=1,
            slot_heads=2, slot_dim_head=4, slot_hidden_dim=16, dropout=0., **common)).double()
        extracted = DCTTransportMechanism(TransportOnlyConfig(host_dim=8, slot_budget=4,
            **common)).double()
        extracted.stage_pair_cost.load_state_dict(reference.stage_pair_cost.state_dict())
        extracted.evidence_gate.load_state_dict(reference.evidence_gate.state_dict())
        extracted.stage_embedding.copy_(reference.stage_embedding)
        generator = torch.Generator().manual_seed(44)
        wsi = torch.randn(2, 3, 8, dtype=torch.float64, generator=generator)
        omics = torch.randn(2, 4, 8, dtype=torch.float64, generator=generator)
        values, gradients = [], []
        for model in (reference, extracted):
            w = wsi.clone().requires_grad_()
            o = omics.clone().requires_grad_()
            costs, rows, cols = model.costs_and_marginals(w, o)
            plan, returned_rows, returned_cols = model.plans_from_costs(costs, rows, cols, 3)
            values.append((costs, rows, cols, plan, returned_rows, returned_cols))
            # Nonseparable, asymmetric weights exercise coupling topology rather
            # than the constant total mass of a balanced plan.
            weights = torch.sin(torch.arange(plan.numel(), dtype=plan.dtype).reshape_as(plan) * .37)
            scalar = (plan * weights).sum() + .11 * (plan.square() * weights.cos()).sum()
            parameters = [parameter for name, parameter in model.named_parameters()
                          if name.startswith(("stage_pair_cost.", "evidence_gate."))]
            gradients.append(torch.autograd.grad(scalar, [*parameters, w, o], allow_unused=True))
        for actual, expected in zip(values[1], values[0]):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        assert len(gradients[0]) == len(gradients[1])
        nonzero = 0
        for actual, expected in zip(gradients[1], gradients[0]):
            if expected is None:
                assert actual is None
            else:
                assert actual is not None and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                nonzero += int(bool(actual.abs().max() > 0))
        assert nonzero > 0
    finally:
        torch.set_num_threads(previous_threads)


def test_default_pure_import_has_no_historical_v2_or_loss_dependency():
    code = ("import sys; from survot_rank.plugins.transport_only.mechanism import DCTTransportMechanism; "
            "assert not any(n == 'survot_rank.plugins.transport_v2' or "
            "n.startswith('survot_rank.plugins.transport_v2.') for n in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
