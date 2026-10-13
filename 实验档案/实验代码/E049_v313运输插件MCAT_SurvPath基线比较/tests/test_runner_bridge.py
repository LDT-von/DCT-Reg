"""Synthetic contract checks; no patient data or training is run."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from runner_bridge import NativeRunnerBridge
from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import _process_data_and_forward


class FakeAdapter(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.last_plugin_output = None
        self.seen = None

    def forward_with_aux(self, **kwargs):
        self.seen = kwargs
        self.last_plugin_output = SimpleNamespace(aux_losses={"weighted_total": self.weight.square()})
        logits = self.weight.expand(1, 4)
        output = (torch.sigmoid(logits), None, None, {}) if self.baseline == "mcat" else logits
        return output, self.weight.square() if kwargs["plugin_compute_aux"] else self.weight * 0


@pytest.mark.parametrize("baseline", ["mcat", "survpath"])
def test_native_batch_bridge_preserves_gradient_and_single_patient_contract(baseline):
    adapter = FakeAdapter(baseline)
    bridge = NativeRunnerBridge(adapter, baseline, [3, 2])
    logits, aux = bridge(x_wsi=torch.ones(1, 4, 1536), x_omic1=torch.ones(1, 3),
                         x_omic2=torch.ones(1, 2), cur_epoch=7)
    assert logits.shape == (1, 4)
    assert torch.allclose(logits, adapter.weight.expand(1, 4), atol=1e-6)
    assert adapter.seen["x_path"].shape == ((4, 1536) if baseline == "mcat" else (1, 4, 1536))
    assert adapter.seen["x_omic1"].shape == (3,)
    assert adapter.seen["cur_epoch"] == 7
    assert adapter.seen["plugin_compute_aux"] is True
    (logits.mean() + aux).backward()
    assert adapter.weight.grad is not None and torch.isfinite(adapter.weight.grad)
    bridge.eval()
    _, eval_aux = bridge(x_wsi=torch.ones(1, 4, 1536), x_omic1=torch.ones(1, 3),
                         x_omic2=torch.ones(1, 2))
    assert eval_aux.item() == 0
    assert adapter.seen["plugin_compute_aux"] is False


def test_bridge_rejects_wrong_batch_and_group_width():
    bridge = NativeRunnerBridge(FakeAdapter("mcat"), "mcat", [3])
    with pytest.raises(ValueError, match="one WSI bag"):
        bridge(x_wsi=torch.ones(2, 4, 1536), x_omic1=torch.ones(2, 3))
    with pytest.raises(ValueError, match="x_omic1"):
        bridge(x_wsi=torch.ones(1, 4, 1536), x_omic1=torch.ones(1, 4))
    with pytest.raises(ValueError, match="Empty WSI bag"):
        bridge(x_wsi=torch.zeros(1, 0, 1536), x_omic1=torch.ones(1, 3))


def test_existing_trainer_input_is_translated_to_native_contract():
    adapter = FakeAdapter("mcat")
    bridge = NativeRunnerBridge(adapter, "mcat", [3, 2])
    data = [torch.ones(1, 4, 1536),
            [[torch.ones(3), torch.ones(2)]],
            torch.tensor([1]), torch.tensor([12.0]), torch.tensor([0.0])]
    args = SimpleNamespace(rna_format="Pathways", cur_epoch=2)
    (logits, aux), y, event_time, censor = _process_data_and_forward(
        args, bridge, data, torch.device("cpu"))
    assert logits.shape == (1, 4)
    assert aux.requires_grad
    assert adapter.seen["x_omic1"].shape == (3,)
    assert y.item() == 1 and event_time.item() == 12.0 and censor.item() == 0.0
