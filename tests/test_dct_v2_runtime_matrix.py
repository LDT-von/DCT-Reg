"""Standalone release stress matrix, synthetic features only."""
from contextlib import nullcontext
import pytest
import torch
from survot_rank.research.methods.dct_v2 import DCTV2, DCTV2Config


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("transport", ["uot", "balanced"])
@pytest.mark.parametrize("plan", ["learned", "independent"])
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("geometry_count", [1, 2, 3])
def test_mixed_availability_and_unequal_slots_finite_backward(transport, plan, amp, geometry_count):
    torch.manual_seed(443)
    cfg = DCTV2Config(omic_sizes=(3,4), wsi_input_dim=6, dim=8, num_classes=3,
        wsi_slots=3, omics_slots=2, stages=2, heads=2, event_layers=1,
        slot_iters=1, slot_heads=2, slot_dim_head=4, slot_hidden_dim=16,
        dropout=0., transport_mode=transport, plan_mode=plan,
        geometries=("cosine","euclidean","positive_dot")[:geometry_count],
        relation_warmup_epochs=0, relation_ramp_epochs=1)
    model = DCTV2(cfg)
    model.set_training_reference(torch.arange(1,7,dtype=torch.float32), torch.tensor([0.,1.,0.,0.,1.,0.]))
    inputs = dict(x_wsi=torch.randn(3,5,6), x_omic1=torch.randn(3,3), x_omic2=torch.randn(3,4),
        wsi_mask=torch.tensor([[False]*5,[True]*5,[True,True,True,False,False]]),
        omics_mask=torch.tensor([[True,True],[False,False],[True,True]]),
        y=torch.tensor([0,1,2]), c=torch.tensor([0.,1.,0.]), event_time=torch.tensor([1.,2.,3.]), epoch=10)
    with torch.autocast("cpu",dtype=torch.bfloat16) if amp else nullcontext():
        result = model.forward_with_losses(**inputs)
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["logits"]).all()
    assert not model.last_state.plans[:2].any()
    torch.testing.assert_close(result["hazards"][:2],model.last_native_hazards[:2],atol=0.,rtol=0.)
    result["loss"].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
