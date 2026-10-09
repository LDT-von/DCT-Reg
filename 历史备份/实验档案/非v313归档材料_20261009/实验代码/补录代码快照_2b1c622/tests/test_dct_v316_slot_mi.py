"""Test suite for DCT v3.16 Slot-based MI Decomposition.

Tests the standalone components without depending on the full survot_rank framework.
"""

from __future__ import annotations

import pytest
import torch

from survot_rank.config import config_to_argv, load_config

from survot_rank.research.methods.dct_v316_slot_mi_interaction.model import (
    MultiHeadSlotAttention,
    MoESlotDecoder,
    IterativeCrossAttention,
    IterativeCrossAttTransformer,
    Transformer,
    InfoNCECritic,
    InteractionProfiler,
    SlotBasedMIDecompositionBlock,
    gumbel_topk_st,
    parallel_topk_st,
)


def framework_args():
    from survot_rank.training.extended_args import build_base_parser

    config = load_config("configs/dct_v316_slot_mi_decomposition.yaml")
    return build_base_parser().parse_args(config_to_argv(config))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def dim():
    return 32


@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def wsi_tokens(batch_size, dim):
    """Simulated WSI tokens: [B, N, D]."""
    return torch.randn(batch_size, 10, dim)


@pytest.fixture
def omic_tokens(batch_size, dim):
    """Simulated omics tokens: [B, M, D]."""
    return torch.randn(batch_size, 5, dim)


@pytest.fixture
def survival_proxy(batch_size, hidden_dim):
    """Simulated survival proxy: [B, hidden_dim]."""
    return torch.randn(batch_size, hidden_dim)


# ============================================================================
# Test: Top-K Functions
# ============================================================================

def test_gumbel_topk_st_shape():
    logits = torch.randn(8, 16)
    k = 4
    hard_hot, indices = gumbel_topk_st(logits, k=k)
    assert hard_hot.shape == logits.shape
    assert indices.shape == (8, k)
    assert torch.allclose(hard_hot.sum(dim=-1), torch.full((8,), float(k)), atol=1e-6)


def test_parallel_topk_st_shape():
    logits = torch.randn(8, 16)
    k = 4
    hard_hot, indices = parallel_topk_st(logits, k=k)
    assert hard_hot.shape == logits.shape
    assert indices.shape == (8, k)


# ============================================================================
# Test: MultiHead Slot Attention
# ============================================================================

def test_multihead_slot_attention_forward_shape(wsi_tokens, dim):
    num_slots = 8
    model = MultiHeadSlotAttention(num_slots=num_slots, dim=dim, heads=4, iters=3)
    slots = model(wsi_tokens)
    assert slots.shape == (wsi_tokens.size(0), num_slots, dim)


def test_multihead_slot_attention_dynamic_num_slots(wsi_tokens, dim):
    model = MultiHeadSlotAttention(num_slots=8, dim=dim, heads=4, iters=3)
    slots = model(wsi_tokens, num_slots=4)
    assert slots.shape == (wsi_tokens.size(0), 4, dim)


def test_multihead_slot_attention_training_mode(wsi_tokens, dim):
    """In training mode, slots are stochastic (reparameterization)."""
    torch.manual_seed(0)
    model = MultiHeadSlotAttention(num_slots=8, dim=dim, heads=4, iters=3)
    model.train()
    slots1 = model(wsi_tokens)
    slots2 = model(wsi_tokens)
    # Different due to stochastic Gaussian initialization
    assert not torch.allclose(slots1, slots2, atol=1e-5)


def test_multihead_slot_attention_eval_is_deterministic_without_slot_collapse(
    wsi_tokens, dim
):
    torch.manual_seed(5)
    model = MultiHeadSlotAttention(num_slots=8, dim=dim, heads=4, iters=3).eval()
    first = model(wsi_tokens)
    second = model(wsi_tokens)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.std(dim=1).mean() > 1e-4
    with pytest.raises(ValueError, match="num_slots"):
        model(wsi_tokens, num_slots=9)


# ============================================================================
# Test: MoE Slot Decoder
# ============================================================================

def test_moe_decoder_forward_shapes(dim):
    num_slots = 8
    num_classes = 4
    batch_size = 4
    slots = torch.randn(batch_size, num_slots, dim)

    model = MoESlotDecoder(dim=dim, num_slots=num_slots, num_classes=num_classes)
    logits, slot_gate, hard_keep = model(slots)

    assert logits.shape == (batch_size, num_classes)
    assert slot_gate.shape == (batch_size, num_slots)
    assert hard_keep.shape == (batch_size, num_slots)
    # Verify hard_keep selects k slots
    k = max(1, int(num_slots * 0.25))
    assert torch.allclose(hard_keep.sum(dim=-1), torch.full((batch_size,), float(k)), atol=1e-6)


def test_moe_decoder_topk_selection(dim):
    num_slots = 8
    num_classes = 4
    batch_size = 4
    slots = torch.randn(batch_size, num_slots, dim)

    model = MoESlotDecoder(
        dim=dim,
        num_slots=num_slots,
        num_classes=num_classes,
        topk_ratio=0.25,
    )

    logits, slot_gate, hard_keep = model(slots)
    k = max(1, int(num_slots * 0.25))
    assert torch.allclose(hard_keep.sum(dim=-1), torch.full((batch_size,), float(k)), atol=1e-6)


def test_moe_decoder_eval_is_deterministic(dim):
    torch.manual_seed(17)
    model = MoESlotDecoder(dim=dim, num_slots=8, num_classes=4).eval()
    slots = torch.randn(4, 8, dim)
    first = model(slots)
    second = model(slots)
    for lhs, rhs in zip(first, second):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


# ============================================================================
# Test: Iterative Cross-Attention
# ============================================================================

def test_iterative_cross_attention_forward_shapes(dim):
    batch_size = 4
    num_wsi_slots = 8
    num_omic_slots = 4

    x1 = torch.randn(batch_size, num_wsi_slots, dim)
    x2 = torch.randn(batch_size, num_omic_slots, dim)

    model = IterativeCrossAttention(dim=dim, num_heads=4, iters=3)
    out1, out2 = model(x1, x2)

    assert out1.shape == x1.shape
    assert out2.shape == x2.shape


def test_iterative_cross_attention_wrapper(dim):
    batch_size = 4
    num_wsi_slots = 8
    num_omic_slots = 4

    x1 = torch.randn(batch_size, num_wsi_slots, dim)
    x2 = torch.randn(batch_size, num_omic_slots, dim)

    model = IterativeCrossAttTransformer(dim=dim, num_heads=4, iters=3)
    output = model(x1, x2)

    assert output.shape == (batch_size, num_wsi_slots + num_omic_slots, dim)


def test_iterative_cross_attention_dynamic_key_values(dim):
    model = IterativeCrossAttention(
        dim=dim, num_heads=4, iters=2, static_kv=False, attn_drop=0.0
    )
    x1 = torch.randn(3, 8, dim, requires_grad=True)
    x2 = torch.randn(3, 4, dim, requires_grad=True)
    out1, out2 = model(x1, x2)
    assert out1.shape == x1.shape
    assert out2.shape == x2.shape
    (out1.square().mean() + out2.square().mean()).backward()
    assert torch.isfinite(x1.grad).all()
    assert torch.isfinite(x2.grad).all()


# ============================================================================
# Test: Transformer
# ============================================================================

def test_transformer_forward_shape(dim):
    batch_size = 4
    num_slots = 8
    slots = torch.randn(batch_size, num_slots, dim)

    model = Transformer(dim=dim, num_heads=4)
    output = model(slots)

    assert output.shape == slots.shape


def test_transformer_with_mask(dim):
    batch_size = 4
    num_slots = 8
    slots = torch.randn(batch_size, num_slots, dim)

    mask = torch.ones(batch_size, num_slots, dtype=torch.bool)
    mask[:, 4:] = False  # Mask out second half

    model = Transformer(dim=dim, num_heads=4)
    output = model(slots, mask=mask)

    assert output.shape == slots.shape


# ============================================================================
# Test: InfoNCE Critic
# ============================================================================

def test_info_nce_critic_shapes(hidden_dim):
    batch_size = 8
    z = torch.randn(batch_size, hidden_dim)
    y = torch.randn(batch_size, hidden_dim)

    model = InfoNCECritic(dim=hidden_dim, temperature=0.1)
    loss = model(z, y)

    assert loss.shape == ()
    assert loss >= 0


def test_info_nce_uses_same_patient_diagonal_targets():
    model = InfoNCECritic(dim=4, target_dim=4, temperature=0.1)
    model.projection = torch.nn.Identity()
    model.target_projection = torch.nn.Identity()
    identity = torch.eye(4)
    loss = model(identity, identity)
    expected = torch.nn.functional.cross_entropy(
        identity @ identity.T / 0.1, torch.arange(4)
    )
    torch.testing.assert_close(loss, expected)
    assert float(loss) < 0.001


# ============================================================================
# Test: Interaction Profiler
# ============================================================================

def test_interaction_profiler_shapes(wsi_tokens, omic_tokens, dim, hidden_dim):
    """Test profiler with matching dim and hidden_dim properly."""
    model = InteractionProfiler(dim=dim, hidden_dim=hidden_dim, prediction_dim=4)
    model.train()
    prediction = torch.randn(wsi_tokens.size(0), 4)
    profile, context, metrics = model(wsi_tokens, omic_tokens, prediction)

    assert profile.shape == (wsi_tokens.size(0), 4)
    assert context.shape == (wsi_tokens.size(0), hidden_dim)
    assert torch.allclose(profile.sum(dim=-1), torch.ones(profile.size(0)), atol=1e-5)
    assert "L_R" in metrics
    assert "L_Up" in metrics
    assert "L_Ug" in metrics
    assert "L_S" in metrics
    assert "L_total" in metrics


def test_interaction_profiler_eval_mode(wsi_tokens, omic_tokens, dim):
    model = InteractionProfiler(dim=dim, hidden_dim=64)
    model.eval()

    profile, context, metrics = model(wsi_tokens, omic_tokens, survival_proxy=None)

    assert profile.shape == (wsi_tokens.size(0), 4)
    assert context.shape == (wsi_tokens.size(0), 64)
    assert len(metrics) == 0


# ============================================================================
# Test: Full Slot-based MI Decomposition Block
# ============================================================================

def test_slot_mi_block_forward_shapes(wsi_tokens, omic_tokens, dim, hidden_dim):
    model = SlotBasedMIDecompositionBlock(
        dim=dim,
        num_wsi_slots=8,
        num_omic_slots=4,
        slot_iters=3,
        cross_iters=3,
        num_heads=4,
        hidden_dim=hidden_dim,
    )

    model.train()
    output, metrics = model(wsi_tokens, omic_tokens)

    batch_size = wsi_tokens.size(0)
    assert output["logits"].shape == (batch_size, 4)
    assert output["profile"].shape == (batch_size, 4)
    assert output["slots_wsi"].shape == (batch_size, 8, dim)
    assert output["slots_omic"].shape == (batch_size, 4, dim)
    assert "L_total" in metrics


def test_slot_mi_block_eval_mode(wsi_tokens, omic_tokens, dim):
    model = SlotBasedMIDecompositionBlock(
        dim=dim,
        num_wsi_slots=8,
        num_omic_slots=4,
        hidden_dim=64,
    )
    model.eval()

    output, metrics = model(wsi_tokens, omic_tokens, survival_proxy=None)

    assert output["profile"] is not None
    assert set(metrics) == {"v316_channel_delta_rms"}


def test_slot_mi_block_produces_different_profiles(batch_size, dim, hidden_dim):
    """Test that different inputs produce different profiles."""
    torch.manual_seed(42)
    wsi_tokens1 = torch.randn(batch_size, 10, dim)
    torch.manual_seed(43)
    wsi_tokens2 = torch.randn(batch_size, 10, dim) * 10
    torch.manual_seed(44)
    omic_tokens = torch.randn(batch_size, 5, dim)
    torch.manual_seed(45)
    torch.manual_seed(46)
    model = SlotBasedMIDecompositionBlock(dim=dim, hidden_dim=hidden_dim)
    model.eval()

    output1, _ = model(wsi_tokens1, omic_tokens)
    output2, _ = model(wsi_tokens2, omic_tokens)

    assert not torch.allclose(output1["profile"], output2["profile"], atol=1e-5)


def test_eval_is_deterministic_and_profile_is_batch_invariant(dim, hidden_dim):
    torch.manual_seed(91)
    model = SlotBasedMIDecompositionBlock(dim=dim, hidden_dim=hidden_dim).eval()
    patient_wsi = torch.randn(1, 10, dim)
    patient_omic = torch.randn(1, 5, dim)
    companion_wsi = torch.randn(1, 10, dim)
    companion_omic = torch.randn(1, 5, dim)

    single_a, _ = model(patient_wsi, patient_omic)
    single_b, _ = model(patient_wsi, patient_omic)
    paired, _ = model(
        torch.cat([patient_wsi, companion_wsi]),
        torch.cat([patient_omic, companion_omic]),
    )

    torch.testing.assert_close(single_a["logits"], single_b["logits"])
    torch.testing.assert_close(single_a["profile"], single_b["profile"])
    torch.testing.assert_close(single_a["profile"][0], paired["profile"][0])
    assert not torch.allclose(single_a["profile"], torch.full((1, 4), 0.25))


# ============================================================================
# Test: Gradient Flow
# ============================================================================

def test_slot_attention_gradient_flow(wsi_tokens, dim):
    model = MultiHeadSlotAttention(num_slots=8, dim=dim, heads=4, iters=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    slots = model(wsi_tokens)
    loss = slots.sum()
    loss.backward()
    optimizer.step()

    assert model.slots_mu.grad is not None
    assert model.slots_logsigma.grad is not None


def test_full_block_gradient_flow(wsi_tokens, omic_tokens, dim, hidden_dim):
    model = SlotBasedMIDecompositionBlock(
        dim=dim,
        num_wsi_slots=8,
        num_omic_slots=4,
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    output, metrics = model(wsi_tokens, omic_tokens)
    loss = output["logits"].sum() + metrics["L_total"]
    loss.backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.slot_attention_wsi.slots_mu.grad is not None
    assert model.moe_decoder_wsi.decoder.weight.grad is not None
    assert model.moe_decoder_wsi.pred_keep_slot[0].weight.grad is not None
    assert model.moe_decoder_omic.decoder.weight.grad is not None
    assert model.moe_decoder_omic.pred_keep_slot[0].weight.grad is not None
    assert model.profiler.proj_R[0].weight.grad is not None
    assert model.profiler.profile_heads[0].weight.grad is not None


def test_framework_config_factory_objective_and_backward():
    from survot_rank.research.methods.catalog import METHOD_CATALOG
    from survot_rank.training.model_factory import get_model
    from survot_rank.training.train_runner import compose_batch_objective, init_loss_function

    args = framework_args()
    assert args.v316_lambda_mi == pytest.approx(0.01)
    assert args.v316_hidden_dim == 256
    assert args.v316_profile_temp == pytest.approx(1.0)
    assert METHOD_CATALOG["dct_v316_slot_mi_decomposition"].status == "candidate"

    # Small dimensions exercise the real registry, constructor, NLL, IPCW
    # reference path, and objective combiner without a production-size tensor.
    args.encoding_dim = 16
    args.wsi_projection_dim = 16
    args.rna_format = "RNASeq"
    args.omic_sizes = None
    args.slot_num_wsi = 3
    args.slot_num_omics = 3
    args.slot_iters = 2
    args.otehv2_heads = 2
    args.v316_hidden_dim = 16
    args.cur_epoch = 0

    model = get_model("dct_v316", args, omic_input_dim=20)
    reference_times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    reference_c = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    model.configure_train_reference(reference_times, reference_c)
    model.train()

    generator = torch.Generator().manual_seed(31)
    payload = {
        "x_wsi": torch.randn(8, 6, 16, generator=generator),
        "x_omics": torch.randn(8, 5, 20, generator=generator),
        "y": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        "event_time": reference_times,
        "c": reference_c,
    }
    logits, auxiliary = model(**payload)
    raw_nll = init_loss_function(args)(
        logits, payload["y"], payload["event_time"], payload["c"]
    )
    total = compose_batch_objective(raw_nll, auxiliary, len(logits))
    expected = (
        raw_nll / len(logits)
        + 0.10 * model.last_training_losses["ipcw_rank"]
        + 0.01 * model.last_mi_metrics["L_total"]
    )
    torch.testing.assert_close(total, expected)
    assert logits.shape == (8, 4)
    assert torch.isfinite(total)
    assert model.objective_weights() == {
        "nll": 1.0,
        "ipcw_rank": 0.10,
        "channel_contrastive": 0.01,
    }

    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    for parameter in (
        model.slot_mi_block.moe_decoder_wsi.decoder.weight,
        model.slot_mi_block.moe_decoder_wsi.pred_keep_slot[0].weight,
        model.slot_mi_block.profiler.profile_heads[0].weight,
        model.slot_mi_block.channel_to_logits.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    model.eval()
    prediction_inputs = {
        "x_wsi": payload["x_wsi"],
        "x_omics": payload["x_omics"],
    }
    eval_first, eval_aux = model(**prediction_inputs)
    eval_second, _ = model(**prediction_inputs)
    torch.testing.assert_close(eval_first, eval_second, rtol=0, atol=0)
    assert eval_aux.item() == 0.0
    assert all(
        not value.requires_grad
        for value in model.last_explanations.values()
        if torch.is_tensor(value)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("amp", [False, True])
def test_config_sized_gpu_forward_backward(amp):
    from survot_rank.training.model_factory import get_model
    from survot_rank.training.train_runner import compose_batch_objective, init_loss_function

    args = framework_args()
    args.rna_format = "Pathways"
    args.omic_sizes = [3, 4, 5]
    args.cur_epoch = 0
    model = get_model("dct_v316", args).cuda().train()
    reference_times = torch.tensor([1.0, 2.0, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    reference_c = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    model.configure_train_reference(reference_times, reference_c)

    generator = torch.Generator().manual_seed(71)
    payload = {
        "x_wsi": torch.randn(2, 2048, 1536, generator=generator).cuda(),
        "y": torch.tensor([0, 3], device="cuda"),
        "event_time": torch.tensor([1.0, 8.0], device="cuda"),
        "c": torch.tensor([0.0, 0.0], device="cuda"),
    }
    for index, width in enumerate(args.omic_sizes, 1):
        payload[f"x_omic{index}"] = torch.randn(
            2, width, generator=generator
        ).cuda()
    with torch.autocast("cuda", enabled=amp, dtype=torch.float16):
        logits, auxiliary = model(**payload)
        raw_nll = init_loss_function(args)(
            logits, payload["y"], payload["event_time"], payload["c"]
        )
        total = compose_batch_objective(raw_nll, auxiliary, len(logits))
    total.backward()

    assert logits.shape == (2, 4)
    assert torch.isfinite(total)
    for parameter in model.slot_mi_block.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


# ============================================================================
# Test: Numerical Stability
# ============================================================================

def test_no_nan_in_forward(wsi_tokens, omic_tokens, dim, hidden_dim):
    model = SlotBasedMIDecompositionBlock(dim=dim, hidden_dim=hidden_dim)

    output, metrics = model(wsi_tokens, omic_tokens)

    for key, value in output.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f"NaN in {key}"

    for key, value in metrics.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f"NaN in metrics[{key}]"


def test_no_nan_with_extreme_inputs(dim, hidden_dim):
    """Test with extreme input values."""
    model = SlotBasedMIDecompositionBlock(dim=dim, hidden_dim=hidden_dim)
    model.eval()

    wsi_tokens = torch.randn(4, 10, dim) * 100
    omic_tokens = torch.randn(4, 5, dim) * 100
    output, _ = model(wsi_tokens, omic_tokens)

    for key, value in output.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f"NaN with extreme inputs in {key}"


# ============================================================================
# Test: Module Counts
# ============================================================================

def test_model_parameter_count(dim):
    model = SlotBasedMIDecompositionBlock(
        dim=dim,
        num_wsi_slots=8,
        num_omic_slots=4,
        hidden_dim=64,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert num_params > 0
    assert num_trainable > 0
    assert num_params == num_trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
