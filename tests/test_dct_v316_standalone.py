"""Standalone test for DCT v3.16 components - no framework dependencies.

This test imports ONLY the model.py file directly.
"""

import sys
import os

# Ensure the package root is in path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import torch
import torch.nn as nn

# Direct import from model.py file
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


def test_gumbel_topk_st_shape():
    print("Testing gumbel_topk_st_shape...")
    logits = torch.randn(8, 16)
    k = 4
    hard_hot, indices = gumbel_topk_st(logits, k=k)
    assert hard_hot.shape == logits.shape
    assert indices.shape == (8, k)
    print(f"  hard_hot sum per row: {hard_hot.sum(dim=-1)}")
    print("  PASSED")


def test_multihead_slot_attention():
    print("Testing MultiHeadSlotAttention...")
    batch_size, num_tokens, dim = 4, 10, 32
    num_slots = 8

    tokens = torch.randn(batch_size, num_tokens, dim)
    model = MultiHeadSlotAttention(
        num_slots=num_slots,
        dim=dim,
        heads=4,
        iters=3
    )

    slots = model(tokens)
    assert slots.shape == (batch_size, num_slots, dim)
    print(f"  Input: {tokens.shape} -> Output: {slots.shape}")
    print("  PASSED")


def test_moe_decoder():
    print("Testing MoESlotDecoder...")
    batch_size, num_slots, dim = 4, 8, 32
    num_classes = 4

    slots = torch.randn(batch_size, num_slots, dim)
    model = MoESlotDecoder(
        dim=dim,
        num_slots=num_slots,
        num_classes=num_classes
    )

    logits, slot_gate, hard_keep = model(slots)

    assert logits.shape == (batch_size, num_classes)
    assert slot_gate.shape == (batch_size, num_slots)
    assert hard_keep.shape == (batch_size, num_slots)
    print(f"  Slots: {slots.shape} -> Logits: {logits.shape}")
    print(f"  hard_keep sum per row: {hard_keep.sum(dim=-1)}")
    print("  PASSED")


def test_cross_attention():
    print("Testing IterativeCrossAttention...")
    batch_size = 4
    num_wsi_slots = 8
    num_omic_slots = 4
    dim = 32

    x1 = torch.randn(batch_size, num_wsi_slots, dim)
    x2 = torch.randn(batch_size, num_omic_slots, dim)

    model = IterativeCrossAttention(dim=dim, num_heads=4, iters=3)
    out1, out2 = model(x1, x2)

    assert out1.shape == x1.shape
    assert out2.shape == x2.shape
    print(f"  WSI: {x1.shape} -> {out1.shape}")
    print(f"  Omic: {x2.shape} -> {out2.shape}")
    print("  PASSED")


def test_interaction_profiler():
    print("Testing InteractionProfiler...")
    batch_size = 4
    dim = 32
    hidden_dim = 64

    slots_wsi = torch.randn(batch_size, 8, dim)
    slots_omic = torch.randn(batch_size, 4, dim)
    survival_proxy = torch.randn(batch_size, 4)

    model = InteractionProfiler(dim=dim, hidden_dim=hidden_dim, prediction_dim=4)
    model.train()

    profile, context, metrics = model(slots_wsi, slots_omic, survival_proxy)

    assert profile.shape == (batch_size, 4)
    assert context.shape == (batch_size, hidden_dim)
    assert "L_total" in metrics
    print(f"  Profile shape: {profile.shape}")
    print(f"  L_total: {metrics['L_total'].item():.4f}")
    print("  PASSED")


def test_full_block():
    print("Testing SlotBasedMIDecompositionBlock...")
    batch_size = 4
    wsi_tokens = torch.randn(batch_size, 10, 32)
    omic_tokens = torch.randn(batch_size, 5, 32)
    hidden_dim = 64
    model = SlotBasedMIDecompositionBlock(
        dim=32,
        num_wsi_slots=8,
        num_omic_slots=4,
        hidden_dim=hidden_dim,
    )
    model.train()

    output, metrics = model(wsi_tokens, omic_tokens)

    assert output["logits"].shape == (batch_size, 4)
    assert output["profile"].shape == (batch_size, 4)
    assert output["slots_wsi"].shape == (batch_size, 8, 32)
    assert output["slots_omic"].shape == (batch_size, 4, 32)
    assert "L_total" in metrics

    print(f"  Logits: {output['logits'].shape}")
    print(f"  Profile: {output['profile'][:2]}")
    print(f"  L_total: {metrics['L_total'].item():.4f}")
    print("  PASSED")


def test_gradient_flow():
    print("Testing gradient flow...")
    batch_size = 2
    wsi_tokens = torch.randn(batch_size, 10, 32)
    omic_tokens = torch.randn(batch_size, 5, 32)
    hidden_dim = 64
    model = SlotBasedMIDecompositionBlock(dim=32, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    output, metrics = model(wsi_tokens, omic_tokens)
    loss = output["logits"].sum() + metrics["L_total"]
    loss.backward()
    optimizer.step()

    # Check gradients exist
    assert model.slot_attention_wsi.slots_mu.grad is not None
    assert model.profiler.proj_R[0].weight.grad is not None
    print("  Gradients computed successfully")
    print("  PASSED")


def test_numerical_stability():
    print("Testing numerical stability...")
    model = SlotBasedMIDecompositionBlock(dim=32, hidden_dim=64)
    model.eval()

    # Normal inputs
    wsi_tokens = torch.randn(4, 10, 32)
    omic_tokens = torch.randn(4, 5, 32)
    output, _ = model(wsi_tokens, omic_tokens)

    has_nan = False
    for key, value in output.items():
        if torch.is_tensor(value):
            if not torch.isfinite(value).all():
                has_nan = True
                print(f"  WARNING: NaN in {key}")

    assert not has_nan
    print("  No NaN values with normal inputs")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("DCT v3.16 Standalone Component Tests")
    print("=" * 60)

    test_gumbel_topk_st_shape()
    test_multihead_slot_attention()
    test_moe_decoder()
    test_cross_attention()
    test_interaction_profiler()
    test_full_block()
    test_gradient_flow()
    test_numerical_stability()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
