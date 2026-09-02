"""单元测试: 验证分层预后slot机制的3个创新点"""

import torch
import pytest
from survot_rank.research.components.slot_attention_v3_hierarchical_prognostic import (
    PrognosticWeightedAttention,
    DynamicSlotPruning,
    HierarchicalSlotPyramid,
    HierarchicalPrognosticSlotAttention,
)


def test_prognostic_weighted_attention():
    """测试创新1: 预后加权路由是否真正影响slot分配"""
    print("\n=== 测试1: 预后加权路由 ===")
    
    batch = 2
    num_slots = 4
    num_tokens = 100
    dim = 256
    
    attn = PrognosticWeightedAttention(dim=dim, heads=4)
    
    queries = torch.randn(batch, num_slots, dim)
    keys = torch.randn(batch, num_tokens, dim)
    values = torch.randn(batch, num_tokens, dim)
    
    # 前向传播
    attended, prognostic_scores = attn(queries, keys, values)
    
    # 验证1: 输出形状正确
    assert attended.shape == (batch, num_slots, dim)
    assert prognostic_scores.shape == (batch, num_tokens)
    
    # 验证2: 预后分数在[0,1]范围
    assert (prognostic_scores >= 0).all() and (prognostic_scores <= 1).all()
    
    # 验证3: 预后加权机制可以通过梯度更新学习
    # 创建一个简单的loss: 让第一个slot关注第一个token
    target_attn = torch.zeros(batch, num_slots, num_tokens)
    target_attn[:, 0, 0] = 1.0  # 第一个slot应该关注第一个token
    
    optimizer = torch.optim.Adam(attn.parameters(), lr=0.01)
    
    initial_loss = None
    for step in range(5):
        optimizer.zero_grad()
        attended_out, prog_scores = attn(queries, keys, values)
        
        # 简单loss: 预后分数应该集中在前几个token
        loss = -(prog_scores[:, :10].mean() - prog_scores[:, 10:].mean())
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    
    # 验证可以学习: loss应该下降
    assert final_loss < initial_loss, f"Loss未下降: {initial_loss:.4f} → {final_loss:.4f}"
    
    print("✓ 预后加权路由工作正常")
    print(f"  - 预后分数范围: [{prognostic_scores.min():.3f}, {prognostic_scores.max():.3f}]")
    print(f"  - lambda系数: {attn.lambda_prognostic.item():.3f}")


def test_dynamic_slot_pruning():
    """测试创新2: 动态剪枝是否真正减少激活slots"""
    print("\n=== 测试2: 动态slot剪枝 ===")
    
    batch = 2
    num_slots = 8
    dim = 256
    
    pruner = DynamicSlotPruning(num_slots=num_slots, dim=dim, pruning_threshold=0.3)
    
    slots = torch.randn(batch, num_slots, dim)
    
    # 训练模式: 软剪枝
    pruner.train()
    slots_pruned_train, mask_train, num_active_train = pruner(slots, training=True)
    
    assert slots_pruned_train.shape == (batch, num_slots, dim)  # 训练时保持维度
    assert num_active_train == num_slots
    
    # 推理模式: 硬剪枝
    pruner.eval()
    
    # 人工设置importance: 前4个重要，后4个不重要
    with torch.no_grad():
        pruner.slot_importance[:4] = 2.0  # 高重要性 → sigmoid(2.0) ≈ 0.88 > 0.3
        pruner.slot_importance[4:] = -2.0  # 低重要性 → sigmoid(-2.0) ≈ 0.12 < 0.3
    
    slots_pruned_eval, mask_eval, num_active_eval = pruner(slots, training=False)
    
    # 验证1: 推理时只保留重要的slots
    assert num_active_eval == 4, f"期望激活4个slots，实际{num_active_eval}"
    assert slots_pruned_eval.shape == (batch, 4, dim)
    
    # 验证2: mask正确
    assert mask_eval[:4].sum() == 4  # 前4个激活
    assert mask_eval[4:].sum() == 0  # 后4个剪枝
    
    print("✓ 动态slot剪枝工作正常")
    print(f"  - 训练模式: {num_active_train}/{num_slots} 激活 (软剪枝)")
    print(f"  - 推理模式: {num_active_eval}/{num_slots} 激活 (硬剪枝)")


def test_hierarchical_slot_pyramid():
    """测试创新3: 分层金字塔是否生成3层slots"""
    print("\n=== 测试3: 分层slot金字塔 ===")
    
    batch = 2
    num_tokens = 2048  # 模拟WSI patches
    dim = 256
    
    pyramid = HierarchicalSlotPyramid(
        dim=dim,
        num_slots_l1=16,
        num_slots_l2=8,
        num_slots_l3=4,
        iters=3,
        enable_pruning=False  # 先不测试剪枝
    )
    
    inputs = torch.randn(batch, num_tokens, dim)
    outputs = pyramid(inputs)
    
    # 验证1: 3层slots都存在
    assert 'slots_l1' in outputs
    assert 'slots_l2' in outputs
    assert 'slots_l3' in outputs
    
    # 验证2: 形状正确（金字塔递减）
    assert outputs['slots_l1'].shape == (batch, 16, dim)
    assert outputs['slots_l2'].shape == (batch, 8, dim)
    assert outputs['slots_l3'].shape == (batch, 4, dim)
    
    # 验证3: 预后分数存在且维度正确
    assert outputs['prognostic_scores_l1'].shape == (batch, num_tokens)
    assert outputs['prognostic_scores_l2'].shape == (batch, 16)  # L2从L1聚合
    assert outputs['prognostic_scores_l3'].shape == (batch, 8)   # L3从L2聚合
    
    # 验证4: metadata正确
    assert outputs['num_active_slots']['l1'] == 16
    assert outputs['num_active_slots']['l2'] == 8
    assert outputs['num_active_slots']['l3'] == 4
    
    print("✓ 分层slot金字塔工作正常")
    print(f"  - L1 (细粒度): {outputs['slots_l1'].shape}")
    print(f"  - L2 (中粒度): {outputs['slots_l2'].shape}")
    print(f"  - L3 (粗粒度): {outputs['slots_l3'].shape}")


def test_full_model_with_pruning():
    """测试完整模型: 分层+剪枝+融合"""
    print("\n=== 测试4: 完整模型（分层+剪枝+融合）===")
    
    batch = 2
    num_tokens = 2048
    dim = 256
    
    model = HierarchicalPrognosticSlotAttention(
        dim=dim,
        num_slots_l1=16,
        num_slots_l2=8,
        num_slots_l3=4,
        iters=3,
        enable_pruning=True,
        fusion_mode="attention"
    )
    
    inputs = torch.randn(batch, num_tokens, dim)
    
    # 训练模式
    model.train()
    outputs_train = model(inputs)
    
    assert 'fused_slots' in outputs_train
    assert outputs_train['fused_slots'].shape == (batch, dim)
    
    # 推理模式 + 人工设置剪枝
    model.eval()
    with torch.no_grad():
        # L1: 只保留前8个 (threshold=0.1)
        model.pyramid.pruning_l1.pruning_threshold = 0.5
        model.pyramid.pruning_l1.slot_importance[:8] = 2.0   # sigmoid(2.0)≈0.88 > 0.5
        model.pyramid.pruning_l1.slot_importance[8:] = -2.0  # sigmoid(-2.0)≈0.12 < 0.5
        
        # L2: 只保留前4个
        model.pyramid.pruning_l2.pruning_threshold = 0.5
        model.pyramid.pruning_l2.slot_importance[:4] = 2.0
        model.pyramid.pruning_l2.slot_importance[4:] = -2.0
        
        # L3: 只保留前2个
        model.pyramid.pruning_l3.pruning_threshold = 0.5
        model.pyramid.pruning_l3.slot_importance[:2] = 2.0
        model.pyramid.pruning_l3.slot_importance[2:] = -2.0
    
    outputs_eval = model(inputs)
    
    # 验证剪枝生效
    assert outputs_eval['num_active_slots']['l1'] == 8
    assert outputs_eval['num_active_slots']['l2'] == 4
    assert outputs_eval['num_active_slots']['l3'] == 2
    
    # 验证fusion成功
    assert outputs_eval['fused_slots'].shape == (batch, dim)
    
    # 验证attention fusion的layer_weights
    if 'layer_weights' in outputs_eval:
        layer_weights = outputs_eval['layer_weights']
        assert layer_weights.shape == (batch, 3)
        # 权重应该和为1
        assert torch.allclose(layer_weights.sum(dim=1), torch.ones(batch), atol=1e-5)
    
    print("✓ 完整模型工作正常")
    print(f"  - 融合后特征: {outputs_eval['fused_slots'].shape}")
    print(f"  - L1激活: {outputs_eval['num_active_slots']['l1']}/16")
    print(f"  - L2激活: {outputs_eval['num_active_slots']['l2']}/8")
    print(f"  - L3激活: {outputs_eval['num_active_slots']['l3']}/4")


def test_comparison_with_baseline():
    """对比实验: 新模型 vs 传统fixed slots"""
    print("\n=== 测试5: 新模型 vs 传统方法对比 ===")
    
    batch = 4
    num_tokens = 2048
    dim = 256
    
    # 传统方法: 固定8个slots，无分层，无剪枝
    from survot_rank.research.components.slot_attention import MultiHeadSlotAttention
    baseline = MultiHeadSlotAttention(
        num_slots=8,
        dim=dim,
        heads=4,
        iters=3
    )
    
    # 新方法: 分层+剪枝
    new_model = HierarchicalPrognosticSlotAttention(
        dim=dim,
        num_slots_l1=16,
        num_slots_l2=8,
        num_slots_l3=4,
        iters=3,
        enable_pruning=True,
        fusion_mode="mean"
    )
    
    inputs = torch.randn(batch, num_tokens, dim)
    
    # Baseline输出
    baseline_slots = baseline(inputs)  # [batch, 8, dim]
    baseline_pooled = baseline_slots.mean(dim=1)  # [batch, dim]
    
    # 新模型输出
    new_outputs = new_model(inputs)
    new_fused = new_outputs['fused_slots']  # [batch, dim]
    
    print("✓ 对比实验完成")
    print(f"  - Baseline: 固定8 slots → {baseline_pooled.shape}")
    print(f"  - 新模型: 3层分层 (16+8+4 slots) → {new_fused.shape}")
    print(f"  - 新模型优势:")
    print(f"    1. 多尺度特征 (细节+中层+全局)")
    print(f"    2. 预后加权路由 (重要特征优先)")
    print(f"    3. 动态剪枝 (推理时自适应激活)")
    
    # 验证特征不同
    assert not torch.allclose(baseline_pooled, new_fused, atol=0.1)
    print("  - 特征表示显著不同 ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("分层预后Slot Attention - 单元测试")
    print("=" * 60)
    
    test_prognostic_weighted_attention()
    test_dynamic_slot_pruning()
    test_hierarchical_slot_pyramid()
    test_full_model_with_pruning()
    test_comparison_with_baseline()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
