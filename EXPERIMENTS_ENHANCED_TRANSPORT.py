#!/usr/bin/env python3
"""
增强 DCT 传输正则化的实验设计

问题诊断:
- 当前: NLL + IPCW rank + 0.05 * direction_loss
- 结果: 预测性能很好 (C-index 0.72)，但传输机制很弱 (DCR 52.6%)

假设:
- 0.05 的权重太小，模型"忽略"了传输目标
- 增加权重可能让模型同时优化预测和传输

实验设计:
1. 固定 anchor (已验证有效，mono_inc_rate 100%)
2. 增加传输权重: 0.05 → {0.1, 0.2, 0.5, 1.0}
3. 对比 C-index 和 DCR 的变化

预期:
- 如果 DCR ↑ 且 C-index 不变或 ↑ → 传输机制确实被加强了
- 如果 C-index ↓ 但 DCR ↑ → 存在权衡
- 如果 C-index ↓ 且 DCR ↓ → 权重太大，破坏了预测能力
"""

# 实验配置
EXPERIMENTS = [
    {
        "name": "baseline",
        "transport_weight": 0.05,
        "fixed_anchor": False,
        "description": "当前配置 (baseline)"
    },
    {
        "name": "fixed_anchor_baseline", 
        "transport_weight": 0.05,
        "fixed_anchor": True,
        "description": "固定 anchor + 当前权重"
    },
    {
        "name": "fixed_anchor_weight_0.1",
        "transport_weight": 0.1,
        "fixed_anchor": True,
        "description": "固定 anchor + 2x 权重"
    },
    {
        "name": "fixed_anchor_weight_0.2",
        "transport_weight": 0.2,
        "fixed_anchor": True,
        "description": "固定 anchor + 4x 权重"
    },
    {
        "name": "fixed_anchor_weight_0.5",
        "transport_weight": 0.5,
        "fixed_anchor": True,
        "description": "固定 anchor + 10x 权重"
    },
    {
        "name": "fixed_anchor_weight_1.0",
        "transport_weight": 1.0,
        "fixed_anchor": True,
        "description": "固定 anchor + 20x 权重"
    },
]

# 关键指标
METRICS = {
    "c_index": "预测性能 (越高越好)",
    "dcr": "方向一致率 (越高越好)",
    "mono_inc_rate": "高风险干预单调性 (越高越好)",
    "mono_dec_rate": "低风险干预单调性 (越高越好)",
}

# 成功标准
SUCCESS_CRITERIA = {
    "c_index": 0.70,  # 不低于当前 0.72 太多
    "dcr": 0.70,      # 目标 DCR > 70%
    "mono_inc_rate": 0.90,
    "mono_dec_rate": 0.90,
}

if __name__ == "__main__":
    print("=" * 80)
    print("DCT 传输正则化增强实验设计")
    print("=" * 80)
    print()
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"实验 {i+1}: {exp['name']}")
        print(f"  描述: {exp['description']}")
        print(f"  传输权重: {exp['transport_weight']}")
        print(f"  固定 anchor: {exp['fixed_anchor']}")
        print()
    
    print("=" * 80)
    print("预期结果模式")
    print("=" * 80)
    print()
    print("模式 A: DCR ↑ 且 C-index 不变 → 最佳！传输机制被有效加强")
    print("模式 B: C-index ↓ 但 DCR ↑ → 存在权衡，可接受")
    print("模式 C: C-index ↓ 且 DCR ↓ → 权重太大，需要调小")
    print()
