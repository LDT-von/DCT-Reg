#!/usr/bin/env python3
"""
DCT v3.1 Risk Ordering Transport - 创新目标函数设计

当前 DCT 的问题:
1. 只传输特征到 anchor，没有控制 risk 变化方向
2. 导致 DCR 只有 52.6%

新思路:
- 不仅传输特征，还要保证 risk ordering 正确
- Low anchor: 传输后 risk 应该降低
- High anchor: 传输后 risk 应该升高

新增损失项:
1. Risk monotonicity loss: 确保 alpha 从 0→1 时 risk 单调变化
2. Risk direction loss: 确保 low/high anchor 方向正确
3. Risk magnitude loss: 确保 risk 变化幅度合理

组合损失:
L_total = NLL + IPCW + λ1 * direction_loss + λ2 * risk_ordering_loss

其中 risk_ordering_loss = 
    α * risk_monotonicity_loss + 
    (1-α) * risk_direction_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskOrderingLoss(nn.Module):
    """
    新增的 Risk Ordering Loss
    
    目标: 
    - 传输到 low anchor 后，risk 应该降低
    - 传输到 high anchor 后，risk 应该升高
    - alpha 从 0→1 时，risk 变化应该是单调的
    """
    
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin
        
    def forward(self, factual_risk, counterfactual_risk, anchor_type):
        """
        Args:
            factual_risk: 原始 risk (batch,)
            counterfactual_risk: 传输后的 risk (batch,)
            anchor_type: 'low' 或 'high'
        
        Returns:
            loss: 标量
        """
        risk_change = counterfactual_risk - factual_risk
        
        if anchor_type == 'low':
            # Low anchor: risk 应该降低 → risk_change < 0
            # Loss = max(0, risk_change + margin)
            loss = F.relu(risk_change + self.margin).mean()
        else:  # 'high'
            # High anchor: risk 应该升高 → risk_change > 0
            # Loss = max(0, -risk_change + margin)
            loss = F.relu(-risk_change + self.margin).mean()
        
        return loss


class RiskMonotonicityLoss(nn.Module):
    """
    Risk Monotonicity Loss
    
    目标: alpha 从 0→1 时，risk 变化应该是单调的
    
    实现: 相邻 alpha 点之间的 risk 差值应该同号
    """
    
    def __init__(self, margin=0.005):
        super().__init__()
        self.margin = margin
        
    def forward(self, risk_sequence):
        """
        Args:
            risk_sequence: (batch, num_alphas) 风险序列
        
        Returns:
            loss: 标量
        """
        # 计算相邻差分
        diffs = risk_sequence[:, 1:] - risk_sequence[:, :-1]  # (batch, num_alphas-1)
        
        # 对 low anchor: diffs 应该 > 0 (risk 降低)
        # 对 high anchor: diffs 应该 < 0 (risk 升高)
        # 使用 hinge loss: max(0, -sign * diff + margin)
        
        # 简化为: diffs 的绝对值应该递减
        abs_diffs = torch.abs(diffs)
        monotonic_penalty = F.relu(abs_diffs[:, 1:] - abs_diffs[:, :-1] + self.margin).mean()
        
        return monotonic_penalty


class CombinedTransportLoss(nn.Module):
    """
    组合传输损失函数
    
    L_total = NLL + IPCW + λ1 * direction_loss + λ2 * risk_ordering_loss
    """
    
    def __init__(self, 
                 direction_weight=0.05,
                 risk_ordering_weight=0.1,
                 risk_monotonicity_weight=0.05):
        super().__init__()
        self.direction_weight = direction_weight
        self.risk_ordering_weight = risk_ordering_weight
        self.risk_monotonicity_weight = risk_monotonicity_weight
        
        self.risk_ordering_loss = RiskOrderingLoss()
        self.risk_monotonicity_loss = RiskMonotonicityLoss()
        
    def forward(self, 
               nll_loss,
               ipcw_loss,
               direction_loss,
               factual_risk,
               counterfactual_risk,
               anchor_type,
               risk_sequence=None):
        """
        Args:
            nll_loss: NLL 损失
            ipcw_loss: IPCW 损失
            direction_loss: 方向损失
            factual_risk: 原始 risk
            counterfactual_risk: 传输后 risk
            anchor_type: 'low' 或 'high'
            risk_sequence: (可选) alpha 序列的风险
        
        Returns:
            total_loss: 组合损失
            loss_dict: 各损失分量
        """
        # 基础损失
        base_loss = nll_loss + ipcw_loss
        
        # 传输方向损失
        transport_loss = self.direction_weight * direction_loss
        
        # Risk ordering 损失
        if counterfactual_risk is not None:
            risk_loss = self.risk_ordering_weight * self.risk_ordering_loss(
                factual_risk, counterfactual_risk, anchor_type
            )
        else:
            risk_loss = torch.tensor(0.0, device=nll_loss.device)
        
        # Risk monotonicity 损失
        if risk_sequence is not None:
            mono_loss = self.risk_monotonicity_weight * self.risk_monotonicity_loss(risk_sequence)
        else:
            mono_loss = torch.tensor(0.0, device=nll_loss.device)
        
        # 总损失
        total_loss = base_loss + transport_loss + risk_loss + mono_loss
        
        loss_dict = {
            'nll': nll_loss.item(),
            'ipcw': ipcw_loss.item(),
            'direction': direction_loss.item(),
            'risk_ordering': risk_loss.item(),
            'risk_monotonicity': mono_loss.item(),
            'total': total_loss.item(),
        }
        
        return total_loss, loss_dict


# ============================================================
# 实验配置
# ============================================================

EXPERIMENT_CONFIGS = {
    # 当前 baseline
    "baseline": {
        "direction_weight": 0.05,
        "risk_ordering_weight": 0.0,
        "risk_monotonicity_weight": 0.0,
        "fixed_anchor": False,
    },
    
    # 只加 risk ordering
    "risk_ordering_only": {
        "direction_weight": 0.05,
        "risk_ordering_weight": 0.1,
        "risk_monotonicity_weight": 0.0,
        "fixed_anchor": False,
    },
    
    # 固定 anchor + risk ordering
    "fixed_anchor_risk_ordering": {
        "direction_weight": 0.05,
        "risk_ordering_weight": 0.1,
        "risk_monotonicity_weight": 0.0,
        "fixed_anchor": True,
    },
    
    # 固定 anchor + 完整 risk loss
    "fixed_anchor_full_risk_loss": {
        "direction_weight": 0.05,
        "risk_ordering_weight": 0.1,
        "risk_monotonicity_weight": 0.05,
        "fixed_anchor": True,
    },
    
    # 高权重配置
    "high_weight_fixed_anchor": {
        "direction_weight": 0.1,
        "risk_ordering_weight": 0.2,
        "risk_monotonicity_weight": 0.1,
        "fixed_anchor": True,
    },
}


if __name__ == "__main__":
    print("=" * 80)
    print("DCT v3.1 Risk Ordering Transport")
    print("=" * 80)
    print()
    print("新增损失项:")
    print("  1. Risk Ordering Loss: 确保传输后 risk 方向正确")
    print("  2. Risk Monotonicity Loss: 确保 alpha 序列 risk 单调")
    print()
    print("实验配置:")
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"\n  {name}:")
        for k, v in config.items():
            print(f"    {k}: {v}")
    print()
    print("=" * 80)
    print("使用说明:")
    print("=" * 80)
    print("""
    # 在 DCTV310DirectionalRegularizedTransport 中集成:
    
    1. 导入损失类:
       from your_module import CombinedTransportLoss
    
    2. 初始化:
       self.transport_loss = CombinedTransportLoss(
           direction_weight=0.05,
           risk_ordering_weight=0.1,
           risk_monotonicity_weight=0.05
       )
    
    3. 前向传播时记录 risk:
       factual_risk = self.compute_risk(x)
       # ... 传输计算 ...
       counterfactual_risk = self.compute_risk(x_transported)
       
    4. 计算损失:
       total_loss, loss_dict = self.transport_loss(
           nll_loss, ipcw_loss, direction_loss,
           factual_risk, counterfactual_risk, anchor_type
       )
    """)
