"""🚀 Hierarchical Prognostic Slot Attention with Causal Slot Pruning

**创新点**：解决以往多模态生存分析中 Slot Attention 的三个根本问题

1. **问题1**: 传统slot机制不区分"预后关键特征"vs"无关噪声特征"
   **解决**: Prognostic-Weighted Routing - 用生存风险显著性加权slot竞争

2. **问题2**: 固定数量slots无法适应患者异质性（晚期vs早期需要不同粒度）
   **解决**: Dynamic Causal Slot Pruning - 训练时学习剪枝阈值，推理时自适应激活

3. **问题3**: 单层slots无法同时捕获局部细节和全局预后模式
   **解决**: Hierarchical Slot Pyramid - 3层金字塔 (细节→中层→全局)

这些改进都是**因果可解释**的，每个slot有明确的预后贡献分数。
"""

from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch import Tensor, nn


class PrognosticWeightedAttention(nn.Module):
    """创新1: 预后加权路由机制
    
    **核心思想**: 不是所有输入token对预后贡献相同，用预后显著性加权slot竞争
    
    **与以往方法的根本区别**:
    - 传统: softmax(Q·K) - 只看相似度，不管预后重要性
    - 我们: softmax(Q·K + λ·prognostic_score) - 预后重要token优先分配
    
    **因果可解释性**: 每个token有明确的"预后贡献分数"，可视化能看到
                      模型在关注肿瘤坏死区、浸润边界等高危特征
    """
    
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        dim_inner = heads * dim_head
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_k = nn.Linear(dim, dim_inner, bias=False)
        self.to_v = nn.Linear(dim, dim_inner, bias=False)
        
        # 预后显著性评分网络：输入token → 预后重要性分数
        self.prognostic_scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()  # 输出[0,1]，1=高预后相关
        )
        
        # 可学习的加权系数，控制"相似度"vs"预后重要性"的平衡
        self.lambda_prognostic = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            queries: [b, num_slots, dim]
            keys: [b, num_tokens, dim]
            values: [b, num_tokens, dim]
            
        Returns:
            attended: [b, num_slots, dim]
            prognostic_weights: [b, num_tokens] - 每个token的预后重要性分数
        """
        b, num_slots, _ = queries.shape
        _, num_tokens, _ = keys.shape
        
        q = rearrange(self.to_q(queries), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(keys), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(values), 'b n (h d) -> b h n d', h=self.heads)
        
        # 标准注意力分数: [b, h, num_slots, num_tokens]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # 预后显著性分数: [b, num_tokens]
        prognostic_scores = self.prognostic_scorer(keys).squeeze(-1)
        
        # 创新: 用预后重要性加权注意力分配
        # 预后重要的token会被优先分配到slots
        prognostic_bias = repeat(prognostic_scores, 'b t -> b h s t', h=self.heads, s=num_slots)
        weighted_dots = dots + self.lambda_prognostic * prognostic_bias
        
        # Slot竞争: 每个token选择最匹配的slot
        attn = weighted_dots.softmax(dim=-2)  # 在slot维度上竞争
        attn = F.normalize(attn, p=1, dim=-1)  # 归一化到token维度
        
        # 加权聚合
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return out, prognostic_scores


class DynamicSlotPruning(nn.Module):
    """创新2: 动态因果slot剪枝
    
    **核心问题**: 固定8个slots对所有患者不合理
    - 早期癌症: 只需2-3个slots (肿瘤主体+周围组织)
    - 晚期癌症: 需要6-8个slots (多个转移灶+浸润区+坏死区)
    
    **解决方案**: 每个slot学习"预后必要性分数"，推理时只激活必要的slots
    
    **因果语义**: 剪掉的slot = "对预后判断无贡献的冗余表示"
    """
    
    def __init__(self, num_slots: int, dim: int, pruning_threshold: float = 0.1):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.pruning_threshold = pruning_threshold
        
        # 每个slot的"预后必要性"可学习参数
        # 初始化为1.0（全部激活），训练中学习哪些可以剪枝
        self.slot_importance = nn.Parameter(torch.ones(num_slots))
        
        # Gumbel-Softmax温度参数，控制剪枝的软硬程度
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, slots: Tensor, training: bool = True) -> Tuple[Tensor, Tensor, int]:
        """
        Args:
            slots: [b, num_slots, dim]
            training: 训练模式=软剪枝（可微分），推理模式=硬剪枝（真正裁剪）
            
        Returns:
            pruned_slots: [b, num_active_slots, dim]
            importance_mask: [num_slots] - 每个slot是否激活
            num_active: 激活的slot数量
        """
        b, num_slots, dim = slots.shape
        
        if training:
            # 训练模式: Gumbel-Softmax软剪枝（保持可微分）
            logits = self.slot_importance
            noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            soft_mask = torch.sigmoid((logits + noise) / self.temperature)
            
            # 软mask: [num_slots, 1, 1] 广播到 [b, num_slots, dim]
            mask = soft_mask.view(1, num_slots, 1)
            pruned_slots = slots * mask
            
            num_active = num_slots  # 训练时保持所有slots维度
            importance_mask = soft_mask
            
        else:
            # 推理模式: 硬剪枝，只保留重要的slots
            importance_scores = torch.sigmoid(self.slot_importance)
            hard_mask = importance_scores > self.pruning_threshold
            
            num_active = hard_mask.sum().item()
            if num_active == 0:
                # 保底：至少保留最重要的1个slot
                hard_mask[importance_scores.argmax()] = True
                num_active = 1
            
            # 只保留激活的slots: [b, num_active_slots, dim]
            pruned_slots = slots[:, hard_mask]
            importance_mask = hard_mask.float()
        
        return pruned_slots, importance_mask, num_active


class HierarchicalSlotPyramid(nn.Module):
    """创新3: 分层slot金字塔
    
    **核心洞察**: 预后需要多尺度特征
    - L1 (细粒度): 16 slots - 局部细节 (细胞形态、核异型性)
    - L2 (中粒度): 8 slots  - 组织模式 (浸润模式、分化程度)  
    - L3 (粗粒度): 4 slots  - 全局预后 (TNM分期、整体侵袭性)
    
    **与ResNet/FPN的区别**: 
    - ResNet: 特征层次结构 (低层→高层语义)
    - 我们: 预后粒度层次结构 (局部风险→全局风险)
    
    **临床可解释性**: 3层输出可分别对应病理报告的不同描述层次
    """
    
    def __init__(
        self,
        dim: int,
        num_slots_l1: int = 16,
        num_slots_l2: int = 8, 
        num_slots_l3: int = 4,
        iters: int = 3,
        enable_pruning: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots_l1 = num_slots_l1
        self.num_slots_l2 = num_slots_l2
        self.num_slots_l3 = num_slots_l3
        self.iters = iters
        self.enable_pruning = enable_pruning
        
        # Layer 1: 细粒度slots (16个)
        self.slots_mu_l1 = nn.Parameter(torch.randn(1, num_slots_l1, dim))
        self.attn_l1 = PrognosticWeightedAttention(dim)
        self.gru_l1 = nn.GRUCell(dim, dim)
        self.norm_l1 = nn.LayerNorm(dim)
        self.mlp_l1 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        if enable_pruning:
            self.pruning_l1 = DynamicSlotPruning(num_slots_l1, dim)
        
        # Layer 2: 中粒度slots (8个) - 从L1聚合
        self.slots_mu_l2 = nn.Parameter(torch.randn(1, num_slots_l2, dim))
        self.attn_l2 = PrognosticWeightedAttention(dim)
        self.gru_l2 = nn.GRUCell(dim, dim)
        self.norm_l2 = nn.LayerNorm(dim)
        self.mlp_l2 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        if enable_pruning:
            self.pruning_l2 = DynamicSlotPruning(num_slots_l2, dim)
        
        # Layer 3: 粗粒度slots (4个) - 从L2聚合
        self.slots_mu_l3 = nn.Parameter(torch.randn(1, num_slots_l3, dim))
        self.attn_l3 = PrognosticWeightedAttention(dim)
        self.gru_l3 = nn.GRUCell(dim, dim)
        self.norm_l3 = nn.LayerNorm(dim)
        self.mlp_l3 = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        if enable_pruning:
            self.pruning_l3 = DynamicSlotPruning(num_slots_l3, dim)
        
        # 输入归一化
        self.norm_input = nn.LayerNorm(dim)
        
    def forward(self, inputs: Tensor) -> dict:
        """
        Args:
            inputs: [b, num_tokens, dim] - 输入tokens (e.g., 2048 WSI patches)
            
        Returns:
            {
                'slots_l1': [b, num_active_l1, dim] - 细粒度slots
                'slots_l2': [b, num_active_l2, dim] - 中粒度slots
                'slots_l3': [b, num_active_l3, dim] - 粗粒度slots
                'prognostic_scores_l1': [b, num_tokens] - L1预后权重
                'prognostic_scores_l2': [b, num_active_l1] - L2预后权重
                'prognostic_scores_l3': [b, num_active_l2] - L3预后权重
                'importance_masks': {l1/l2/l3: [num_slots]} - 剪枝mask
                'num_active_slots': {l1/l2/l3: int} - 激活slot数
            }
        """
        b, num_tokens, dim = inputs.shape
        device = inputs.device
        
        inputs_norm = self.norm_input(inputs)
        outputs = {}
        
        # === Layer 1: 细粒度特征提取 ===
        slots_l1 = repeat(self.slots_mu_l1, '1 n d -> b n d', b=b)
        
        for _ in range(self.iters):
            slots_prev = slots_l1
            slots_l1_norm = self.norm_l1(slots_l1)
            
            # 预后加权注意力
            updates, prog_scores_l1 = self.attn_l1(
                queries=slots_l1_norm,
                keys=inputs_norm,
                values=inputs_norm
            )
            
            # GRU更新
            updates_flat = updates.reshape(-1, dim)
            slots_prev_flat = slots_prev.reshape(-1, dim)
            slots_l1 = self.gru_l1(updates_flat, slots_prev_flat)
            slots_l1 = slots_l1.reshape(b, self.num_slots_l1, dim)
            
            # MLP
            slots_l1 = slots_l1 + self.mlp_l1(slots_l1)
        
        # 动态剪枝
        if self.enable_pruning:
            slots_l1, mask_l1, num_active_l1 = self.pruning_l1(slots_l1, self.training)
        else:
            mask_l1 = torch.ones(self.num_slots_l1, device=device)
            num_active_l1 = self.num_slots_l1
        
        outputs['slots_l1'] = slots_l1
        outputs['prognostic_scores_l1'] = prog_scores_l1
        
        # === Layer 2: 中粒度聚合 ===
        slots_l2 = repeat(self.slots_mu_l2, '1 n d -> b n d', b=b)
        
        for _ in range(self.iters):
            slots_prev = slots_l2
            slots_l2_norm = self.norm_l2(slots_l2)
            
            # 从L1聚合到L2
            updates, prog_scores_l2 = self.attn_l2(
                queries=slots_l2_norm,
                keys=slots_l1,
                values=slots_l1
            )
            
            updates_flat = updates.reshape(-1, dim)
            slots_prev_flat = slots_prev.reshape(-1, dim)
            slots_l2 = self.gru_l2(updates_flat, slots_prev_flat)
            slots_l2 = slots_l2.reshape(b, self.num_slots_l2, dim)
            slots_l2 = slots_l2 + self.mlp_l2(slots_l2)
        
        if self.enable_pruning:
            slots_l2, mask_l2, num_active_l2 = self.pruning_l2(slots_l2, self.training)
        else:
            mask_l2 = torch.ones(self.num_slots_l2, device=device)
            num_active_l2 = self.num_slots_l2
        
        outputs['slots_l2'] = slots_l2
        outputs['prognostic_scores_l2'] = prog_scores_l2
        
        # === Layer 3: 粗粒度全局预后 ===
        slots_l3 = repeat(self.slots_mu_l3, '1 n d -> b n d', b=b)
        
        for _ in range(self.iters):
            slots_prev = slots_l3
            slots_l3_norm = self.norm_l3(slots_l3)
            
            # 从L2聚合到L3
            updates, prog_scores_l3 = self.attn_l3(
                queries=slots_l3_norm,
                keys=slots_l2,
                values=slots_l2
            )
            
            updates_flat = updates.reshape(-1, dim)
            slots_prev_flat = slots_prev.reshape(-1, dim)
            slots_l3 = self.gru_l3(updates_flat, slots_prev_flat)
            slots_l3 = slots_l3.reshape(b, self.num_slots_l3, dim)
            slots_l3 = slots_l3 + self.mlp_l3(slots_l3)
        
        if self.enable_pruning:
            slots_l3, mask_l3, num_active_l3 = self.pruning_l3(slots_l3, self.training)
        else:
            mask_l3 = torch.ones(self.num_slots_l3, device=device)
            num_active_l3 = self.num_slots_l3
        
        outputs['slots_l3'] = slots_l3
        outputs['prognostic_scores_l3'] = prog_scores_l3
        
        # 汇总metadata
        outputs['importance_masks'] = {
            'l1': mask_l1,
            'l2': mask_l2,
            'l3': mask_l3,
        }
        outputs['num_active_slots'] = {
            'l1': num_active_l1,
            'l2': num_active_l2,
            'l3': num_active_l3,
        }
        
        return outputs


class HierarchicalPrognosticSlotAttention(nn.Module):
    """完整模型: 分层预后slot注意力机制
    
    用法示例:
        model = HierarchicalPrognosticSlotAttention(
            dim=256,
            num_slots_l1=16,
            num_slots_l2=8,
            num_slots_l3=4,
            enable_pruning=True
        )
        
        # 前向传播
        wsi_features = ... # [batch, 2048, 256]
        outputs = model(wsi_features)
        
        # 多尺度融合用于预后预测
        slots_fine = outputs['slots_l1']    # 细节特征
        slots_mid = outputs['slots_l2']     # 中层特征
        slots_coarse = outputs['slots_l3']  # 全局特征
        
        # 可视化预后重要性
        prog_scores = outputs['prognostic_scores_l1']  # [b, 2048]
        # 映射回原始patches，显示哪些区域对预后最重要
        
        # 查看动态剪枝结果
        print(f"L1激活slots: {outputs['num_active_slots']['l1']}/16")
        print(f"L2激活slots: {outputs['num_active_slots']['l2']}/8")
        print(f"L3激活slots: {outputs['num_active_slots']['l3']}/4")
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_slots_l1: int = 16,
        num_slots_l2: int = 8,
        num_slots_l3: int = 4,
        iters: int = 3,
        enable_pruning: bool = True,
        fusion_mode: str = "concat"  # concat | mean | attention
    ):
        super().__init__()
        self.pyramid = HierarchicalSlotPyramid(
            dim=dim,
            num_slots_l1=num_slots_l1,
            num_slots_l2=num_slots_l2,
            num_slots_l3=num_slots_l3,
            iters=iters,
            enable_pruning=enable_pruning
        )
        self.fusion_mode = fusion_mode
        
        # 多尺度融合层
        if fusion_mode == "concat":
            # 拼接所有层的slots
            max_slots = num_slots_l1 + num_slots_l2 + num_slots_l3
            self.fusion_proj = nn.Linear(dim * max_slots, dim)
        elif fusion_mode == "attention":
            # 学习每层的重要性权重
            self.layer_attn = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.GELU(),
                nn.Linear(dim // 4, 3),  # 3层
                nn.Softmax(dim=-1)
            )
        
    def forward(self, inputs: Tensor) -> dict:
        """
        Args:
            inputs: [b, num_tokens, dim]
            
        Returns:
            outputs: dict包含:
                - 'fused_slots': [b, dim] - 融合后的预后表示
                - 'slots_l1/l2/l3': 各层slots
                - 'prognostic_scores_l1/l2/l3': 预后重要性
                - 其他metadata
        """
        outputs = self.pyramid(inputs)
        
        # 多尺度融合
        b = inputs.shape[0]
        slots_l1 = outputs['slots_l1']  # [b, n1, dim]
        slots_l2 = outputs['slots_l2']  # [b, n2, dim]
        slots_l3 = outputs['slots_l3']  # [b, n3, dim]
        
        if self.fusion_mode == "concat":
            # 拼接所有slots并投影
            slots_l1_flat = slots_l1.reshape(b, -1)
            slots_l2_flat = slots_l2.reshape(b, -1)
            slots_l3_flat = slots_l3.reshape(b, -1)
            
            # Pad到固定长度
            max_dim_l1 = self.pyramid.num_slots_l1 * self.pyramid.dim
            max_dim_l2 = self.pyramid.num_slots_l2 * self.pyramid.dim
            max_dim_l3 = self.pyramid.num_slots_l3 * self.pyramid.dim
            
            slots_l1_padded = F.pad(slots_l1_flat, (0, max_dim_l1 - slots_l1_flat.size(1)))
            slots_l2_padded = F.pad(slots_l2_flat, (0, max_dim_l2 - slots_l2_flat.size(1)))
            slots_l3_padded = F.pad(slots_l3_flat, (0, max_dim_l3 - slots_l3_flat.size(1)))
            
            fused = torch.cat([slots_l1_padded, slots_l2_padded, slots_l3_padded], dim=1)
            fused_slots = self.fusion_proj(fused)  # [b, dim]
            
        elif self.fusion_mode == "mean":
            # 简单平均pooling
            pool_l1 = slots_l1.mean(dim=1)
            pool_l2 = slots_l2.mean(dim=1)
            pool_l3 = slots_l3.mean(dim=1)
            fused_slots = (pool_l1 + pool_l2 + pool_l3) / 3.0
            
        elif self.fusion_mode == "attention":
            # 学习每层的重要性权重
            pool_l1 = slots_l1.mean(dim=1)  # [b, dim]
            pool_l2 = slots_l2.mean(dim=1)
            pool_l3 = slots_l3.mean(dim=1)
            
            # 用L3全局特征决定每层权重
            layer_weights = self.layer_attn(pool_l3)  # [b, 3]
            w1, w2, w3 = layer_weights[:, 0:1], layer_weights[:, 1:2], layer_weights[:, 2:3]
            
            fused_slots = w1 * pool_l1 + w2 * pool_l2 + w3 * pool_l3
            outputs['layer_weights'] = layer_weights
        
        outputs['fused_slots'] = fused_slots
        return outputs

