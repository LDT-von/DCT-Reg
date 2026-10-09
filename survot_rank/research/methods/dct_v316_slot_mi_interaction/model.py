"""Standalone components for the DCT v3.16 slot-interaction candidate.

R/Up/Ug/S are hypothesis labels for learned channels rather than a formal
partial-information decomposition. Their meaning must be tested empirically.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, init


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))


def relaxed_topk(logits, k, temperature=1.0):
    """Sequential soft top-k: 迭代选择top-1并抑制它。"""
    scores = logits
    soft_k_hot = torch.zeros_like(logits)
    for _ in range(k):
        probs = F.softmax(scores / temperature, dim=-1)
        soft_k_hot = soft_k_hot + probs
        scores = scores + torch.log((1.0 - probs).clamp(min=1e-20))
    return soft_k_hot


def gumbel_topk_st(logits, k=1, temperature=1.0):
    """Standard straight-through Gumbel-TopK. Forward: hard, Backward: relaxed."""
    noised_logits = logits + gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft_k_hot = relaxed_topk(noised_logits, k=k, temperature=temperature)
    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


def parallel_topk_st(logits, k=1, temperature=1.0):
    """Parallel approximate top-k relaxation."""
    noised_logits = logits + gumbel_noise(logits)
    topk_indices = noised_logits.topk(k=k, dim=-1).indices
    hard_k_hot = torch.zeros_like(logits)
    hard_k_hot.scatter_(1, topk_indices, 1.0)
    soft = F.softmax(noised_logits / temperature, dim=-1)
    soft_k_hot = k * soft
    y = hard_k_hot + soft_k_hot - soft_k_hot.detach()
    return y, topk_indices


def split_heads(x: Tensor, heads: int) -> Tensor:
    """Split last dimension into heads: [B, N, H*D] -> [B, H, N, D]"""
    b, n, hd = x.shape
    d = hd // heads
    x = x.view(b, n, heads, d)
    return x.permute(0, 2, 1, 3).contiguous()


def merge_heads(x: Tensor) -> Tensor:
    """Merge heads back: [B, H, N, D] -> [B, N, H*D]"""
    b, h, n, d = x.shape
    x = x.permute(0, 2, 1, 3).contiguous()  # [B, N, H, D]
    return x.view(b, n, h * d)


# ============================================================================
# 1. MultiHead Slot Attention (from SlotSPE)
# ============================================================================


class MultiHeadSlotAttention(Module):
    """Multi-Head Slot Attention with GRU-based Slot Update.

    来自 SlotSPE 的专业实现：
    - Slot initialization: 重参数化高斯 mu + sigma * randn
    - Slot update: GRU单元 + 残差MLP
    - Iterative refinement: 迭代iters次更新

    Args:
        num_slots: Slot数量
        dim: 特征维度
        heads: 注意力头数
        dim_head: 每个头的维度
        iters: 迭代更新次数
        eps: 归一化常数
        hidden_dim: MLP隐藏层维度
    """

    def __init__(
        self,
        num_slots: int,
        dim: int,
        heads: int = 4,
        dim_head: int = 64,
        iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.heads = heads
        self.scale = dim_head ** -0.5

        # Per-slot posterior parameters preserve deterministic slot identities
        # at evaluation. A shared [1,1,D] mean makes every eval slot identical,
        # and all subsequent permutation-equivariant updates remain collapsed.
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)

        dim_inner = dim_head * heads
        self.to_q = nn.Linear(dim, dim_inner)
        self.to_k = nn.Linear(dim, dim_inner)
        self.to_v = nn.Linear(dim, dim_inner)
        self.combine_heads = nn.Linear(dim_inner, dim)

        # GRU-based slot update
        self.gru = nn.GRUCell(dim, dim)

        hidden_dim_mlp = max(dim, hidden_dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Linear(hidden_dim_mlp, dim),
        )

    def forward(self, inputs: Tensor, num_slots: Optional[int] = None) -> Tensor:
        """Slot extraction.

        Args:
            inputs: [B, N, D] 输入token序列
            num_slots: 可选的动态slot数量

        Returns:
            slots: [B, num_slots, D] 提取的slot表征
        """
        b, n, device, dtype = inputs.shape[0], inputs.shape[1], inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self.num_slots
        if n_s < 1 or n_s > self.num_slots:
            raise ValueError(
                f"num_slots must be in [1, {self.num_slots}], received {n_s}"
            )

        # Slot initialization: reparameterized Gaussian
        # Use expand instead of einops for better compatibility
        mu = self.slots_mu[:, :n_s].expand(b, -1, -1).contiguous()
        sigma = self.slots_logsigma[:, :n_s].exp().expand(b, -1, -1).contiguous()
        # Training keeps the stochastic posterior sample.  Validation and
        # inference use its mean so repeated predictions are deterministic.
        slots = mu + sigma * torch.randn_like(mu) if self.training else mu.clone()

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)  # [B, N, H*D]
        v = self.to_v(inputs)  # [B, N, H*D]
        k = split_heads(k, self.heads)  # [B, H, N, D]
        v = split_heads(v, self.heads)  # [B, H, N, D]

        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # [B, n_s, H*D]
            q = split_heads(q, self.heads)  # [B, H, n_s, D]

            # Attention weights: [B, H, n_s, N]. Slots compete for each
            # token, then each slot's mass is normalized across tokens.
            dots = torch.einsum("... i d, ... j d -> ... i j", q, k) * self.scale
            attn = dots.softmax(dim=-2)
            attn = F.normalize(attn + self.eps, p=1, dim=-1)  # L1-normalized

            # Value aggregation: [B, H, n_s, D]
            updates = torch.einsum("... j d, ... i j -> ... i d", v, attn)
            updates = merge_heads(updates)  # [B, n_s, H*D]
            updates = self.combine_heads(updates)  # [B, n_s, D]

            # Slot update via GRU
            slots_flat = slots.reshape(-1, self.dim)  # [B*n_s, D]
            updates_flat = updates.reshape(-1, self.dim)
            slots_prev_flat = slots_prev.reshape(-1, self.dim)
            slots_flat = self.gru(updates_flat, slots_prev_flat)
            slots = slots_flat.view(b, n_s, self.dim)

            # Residual MLP
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots  # (B, num_slots, D)


# ============================================================================
# 2. MoE Slot Decoder with Top-K Gating (from SlotSPE)
# ============================================================================


class MoESlotDecoder(Module):
    """MoE-style Slot Decoder with Top-K Gating.

    来自 SlotSPE：
    - 预测每个slot的keep_score
    - Gumbel-TopK选择最重要的slot
    - 软门控加权池化

    Args:
        dim: Slot维度
        num_slots: Slot数量
        num_classes: 预测类别数
        temperature: Top-K温度
        topk_ratio: 选择比例
        top_k_method: 'parallel_topk_st' 或 'gumbel_topk_st'
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        num_classes: int = 4,
        temperature: float = 0.01,
        topk_ratio: float = 0.25,
        top_k_method: str = "parallel_topk_st",
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes
        self.dim = dim
        self.temperature = temperature
        self.top_k_method = top_k_method
        self.k = max(1, int(num_slots * topk_ratio))

        self.map = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.decoder = nn.Linear(dim, num_classes)
        self.pred_keep_slot = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Slot解码与选择。

        Args:
            slots: [B, S, D] Slot表征

        Returns:
            logits: [B, C] 加权融合后的logits
            slot_gate: [B, S] 软门控权重
            hard_keep: [B, S] 二元掩码 (Top-K选择的slot)
        """
        slots = self.map(slots)
        slot_logits = self.decoder(slots)  # (B, S, C)
        keep_score = self.pred_keep_slot(slots).squeeze(-1)  # (B, S)

        # Top-K gating is stochastic only while training.  Evaluation must not
        # depend on a fresh Gumbel draw, otherwise validation scores and saved
        # explanations change when the same patient is evaluated twice.
        if self.training:
            if self.top_k_method == "gumbel_topk_st":
                hard_keep, _ = gumbel_topk_st(
                    keep_score, temperature=self.temperature, k=self.k
                )
            else:
                hard_keep, _ = parallel_topk_st(
                    keep_score, temperature=self.temperature, k=self.k
                )
        else:
            topk_indices = keep_score.topk(k=self.k, dim=-1).indices
            hard_keep = torch.zeros_like(keep_score)
            hard_keep.scatter_(1, topk_indices, 1.0)

        # Soft slot gate
        slot_gate = F.softmax(keep_score / self.temperature, dim=-1)
        slot_gate = slot_gate * hard_keep
        slot_gate = slot_gate / (slot_gate.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted prediction
        x = torch.einsum("bs,bsc->bc", slot_gate, slot_logits)  # (B, C)
        return x, slot_gate, hard_keep


# ============================================================================
# 3. Iterative Cross-Attention (from SlotSPE)
# ============================================================================


class IterativeCrossAttention(Module):
    """Iterative Bidirectional Cross-Attention.

    来自 SlotSPE：
    - 双向交叉注意力：x1 attends to x2, x2 attends to x1
    - 分离的GRU更新单元
    - 迭代iters次更新

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        iters: 迭代次数
        static_kv: 是否预计算K/V
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        iters: int = 3,
        static_kv: bool = True,
        attn_drop: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.iters = iters
        self.static_kv = static_kv
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

        # TWO separate GRUs -- one per modality
        self.gru1 = nn.GRUCell(dim, dim)
        self.gru2 = nn.GRUCell(dim, dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.norm1a = nn.LayerNorm(dim)
        self.norm1b = nn.LayerNorm(dim)
        self.norm2a = nn.LayerNorm(dim)
        self.norm2b = nn.LayerNorm(dim)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        mask1: Optional[Tensor] = None,
        mask2: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """双向交叉注意力更新。

        Args:
            x1: [B, N1, D] modality 1 (e.g., WSI slots)
            x2: [B, N2, D] modality 2 (e.g., Omics slots)
            mask1: [B, N1] 可选的掩码
            mask2: [B, N2] 可选的掩码

        Returns:
            x1: [B, N1, D] 更新后的modality 1
            x2: [B, N2, D] 更新后的modality 2
        """
        b, n1, n2 = x1.size(0), x1.size(1), x2.size(1)

        # Precompute static k/v for all iterations
        if self.static_kv:
            k1 = split_heads(self.to_k(x1), self.num_heads)  # [B, H, N1, D_h]
            v1 = split_heads(self.to_v(x1), self.num_heads)
            k2 = split_heads(self.to_k(x2), self.num_heads)
            v2 = split_heads(self.to_v(x2), self.num_heads)

        for _ in range(self.iters):
            x1_prev, x2_prev = x1, x2
            x1 = self.norm1a(x1)
            x2 = self.norm2a(x2)

            if not self.static_kv:
                k1 = split_heads(self.to_k(x1), self.num_heads)
                v1 = split_heads(self.to_v(x1), self.num_heads)
                k2 = split_heads(self.to_k(x2), self.num_heads)
                v2 = split_heads(self.to_v(x2), self.num_heads)

            q1 = split_heads(self.to_q(x1), self.num_heads)  # [B, H, N1, D_h]
            q2 = split_heads(self.to_q(x2), self.num_heads)  # [B, H, N2, D_h]

            # Bidirectional cross-attention
            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale  # x1 attends to x2: [B, H, N1, N2]
            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale  # x2 attends to x1: [B, H, N2, N1]

            # Apply masks
            if mask1 is not None and mask2 is not None:
                mask_q1 = mask1.unsqueeze(1).unsqueeze(-1)  # [B, 1, N1, 1]
                mask_k2 = mask2.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N2]
                attn1 = attn1.masked_fill(mask_q1 * mask_k2 == 0, -1e9)

                mask_q2 = mask2.unsqueeze(1).unsqueeze(-1)  # [B, 1, N2, 1]
                mask_k1 = mask1.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N1]
                attn2 = attn2.masked_fill(mask_q2 * mask_k1 == 0, -1e9)

            attn1 = self.attn_drop(F.softmax(attn1, dim=-1))
            attn2 = self.attn_drop(F.softmax(attn2, dim=-1))

            update1 = (attn1 @ v2)  # [B, H, N1, D_h]
            update1 = merge_heads(update1)  # [B, N1, D]
            update1 = self.proj_drop(self.proj(update1))

            update2 = (attn2 @ v1)  # [B, H, N2, D_h]
            update2 = merge_heads(update2)  # [B, N2, D]
            update2 = self.proj_drop(self.proj(update2))

            # GRU slot update (separate GRUs per modality)
            update1_flat = update1.reshape(-1, self.dim)
            update2_flat = update2.reshape(-1, self.dim)
            x1_prev_flat = x1_prev.reshape(-1, self.dim)
            x2_prev_flat = x2_prev.reshape(-1, self.dim)

            x1_flat = self.gru1(update1_flat, x1_prev_flat)
            x2_flat = self.gru2(update2_flat, x2_prev_flat)

            x1 = x1_flat.view(b, n1, self.dim)
            x2 = x2_flat.view(b, n2, self.dim)

            # MLP residual
            x1 = x1 + self.mlp1(self.norm1b(x1))
            x2 = x2 + self.mlp2(self.norm2b(x2))

        return x1, x2


class IterativeCrossAttTransformer(Module):
    """Wrapper: Iterative Cross-Attention返回拼接输出。"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        iters: int = 3,
        static_kv: bool = True,
        attn_drop: float = 0.1,
    ):
        super().__init__()
        self.attn = IterativeCrossAttention(
            dim=dim,
            num_heads=num_heads,
            iters=iters,
            static_kv=static_kv,
            attn_drop=attn_drop,
        )

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        mask1: Optional[Tensor] = None,
        mask2: Optional[Tensor] = None,
    ) -> Tensor:
        """返回拼接的交叉注意力输出。"""
        attn1, attn2 = self.attn(x1, x2, mask1=mask1, mask2=mask2)
        x = torch.cat([attn1, attn2], dim=1)  # (B, N1+N2, D)
        return x


# ============================================================================
# 4. Transformer (from SlotSPE)
# ============================================================================


class Transformer(Module):
    """Self-Attention Transformer with Slot-Level Masking."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """自注意力更新，支持slot级掩码。

        Args:
            x: [B, S, D] Slot序列
            mask: [B, S] True表示keep, False表示mask out
        """
        if mask is not None:
            key_padding_mask = ~mask  # True = masked positions
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            self.norm1(x), x, x,
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# 5. Per-sample interaction-channel profiler
# ============================================================================


class InfoNCECritic(Module):
    """Contrast a learned channel with the same patient's prediction context."""

    def __init__(
        self,
        dim: int,
        target_dim: Optional[int] = None,
        temperature: float = 0.1,
    ):
        super().__init__()
        if dim < 1 or (target_dim is not None and target_dim < 1):
            raise ValueError("InfoNCE dimensions must be positive")
        if temperature <= 0:
            raise ValueError("InfoNCE temperature must be positive")
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.target_projection = nn.Linear(target_dim or dim, dim)

    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        """Return a diagonal in-batch contrastive loss."""
        if z.ndim != 2 or y.ndim != 2 or z.size(0) != y.size(0):
            raise ValueError("InfoNCE inputs must be [B,D] with matching batches")
        if z.size(0) < 2:
            return z.sum() * 0.0
        z_proj = F.normalize(self.projection(z), dim=-1)
        y_proj = F.normalize(self.target_projection(y), dim=-1)
        logits = (z_proj @ y_proj.T) / self.temperature
        # Patient i is paired with prediction context i.  The previous all-zero
        # target incorrectly trained every row toward patient 0.
        labels = torch.arange(z.size(0), dtype=torch.long, device=z.device)
        return F.cross_entropy(logits, labels)


class InteractionProfiler(Module):
    """Learn a patient-local distribution over four interaction channels.

    R, Up, Ug, and S are provisional semantic labels. The architecture gives
    the channels different inputs, but does not by itself identify a formal
    redundant/unique/synergistic information decomposition.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        prediction_dim: int = 4,
        profile_temp: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        if profile_temp <= 0:
            raise ValueError("profile_temp must be positive")
        self.profile_temp = profile_temp

        # 四个通道的投影
        self.proj_R = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_Up = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_Ug = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.proj_S = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # InfoNCE批评器
        self.nce_R = InfoNCECritic(hidden_dim, prediction_dim)
        self.nce_Up = InfoNCECritic(hidden_dim, prediction_dim)
        self.nce_Ug = InfoNCECritic(hidden_dim, prediction_dim)
        self.nce_S = InfoNCECritic(hidden_dim, prediction_dim)
        self.profile_heads = nn.ModuleList(
            nn.Linear(hidden_dim, 1) for _ in range(4)
        )

    def forward(
        self,
        slots_wsi: Tensor,
        slots_omic: Tensor,
        survival_proxy: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, dict]:
        """计算每个样本的R/U/S profile。

        Args:
            slots_wsi: [B, S_w, D] WSI slots
            slots_omic: [B, S_o, D] Omics slots
            survival_proxy: [B, prediction_dim] detached prediction context

        Returns:
            profile: [B, 4] 每个样本的(r, up, ug, s)路由权重
            context: [B, H] profile加权的通道表征
            metrics: dict with losses
        """
        # Pool slots
        wsi_pooled = slots_wsi.mean(dim=1)  # [B, D]
        omic_pooled = slots_omic.mean(dim=1)  # [B, D]
        joint = torch.cat([wsi_pooled, omic_pooled], dim=-1)  # [B, 2D]

        # 四个通道
        z_R = self.proj_R(joint)  # 冗余
        z_Up = self.proj_Up(wsi_pooled)  # 病理特异
        z_Ug = self.proj_Ug(omic_pooled)  # 基因特异
        z_S = self.proj_S(joint)  # 协同

        metrics = {}

        if self.training and survival_proxy is not None:
            # 使用InfoNCE最大化每个通道对预测的贡献
            L_R = self.nce_R(z_R, survival_proxy)
            L_Up = self.nce_Up(z_Up, survival_proxy)
            L_Ug = self.nce_Ug(z_Ug, survival_proxy)
            L_S = self.nce_S(z_S, survival_proxy)

            metrics["L_R"] = L_R
            metrics["L_Up"] = L_Up
            metrics["L_Ug"] = L_Ug
            metrics["L_S"] = L_S
            metrics["L_total"] = L_R + L_Up + L_Ug + L_S

        channels = torch.stack([z_R, z_Up, z_Ug, z_S], dim=1)  # [B,4,H]
        raw_profile = torch.cat(
            [
                head(channel)
                for head, channel in zip(self.profile_heads, channels.unbind(1))
            ],
            dim=-1,
        )
        # Patient-local scoring keeps the profile invariant to batch companions.
        profile = F.softmax(raw_profile / self.profile_temp, dim=-1)
        context = torch.einsum("bc,bch->bh", profile, channels)
        return profile, context, metrics


# ============================================================================
# 6. Complete Slot-based MI Decomposition Block
# ============================================================================


class SlotBasedMIDecompositionBlock(Module):
    """Slot-based Mutual Information Decomposition Block.

    整合:
    - MultiHeadSlotAttention: 独立Slot提取
    - IterativeCrossAttention: 双向交叉注意力
    - InteractionProfiler: Per-Sample R/U/S分解
    - MoESlotDecoder: Top-K门控解码
    """

    def __init__(
        self,
        dim: int,
        num_wsi_slots: int = 8,
        num_omic_slots: int = 8,
        slot_iters: int = 3,
        cross_iters: int = 3,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_classes: int = 4,
        profile_temp: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for discrete survival")
        self.num_classes = num_classes

        # Slot Attention for each modality
        self.slot_attention_wsi = MultiHeadSlotAttention(
            num_slots=num_wsi_slots,
            dim=dim,
            heads=num_heads,
            iters=slot_iters,
        )
        self.slot_attention_omic = MultiHeadSlotAttention(
            num_slots=num_omic_slots,
            dim=dim,
            heads=num_heads,
            iters=slot_iters,
        )

        # Cross-modal interaction
        self.cross_attention = IterativeCrossAttTransformer(
            dim=dim,
            num_heads=num_heads,
            iters=cross_iters,
        )

        # MoE Decoders
        self.moe_decoder_wsi = MoESlotDecoder(
            dim=dim,
            num_slots=num_wsi_slots,
            num_classes=num_classes,
        )
        self.moe_decoder_omic = MoESlotDecoder(
            dim=dim,
            num_slots=num_omic_slots,
            num_classes=num_classes,
        )

        # Intra-modal self-attention (masked by Top-K)
        self.self_attention_wsi = Transformer(dim=dim, num_heads=num_heads)
        self.self_attention_omic = Transformer(dim=dim, num_heads=num_heads)

        # R/U/S Interaction Profiler
        self.profiler = InteractionProfiler(
            dim=dim,
            hidden_dim=hidden_dim,
            prediction_dim=num_classes,
            profile_temp=profile_temp,
        )

        # Final fusion
        self.to_logits = nn.Linear(dim * 3, num_classes)
        self.channel_to_logits = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(
        self,
        wsi_tokens: Tensor,
        omic_tokens: Tensor,
        survival_proxy: Optional[Tensor] = None,
    ) -> tuple[dict, dict]:
        """前向传播。

        Args:
            wsi_tokens: [B, N, D] WSI token序列
            omic_tokens: [B, M, D] Omics token序列
            survival_proxy: optional [B, num_classes] prediction context

        Returns:
            output: dict with 'logits', 'profile', 'slots'
            metrics: dict with losses
        """
        # Slot extraction (independent)
        slots_wsi = self.slot_attention_wsi(wsi_tokens)  # [B, S_w, D]
        slots_omic = self.slot_attention_omic(omic_tokens)  # [B, S_o, D]

        # Slot-level predictions
        logits_wsi, slot_gate_wsi, hard_keep_wsi = self.moe_decoder_wsi(slots_wsi)
        logits_omic, slot_gate_omic, hard_keep_omic = self.moe_decoder_omic(slots_omic)

        # Cross-modal interaction
        cross_output = self.cross_attention(slots_wsi, slots_omic)  # [B, S_w+S_o, D]

        # Intra-modal self-attention (masked by Top-K selected slots)
        wsi_intra = self.self_attention_wsi(slots_wsi, mask=hard_keep_wsi.bool())
        omic_intra = self.self_attention_omic(slots_omic, mask=hard_keep_omic.bool())

        # Final fusion: concatenate cross + WSI intra + Omic intra
        fused = torch.cat(
            [
                cross_output.mean(dim=1),
                torch.einsum("bs,bsd->bd", slot_gate_wsi, wsi_intra),
                torch.einsum("bs,bsd->bd", slot_gate_omic, omic_intra),
            ],
            dim=1,
        )
        # Keep both sparse per-modality decoders on the survival-loss path.
        # Previously their logits were diagnostic-only and converting the
        # Top-K mask to bool cut the remaining gradient to the gate networks.
        base_logits = self.to_logits(fused) + 0.5 * (logits_wsi + logits_omic)

        # The detached NLL-trained prediction is the contrastive context.  This
        # avoids the old 2D-vs-H mismatch and keeps the auxiliary target branch
        # from updating the base prediction head.
        prediction_context = base_logits.detach() if survival_proxy is None else survival_proxy
        profile, channel_context, mi_metrics = self.profiler(
            slots_wsi, slots_omic, prediction_context
        )
        channel_delta = self.channel_to_logits(channel_context)
        logits = base_logits + channel_delta
        mi_metrics["v316_channel_delta_rms"] = (
            channel_delta.detach().square().mean().sqrt()
        )

        output = {
            "logits": logits,
            "profile": profile,
            "slots_wsi": slots_wsi,
            "slots_omic": slots_omic,
            "slot_gate_wsi": slot_gate_wsi,
            "slot_gate_omic": slot_gate_omic,
            "hard_keep_wsi": hard_keep_wsi,
            "hard_keep_omic": hard_keep_omic,
            "logits_wsi": logits_wsi,
            "logits_omic": logits_omic,
            "base_logits": base_logits,
            "channel_logits": channel_delta,
        }

        return output, mi_metrics
