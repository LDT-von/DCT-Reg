"""Optional query-conditioned WSI feature reconstruction for v3.13."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WSISlotReconstructionDecoder(nn.Module):
    """Decode detached patch targets using semantic slots as the only values.

    A frozen random projection of each target supplies its query. There is no
    target/query residual to the output. This is a self-reconstruction
    regularizer, not a masked-patch prediction or image-generation model.
    """

    def __init__(self, dim: int, num_heads: int, chunk_size: int = 256):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("WSI reconstruction chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.query_projection.requires_grad_(False)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim),
        )

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        keys = self.memory_norm(memory)
        decoded, _ = self.cross_attention(
            self.query_norm(self.query_projection(queries.detach())),
            keys, keys, need_weights=False,
        )
        return decoded + self.output_mlp(self.output_norm(decoded))

    def reconstruction_loss(
        self, memory: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        """Mean cosine distance per patient, then mean over available patients.

        Chunking bounds attention workspace; autograd still retains activations
        for all valid patches until backward. Zero padding never contributes.
        """
        if memory.ndim != 3 or target.ndim != 3:
            raise ValueError("WSI slots and targets must be [B, K/N, D]")
        if memory.size(0) != target.size(0) or memory.size(-1) != target.size(-1):
            raise ValueError("WSI slots and targets must share batch and feature dimensions")
        if valid.shape != target.shape[:2]:
            raise ValueError("WSI valid patch mask must have shape [B, N]")
        target = target.detach()
        valid = valid.to(device=target.device, dtype=torch.bool)
        counts = valid.sum(dim=1)
        available = counts > 0
        if not bool(available.any()):
            return memory.sum() * 0.0
        target = target.masked_fill(~valid.unsqueeze(-1), 0.0)
        sums = memory.new_zeros(memory.size(0))
        for start in range(0, target.size(1), self.chunk_size):
            stop = start + self.chunk_size
            chunk = target[:, start:stop]
            prediction = self(memory, chunk)
            distances = 1.0 - F.cosine_similarity(prediction, chunk, dim=-1)
            sums = sums + distances.masked_fill(~valid[:, start:stop], 0.0).sum(dim=1)
        return (sums[available] / counts[available].to(sums.dtype)).mean()
