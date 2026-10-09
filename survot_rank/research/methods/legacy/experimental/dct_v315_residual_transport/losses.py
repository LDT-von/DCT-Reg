"""Version-local stable discrete survival likelihood (c=1 means censored).

The standard cumulative-likelihood algebra is retained from the v3.14 audit;
there is no runtime dependency on that version. No auxiliary loss lives here.
"""
import torch
from torch import nn
import torch.nn.functional as F


def discrete_nll(logits, y, c, alpha=0.0):
    if logits.ndim != 2 or min(logits.shape) < 1:
        raise ValueError("logits must be nonempty [B, C]")
    y = torch.as_tensor(y, device=logits.device)
    c = torch.as_tensor(c, device=logits.device).reshape(-1)
    if y.ndim == 2 and y.size(-1) > 1:
        if y.shape != logits.shape or not torch.all((y == 0) | (y == 1)) or not torch.all(y.sum(-1) == 1):
            raise ValueError("labels must be time bins or valid one-hot labels")
        y = y.argmax(-1)
    y = y.reshape(-1)
    if y.numel() != len(logits) or c.numel() != len(logits):
        raise ValueError("labels and censorship must match batch size")
    if not torch.isfinite(y).all() or not torch.all(y == y.long()) or (y < 0).any() or (y >= logits.size(1)).any():
        raise ValueError("invalid discrete time bin")
    if not torch.all((c == 0) | (c == 1)) or not 0 <= alpha <= 1:
        raise ValueError("binary censorship and alpha in [0,1] required")
    h = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    logs = F.logsigmoid(-h).cumsum(-1)
    index = y.long()[:, None]
    event = -(F.pad(logs, (1, 0)).gather(1, index) + F.logsigmoid(h).gather(1, index)).squeeze(1)
    censored = -logs.gather(1, index).squeeze(1)
    return (1-c)*event + (1-alpha)*c*censored


class StableNLLSurvLoss(nn.Module):
    """Sum reduction: the shared trainer divides by B exactly once."""
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, h, y, t=None, c=None):
        if c is None:
            raise ValueError("censorship is required")
        return discrete_nll(h, y, c, self.alpha).sum()
