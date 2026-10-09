"""Stable discrete-time survival likelihood; c=1 means right censored."""
import torch
from torch import nn
import torch.nn.functional as F


def discrete_nll(logits, y, c, alpha=0.0):
    """Return [B, ...] NLL, including survival through the censoring bin."""
    if logits.ndim not in (2, 3):
        raise ValueError("survival logits must be [B, C] or [B, K, C]")
    y = torch.as_tensor(y, device=logits.device)
    c = torch.as_tensor(c, device=logits.device).reshape(-1)
    if y.ndim == 2 and y.size(-1) == logits.size(-1):
        if not torch.all((y == 0) | (y == 1)) or not torch.all(y.sum(-1) == 1):
            raise ValueError("survival labels must be one-hot")
        y = y.argmax(-1)
    y = y.reshape(-1)
    if y.numel() != logits.size(0) or c.numel() != logits.size(0):
        raise ValueError("survival labels and censorship must match batch size")
    if not torch.isfinite(y).all() or not torch.all(y == y.long()) or (y < 0).any() or (y >= logits.size(-1)).any():
        raise ValueError("invalid discrete survival time bin")
    if not torch.all((c == 0) | (c == 1)):
        raise ValueError("c must be binary: 0=event, 1=censored")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha_surv must be in [0, 1]")
    h = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    log_survival = F.logsigmoid(-h).cumsum(-1)
    survival_before = F.pad(log_survival, (1, 0), value=0.)
    shape = [len(y)] + [1] * (h.ndim - 1)
    index = y.long().view(shape).expand(*h.shape[:-1], 1)
    event = -(survival_before.gather(-1, index) + F.logsigmoid(h).gather(-1, index)).squeeze(-1)
    censored = -log_survival.gather(-1, index).squeeze(-1)
    c = c.view([len(c)] + [1] * (h.ndim - 2))
    return (1-c) * event + (1-alpha) * c * censored


class StableNLLSurvLoss(nn.Module):
    """Sum reduction matches compose_batch_objective's one division by B."""
    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, h, y, t=None, c=None):
        if c is None:
            raise ValueError("censorship is required")
        return discrete_nll(h, y, c, self.alpha).sum()
