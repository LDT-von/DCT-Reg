"""Censor-aware six-objective helpers for independent DCT v2.

Main, slot and native likelihoods include cumulative survival through all
preceding bins. IPCW statistics must be fitted on training patients only.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _labels(prediction, y, c):
    if prediction.ndim not in (2, 3) or prediction.shape[0] < 1 or prediction.shape[-1] < 1:
        raise ValueError("predictions must be [B,C] or [B,K,C]")
    if not prediction.is_floating_point() or not bool(torch.isfinite(prediction).all()):
        raise ValueError("predictions must be finite floating tensors")
    y = torch.as_tensor(y, device=prediction.device)
    c = torch.as_tensor(c, device=prediction.device)
    if y.ndim != 1 or c.ndim != 1 or y.numel() != prediction.size(0) or c.numel() != prediction.size(0):
        raise ValueError("y and censorship must be [B]")
    if not bool(torch.isfinite(y).all()) or bool((y != y.long()).any()) or bool((y < 0).any()) or bool((y >= prediction.size(-1)).any()):
        raise ValueError("y must contain valid integer time-bin labels")
    if not bool(torch.isfinite(c).all()) or bool(((c != 0) & (c != 1)).any()):
        raise ValueError("censorship must be binary (0=event, 1=censored)")
    return y.long(), c.to(prediction.dtype)


def _nll_from_logs(log_h, log_s, y, c, alpha, reduction):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite in [0,1]")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("invalid NLL reduction")
    y, c = _labels(log_h, y, c)
    shape = [log_h.size(0)] + [1] * (log_h.ndim-1)
    indices = y.reshape(shape).expand(*log_h.shape[:-1], 1)
    cum = log_s.cumsum(-1)
    padded = torch.cat((torch.zeros_like(cum[..., :1]), cum), -1)
    event = -(padded.gather(-1, indices).squeeze(-1) + log_h.gather(-1, indices).squeeze(-1))
    censored = -padded.gather(-1, indices+1).squeeze(-1)
    cc = c.reshape([log_h.size(0)] + [1]*(log_h.ndim-2))
    loss = (1-cc)*event + (1-alpha)*cc*censored
    return loss if reduction == "none" else (loss.mean() if reduction == "mean" else loss.sum())


def logit_nll(logits, y, c, alpha=0., reduction="mean"):
    """Stable discrete NLL for raw logits, matching the shared DCT trainer."""
    _labels(logits, y, c)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    z = logits.to(dtype)
    return _nll_from_logs(F.logsigmoid(z), F.logsigmoid(-z), y, c, alpha, reduction)


def discrete_nll(hazards, y, c, alpha=0., reduction="mean"):
    """Probability-input counterpart, useful for exact hand-calculated checks."""
    _labels(hazards, y, c)
    if bool(((hazards < 0) | (hazards > 1)).any()):
        raise ValueError("hazards must be probabilities")
    dtype = torch.float64 if hazards.dtype == torch.float64 else torch.float32
    h = hazards.to(dtype).clamp(1e-7, 1-1e-7)
    return _nll_from_logs(h.log(), torch.log1p(-h), y, c, alpha, reduction)


class StableNLLSurvLoss(nn.Module):
    """Sum over patients; the shared trainer divides by B exactly once."""
    def __init__(self, alpha=0.):
        super().__init__()
        self.alpha = alpha

    def forward(self, h, y, t=None, c=None):
        if c is None:
            raise ValueError("censorship is required")
        return logit_nll(h, y, c, self.alpha, "sum")


def censor_km(times, censorship):
    times = torch.as_tensor(times, dtype=torch.float32).flatten()
    c = torch.as_tensor(censorship, device=times.device, dtype=torch.float32).flatten()
    if not times.numel() or times.shape != c.shape or not bool(torch.isfinite(times).all()) or bool((times < 0).any()):
        raise ValueError("training follow-up times must be finite nonnegative and nonempty")
    if not bool(torch.isfinite(c).all()) or bool(((c != 0) & (c != 1)).any()):
        raise ValueError("training censorship must be binary and paired with times")
    unique = torch.unique(times, sorted=True)
    values = []
    value = times.new_ones(())
    for t in unique:
        risk = (times >= t).sum()
        count = ((times == t) & (c == 1)).sum()
        value = value * (1-count.to(times.dtype)/risk)
        values.append(value)
    return unique, torch.stack(values)


def inverse_censor_weight(query_times, reference_times, reference_survival, *, power=1., max_weight=10.):
    """G(t-) at tied event/censor times; stabilization is explicit clipping."""
    if not reference_times.numel():
        raise RuntimeError("fit the training-fold censor reference before IPCW objectives")
    query_times = query_times.to(reference_times.dtype)
    indices = torch.searchsorted(reference_times, query_times.to(reference_times.dtype).contiguous(), right=False)-1
    g = torch.ones_like(query_times)
    valid = indices >= 0
    g[valid] = reference_survival[indices[valid]].to(g.dtype)
    return g.clamp_min(1e-7).pow(-power).clamp_max(max_weight)


def risk_from_logits(logits):
    return -torch.cumprod(1-torch.sigmoid(logits.float()), -1).sum(-1)


def ipcw_pairwise_ranking_loss(logits, times, c, weights, *, margin=.02, temperature=.5, memory=None):
    """Current-current/current-memory comparable pairs, never memory-memory."""
    risk = risk_from_logits(logits)
    current_count = risk.numel()
    if memory is not None:
        mr, mt, mc, mw = (v.detach().to(risk.device) for v in memory)
        risk = torch.cat((risk, mr))
        times, c, weights = torch.cat((times, mt)), torch.cat((c, mc)), torch.cat((weights, mw))
    comparable = (c[:, None] == 0) & (times[:, None] < times[None, :])
    mask = torch.arange(risk.numel(), device=risk.device) < current_count
    comparable &= mask[:, None] | mask[None, :]
    count = comparable.sum().detach()
    if not bool(comparable.any()):
        return risk.sum()*0., count
    pair_weights = weights[:, None].expand_as(comparable)[comparable]
    differences = risk[:, None]-risk[None, :]
    losses = temperature*F.softplus((margin-differences)/temperature)
    return (losses[comparable]*pair_weights).sum()/pair_weights.sum().clamp_min(1e-7), count
