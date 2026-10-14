"""Parameter-free centered pair reading and collision-safe feature separation.

The collision direction is an explicit generalized-gradient convention for an
actual nonsmooth Euclidean-distance hinge, not a fabricated derivative of the
old squared-distance objective. It is used only at exact represented equality.
"""
import math

import torch
from torch.nn import functional as F

from survot_rank.research.methods.dct_v2.transport import realized_independent_plan


def centered_pair_coactivation(plans, wsi_logits, omics_logits, *, mass_aware=True):
    """Return [B,S,C] mass-weighted risk covariance averaged over geometries.

    Each geometry is centered against its own exact realized-margin independent
    plan. No local ratio gates or second mass multiplier are applied. Constant
    slot responses and independent matching therefore give zero up to rounding.
    """
    if plans.ndim != 5 or wsi_logits.ndim != 3 or omics_logits.ndim != 3:
        raise ValueError("expected plans [B,S,G,W,O] and per-modality logits [B,K,C]")
    if (plans.size(0) != wsi_logits.size(0) or plans.size(0) != omics_logits.size(0)
            or plans.size(-2) != wsi_logits.size(1) or plans.size(-1) != omics_logits.size(1)
            or wsi_logits.size(-1) != omics_logits.size(-1)):
        raise ValueError("plan and slot logit shapes disagree")
    if not isinstance(mass_aware, bool):
        raise ValueError("mass_aware must be boolean")
    if not all(v.is_floating_point() and v.device == plans.device and bool(torch.isfinite(v).all())
               for v in (plans, wsi_logits, omics_logits)) or bool((plans < 0).any()):
        raise ValueError("finite floating logits and nonnegative plans on one device are required")
    dtype = torch.float64 if plans.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=plans.device.type, enabled=False):
        p = plans.to(dtype)
        difference = p-realized_independent_plan(p)
        fw, fo = torch.tanh(wsi_logits.to(dtype)), torch.tanh(omics_logits.to(dtype))
        result = torch.einsum("bsgwo,bwc,boc->bsgc", difference, fw, fo)
        if not mass_aware:
            mass = p.sum((-2, -1))
            safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
            result = result/safe_mass[..., None]
        return result.mean(2)


def _stable_norm(difference):
    scale = difference.abs().amax(-1, keepdim=True)
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale.squeeze(-1)*torch.linalg.vector_norm(difference/safe, dim=-1)


def _slot_codes(count, dim, *, device, dtype):
    if count <= dim:
        return torch.eye(count, dim, device=device, dtype=dtype)
    # Configurations have dim>=2. Distinct circular codes also cover K>D.
    angle = torch.arange(count, device=device, dtype=torch.float64)*(2*math.pi/count)
    result = torch.zeros(count, dim, device=device, dtype=torch.float64)
    result[:, 0], result[:, 1] = torch.cos(angle), torch.sin(angle)
    return result.to(dtype)


class _CollisionDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, difference, tie_direction):
        distance = _stable_norm(difference)
        exact_collision = difference.eq(0).all(-1)
        safe_distance = torch.where(distance > 0, distance, torch.ones_like(distance))
        normal = difference/safe_distance[..., None]
        direction = torch.where(exact_collision[..., None], tie_direction, normal)
        ctx.save_for_backward(direction)
        return distance

    @staticmethod
    def backward(ctx, gradient):
        (direction,) = ctx.saved_tensors
        return gradient[..., None]*direction, None


def collision_safe_separation(slots, margin=1.):
    """True L2-margin hinge; exact-zero directions follow distinct slot codes.

    At nonzero distances this has the ordinary L2 gradient. At a complete
    collision the chosen gradient is the limit along s_i=s+epsilon*code_i.
    Codes never enter the forward loss value or the model representation.
    Upstream degenerate Jacobians can still block escape; no global guarantee.
    """
    if slots.ndim != 3 or slots.size(0) < 1 or slots.size(1) < 1 or slots.size(2) < 2:
        raise ValueError("slots must be nonempty floating [B,K,D], D>=2")
    if not slots.is_floating_point() or not bool(torch.isfinite(slots).all()):
        raise ValueError("slots must be finite floating tensors")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    if slots.size(1) == 1:
        return slots.sum()*0.
    dtype = torch.float64 if slots.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=slots.device.type, enabled=False):
        s = slots.to(dtype)
        i, j = torch.triu_indices(s.size(1), s.size(1), offset=1, device=s.device)
        codes = _slot_codes(s.size(1), s.size(2), device=s.device, dtype=dtype)
        tie = codes[i]-codes[j]
        tie = tie/_stable_norm(tie)[..., None]
        distance = _CollisionDistance.apply(s[:, i]-s[:, j], tie[None])
        return F.relu(margin-distance).mean()
