"""Transport cost and solver kernels with no historical v2 runtime dependency.

The independent-plan helper and both solver methods were extracted verbatim
by AST boundaries from transport_v2/core.py. Balanced costs and plans continue
to use the published v1 TransportCore; no predictor or objective is introduced.
"""
import math

import torch
from torch.nn import functional as F

from ..transport.core import TransportCore, pair_tokens


def realized_independent_plan(plan):
    """Remove topology without changing positive realized mass or marginals."""
    if plan.ndim < 2 or not plan.is_floating_point():
        raise ValueError("plan must be a floating transport matrix")
    if not bool(torch.isfinite(plan).all()) or bool((plan < 0).any()):
        raise ValueError("plan must be finite and nonnegative")
    rows, cols = plan.sum(-1), plan.sum(-2)
    mass = rows.sum(-1)
    safe_mass = torch.where(mass > 0, mass, torch.ones_like(mass))
    # Normalized factors avoid underflow in r*c for very small positive mass.
    return mass[..., None, None] * (rows / safe_mass[..., None])[..., :, None] * (
        cols / safe_mass[..., None])[..., None, :]


class DCTTransportSolver(TransportCore):
    """Reuse balanced kernels and supply bounded costs and KL-relaxed plans."""

    def _costs_and_marginals(self, wsi, omics):
        cfg = self.config
        if cfg.cost_mode == "legacy":
            return super()._costs_and_marginals(wsi, omics)
        contents = pair_tokens(wsi, omics)
        stage_cost = torch.sigmoid(self.stage_pair_cost(contents)).permute(0, 3, 1, 2)
        cosine = torch.bmm(F.normalize(wsi, dim=-1), F.normalize(omics, dim=-1).transpose(1, 2))
        distance = torch.cdist(wsi, omics, p=2)
        base = {
            "cosine": (1. - cosine.clamp(-1., 1.)) / 2.,
            # Raw norms remain informative; this is not normalized squared L2,
            # which would be algebraically identical to the cosine branch.
            "euclidean": distance / (distance + cfg.euclidean_scale),
            "positive_dot": torch.sigmoid(-torch.bmm(wsi, omics.transpose(1, 2)) / math.sqrt(cfg.dim)),
        }
        costs, rows, cols = [], [], []
        for stage in range(cfg.stages):
            code = self.stage_embedding[stage].view(1, 1, 1, -1).expand(*contents.shape[:-1], cfg.dim)
            gate = torch.sigmoid(self.evidence_gate(torch.cat((contents, code), -1)).squeeze(-1))
            costs.append(cfg.cost_scale * torch.stack([
                base[name] + cfg.stage_cost_weight * stage_cost[:, stage]
                + cfg.evidence_cost_weight * (1. - gate) for name in cfg.geometries], 1))
            rows.append(gate.mean(-1).clamp_min(cfg.evidence_mass_floor))
            cols.append(gate.mean(-2).clamp_min(cfg.evidence_mass_floor))
        rows, cols = torch.stack(rows, 1), torch.stack(cols, 1)
        rows, cols = rows / rows.sum(-1, keepdim=True), cols / cols.sum(-1, keepdim=True)
        strength = cfg.marginal_strength
        return torch.stack(costs, 1), ((1.-strength) / rows.size(-1) + strength * rows), (
            (1.-strength) / cols.size(-1) + strength * cols)

    def plans_from_costs(self, costs, rows, cols, epoch):
        cfg = self.config
        if cfg.transport_mode == "balanced":
            self.last_solver_diagnostics = {"mode": "balanced", "actual_marginal_control": False}
            return super().plans_from_costs(costs, rows, cols, epoch)
        if self.plan_mode not in {"learned", "independent"}:
            raise ValueError("v2 plan_mode must be learned or independent")
        if costs.ndim != 5 or rows.shape != (costs.size(0), costs.size(1), costs.size(3)) or (
            cols.shape != (costs.size(0), costs.size(1), costs.size(4))):
            raise ValueError("expected costs [B,S,G,W,O], rows [B,S,W], cols [B,S,O]")
        if not bool(torch.isfinite(costs).all()) or not bool(torch.isfinite(rows).all()) or not bool(
            torch.isfinite(cols).all()) or bool((rows <= 0).any()) or bool((cols <= 0).any()):
            raise ValueError("costs must be finite and reference marginals finite and strictly positive")
        dtype = torch.float64 if costs.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=costs.device.type, enabled=False):
            c, a, b = costs.to(dtype), rows.to(dtype), cols.to(dtype)
            a, b = a / a.sum(-1, keepdim=True), b / b.sum(-1, keepdim=True)
            log_a, log_b = a.log().unsqueeze(2), b.log().unsqueeze(2)
            eps = self.epsilon(epoch)
            log_kernel = log_a[..., :, None] + log_b[..., None, :] - c / eps
            theta_h, theta_g = cfg.tau_wsi / (cfg.tau_wsi + eps), cfg.tau_omics / (cfg.tau_omics + eps)
            u, v = torch.zeros_like(log_kernel[..., 0]), torch.zeros_like(log_kernel[..., 0, :])
            effective_tolerance = max(cfg.uot_tolerance, 8. * torch.finfo(dtype).eps)
            potential_scale = 1.
            change = float("inf")
            for iteration in range(cfg.uot_iters):
                old_u, old_v = u, v
                u = theta_h * (log_a - torch.logsumexp(log_kernel + v[..., None, :], -1))
                v = theta_g * (log_b - torch.logsumexp(log_kernel + u[..., :, None], -2))
                if (iteration + 1) % 10 == 0 or iteration + 1 == cfg.uot_iters:
                    potential_scale = float(torch.maximum(u.abs().amax(), v.abs().amax()).detach().clamp_min(1.))
                    effective_tolerance = max(cfg.uot_tolerance, 8. * torch.finfo(dtype).eps * potential_scale)
                    change = torch.maximum((u-old_u).abs().amax(), (v-old_v).abs().amax()).detach()
                    if bool(change <= effective_tolerance):
                        break
            if not bool(change <= effective_tolerance):
                raise RuntimeError("UOT did not converge within uot_iters; increase the budget or recalibrate tau/eps")
            plan = (log_kernel + u[..., :, None] + v[..., None, :]).exp()
            if not bool(torch.isfinite(plan).all()):
                raise RuntimeError("UOT produced a nonfinite plan; check cost/tau/eps scales")
            self.last_solver_diagnostics = {"mode": "uot", "iterations": iteration + 1,
                "max_log_update": float(change), "effective_tolerance": effective_tolerance,
                "requested_tolerance": cfg.uot_tolerance, "max_log_potential": potential_scale,
                "converged": True,
                "actual_marginal_control": self.plan_mode == "independent"}
            if self.plan_mode == "independent":
                plan = realized_independent_plan(plan)
            return plan, a, b
