"""Independent mathematical audit; no DCT module imports or model edits.

At an exact Euclidean-norm cusp, backward explicitly selects a convex norm
subgradient. This is not a smooth derivative, a finite-difference surrogate,
an anchor penalty, or a claim about patient outcomes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def fixed_codes(k: int, d: int, *, dtype: torch.dtype, device=None):
    if k <= 1 or d < 2:
        raise ValueError("This audit requires K >= 2 and D >= 2")
    if k <= d:
        return torch.eye(k, d, dtype=dtype, device=device)
    angles = torch.arange(k, dtype=dtype, device=device) * (2 * math.pi / k)
    codes = torch.zeros(k, d, dtype=dtype, device=device)
    codes[:, 0] = angles.cos()
    codes[:, 1] = angles.sin()
    return codes


def stable_length_and_direction(delta):
    """No near-zero threshold; exact zero checks represented components."""
    scale = delta.abs().amax(dim=-1)
    zero = scale == 0
    scaled = delta / torch.where(zero, torch.ones_like(scale), scale)[..., None]
    scaled_length = torch.linalg.vector_norm(scaled, dim=-1)
    direction = scaled / torch.where(zero, torch.ones_like(scaled_length), scaled_length)[..., None]
    return scale * scaled_length, direction, zero


class SelectedNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, zero_direction):
        length, direction, zero = stable_length_and_direction(delta)
        chosen = torch.where(zero[..., None], zero_direction, direction)
        ctx.save_for_backward(chosen)
        return length

    @staticmethod
    def backward(ctx, grad_output):
        (chosen,) = ctx.saved_tensors
        return grad_output[..., None] * chosen, None


def selected_distance_hinge(slots, margin=1.0):
    """The forward is exactly mean relu(margin - Euclidean distance)."""
    dtype = torch.float64 if slots.dtype == torch.float64 else torch.float32
    calculation = slots.to(dtype=dtype)
    k, d = calculation.shape[-2:]
    codes = fixed_codes(k, d, dtype=dtype, device=slots.device)
    ij = torch.triu_indices(k, k, offset=1, device=slots.device)
    _, zero_direction, _ = stable_length_and_direction(codes[ij[0]] - codes[ij[1]])
    delta = calculation[..., ij[0], :] - calculation[..., ij[1], :]
    distances = SelectedNorm.apply(delta, zero_direction)
    return torch.relu(margin - distances).mean()


def ordinary_squared_hinge(slots, margin=1.0):
    k = slots.shape[-2]
    ij = torch.triu_indices(k, k, offset=1)
    sq = (slots[ij[0]] - slots[ij[1]]).square().sum(-1)
    return torch.relu(margin - sq).mean()


def main():
    cases = []
    for dtype in (torch.float16, torch.float32, torch.float64):
        for k, d in ((2, 2), (4, 8), (6, 4), (32, 16)):
            parameter = torch.nn.Parameter(torch.full((k, d), 0.25, dtype=dtype))
            before = selected_distance_hinge(parameter)
            before.backward()
            grad = parameter.grad.detach().clone()
            finite = bool(torch.isfinite(grad).all())
            norms = grad.float().norm(dim=-1)
            # A direct slot-parameter step is a constructive escape witness.
            with torch.no_grad():
                parameter -= 0.05 * grad
            after = selected_distance_hinge(parameter)
            record = dict(dtype=str(dtype), k=k, d=d, before=float(before),
                          after=float(after), min_slot_grad=float(norms.min()),
                          net_force=float(grad.double().sum(0).norm()), finite=finite)
            assert finite and bool((norms > 0).all())
            assert float(after) < float(before), record
            cases.append(record)

    collapse = torch.zeros(4, 8, dtype=torch.float64, requires_grad=True)
    ordinary_squared_hinge(collapse).backward()
    assert not bool(collapse.grad.any())

    # Ordinary differentiable domain: custom backward must match gradcheck.
    torch.manual_seed(7)
    away = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    gradcheck = torch.autograd.gradcheck(lambda x: selected_distance_hinge(x, 10), (away,))
    assert gradcheck

    near = []
    for dtype, value in ((torch.float16, 1e-4), (torch.float16, 1e-8),
                         (torch.float32, 1e-7), (torch.float32, 1e-40),
                         (torch.float64, 1e-100)):
        delta = torch.tensor([[value, 0.0]], dtype=dtype).to(
            torch.float64 if dtype == torch.float64 else torch.float32
        ).requires_grad_()
        tie = torch.tensor([[0.0, 1.0]], dtype=delta.dtype)
        length = SelectedNorm.apply(delta, tie)
        length.sum().backward()
        represented_zero = bool((delta.detach() == 0).all())
        expected = tie if represented_zero else torch.tensor([[1.0, 0.0]], dtype=delta.dtype)
        assert torch.allclose(delta.grad, expected)
        assert bool(torch.isfinite(delta.grad).all())
        near.append(dict(input_dtype=str(dtype), requested_value=value,
                         represented_zero=represented_zero, length=float(length),
                         selected_direction=delta.grad.tolist()))

    # Permutation is equivariant when the fixed slot identities permute too.
    # Scalar forward itself is still invariant; no code participates in forward.
    sample = torch.randn(6, 4, dtype=torch.float64)
    perm = torch.tensor([5, 1, 3, 0, 2, 4])
    assert torch.allclose(selected_distance_hinge(sample), selected_distance_hinge(sample[perm]))

    # Anchored slotwise nonlinearity preserves a same-budget topology contrast.
    # g=h=(-1,+1), a=b=(.5,.5), q=1; phi is ReLU solely for this witness.
    h = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    g = h.clone()
    diag = 0.5 * torch.eye(2, dtype=torch.float64)
    anti = diag.flip(1)

    def anchored(plan):
        rows, cols = plan.sum(1), plan.sum(0)
        moved_h = plan.T @ h / cols
        moved_g = plan @ g / rows
        qrow = rows / 0.5
        qcol = cols / 0.5
        left = (torch.relu(g + 0.5 * qcol * (moved_h - g)) - torch.relu(g)).mean()
        right = (torch.relu(h + 0.5 * qrow * (moved_g - h)) - torch.relu(h)).mean()
        return 0.5 * (left + right)

    topology = dict(diagonal=float(anchored(diag)), antidiagonal=float(anchored(anti)),
                    mass=float(diag.sum()), rows=diag.sum(1).tolist(), cols=diag.sum(0).tolist())
    assert topology["diagonal"] == 0.0 and topology["antidiagonal"] == -0.5

    output = dict(
        scope="independent mathematical witness; no DCT architecture or patient execution",
        exact_collapse_step_cases=cases,
        old_squared_hinge_exact_collapse_gradient=float(collapse.grad.norm()),
        gradcheck_away_from_cusps=gradcheck,
        exact_cusp_gradcheck="intentionally not asserted: there is no classical derivative",
        near_zero_cases=near,
        anchored_nonlinear_topology_witness=topology,
        caveats=["A direct slot step is not a guarantee through an arbitrary model Jacobian.",
                 "Fixed codes select a backward direction only at exact collisions.",
                 "Distinct representations do not establish biological semantics.",
                 "An anchored nonlinear readout can retain topology; arbitrary odd nonlinearities may still cancel."])
    path = Path(__file__).with_name("symmetry_math_results.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(cases=len(cases), gradcheck=gradcheck, topology=topology,
                          output=str(path)), ensure_ascii=False))


if __name__ == "__main__":
    main()
