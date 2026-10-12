"""Padding invariance and empty-observation contracts for slot extraction."""

from __future__ import annotations

import pytest
import torch

from survot_rank.research.components.slot_attention import MultiHeadSlotAttention


def make_attention(init_mode="deterministic", *, training=False):
    torch.manual_seed(101)
    module = MultiHeadSlotAttention(
        3, 8, heads=2, dim_head=4, iters=2, hidden_dim=12,
        init_mode=init_mode,
    )
    module.train(training)
    return module


def input_tokens(batch=2, tokens=5):
    return torch.randn(batch, tokens, 8, generator=torch.Generator().manual_seed(19))


@pytest.mark.parametrize("init_mode", ["gaussian", "deterministic", "learned"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("padding_value", ["arbitrary", "nan"])
def test_masked_padding_preserves_outputs_and_attention(init_mode, training, padding_value):
    module = make_attention(init_mode, training=training)
    module.capture_attention = True
    inputs = input_tokens()
    padding = torch.randn(2, 4, 8) * 1e4
    if padding_value == "nan":
        padding.fill_(float("nan"))
    padded = torch.cat((inputs, padding), dim=1)
    mask = torch.arange(9)[None, :].expand(2, -1) < 5

    torch.manual_seed(307)
    expected = module(inputs)
    expected_assignment = module.last_token_assignment.clone()
    expected_pooling = module.last_pooling_attention.clone()
    torch.manual_seed(307)
    actual = module(padded, token_mask=mask)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(module.last_token_assignment[:, :, :5], expected_assignment)
    torch.testing.assert_close(module.last_pooling_attention[:, :, :5], expected_pooling)
    assert torch.count_nonzero(module.last_token_assignment[:, :, 5:]) == 0
    assert torch.count_nonzero(module.last_pooling_attention[:, :, 5:]) == 0
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("init_mode", ["gaussian", "deterministic", "learned"])
@pytest.mark.parametrize("training", [False, True])
def test_all_valid_mask_is_exactly_compatible_with_omitted_mask(init_mode, training):
    module = make_attention(init_mode, training=training)
    inputs = input_tokens()
    state_keys = tuple(module.state_dict())
    torch.manual_seed(311)
    expected = module(inputs)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(311)
    actual = module(inputs, token_mask=torch.ones(2, 5, dtype=torch.bool))
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert tuple(module.state_dict()) == state_keys


def test_unmasked_historical_output_fixture():
    # Recorded before adding token_mask; this guards the pre-existing forward
    # independently of the new masked branch.
    expected = torch.tensor([
        [
            [-.5428754091, -.0695136189, -.3891715705, -.1564584225, -1.6005306244, -.1978270859, -.7108727694, -.6062488556],
            [-.0933766365, -.0287957937, .0923247039, -.0139996409, -1.6211810112, -.7326570749, -.6030507088, -.0850786865],
            [-.3938045800, .3868255913, .0288008451, .3865168393, -1.2594300508, -.8097719550, -.9216445684, -.1782594621],
        ],
        [
            [-.4394833744, -.1812767833, -.3075073659, -.2144044638, -1.5335524082, -.1852719784, -.8037527800, -.6953182220],
            [-.0134738982, -.0861871243, .2018120140, -.0783393607, -1.5408656597, -.7017118931, -.6707814336, -.1003854573],
            [-.2596288323, .3493655622, .1757009029, .3168180585, -1.1544508934, -.7940800190, -1.0428733826, -.1913622916],
        ],
    ])
    torch.testing.assert_close(make_attention()(input_tokens()), expected, atol=2e-6, rtol=2e-6)


def test_padding_has_zero_gradient_and_preserves_valid_token_gradient():
    module = make_attention(training=True)
    inputs = input_tokens().requires_grad_()
    torch.manual_seed(313)
    module(inputs).square().sum().backward()
    expected_gradient = inputs.grad.clone()
    module.zero_grad(set_to_none=True)

    padded = torch.cat((inputs.detach(), torch.full((2, 3, 8), float("nan"))), dim=1)
    padded.requires_grad_()
    mask = torch.arange(8)[None, :].expand(2, -1) < 5
    torch.manual_seed(313)
    module(padded, token_mask=mask).square().sum().backward()
    torch.testing.assert_close(padded.grad[:, :5], expected_gradient, atol=3e-6, rtol=3e-6)
    assert torch.count_nonzero(padded.grad[:, 5:]) == 0
    assert torch.isfinite(padded.grad).all()


@pytest.mark.parametrize("tokens", [0, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_all_invalid_tokens_return_graph_connected_zeros(tokens, dtype):
    module = make_attention(training=True).to(dtype=dtype)
    module.capture_attention = True
    inputs = torch.full((2, tokens, 8), float("nan"), dtype=dtype, requires_grad=True)
    output = module(inputs, token_mask=torch.zeros(2, tokens, dtype=torch.bool))
    assert output.requires_grad
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(module.last_token_assignment) == 0
    assert torch.count_nonzero(module.last_pooling_attention) == 0
    output.sum().backward()
    assert inputs.grad is not None
    assert torch.count_nonzero(inputs.grad) == 0
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) == 0


def test_variable_valid_counts_and_empty_patients_in_one_batch():
    module = make_attention()
    module.capture_attention = True
    inputs = input_tokens(batch=3, tokens=7)
    mask = torch.tensor([
        [False, True, False, True, True, False, False],
        [True, False, False, False, False, False, False],
        [False, False, False, False, False, False, False],
    ])
    inputs[~mask] = float("nan")
    output = module(inputs, token_mask=mask)
    assignment = module.last_token_assignment.clone()
    pooling = module.last_pooling_attention.clone()
    for patient in range(2):
        expected = module(inputs[patient:patient + 1, mask[patient]])
        torch.testing.assert_close(output[patient:patient + 1], expected, atol=2e-6, rtol=2e-6)
    assert torch.count_nonzero(output[2]) == 0
    assert torch.count_nonzero(assignment.masked_select(~mask[:, None, :])) == 0
    assert torch.count_nonzero(pooling.masked_select(~mask[:, None, :])) == 0
    torch.testing.assert_close(pooling[:2].sum(dim=-1), torch.ones(2, 3))
    assert torch.count_nonzero(pooling[2]) == 0


@pytest.mark.parametrize("bad_mask", [None, [[True] * 5] * 2, torch.ones(2, 5), torch.ones(2, 5, dtype=torch.int64)])
def test_mask_requires_boolean_tensor(bad_mask):
    if bad_mask is None:
        # Explicit None is the public legacy default, not an invalid mask.
        assert torch.isfinite(make_attention()(input_tokens(), token_mask=None)).all()
    else:
        with pytest.raises(TypeError, match="token_mask.*torch.bool"):
            make_attention()(input_tokens(), token_mask=bad_mask)


@pytest.mark.parametrize("shape", [(5,), (2, 4), (1, 5), (2, 5, 1)])
def test_mask_shape_validation(shape):
    with pytest.raises(ValueError, match="token_mask must have shape"):
        make_attention()(input_tokens(), token_mask=torch.ones(shape, dtype=torch.bool))


def test_nonmaterialized_mask_device_has_clear_error():
    mask = torch.ones(2, 5, dtype=torch.bool, device="meta")
    with pytest.raises(ValueError, match="token_mask could not be moved.*inputs device"):
        make_attention()(input_tokens(), token_mask=mask)
