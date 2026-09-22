from __future__ import annotations

import io
import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from survot_rank.research.methods.legacy.experimental.dct_v315_residual_transport.model import (
    DCTV315ResidualTransport, OneShotSlots, ResidualTransportInteraction, uniform_sinkhorn,
)
from survot_rank.research.methods.legacy.experimental.dct_v315_residual_transport.losses import (
    StableNLLSurvLoss, discrete_nll,
)


def arguments(**overrides):
    options = dict(encoding_dim=16, wsi_projection_dim=32, n_classes=4,
        omic_sizes=[3, 4, 5], slot_num_wsi=4, slot_num_omics=3,
        alpha_surv=0., bag_loss='nll_surv', rna_format='Pathways', survot_method='dct_v315',
        dct_v315_dropout=0., dct_v315_ot_epsilon=.2, dct_v315_ot_iters=40,
        dct_v315_transport_mode='ot', cur_epoch=7)
    options.update(overrides)
    return SimpleNamespace(**options)


def batch(batch_size=4, patches=17, encoding_dim=16, widths=(3, 4, 5)):
    generator = torch.Generator().manual_seed(102)
    result = dict(x_wsi=torch.randn(batch_size, patches, encoding_dim, generator=generator),
                  y=torch.arange(batch_size) % 4, event_time=torch.arange(1, batch_size+1).float(),
                  c=(torch.arange(batch_size) % 3 == 0).float())
    for i, width in enumerate(widths, 1):
        result[f'x_omic{i}'] = torch.randn(batch_size, width, generator=generator)
    return result


def test_catalog_config_factory_and_nll_integration():
    from survot_rank.config import load_config, config_to_argv, apply_overrides
    from survot_rank.training.extended_args import build_base_parser
    from survot_rank.training.model_factory import get_model
    from survot_rank.training.train_runner import init_loss_function
    from survot_rank.research.methods.catalog import METHOD_CATALOG, PRIMARY_METHOD
    config = load_config('configs/dct_v315_blca_uni.yaml')
    parsed = build_base_parser().parse_args(config_to_argv(config))
    assert parsed.alpha_surv == 0 and parsed.slot_num_wsi == 8
    assert isinstance(init_loss_function(parsed), StableNLLSurvLoss)
    assert type(get_model('dct_v315', arguments())).__name__ == 'DCTV315ResidualTransport'
    assert METHOD_CATALOG['dct_v315_residual_transport'].status == 'candidate'
    assert PRIMARY_METHOD == 'dct_v310_directional_regularized_transport'
    alternate = build_base_parser().parse_args(config_to_argv(apply_overrides(config, ['dct_v315_transport_mode=independent'])))
    assert alternate.dct_v315_transport_mode == 'independent'

    ranked = load_config('configs/dct_v315_rti_rank_blca_uni2h.yaml')
    ranked_args = build_base_parser().parse_args(config_to_argv(ranked))
    control_args = build_base_parser().parse_args(config_to_argv(
        load_config('configs/dct_v315_blca_uni2h.yaml')
    ))
    assert ranked_args.dct_v315_lambda_rti_rank == pytest.approx(0.10)
    assert ranked_args.dct_v315_rti_rank_margin == pytest.approx(0.02)
    assert ranked_args.dct_v315_rti_rank_temperature == pytest.approx(0.50)
    assert ranked_args.dct_v315_ipcw_max_weight == pytest.approx(10.0)
    differences = {
        key for key, value in vars(control_args).items()
        if value != vars(ranked_args)[key]
    }
    assert differences == {
        'results_dir', 'specific_simple', 'dct_v315_lambda_rti_rank'
    }


def test_no_old_version_runtime_dependency():
    program = '''
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if 'survot_rank.research.methods' in name and 'dct_v315' not in name:
        raise AssertionError('Old method imported: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from survot_rank.research.methods.legacy.experimental.dct_v315_residual_transport.model import DCTV315ResidualTransport
import torch
assert DCTV315ResidualTransport.__bases__ == (torch.nn.Module,)
'''
    result = subprocess.run([sys.executable, '-c', program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('censor', [0., 1.])
@pytest.mark.parametrize('time_bin', range(4))
def test_nll_by_hand(censor, time_bin):
    logits = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    loss = discrete_nll(logits, torch.tensor([time_bin]*2), torch.tensor([censor]*2))
    torch.testing.assert_close(loss, torch.full((2,), (time_bin+1)*math.log(2), dtype=torch.float64))
    loss.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_extreme_logits_and_invalid_labels():
    logits = torch.tensor([[100., -100., 100., -100.]], requires_grad=True)
    loss = discrete_nll(logits, [3], [0]).sum()
    loss.backward()
    assert loss > 200 and logits.grad[0, 3] < -.9
    for y, c in (([4], [0]), ([.5], [0]), ([0], [2])):
        with pytest.raises(ValueError):
            discrete_nll(logits, y, c)
    plain = discrete_nll(torch.zeros(2, 4), [0, 3], [0, 1])
    one_hot = discrete_nll(torch.zeros(2, 4), torch.nn.functional.one_hot(torch.tensor([0, 3]), 4), [0, 1])
    torch.testing.assert_close(plain, one_hot)


@pytest.mark.parametrize('shape', [(1, 1), (2, 4), (8, 8), (16, 3)])
def test_sinkhorn_marginals_and_gradients(shape):
    cost = (torch.rand(2, *shape)*2).requires_grad_()
    plan = uniform_sinkhorn(cost)
    assert (plan >= 0).all()
    torch.testing.assert_close(plan.sum(-1), torch.full((2, shape[0]), 1/shape[0]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(plan.sum(-2), torch.full((2, shape[1]), 1/shape[1]), atol=2e-6, rtol=1e-5)
    (plan*cost).sum().backward()
    assert torch.isfinite(cost.grad).all()


def test_independent_coupling_zeroes_only_interaction():
    model = DCTV315ResidualTransport(arguments(dct_v315_transport_mode='independent')).eval()
    logits, aux = model(**batch())
    evidence = model.explain_last_batch()
    assert aux.item() == 0
    assert torch.count_nonzero(evidence['interaction_logits']) == 0
    assert torch.count_nonzero(evidence['pair_logit_contributions']) == 0
    torch.testing.assert_close(logits, evidence['global_logits'], rtol=0, atol=0)


def test_plan_really_changes_bilinear_interaction():
    w = torch.tensor([[[1., -1.], [-1., 1.]]])
    diagonal = torch.eye(2)[None]/2
    reversed_plan = diagonal.flip(-1)
    first = ResidualTransportInteraction.interaction(w, w, diagonal)
    second = ResidualTransportInteraction.interaction(w, w, reversed_plan)
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, -torch.ones_like(second))


def test_transport_path_has_gradients_through_plan_and_cost():
    torch.manual_seed(119)
    layer = ResidualTransportInteraction(16, 4)
    w = torch.randn(3, 4, 16, requires_grad=True)
    o = torch.randn(3, 3, 16, requires_grad=True)
    delta, _, plan, *_ = layer(w, o, torch.ones(3, dtype=torch.bool))
    plan_gradient = torch.autograd.grad(delta.square().sum(), plan, retain_graph=True)[0]
    cost_path = torch.autograd.grad(plan.square().sum(), [w, o])
    assert plan_gradient.abs().sum() > 0 and torch.isfinite(plan_gradient).all()
    assert all(g.abs().sum() > 0 and torch.isfinite(g).all() for g in cost_path)


def test_switching_to_independent_keeps_base_but_removes_rti_increment():
    torch.manual_seed(118)
    model = DCTV315ResidualTransport(arguments()).eval()
    data = batch()
    ot = model(**data)[0]
    evidence = model.explain_last_batch()
    model.transport.mode = 'independent'
    null = model(**data)[0]
    torch.testing.assert_close(null, evidence['global_logits'], rtol=0, atol=0)
    assert (ot-null).abs().max() > 1e-5
    torch.testing.assert_close(ot-null, evidence['interaction_logits'], atol=1e-6, rtol=1e-5)


def test_content_offset_does_not_create_interaction():
    torch.manual_seed(5)
    layer = ResidualTransportInteraction(16, 4)
    w, o = torch.randn(3, 4, 16), torch.randn(3, 3, 16)
    paired = torch.ones(3, dtype=torch.bool)
    original = layer(w, o, paired)
    shifted = layer(w+torch.randn(3, 1, 16), o+torch.randn(3, 1, 16), paired)
    torch.testing.assert_close(original[0], shifted[0], atol=2e-6, rtol=1e-5)


def test_identical_tokens_cannot_fake_distinct_slots_or_interactions():
    slots = OneShotSlots(16, 4)
    tokens = torch.randn(2, 1, 16).expand(-1, 10, -1)
    result, attention = slots(tokens)
    torch.testing.assert_close(result, tokens[:, :1].expand(-1, 4, -1))
    layer = ResidualTransportInteraction(16, 4)
    delta, moment, *_ = layer(result, result, torch.ones(2, dtype=torch.bool))
    torch.testing.assert_close(moment, torch.zeros_like(moment), atol=1e-7, rtol=0)
    assert torch.isfinite(delta).all()


def test_patch_and_slot_permutation_invariance():
    torch.manual_seed(90)
    model = DCTV315ResidualTransport(arguments()).eval()
    data = batch()
    before = model(**data)[0]
    data['x_wsi'] = data['x_wsi'].flip(1)
    with torch.no_grad():
        model.wsi_slots.queries.copy_(model.wsi_slots.queries.flip(0))
        model.omic_slots.queries.copy_(model.omic_slots.queries.flip(0))
    after = model(**data)[0]
    torch.testing.assert_close(before, after, atol=3e-6, rtol=1e-5)


def test_logits_additive_explanation_and_cached_weight():
    model = DCTV315ResidualTransport(arguments()).eval()
    logits, _ = model(**batch())
    with torch.no_grad():
        model.transport.readout.weight.add_(10.)
    evidence = model.explain_last_batch()
    pair_sum = evidence['pair_logit_contributions'].sum((1, 2))
    torch.testing.assert_close(pair_sum, evidence['interaction_logits'], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits, evidence['global_logits']+pair_sum, atol=1e-6, rtol=1e-5)


def test_one_nll_reaches_every_intended_parameter():
    torch.manual_seed(45)
    model = DCTV315ResidualTransport(arguments()).train()
    data = batch()
    logits, aux = model(**data)
    loss = StableNLLSurvLoss()(logits, data['y'], c=data['c'])/len(logits) + aux
    loss.backward()
    assert aux.item() == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert model.objective_weights() == dict(nll=1., rti_ipcw_rank=0., per_slot_nll=0., slot_diversity=0., reconstruction=0.)


def test_rti_directed_ipcw_rank_uses_hard_pairs_and_only_updates_correction():
    model = DCTV315ResidualTransport(arguments(
        dct_v315_lambda_rti_rank=.1,
        dct_v315_rti_rank_margin=.02,
        dct_v315_rti_rank_temperature=.5,
        dct_v315_ipcw_max_weight=10.,
    ))
    model.configure_train_reference(
        torch.tensor([1., 2., 2., 3., 4.]),
        torch.tensor([0., 1., 0., 0., 1.]),
    )
    # G(2-) excludes censoring tied at time 2; G(3-) includes it.
    torch.testing.assert_close(
        model._ipcw(torch.tensor([2., 3.])), torch.tensor([1., 4. / 3.])
    )

    base = torch.zeros(3, 4, requires_grad=True)
    delta = torch.zeros(3, 4, requires_grad=True)
    loss, pairs, hard_pairs = model._rti_directed_ipcw_rank(
        base, delta, torch.tensor([1., 2., 3.]), torch.zeros(3), torch.ones(3, dtype=torch.bool)
    )
    assert pairs.item() == 3
    assert hard_pairs.item() == 3
    assert loss > 0
    loss.backward()
    assert base.grad is None
    assert delta.grad[0].sum() < 0
    assert delta.grad[-1].sum() > 0

    correct_base = torch.tensor([[5., 5., 5., 5.], [-5., -5., -5., -5.]])
    zero_delta = torch.zeros_like(correct_base, requires_grad=True)
    zero_loss, pairs, hard_pairs = model._rti_directed_ipcw_rank(
        correct_base, zero_delta, torch.tensor([1., 2.]), torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
    )
    assert pairs.item() == 1
    assert hard_pairs.item() == 0
    assert zero_loss.item() == 0


def test_ranked_recipe_requires_fold_reference_and_composes_auxiliary():
    args = arguments(dct_v315_lambda_rti_rank=.1)
    model = DCTV315ResidualTransport(args).train()
    data = batch()
    missing_time = dict(data)
    missing_time.pop('event_time')
    with pytest.raises(ValueError, match='event_time and censorship'):
        model(**missing_time)
    with pytest.raises(RuntimeError, match='configure_train_reference'):
        model(**data)

    model.configure_train_reference(
        torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.]),
        torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]),
    )
    # Equal detached base risk makes all comparable pairs hard, so the RTI
    # branch receives an explicit correction signal.
    with torch.no_grad():
        model.global_head[-1].weight.zero_()
        model.global_head[-1].bias.zero_()
    logits, auxiliary = model(**data)
    assert auxiliary > 0
    torch.testing.assert_close(
        auxiliary, .1 * model.last_training_losses['v315_rti_ipcw_rank']
    )
    assert model.last_training_losses['v315_hard_rank_pairs'] > 0
    loss = StableNLLSurvLoss()(logits, data['y'], c=data['c']) / len(logits) + auxiliary
    loss.backward()
    assert model.transport.readout.weight.grad is not None
    assert torch.isfinite(model.transport.readout.weight.grad).all()
    assert model.objective_weights()['rti_ipcw_rank'] == pytest.approx(.1)


@pytest.mark.parametrize('missing', ['wsi', 'omic'])
def test_missing_modality_ignores_nan_and_zeroes_cross_term(missing):
    model = DCTV315ResidualTransport(arguments()).eval()
    data = batch()
    flags = {f'{missing}_missing': True}
    before = model(**data, **flags)[0]
    keys = ['x_wsi'] if missing == 'wsi' else [f'x_omic{i}' for i in range(1, 4)]
    for key in keys:
        data[key][:] = float('nan')
    after, _ = model(**data, **flags)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.count_nonzero(model.last_explanations['interaction_logits']) == 0
    with pytest.raises(ValueError, match='nonfinite'):
        model(**data)


def test_invalid_hyperparameters_inputs_and_both_missing():
    for params in ({'dct_v315_ot_epsilon': float('nan')}, {'slot_num_wsi': 33},
                   {'omic_sizes': []}, {'dct_v315_dropout': 1.}, {'bag_loss': 'cox_surv'},
                   {'dct_v315_ot_iters': 0}, {'dct_v315_transport_mode': 'bad'},
                   {'dct_v315_lambda_rti_rank': -1.},
                   {'dct_v315_rti_rank_temperature': 0.}):
        with pytest.raises(ValueError):
            DCTV315ResidualTransport(arguments(**params))
    model = DCTV315ResidualTransport(arguments())
    with pytest.raises(ValueError, match='at least one'):
        model(**batch(), wsi_missing=True, omic_missing=True)
    with pytest.raises(ValueError, match='availability'):
        model(**batch(), omic_available=[1, 0])


def test_labels_and_epoch_never_enter_prediction():
    model = DCTV315ResidualTransport(arguments()).eval()
    data = batch()
    before = model(**data, cur_epoch=0)[0]
    data.update(y=torch.full((4,), 999), c=torch.ones(4), event_time=torch.rand(4))
    after = model(**data, cur_epoch=100)[0]
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_batch_one_single_pathway_single_slot_and_constant_values():
    model = DCTV315ResidualTransport(arguments(omic_sizes=[3], slot_num_wsi=1, slot_num_omics=1))
    data = batch(batch_size=1, patches=1, widths=[3])
    logits, aux = model(**data)
    loss = StableNLLSurvLoss()(logits, data['y'], c=data['c']) + aux
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert torch.count_nonzero(model.last_explanations['interaction_logits']) == 0


@pytest.mark.parametrize('rank_weight', [0., .1])
def test_checkpoint_roundtrip_recipe_and_predictions(rank_weight):
    recipe = arguments(dct_v315_lambda_rti_rank=rank_weight)
    model = DCTV315ResidualTransport(recipe).eval()
    before = model(**batch())[0]
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    other = DCTV315ResidualTransport(arguments(
        cur_epoch=0, dct_v315_lambda_rti_rank=rank_weight,
    )).eval()
    other.load_state_dict(state)
    torch.testing.assert_close(before, other(**batch())[0], rtol=0, atol=0)
    wrong = DCTV315ResidualTransport(arguments(dct_v315_ot_epsilon=.1))
    with pytest.raises(RuntimeError, match='recipe'):
        wrong.load_state_dict(state)
    wrong_rank = DCTV315ResidualTransport(arguments(
        dct_v315_lambda_rti_rank=.2 if rank_weight else .1,
    ))
    with pytest.raises(RuntimeError, match='recipe'):
        wrong_rank.load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('amp', [False, True])
def test_config_sized_gpu_forward_backward(amp):
    model = DCTV315ResidualTransport(arguments(encoding_dim=1024, wsi_projection_dim=256,
                      slot_num_wsi=8, slot_num_omics=8)).cuda()
    data = {k:v.cuda() for k,v in batch(batch_size=2, patches=4096, encoding_dim=1024).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=.0005)
    with torch.autocast('cuda', enabled=amp, dtype=torch.float16):
        logits, aux = model(**data)
        loss = StableNLLSurvLoss()(logits, data['y'], c=data['c'])/len(logits)+aux
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('amp', [False, True])
def test_uni2h_sized_ranked_gpu_forward_backward(amp):
    args = arguments(
        encoding_dim=1536,
        wsi_projection_dim=256,
        slot_num_wsi=8,
        slot_num_omics=8,
        dct_v315_lambda_rti_rank=.1,
    )
    model = DCTV315ResidualTransport(args).cuda().train()
    model.configure_train_reference(
        torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.]),
        torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.]),
    )
    with torch.no_grad():
        model.global_head[-1].weight.zero_()
        model.global_head[-1].bias.zero_()
    data = {
        key: value.cuda()
        for key, value in batch(
            batch_size=4, patches=2048, encoding_dim=1536
        ).items()
    }
    with torch.autocast('cuda', enabled=amp, dtype=torch.float16):
        logits, auxiliary = model(**data)
        loss = StableNLLSurvLoss()(logits, data['y'], c=data['c']) / len(logits) + auxiliary
    assert auxiliary > 0
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shared_train_epoch_and_batch_adapter():
    from survot_rank.training.train_runner import train_one_epoch, init_loss_function, _process_data_and_forward
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = arguments(batch_size=4, cur_fold=0, max_epochs=30, grad_clip_norm=1., grad_accum_steps=1)
    model = DCTV315ResidualTransport(args).to(device)
    data = batch()
    groups = [[data[f'x_omic{j}'][i] for j in range(1, 4)] for i in range(4)]
    packed = (data['x_wsi'], groups, data['y'], data['event_time'], data['c'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    result = train_one_epoch(args, 0, model, [packed, packed], optimizer, scheduler, init_loss_function(args), io.StringIO())
    assert all(math.isfinite(float(v)) for v in result.values())
    assert result['v315_auxiliary'] == 0
    model.eval()
    out, *_ = _process_data_and_forward(args, model, packed, device, test=True)
    assert torch.isfinite(out[0]).all()
