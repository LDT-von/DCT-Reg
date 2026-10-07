"""Reproducible synthetic structural/gradient audit; not a performance experiment."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from survot_rank.research.methods.legacy.experimental.dct_v314_masked_transport_reconstruction.model import DCTV314MaskedTransportReconstruction
from survot_rank.training.train_runner import init_loss_function, compose_batch_objective


def settings(seed=3, **overrides):
    values = dict(bag_loss='nll_surv', survot_method='dct_v314', alpha_surv=0.15, omic_sizes=[16]*12,
                  n_classes=4, encoding_dim=32, wsi_projection_dim=32,
                  rna_format='Pathways', slot_num_wsi=16, slot_num_omics=16,
                  slot_iters=10, otehv2_heads=4, otehv2_layers=1,
                  otehv2_dropout=0.0, otehv2_iter=20, otehv2_eps=0.05,
                  dct_num_stages=4, dct_coupling_projection_iters=100,
                  dct_coupling_projection_tol=1e-4, cur_epoch=7,
                  dct_slot_eval_seed=seed)
    values.update(overrides)
    return SimpleNamespace(**values)


def payload(device, batch_size=16, patches=48, encoding_dim=32):
    # A learnable survival signal shared by heterogeneous modality tokens.
    latent = torch.randn(batch_size, 1, 1, device=device)
    result = {'x_wsi': torch.randn(batch_size, patches, encoding_dim, device=device) + latent}
    for i in range(1, 13):
        result[f'x_omic{i}'] = torch.randn(batch_size, 16, device=device) + latent[:, 0]
    result['y'] = torch.bucketize(latent.flatten(), latent.new_tensor([-0.6, 0., 0.6])).long()
    result['event_time'] = (result['y'].float() + 1) * 10 + torch.arange(batch_size, device=device).float()/20
    result['c'] = (torch.arange(batch_size, device=device) % 4 == 0).float()
    return result


def scalar(value):
    return float(value.detach())


def gradient_audit(model, data):
    model.train()
    xw = model.wsi_mlp(data['x_wsi'])
    xo = model._encode_omics(data)
    w, o, _, _ = model._encode_transport_slots(xw, xo, data)
    costs, rows, cols, _ = model._cost_tensor(w, o)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, 7)
    logits, gate = model._encode_logits_from_plans(w, o, plans)
    nll = init_loss_function(model.args)(logits, data['y'], data['event_time'], data['c'])/len(logits)
    terms = {'nll': nll,
             'ipcw': 0.1*model._ipcw_pairwise_ranking_loss(logits, data['event_time'], data['c']),
             'slot_nll': 0.05*model.per_slot_nll_loss(w, o, data['y'], data['event_time'], data['c']),
             'diversity': 0.02*model.slot_diversity_loss(w, o)}
    target = model._clean_reconstruction_target(data) if hasattr(model, '_clean_reconstruction_target') else xo.detach()
    mtr = model.masked_transport_reconstruction_losses(x_omics=xo, slots_wsi=w, factual_gate=gate,
            target=target, available=torch.ones(len(w), device=w.device, dtype=torch.bool), epoch=7)
    terms['mtr'] = model.dct_v314_lambda_mtr*mtr[2]
    groups = ['wsi_mlp', 'sig_networks', 'slot_attention_wsi', 'slot_attention_omic',
              'shared_wsi_prototypes', 'shared_omic_prototypes', 'stage_pair_cost',
              'pathway_reconstruction_decoder', 'event_gate']
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    report = {}
    for term, loss in terms.items():
        gradients = torch.autograd.grad(loss, [p for _, p in parameters], retain_graph=True, allow_unused=True)
        report[term] = {'weighted_loss': scalar(loss), 'gradient_norm': {}}
        for prefix in groups:
            squares = [g.detach().float().square().sum() for (n, _), g in zip(parameters, gradients)
                       if n.startswith(prefix) and g is not None]
            report[term]['gradient_norm'][prefix] = scalar(torch.stack(squares).sum().sqrt()) if squares else 0.
    return report


@torch.no_grad()
def snapshot(model, data):
    model.eval()
    xw = model.wsi_mlp(data['x_wsi'])
    xo = model._encode_omics(data)
    w, o, _, _ = model._encode_transport_slots(xw, xo, data)
    result = {}
    for name, slots, head in [('wsi', w, model.per_slot_hazard_wsi), ('omic', o, model.per_slot_hazard_omic)]:
        norm = F.normalize(slots.float(), dim=-1)
        sim = norm @ norm.transpose(-1, -2)
        off = ~torch.eye(slots.size(1), device=slots.device, dtype=torch.bool)
        singular = torch.linalg.svdvals(slots.float() - slots.mean(1, keepdim=True))
        probs = singular / singular.sum(-1, keepdim=True).clamp_min(1e-12)
        result[name] = {'off_diagonal_cosine': scalar(sim[:, off].mean()),
                        'slot_std': scalar(slots.std(1, unbiased=False).mean()),
                        'effective_rank': scalar((-(probs*probs.clamp_min(1e-12).log()).sum(-1)).exp().mean()),
                        'hazard_variance': scalar(head(slots).sigmoid().var(1, unbiased=False).mean())}
    logits, aux = model(**data)
    result['nll'] = scalar(init_loss_function(model.args)(logits, data['y'], data['event_time'], data['c'])/len(logits))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 17, 29])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', choices=['masked', 'full', 'hybrid', 'off'], default='masked')
    parser.add_argument('--projection-dim', type=int, default=32)
    parser.add_argument('--encoding-dim', type=int, default=32)
    parser.add_argument('--patches', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--alpha-surv', type=float, default=0.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_dir = Path(__file__).resolve().parents[1]/'survot_rank/research/methods/legacy/experimental/dct_v314_masked_transport_reconstruction'
    report = {'kind': 'synthetic structural audit, not clinical performance', 'device': args.device,
              'mode': args.mode, 'steps': args.steps, 'parameters': vars(args),
              'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.glob('*.py')},
              'seeds': {}}
    for seed in args.seeds:
        torch.manual_seed(seed)
        model = DCTV314MaskedTransportReconstruction(settings(seed, dct_v314_reconstruction_mode=args.mode,
                    encoding_dim=args.encoding_dim, wsi_projection_dim=args.projection_dim,
                    alpha_surv=args.alpha_surv)).to(args.device)
        # Independent of model parameter initialization; all mode ablations see identical data.
        torch.manual_seed(seed+1000)
        data = payload(args.device, args.batch_size, args.patches, args.encoding_dim)
        # Small GPU smoke batches may not include enough uncensored events;
        # fit their reference on an independent synthetic training population.
        reference = data if (data['c'] == 0).sum() >= model.spt_num_stages else payload(args.device)
        model.configure_train_reference(reference['event_time'], reference['c'])
        record = {'settings': vars(model.args).copy(), 'initial': snapshot(model, data),
                  'gradient_audit': gradient_audit(model, data)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0005)
        record['finite_steps'] = 0
        for step in range(args.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, aux = model(**data)
            loss = compose_batch_objective(init_loss_function(model.args)(logits, data['y'], data['event_time'], data['c']), aux, len(logits))
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f'nonfinite loss/gradient at seed={seed}, step={step}')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            record['finite_steps'] += 1
        record['final_losses'] = {k: scalar(v) for k, v in model.last_training_losses.items()}
        record['final'] = snapshot(model, data)
        report['seeds'][str(seed)] = record
        print(json.dumps({'seed': seed, 'initial': record['initial'], 'final': record['final']}, ensure_ascii=False), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
