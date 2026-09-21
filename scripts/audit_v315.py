"""Reproducible v3.15 structural audit, not real-data performance evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from survot_rank.research.methods.legacy.experimental.dct_v315_residual_transport.model import DCTV315ResidualTransport
from survot_rank.research.methods.legacy.experimental.dct_v315_residual_transport.losses import StableNLLSurvLoss


def settings(options):
    return SimpleNamespace(encoding_dim=options.encoding_dim, wsi_projection_dim=options.dim,
        n_classes=4, omic_sizes=[16]*12, slot_num_wsi=options.slots, slot_num_omics=options.slots,
        rna_format='Pathways', bag_loss='nll_surv', alpha_surv=options.alpha_surv,
        dct_v315_dropout=0., dct_v315_ot_epsilon=.2, dct_v315_ot_iters=40,
        dct_v315_transport_mode=options.mode, survot_method='dct_v315')


def payload(options, device):
    b = options.batch_size
    latent = torch.randn(b, 1, 1, device=device)
    data = {'x_wsi': torch.randn(b, options.patches, options.encoding_dim, device=device)+latent}
    for i in range(1, 13):
        data[f'x_omic{i}'] = torch.randn(b, 16, device=device)+latent[:, 0]
    data['y'] = torch.bucketize(latent.flatten(), latent.new_tensor([-.6, 0., .6])).long()
    data['c'] = (torch.arange(b, device=device) % 4 == 0).float()
    return data


@torch.no_grad()
def snapshot(model, data, criterion):
    model.eval()
    logits, _ = model(**data)
    result = {k:float(v) for k,v in model.last_training_losses.items()}
    result['nll'] = float(criterion(logits, data['y'], c=data['c'])/len(logits))
    for name in ('wsi', 'omic'):
        slots = model.last_explanations[name+'_slots'].float()
        residual = slots-slots.mean(1, keepdim=True)
        singular = torch.linalg.svdvals(residual)
        prob = singular/singular.sum(-1, keepdim=True).clamp_min(1e-12)
        # A zero matrix has rank zero, not exp(entropy=0)=1.
        rank = (-(prob*prob.clamp_min(1e-12).log()).sum(-1)).exp()
        rank = torch.where(singular.sum(-1)>1e-8, rank, 0.)
        result[name+'_effective_rank'] = float(rank.mean())
    evidence = model.last_explanations
    plan = evidence['transport_plan']
    result['plan_independence_distance'] = float((plan-1./(plan.size(1)*plan.size(2))).square().sum((1, 2)).sqrt().mean())
    result['global_logit_rms'] = float(evidence['global_logits'].square().mean().sqrt())
    return result


def gradient_audit(model, data, criterion):
    model.train()
    logits, _ = model(**data)
    loss = criterion(logits, data['y'], c=data['c'])/len(logits)
    named = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _,p in named], allow_unused=True)
    result = {}
    for group in ('wsi_encoder', 'omic_encoders', 'wsi_slots', 'omic_slots', 'global_head', 'transport'):
        values = [g.detach().float().square().sum() for (n,p),g in zip(named, gradients)
                  if n.startswith(group) and g is not None]
        result[group] = float(torch.stack(values).sum().sqrt()) if values else 0.
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 17, 29])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--encoding-dim', type=int, default=32)
    parser.add_argument('--patches', type=int, default=48)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--mode', choices=['ot', 'independent'], default='ot')
    parser.add_argument('--alpha-surv', type=float, default=0.)
    options = parser.parse_args()
    if min(options.steps, options.batch_size, options.patches) < 1:
        parser.error('steps, batch size and patch count must be positive')
    torch.set_num_threads(1)
    device = torch.device(options.device)
    source = Path(__file__).resolve().parents[1]/'survot_rank/research/methods/legacy/experimental/dct_v315_residual_transport'
    report = {'kind':'synthetic structure only; no clinical performance claim', 'options':vars(options),
              'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}, 'seeds':{}}
    for seed in options.seeds:
        torch.manual_seed(seed)
        model = DCTV315ResidualTransport(settings(options)).to(device)
        torch.manual_seed(seed+1000)
        data = payload(options, device)
        criterion = StableNLLSurvLoss(options.alpha_surv)
        record = {'parameters':sum(p.numel() for p in model.parameters()),
                  'initial':snapshot(model, data, criterion), 'nll_gradient_norms':gradient_audit(model, data, criterion)}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.0005)
        elapsed, max_gradient = [], 0.
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for step in range(options.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits, auxiliary = model(**data)
            loss = criterion(logits, data['y'], c=data['c'])/len(logits)+auxiliary
            loss.backward()
            if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError(f'nonfinite loss or gradient: seed={seed}, step={step}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            max_gradient = max(max_gradient, float(norm))
            optimizer.step()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter()-started)
        record.update(finite_steps=options.steps, max_preclip_gradient_norm=max_gradient,
                      median_step_seconds=statistics.median(elapsed), final=snapshot(model, data, criterion))
        if device.type == 'cuda':
            record['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device)/1024**2
        report['seeds'][str(seed)] = record
        print(json.dumps({'seed':seed, 'final':record['final']}, ensure_ascii=False), flush=True)
    path = Path(options.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
