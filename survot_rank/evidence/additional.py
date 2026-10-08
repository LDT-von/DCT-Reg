"""Additional v3.13 mechanism tests. Real data run only through the explicit CLI.

All endpoints are frozen-checkpoint diagnostics, conditional on legacy_val
checkpoint selection. They neither retrain a model nor claim independent test
performance. Cross reconstruction still observes recipient omics through OT.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .manifest import audit, cindex, predictions, revision, save_json, sha256
from .v313 import indexed_sample, load_run, mix_plans, pathway_error, payload_for, replay

RECON_MODES = ("native", "product", "direct", "shuffled", "train_mean", "self")
RETRIEVAL_VERSION = 2
RETRIEVAL_CENTER = "arithmetic mean of individually layer-normalized training tokens"


def stable_rng(seed, *keys):
    """Independent of Python hash randomization, run order, arm and device."""
    token = "\0".join(map(str, (seed, *keys))).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def validate_options(experiment, fractions, repeats):
    fractions = tuple(float(x) for x in fractions)
    if experiment not in ("all", "reconstruction", "patches"):
        raise ValueError("Unknown experiment")
    if repeats < 1 or len(fractions) < 2 or len(set(fractions)) != len(fractions):
        raise ValueError("Require positive repeats and distinct fractions including zero")
    if fractions[0] != 0 or any(not math.isfinite(x) or not 0 <= x < 1 for x in fractions):
        raise ValueError("Fractions must start with 0 and lie in [0,1)")
    if tuple(sorted(fractions)) != fractions:
        raise ValueError("Fractions must increase")
    return fractions


def donor_indices(case_ids, seed, repeat):
    """One seeded cycle: no self pairs, no donor reused in the same repeat."""
    if len(case_ids) < 2 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Pairing control needs at least two distinct patients")
    order = stable_rng(seed, "donors", repeat, *case_ids).permutation(len(case_ids))
    donor = np.empty(len(case_ids), dtype=np.int64)
    donor[order] = np.roll(order, -1)
    return donor


def patch_orders(scores, valid, seed, case_id, repeat):
    """Return real row indices; padding never enters a subset or a ranking."""
    scores, valid = np.asarray(scores), np.asarray(valid, dtype=bool)
    if scores.ndim != 1 or scores.shape != valid.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid patch scores or identity mask")
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError("Need at least two verified real patches")
    # Seeded tie breaking avoids equating row order to relevance when scores tie.
    tie = stable_rng(seed, "ties", case_id).random(len(indices))
    top = indices[np.lexsort((tie, -scores[indices]))]
    bottom = indices[np.lexsort((tie, scores[indices]))]
    random = stable_rng(seed, "patches", case_id, repeat).permutation(indices)
    return top, bottom, random


def keep_rows(order, fraction, *, deletion):
    n = len(order)
    removed = min(n - 1, int(math.floor(n * fraction)))
    chosen = order[removed:] if deletion else order[: n - removed]
    return np.sort(chosen)


def retrieval_metrics(decoded, targets, train_mean):
    """Centered, normalized latent patient retrieval with conservative ties.

No gene-expression correlation is computed across arbitrary embedding axes.
Ranks are within one fold; all ties count against the correct match. Zero
centered vectors score zero and are explicitly counted.
"""
    decoded, targets, train_mean = [np.asarray(x, dtype=np.float64) for x in (decoded, targets, train_mean)]
    if decoded.shape != targets.shape or decoded.ndim != 3 or train_mean.shape != targets.shape[1:]:
        raise ValueError("Retrieval expects [patients,pathways,embedding] and [pathways,embedding]")
    if len(targets) < 2 or not all(np.isfinite(x).all() for x in (decoded, targets, train_mean)):
        raise ValueError("Need at least two finite target patients")
    q = (decoded - train_mean).reshape(len(decoded), -1)
    t = (targets - train_mean).reshape(len(targets), -1)
    qnorm, tnorm = np.linalg.norm(q, axis=1), np.linalg.norm(t, axis=1)
    scores = (q / np.maximum(qnorm[:, None], 1e-12)) @ (t / np.maximum(tnorm[:, None], 1e-12)).T
    diagonal = scores.diagonal()
    competitors = scores >= diagonal[:, None] - 1e-10
    np.fill_diagonal(competitors, False)
    ranks = 1 + competitors.sum(1)
    return dict(top1=float(np.mean(ranks == 1)), mrr=float(np.mean(1 / ranks)),
                chance_top1=1 / len(targets), ranks=ranks, matched_cosine=diagonal,
                zero_query_count=int((qnorm < 1e-12).sum()), zero_target_count=int((tnorm < 1e-12).sum()))


@torch.no_grad()
def encode_patient(model, payload):
    if model.training or any(payload.get(key) is not None for key in ("y", "c", "event_time")):
        raise ValueError("Require eval mode and no survival-label input")
    xo = model._encode_omics(payload)
    sw, so, _, _ = model._encode_transport_slots(model.wsi_mlp(payload["x_wsi"]), xo, payload)
    return sw, so, xo


@torch.no_grad()
def solve(model, sw, so):
    costs, rows, cols, _ = model._cost_tensor(sw, so)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, int(model.args.cur_epoch), replay_fixed=False)
    logits, gate = model._encode_logits_from_plans(sw, so, plans)
    return plans, logits, gate


@torch.no_grad()
def decode_controls(model, sw, so, target, donor_sw, train_mean):
    """Only product freezes the factual gate: isolates coupling organization.

    Shuffled WSI slots are already modality-specific: cost and event gate are
    recomputed against recipient omics, without changing recipient targets.
    Direct feeds WSI slots to the same trained decoder (a diagnostic intervention).
    """
    plans, _, gate = solve(model, sw, so)
    transported = model._transport_wsi_to_omic(sw, plans, gate)[0]
    product = model._transport_wsi_to_omic(sw, mix_plans(plans, 1), gate)[0]
    shuffled_plans, shuffled_logits, shuffled_gate = solve(model, donor_sw, so)
    shuffled = model._transport_wsi_to_omic(donor_sw, shuffled_plans, shuffled_gate)[0]
    decoder = model.pathway_reconstruction_decoder
    decoded = dict(native=decoder(sw if model._cross_mode == "direct" else transported),
                   product=decoder(product), direct=decoder(sw),
                   shuffled=decoder(donor_sw if model._cross_mode == "direct" else shuffled),
                   train_mean=train_mean.unsqueeze(0), self=decoder(so))
    errors = {key: pathway_error(value, target).cpu().numpy()[0] for key, value in decoded.items()}
    # training_mean is already the arithmetic mean in the normalized target
    # space. Normalizing it again moves the retrieval center and invents a
    # patient residual for the constant-mean control. Loss normalization above
    # remains exactly the trained model's reconstruction distance.
    normalized = {key: (value if key == "train_mean" else
                       F.layer_norm(value, (value.size(-1),))).cpu().numpy()[0]
                  for key, value in decoded.items()}
    return errors, normalized, float(model._risk(shuffled_logits).item())


@torch.no_grad()
def risk_for_rows(model, payload, rows):
    changed = dict(payload, x_wsi=payload["x_wsi"][:, torch.as_tensor(rows, device=payload["x_wsi"].device)])
    # Re-encode slots, costs, all OT plans and survival head from the changed bag.
    sw, so, _ = encode_patient(model, changed)
    _, logits, _ = solve(model, sw, so)
    return float(model._risk(logits).item())


@torch.no_grad()
def patch_experiment(model, payload, attention, valid, case_id, fractions, repeats, seed):
    if payload["x_wsi"].size(0) != 1:
        raise ValueError("Patch experiments use one identified patient at a time")
    scores = np.asarray(attention).mean(axis=0)
    first_orders = patch_orders(scores, valid, seed, case_id, 0)
    reference = risk_for_rows(model, payload, np.flatnonzero(valid))
    deletion = np.empty((len(fractions), repeats, 3), dtype=float)
    budget = np.empty((len(fractions), repeats), dtype=float)
    removed_counts = np.asarray([len(first_orders[0]) - len(keep_rows(first_orders[0], f, deletion=True)) for f in fractions])
    for j, fraction in enumerate(fractions):
        for repeat in range(repeats):
            orders = patch_orders(scores, valid, seed, case_id, repeat)
            if fraction == 0:
                deletion[j, repeat] = reference
                budget[j, repeat] = reference
                continue
            # Top/bottom are deterministic; do not repeat their model evaluation.
            for mode, order in enumerate(orders):
                deletion[j, repeat, mode] = (deletion[j, 0, mode] if repeat and mode < 2 else
                    risk_for_rows(model, payload, keep_rows(order, fraction, deletion=True)))
            budget[j, repeat] = risk_for_rows(model, payload, keep_rows(orders[2], fraction, deletion=False))
    real_scores = scores[np.asarray(valid, dtype=bool)]
    return dict(deletion_risk=deletion, budget_risk=budget, unpadded_risk=reference,
                real_patches=int(np.sum(valid)), removed_counts=removed_counts,
                attention_score_std=float(np.std(real_scores)),
                attention_relative_range=float(np.ptp(real_scores) / max(abs(real_scores.mean()), 1e-12)))


@torch.no_grad()
def training_mean(model, args, train, device):
    reference = copy.copy(train)
    reference.split_key = "val"
    if not len(reference):
        raise ValueError("Empty training reference")
    total = None
    for i in range(len(reference)):
        sample, _, _, _ = indexed_sample(reference, i)
        target = model._encode_omics(payload_for(sample, args, device))
        normalized = F.layer_norm(target, (target.size(-1),))[0].double()
        total = normalized.clone() if total is None else total + normalized
    return (total / len(reference)).float()


def summarize_patches(arrays):
    baseline = arrays["unpadded_risk"]
    result = dict(unpadded_cindex=cindex(arrays["time"], arrays["censor"], baseline),
                  native_cindex=cindex(arrays["time"], arrays["censor"], arrays["risk"]), deletion=[], budget=[])
    for j, fraction in enumerate(arrays["fractions"]):
        row = dict(removed_fraction=float(fraction), mean_removed_patches=float(arrays["removed_counts"][:, j].mean()))
        for k, name in enumerate(("top", "bottom", "random")):
            risks = arrays["deletion_risk"][:, j, :, k]
            row[name + "_cindex"] = [cindex(arrays["time"], arrays["censor"], risks[:, r]) for r in range(risks.shape[1])]
            row[name + "_abs_delta"] = float(np.abs(risks - baseline[:, None]).mean())
        result["deletion"].append(row)
        risks = arrays["budget_risk"][:, j]
        result["budget"].append(dict(retained_fraction=1 - float(fraction),
            cindex=[cindex(arrays["time"], arrays["censor"], risks[:, r]) for r in range(risks.shape[1])],
            abs_delta=float(np.abs(risks - baseline[:, None]).mean())))
    result["attention_relative_range_median"] = float(np.median(arrays["attention_relative_range"]))
    result["padding_abs_delta_mean"] = float(np.abs(arrays["risk"] - baseline).mean())
    return result


def _assert_buffers(model, before):
    for name, tensor in model.named_buffers():
        if not torch.equal(tensor, before[name]):
            raise ValueError(f"Diagnostic changed model buffer {name}")


def export_additional(run, output_root, *, experiment="all", device="cuda:0",
                      fractions=(0, .1, .25, .5, .75), repeats=5, diagnostic_seed=20261008):
    fractions = validate_options(experiment, fractions, repeats)
    report = audit({"runs": [run]})
    if not report["passed"]:
        raise ValueError("; ".join(report["errors"]))
    if not run.get("checkpoint") or Path(run["id"]).name != run["id"] or run["id"] in (".", ".."):
        raise ValueError("Require a best checkpoint and simple run ID")
    out = Path(output_root) / run["id"]
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refuse to overwrite {out}; use a fresh output directory")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(device)
    model, args, factory, train, val = load_run(run, device)
    saved = predictions(run["predictions"])
    ids = val.label_df["case id"].astype(str).tolist()
    if len(set(ids)) != len(ids) or set(ids) != set(saved["case_ids"]):
        raise ValueError("Validation patient identity differs from audit")
    expected = {cid: (risk, time, censor) for cid, risk, time, censor in
                zip(saved["case_ids"], saved["risk"], saved["time"], saved["censor"])}
    # Order does not depend on the training dataset row ordering.
    val_index = {cid: i for i, cid in enumerate(ids)}
    ids = sorted(ids)
    before = {key: value.clone() for key, value in model.named_buffers()}
    reconstruction = experiment in ("all", "reconstruction")
    patches = experiment in ("all", "patches")
    if reconstruction and model.dct_v313_reconstruction_cross_coef <= 0:
        raise ValueError("Cross decoder was not supervised in this arm; select a trained cross-reconstruction arm")
    train_mean = training_mean(model, args, train, device) if reconstruction else None
    donors = np.stack([donor_indices(ids, diagnostic_seed, r) for r in range(repeats)]) if reconstruction else None
    arrays = dict(case_ids=np.asarray(ids), pathway_names=np.asarray(factory.pathway_names, dtype=str),
                  fractions=np.asarray(fractions), risk=[], time=[], censor=[])
    cached = []
    feature_records = []
    for i, cid in enumerate(ids):
        sample, features, slide_index, patch_index = indexed_sample(val, val_index[cid])
        payload = payload_for(sample, args, device)
        payload["event_time"] = None
        values = replay(model, payload, (0,))
        # Also prove the manual encoder/head path equals the standard model.
        with torch.no_grad():
            standard = float(model._risk(model(**payload)[0]).item())
        risk = float(values["risk"][0])
        time, censor = float(sample[3]), float(sample[4])
        if not np.allclose((risk, time, censor), expected[cid], atol=2e-5, rtol=1e-5) or not np.isclose(standard, risk, atol=1e-6, rtol=1e-5):
            raise ValueError(f"Best prediction/outcome/standard-forward replay mismatch: {cid}")
        for key, value in (("risk", risk), ("time", time), ("censor", censor)):
            arrays[key].append(value)
        for record in features:
            record["sha256"] = sha256(record["feature_path"])
        feature_records.append(dict(case_id=cid, features=features,
            selected_rows_sha256=hashlib.sha256(np.stack((slide_index, patch_index)).tobytes()).hexdigest()))
        if reconstruction:
            with torch.no_grad():
                cached.append(tuple(x.cpu() for x in encode_patient(model, payload)))
        if patches:
            result = patch_experiment(model, payload, values["attention_wsi"][0], patch_index >= 0,
                                      cid, fractions, repeats, diagnostic_seed)
            for key, value in result.items():
                arrays.setdefault(key, []).append(value)
        print(f"[additional] {run['id']} factual {i + 1}/{len(ids)}", flush=True)
    summary = dict(schema_version=1, run=run, hashes=report["runs"][0]["hashes"],
        export_commit=revision(Path(__file__).resolve().parents[2]), best_epoch=args.cur_epoch,
        experiment=experiment, diagnostic_seed=diagnostic_seed, repeats=repeats, fractions=list(fractions),
        coefficients=model.effective_recon_coefficients(), feature_records=feature_records,
        note="Frozen best-validation diagnostics; paired omics observed, not raw-gene imputation, causal effects, or independent test evidence")
    if reconstruction:
        errors = {key: [] for key in RECON_MODES}
        decoded = {key: [] for key in RECON_MODES}
        # Store shuffled errors/ranks for every replicate, not only a favorable shuffle.
        shuffled_errors, shuffled_risks, shuffled_decoded = [], [], [[] for _ in range(repeats)]
        for i, (sw_cpu, so_cpu, xo_cpu) in enumerate(cached):
            sw, so, xo = (x.to(device) for x in (sw_cpu, so_cpu, xo_cpu))
            patient_shuffles, patient_shuffled_risks = [], []
            for r in range(repeats):
                donor = cached[int(donors[r, i])][0].to(device)
                err, norm, shuffled_risk = decode_controls(model, sw, so, xo, donor, train_mean)
                patient_shuffles.append(err["shuffled"])
                patient_shuffled_risks.append(shuffled_risk)
                shuffled_decoded[r].append(norm["shuffled"])
                if r == 0:
                    for key in RECON_MODES:
                        errors[key].append(err[key])
                        decoded[key].append(norm[key])
            shuffled_errors.append(patient_shuffles)
            shuffled_risks.append(patient_shuffled_risks)
        target = torch.cat([F.layer_norm(x[2], (x[2].size(-1),)) for x in cached]).numpy()
        center = train_mean.cpu().numpy()
        arrays["retrieval_training_mean"] = center
        summary["retrieval_metric_version"] = RETRIEVAL_VERSION
        summary["retrieval_center_definition"] = RETRIEVAL_CENTER
        arrays["target_centered_energy"] = np.square(target - center).mean(-1)
        arrays["target_patient_variance"] = target.var(0).mean(-1)
        arrays["donor_case_ids"] = np.asarray(ids)[donors]
        arrays["reconstruction_shuffled_all_error"] = np.asarray(shuffled_errors)
        arrays["pairing_shuffled_risk"] = np.asarray(shuffled_risks)
        summary["reconstruction"] = {}
        for key in RECON_MODES:
            arrays["reconstruction_" + key + "_error"] = np.asarray(errors[key])
            metric = retrieval_metrics(np.asarray(decoded[key]), target, center)
            arrays["retrieval_" + key + "_rank"] = metric.pop("ranks")
            arrays["retrieval_" + key + "_cosine"] = metric.pop("matched_cosine")
            metric["mean_error"] = float(np.asarray(errors[key]).mean())
            summary["reconstruction"][key] = metric
        shuffled_metrics = [retrieval_metrics(np.asarray(x), target, center) for x in shuffled_decoded]
        summary["reconstruction"]["shuffled"]["mean_error"] = float(arrays["reconstruction_shuffled_all_error"].mean())
        summary["reconstruction"]["shuffled"]["top1"] = float(np.mean([x["top1"] for x in shuffled_metrics]))
        summary["reconstruction"]["shuffled"]["mrr"] = float(np.mean([x["mrr"] for x in shuffled_metrics]))
        arrays["retrieval_shuffled_all_rank"] = np.stack([x["ranks"] for x in shuffled_metrics], axis=1)
        summary["reconstruction_target_variance_mean"] = float(arrays["target_patient_variance"].mean())
        summary["training_reference_patient_count"] = len(train)
        summary["pairing"] = dict(native_cindex=cindex(arrays["time"], arrays["censor"], arrays["risk"]),
            shuffled_cindex=[cindex(arrays["time"], arrays["censor"], arrays["pairing_shuffled_risk"][:, r]) for r in range(repeats)],
            mean_abs_delta=float(np.abs(arrays["pairing_shuffled_risk"] - np.asarray(arrays["risk"])[:, None]).mean()))
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind in "fc" and not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Nonfinite diagnostic output")
    if patches:
        summary["patches"] = summarize_patches(arrays)
    _assert_buffers(model, before)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "additional.npz", **arrays)
    summary["array_sha256"] = sha256(out / "additional.npz")
    save_json(out / "additional.json", summary)
    return summary
