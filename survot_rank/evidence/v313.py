"""Read-only checkpoint replay and interventions for the actual v3.13 model."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .manifest import audit, best_epoch, cindex, predictions, revision, save_json, sha256


def independent_plan(plan):
    """Product of *actual factual* marginals, preserving numerical plan mass."""
    mass = plan.sum(dim=(-2, -1), keepdim=True)
    if (mass <= 0).any() or not torch.isfinite(plan).all() or (plan < 0).any():
        raise ValueError("Invalid factual transport mass")
    return plan.sum(-1, keepdim=True) * plan.sum(-2, keepdim=True) / mass


def mix_plans(plans, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    return [tuple((1 - alpha) * plan + alpha * independent_plan(plan) for plan in stage) for stage in plans]


class SlotAttentionCapture:
    """Final pooling weights, then prototype rollout; no training hooks changed."""
    def __init__(self, module):
        self.module, self.q, self.k = module, None, None
        self.handles = [module.to_q.register_forward_hook(self._q), module.to_k.register_forward_hook(self._k)]

    def _q(self, module, inputs, output):
        self.q = output.detach()

    def _k(self, module, inputs, output):
        self.k = output.detach()

    def weights(self):
        if self.q is None or self.k is None:
            raise ValueError("Slot pooling projections were not observed")
        q, k = self.module.split_heads(self.q), self.module.split_heads(self.k)
        dots = torch.einsum("bhkd,bhnd->bhkn", q, k) * self.module.scale
        return F.normalize(dots.softmax(dim=-2) + self.module.eps, p=1, dim=-1).mean(dim=1)

    def close(self):
        for handle in self.handles:
            handle.remove()


def pathway_error(prediction, target):
    """[B,P] diagnostic matching the model's normalized reconstruction distance."""
    pred = F.layer_norm(prediction, (prediction.size(-1),))
    target = F.layer_norm(target.detach(), (target.size(-1),))
    return 0.5 * (1 - F.cosine_similarity(pred, target, dim=-1) + F.smooth_l1_loss(pred, target, reduction="none").mean(-1))


def slot_diagnostics(slots, hazards):
    normalized = F.normalize(slots, dim=-1)
    cosine = normalized @ normalized.transpose(-1, -2)
    distance = torch.cdist(slots, slots) / slots.size(-1) ** 0.5
    mask = torch.triu(torch.ones(slots.size(1), slots.size(1), dtype=torch.bool, device=slots.device), diagonal=1)
    return dict(cosine=cosine, distance=distance,
                mean_cosine=cosine[:, mask].mean(-1), mean_distance=distance[:, mask].mean(-1),
                hazard_variance=hazards.var(dim=1, unbiased=False).mean(-1))


@torch.no_grad()
def replay(model, payload, alphas=(0, .25, .5, .75, 1)):
    if model.training:
        raise ValueError("Evidence replay requires eval mode")
    if any(payload.get(key, False) for key in ("wsi_missing", "omic_missing")):
        raise ValueError("This paired-modality evidence export requires both modalities")
    if not alphas or list(alphas)[0] != 0 or len(set(alphas)) != len(alphas):
        raise ValueError("Distinct alphas must begin with factual alpha=0")
    captures = [SlotAttentionCapture(model.slot_attention_wsi), SlotAttentionCapture(model.slot_attention_omic)]
    try:
        xw, xo = model.wsi_mlp(payload["x_wsi"]), model._encode_omics(payload)
        sw, so, cw, co = model._encode_transport_slots(xw, xo, payload)
        aw, ao = captures[0].weights(), captures[1].weights()
    finally:
        for capture in captures:
            capture.close()
    costs, rows, cols, evidence_gate = model._cost_tensor(sw, so)
    plans, _ = model._plans_from_cost_tensor(costs, rows, cols, int(model.args.cur_epoch), replay_fixed=False)
    logits, gate = model._encode_logits_from_plans(sw, so, plans)
    risk = model._risk(logits)
    hs = [torch.sigmoid(model.per_slot_hazard_wsi(sw)), torch.sigmoid(model.per_slot_hazard_omic(so))]
    output = dict(logits=logits, risk=risk, survival=torch.cumprod(1 - torch.sigmoid(logits), dim=-1),
                  plans=torch.stack([torch.stack(stage, dim=1) for stage in plans], dim=1),
                  rows=rows, cols=cols, costs=costs, evidence_gate=evidence_gate, stage_gate=gate,
                  slots_wsi=sw, slots_omic=so, hazard_wsi=hs[0], hazard_omic=hs[1],
                  attention_wsi=torch.bmm(cw, aw), attention_omic=torch.bmm(co, ao),
                  self_pathway_error=pathway_error(model.pathway_reconstruction_decoder(so), xo))
    for modality, slots, hazard in zip(("wsi", "omic"), (sw, so), hs):
        output.update({f"{modality}_{key}": value for key, value in slot_diagnostics(slots, hazard).items()})
    sweep_risk, errors, row_residual, col_residual = [], [], [], []
    for alpha in alphas:
        mixed = mix_plans(plans, float(alpha))
        modified_logits, modified_gate = model._encode_logits_from_plans(sw, so, mixed)
        if alpha == 0 and not torch.allclose(modified_logits, logits, atol=1e-6, rtol=1e-5):
            raise ValueError("alpha=0 does not replay the factual predictor")
        # Direct changes only the reconstruction branch; its predictor still uses OT.
        memory = sw if model._cross_mode == "direct" else model._transport_wsi_to_omic(sw, mixed, modified_gate)[0]
        sweep_risk.append(model._risk(modified_logits))
        errors.append(pathway_error(model.pathway_reconstruction_decoder(memory), xo))
        row_residual.append(torch.stack([(p.sum(-1) - q.sum(-1)).abs().amax(-1) for stage, original in zip(mixed, plans) for p, q in zip(stage, original)]).amax(0))
        col_residual.append(torch.stack([(p.sum(-2) - q.sum(-2)).abs().amax(-1) for stage, original in zip(mixed, plans) for p, q in zip(stage, original)]).amax(0))
    output.update(sweep_risk=torch.stack(sweep_risk, dim=1),
                  cross_pathway_error=torch.stack(errors, dim=1),
                  sweep_row_residual=torch.stack(row_residual, dim=1), sweep_col_residual=torch.stack(col_residual, dim=1))
    return {key: value.detach().cpu().numpy() for key, value in output.items()}


def indexed_sample(dataset, index):
    """Capture feature rows during the real dataset load, including multi-slide bags."""
    records, raw_bags = [], []
    load_feature, load_wsi = dataset._load_wsi_feature, dataset.load_wsi

    def feature(path):
        bag = load_feature(path)
        records.append({"feature_path": str(Path(path).resolve()), "patch_count": int(bag.size(0))})
        return bag

    def wsi(slides):
        bag = load_wsi(slides)
        raw_bags.append(bag)
        return bag

    dataset._load_wsi_feature, dataset.load_wsi = feature, wsi
    try:
        sample = dataset[index]
    finally:
        dataset._load_wsi_feature, dataset.load_wsi = load_feature, load_wsi
    if sample is None or len(raw_bags) != 1:
        raise ValueError(f"Missing/corrupt input for patient index {index}; export stops instead of shifting IDs")
    raw = raw_bags[0]
    if raw.ndim != 2 or sum(record["patch_count"] for record in records) != raw.size(0):
        raise ValueError("No verified real WSI bag (zero fallback and missing-modality cases are unsupported)")
    count = dataset.dataset_factory.num_patches
    n = min(count, raw.size(0)) if count is not None else raw.size(0)
    selected = np.floor(np.arange(n) * raw.size(0) / n).astype(np.int64)
    if not torch.equal(sample[0][:n], raw[selected]):
        raise ValueError("Dataset patch sampling differs from the deterministic evaluation rule")
    padded = np.full(sample[0].size(0), -1, dtype=np.int64)
    padded[:n] = selected
    slide_index, patch_index = np.full_like(padded, -1), np.full_like(padded, -1)
    offset = 0
    for j, record in enumerate(records):
        match = (padded >= offset) & (padded < offset + record["patch_count"])
        slide_index[match], patch_index[match] = j, padded[match] - offset
        offset += record["patch_count"]
    return sample, records, slide_index, patch_index


def load_checkpoint_strict(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Never drop mismatched parameters or replace checkpoint reference buffers.
    for key in ("dct_stage_edges", "dct_censor_times", "dct_censor_survival"):
        if key in state and key in model.state_dict():
            reference = model.state_dict()[key]
            if state[key].shape != reference.shape or not torch.allclose(state[key], reference, atol=1e-5, rtol=1e-5):
                raise ValueError(f"Train-fold reference buffer {key} differs from checkpoint")
    model.load_state_dict(state, strict=True)


def load_run(run, device):
    from survot_rank.config import apply_overrides, config_to_argv, load_config
    from survot_rank.training.extended_args import process_args_extended
    from survot_rank.training.train_runner import SurvivalDatasetFactory, get_split, get_model, set_global_seed
    config = apply_overrides(load_config(run["config"]), run.get("overrides", []))
    args = process_args_extended(config_to_argv(config))
    args._dct_user_overrides = set(config.get("_dct_user_overrides", []))
    if args.study != run["cancer"] or args.seed != run["seed"]:
        raise ValueError("Manifest cancer/seed disagrees with config+overrides")
    if args.survot_method not in ("dct_v313", "dct_v313_transport_reconstruction") or args.rna_format != "Pathways":
        raise ValueError("Checkpoint exporter supports actual v3.13 Pathways checkpoints only")
    if getattr(args, "evaluation_protocol", "legacy_val") != "legacy_val" or args.on_missing_wsi != "error":
        raise ValueError("Require legacy_val and on_missing_wsi=error")
    set_global_seed(args.seed)
    args.cur_epoch = best_epoch(run["curve"])[0]
    factory = SurvivalDatasetFactory(study=args.study, data_path=args.data_path, rna_format=args.rna_format,
        signature=args.signature, n_bins=args.n_classes, label_col=args.label_col, num_genes=args.num_genes,
        num_patches=args.num_patches, binning_mode=args.binning_mode, which_splits=args.which_splits)
    expected_split = Path(args.data_path) / "splits" / args.which_splits / args.study / f"fold_{run['fold']}.csv"
    if sha256(expected_split) != sha256(run["split_csv"]):
        raise ValueError("Configured split differs from audited split")
    factory.clinical_df = factory.clinical_df[factory.clinical_df["case id"].isin(set(factory.gene_data_df.columns))].reset_index(drop=True)
    train, val, _, _ = get_split(args, factory, int(run["fold"]))
    args.omic_sizes, args.omic_names = factory.omic_sizes, factory.omic_names
    args.pathway_names = factory.pathway_names
    model = get_model(args.survot_method, args, omic_names=factory.omic_names, pathway_names=factory.pathway_names)
    expected_modes = {"direct": ("direct", "learned"), "independent": ("transport", "independent")}
    if run["arm"] in expected_modes or run["arm"].startswith("exp"):
        expected = expected_modes.get(run["arm"], ("transport", "learned"))
        if (model._cross_mode, model._plan_mode) != expected:
            raise ValueError("Manifest arm disagrees with effective model cross/plan mode")
    model.configure_train_reference(train.label_df[factory.label_col].to_numpy(), train.label_df[factory.censorship_var].to_numpy())
    load_checkpoint_strict(model, run["checkpoint"])
    model.to(device).eval()
    return model, args, factory, train, val


def payload_for(sample, args, device):
    from survot_rank.training.train_runner import _collate_pathways
    from utils.core_utils import _unpack_data
    xw, xo, y, event_time, censor, _ = _unpack_data(_collate_pathways([sample]), device, args.rna_format)
    return dict(x_wsi=xw, event_time=event_time, y=None, c=None, cur_epoch=args.cur_epoch,
                wsi_missing=False, omic_missing=False, **{f"x_omic{i}": x for i, x in enumerate(xo, 1)})


@torch.no_grad()
def profile_model(model, payload, repeats=30, warmup=5):
    device = next(model.parameters()).device
    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        model(**payload)
    synchronize()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        model(**payload)
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return dict(device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                torch_version=torch.__version__, batch_size=payload["x_wsi"].size(0), patches=payload["x_wsi"].size(1),
                repeats=repeats, warmup=warmup, forward="standard model.eval() forward including built-in explanations",
                parameters=sum(p.numel() for p in model.parameters()), latency_ms_median=float(np.median(samples)),
                latency_ms_p95=float(np.quantile(samples, .95)),
                peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)


def export_run(run, output_root, *, device="cuda:0", alphas=(0, .25, .5, .75, 1), case_ids=(), km=False, profile=False):
    # Enforce deterministic cuDNN so checkpoint replay matches the saved best
    # predictions within the audit's per-patient tolerance. Without this,
    # floating-point reorder from non-deterministic algorithms can push the
    # |replay - saved| difference above the 2e-5 evidence threshold on a few
    # patients (e.g. TCGA-4Z-AA7S, TCGA-A3-3363) even though the audit-level
    # C-index is unchanged.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, RuntimeError):
        pass
    torch.manual_seed(int(run.get("seed", 3)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run.get("seed", 3)))

    report = audit({"runs": [run]})
    if not report["passed"]:
        raise ValueError("; ".join(report["errors"]))
    if not run.get("checkpoint"):
        raise ValueError("Export requires the best checkpoint")
    if Path(run["id"]).name != run["id"] or run["id"] in (".", ".."):
        raise ValueError("Run ID must be a simple directory name")
    out = Path(output_root) / run["id"]
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Export directory is not empty: {out}; use a fresh output directory")
    model, args, factory, train, val = load_run(run, torch.device(device))
    baseline = predictions(run["predictions"])
    expected = dict(zip(baseline["case_ids"], baseline["risk"]))
    expected_outcome = {cid: (t, c) for cid, t, c in zip(baseline["case_ids"], baseline["time"], baseline["censor"])}
    actual_ids = val.label_df["case id"].astype(str).tolist()
    if set(actual_ids) != set(expected) or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Loaded validation patients disagree with audited predictions")
    selected = set(case_ids) if case_ids else set(sorted(actual_ids)[:3])
    if not selected <= set(actual_ids):
        raise ValueError("Requested case IDs must belong to this validation fold")
    before = {key: tensor.clone() for key, tensor in model.named_buffers()}
    collected, times, censor, cases = {}, [], [], []
    first_payload = None
    for i, cid in enumerate(actual_ids):
        sample, features, slide_indices, patch_indices = indexed_sample(val, i)
        payload = payload_for(sample, args, torch.device(device))
        values = replay(model, payload, alphas)
        if not np.isclose(float(values["risk"][0]), expected[cid], rtol=1e-5, atol=2e-5):
            raise ValueError(f"Checkpoint replay differs from saved best prediction for {cid}; verify source/config/best epoch")
        t, c = float(sample[3]), float(sample[4])
        if not np.allclose((t, c), expected_outcome[cid], rtol=1e-5, atol=1e-5):
            raise ValueError(f"Loaded outcomes differ from audited outcomes for {cid}")
        if first_payload is None:
            first_payload = payload
        for key, value in values.items():
            if key not in ("attention_wsi",):  # exclude large per-patch WSI attention; keep attention_omic for cohort heatmap
                collected.setdefault(key, []).append(value)
        if cid in selected:
            case_index = len(cases)
            name = f"case_{case_index:03d}"
            for feature in features:
                feature["sha256"] = sha256(feature["feature_path"])
            np.savez_compressed(out / f"{name}.npz", **values, slide_index=slide_indices, patch_index=patch_indices)
            cases.append(dict(case_id=cid, file=f"{name}.npz", features=features,
                              note="Final slot pooling + prototype rollout; attention is descriptive, not causal attribution"))
        times.append(t)
        censor.append(c)
        print(f"[export] {run['id']} {i + 1}/{len(actual_ids)}", flush=True)
    merged = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    merged.update(case_ids=np.asarray(actual_ids), time=np.asarray(times), censor=np.asarray(censor), alphas=np.asarray(alphas),
                  pathway_names=np.asarray(factory.pathway_names, dtype=str), bins=np.asarray(factory.bins))
    km_threshold = None
    if km:
        # Deterministic evaluation sampling on the training patients; outcomes do not select the threshold.
        reference = copy.copy(train)
        reference.split_key = "val"
        train_risks = []
        for i in range(len(reference)):
            sample, _, _, _ = indexed_sample(reference, i)
            payload = payload_for(sample, args, torch.device(device))
            with torch.no_grad():
                logits = model(**payload)[0]
                train_risks.append(float(model._risk(logits)[0]))
        km_threshold = float(np.median(train_risks))
        merged["km_high"] = merged["risk"] >= km_threshold
        merged["train_time"] = train.label_df[factory.label_col].to_numpy(float)
        merged["train_censor"] = train.label_df[factory.censorship_var].to_numpy(float)
    profile_result = profile_model(model, first_payload) if profile else None
    for key, tensor in model.named_buffers():
        if not torch.equal(tensor, before[key]):
            raise ValueError(f"Read-only replay mutated buffer {key}")
    np.savez_compressed(out / "patients.npz", **merged)
    summary = dict(schema_version=1, run=run, hashes=report["runs"][0]["hashes"],
                   best_epoch=args.cur_epoch, export_commit=revision(Path(__file__).resolve().parents[2]),
                   objective_weights=model.objective_weights_effective(), cases=cases, km_train_median=km_threshold,
                   profile=profile_result, sweep=[dict(alpha=float(alpha), cindex=cindex(times, censor, merged["sweep_risk"][:, j]),
                       mean_abs_risk_change=float(np.abs(merged["sweep_risk"][:, j] - merged["risk"]).mean()),
                       mean_cross_error=float(merged["cross_pathway_error"][:, j].mean())) for j, alpha in enumerate(alphas)],
                   note="legacy_val best-validation replay; alpha changes all stage/geometry plans using factual-marginal products; latent pathway reconstruction is not raw-gene recovery")
    save_json(out / "export.json", summary)
    return summary
