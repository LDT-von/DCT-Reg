"""Target-blind gene masking, shared decoder and plan-sensitive readout.

No raw pathology skip is present: donor features reach each pathway only
through its visible-token receiver assignment and a transport matrix.
"""
from dataclasses import dataclass
import re

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GlobalGeneSchema:
    pathway_gene_ids: tuple[tuple[str, ...], ...]
    gene_ids: tuple[str, ...]
    indices: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, pathway_gene_ids, widths, gene_ids=None):
        if not isinstance(pathway_gene_ids, (list, tuple)) or len(pathway_gene_ids) != len(widths):
            raise ValueError("actual pathway gene-ID metadata is required")
        paths, unique, positions, indices = [], [], {}, []
        for names, width in zip(pathway_gene_ids, widths):
            if not isinstance(names, (list, tuple)) or len(names) != width:
                raise ValueError("pathway gene-ID widths must match model inputs")
            names = tuple(names)
            if any(not isinstance(name, str) or not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError("gene IDs must be nonempty strings and unique within each pathway")
            local = []
            for name in names:
                if name not in positions:
                    positions[name] = len(unique)
                    unique.append(name)
                local.append(positions[name])
            paths.append(names)
            indices.append(tuple(local))
        if gene_ids is not None and tuple(gene_ids) != tuple(unique):
            raise ValueError("normalization gene-ID order must equal pathway first-occurrence order")
        return cls(tuple(paths), tuple(unique), tuple(indices))

    def validate_duplicate_values(self, values, available=None):
        """A shared gene must carry the same observed value in every copy."""
        owners = {}
        for pathway, (names, value) in enumerate(zip(self.pathway_gene_ids, values)):
            for column, name in enumerate(names):
                if name in owners:
                    opath, ocolumn = owners[name]
                    active = torch.ones(value.size(0), dtype=torch.bool, device=value.device)
                    if available is not None:
                        active = available[:, pathway] & available[:, opath]
                    if bool(active.any()) and not torch.allclose(value[active, column], values[opath][active, ocolumn], atol=1e-5, rtol=1e-5):
                        raise ValueError("duplicated raw gene has inconsistent values: " + name)
                else:
                    owners[name] = (pathway, column)

    def expand_mask(self, global_mask):
        if global_mask.dtype != torch.bool or global_mask.ndim != 2 or global_mask.size(1) != len(self.gene_ids):
            raise ValueError("global gene mask must be boolean [B,unique_genes]")
        return tuple(global_mask[:, torch.tensor(ids, device=global_mask.device)] for ids in self.indices)


def validate_gain_provenance(provenance, schema, train_patient_ids, wsi_input_dim):
    """Validate the recorded held-patient design, not just asserted flags.

    The crossfit loader additionally recomputes fingerprints/statistics against
    actual source arrays. A model with no source arrays can validate structure
    and identities, but cannot independently verify data fingerprints.
    """
    if not isinstance(provenance, dict) or provenance.get("schema") != "dct_v23_oof_pathway_gain_v1":
        raise ValueError("OOF gain provenance schema is required")
    if provenance.get("held_out_gain") is not True or provenance.get("no_survival_labels") is not True:
        raise ValueError("gain labels must come from held-patient reconstruction without survival labels")
    train = set(train_patient_ids)
    recorded = provenance.get("train_patient_ids")
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(set(recorded)) or set(recorded) != train:
        raise ValueError("OOF gain training identities do not match the actual train fold")
    if tuple(provenance.get("gene_ids", ())) != schema.gene_ids:
        raise ValueError("OOF gene order does not match reconstruction coordinates")
    paths = provenance.get("pathway_gene_ids")
    if not isinstance(paths, (list, tuple)) or tuple(tuple(path) for path in paths) != schema.pathway_gene_ids:
        raise ValueError("OOF pathway gene order does not match reconstruction coordinates")
    if provenance.get("teacher_type") != "fixed_coordinate_ridge_native_vs_conditional":
        raise ValueError("unsupported OOF reconstruction teacher")
    if provenance.get("gain_definition") != "native_masked_mse_minus_conditional_masked_mse_teacher_standardized_raw_genes":
        raise ValueError("OOF signed gain definition must be explicit")
    for name in ("raw_expression_sha256", "pooled_wsi_sha256"):
        value = provenance.get(name)
        if not isinstance(value, str) or re.fullmatch("[0-9a-fA-F]{64}", value) is None:
            raise ValueError("OOF source fingerprints are required")
    masks = provenance.get("global_masks")
    if not isinstance(masks, (list, tuple)) or not masks:
        raise ValueError("OOF global gene masks are required")
    matrix = torch.as_tensor(masks)
    if matrix.ndim != 2 or matrix.size(1) != len(schema.gene_ids) or not bool(((matrix == 0) | (matrix == 1)).all()):
        raise ValueError("OOF global gene mask dimensions or values are invalid")
    matrix = matrix.bool()
    if not bool(matrix.any(0).all()) or not bool(matrix.any(1).all()) or not bool((~matrix).any(1).all()):
        raise ValueError("OOF masks must cover every gene and retain visible genes in every round")
    teachers = provenance.get("teachers")
    if not isinstance(teachers, (list, tuple)) or len(teachers) < 2:
        raise ValueError("actual OOF teacher fit/query records are required")
    scored, teacher_ids = set(), set()
    for teacher in teachers:
        if not isinstance(teacher, dict) or "teacher_id" not in teacher:
            raise ValueError("OOF teacher records require identities")
        teacher_id = teacher["teacher_id"]
        if teacher_id in teacher_ids:
            raise ValueError("OOF teacher identities must be unique")
        teacher_ids.add(teacher_id)
        fit = teacher.get("fit_patient_ids", ())
        query = teacher.get("score_patient_ids", ())
        if (not isinstance(fit, (list, tuple)) or not isinstance(query, (list, tuple))
                or not fit or not query or len(fit) != len(set(fit)) or len(query) != len(set(query))):
            raise ValueError("OOF teacher fit/query identities must be actual unique nonempty lists")
        fit, query = set(fit), set(query)
        if fit & query or fit | query != train or query & scored:
            raise ValueError("held-patient gain requires disjoint complementary teacher fit/query sets")
        scored.update(query)
        if set(teacher.get("normalization_fit_patient_ids", ())) != fit:
            raise ValueError("OOF teacher normalization must use only its fitted patients")
        for name, width, positive in (("gene_mean", len(schema.gene_ids), False), ("gene_std", len(schema.gene_ids), True),
                                       ("wsi_mean", wsi_input_dim, False), ("wsi_std", wsi_input_dim, True)):
            values = torch.as_tensor(teacher.get(name, ()), dtype=torch.float64)
            if values.shape != (width,) or not bool(torch.isfinite(values).all()) or positive and bool((values <= 0).any()):
                raise ValueError("invalid fitted OOF teacher normalization: " + name)
    if scored != train:
        raise ValueError("OOF teacher queries must cover every training patient exactly once")


def sample_global_mask(batch, genes, ratio, device):
    if genes < 2:
        raise ValueError("masked reconstruction needs at least two unique genes")
    mask = torch.rand(batch, genes, device=device) < ratio
    # Guarantee one held-out and one visible unique gene for every patient.
    empty = ~mask.any(-1)
    full = mask.all(-1)
    if bool(empty.any()):
        mask[empty, torch.randint(genes, (int(empty.sum()),), device=device)] = True
    if bool(full.any()):
        mask[full, torch.randint(genes, (int(full.sum()),), device=device)] = False
    return mask


def pathway_receiver_assignment(visible_tokens, receiver_slots, temperature):
    with torch.autocast(device_type=visible_tokens.device.type, enabled=False):
        score = torch.bmm(F.normalize(visible_tokens.float(), dim=-1),
                          F.normalize(receiver_slots.float(), dim=-1).transpose(1, 2))
        return torch.softmax(score / temperature, -1)


def conditional_donor_context(state, plans, assignment):
    """Keep receiver-local conditional donor barycenters before pathway readout.

    Normalize within each receiver rather than globally pooling row mass.
    Fixed reconstruction gain is one, independent of deployment residual_gain;
    lowering total exchange therefore cannot lower the scored error weight.
    """
    with torch.autocast(device_type=plans.device.type, enabled=False):
        consensus = plans.float().mean(2)
        cols = consensus.sum(-2)
        moved = torch.einsum("bswo,bwd->bsod", consensus, state.wsi_slots.float())
        moved = moved / cols[..., None].clamp_min(1e-8)
        stage_weights = state.stage_weights.detach().float()
        per_receiver = torch.einsum("bs,bsod->bod", stage_weights, moved)
        return torch.bmm(assignment.float(), per_receiver)


class RawGeneReconstructionDecoder(nn.Module):
    """One common trunk and pathway output projections, reused for both arms."""
    def __init__(self, dim, hidden_dim, widths):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, hidden_dim), nn.GELU())
        self.pathway_outputs = nn.ModuleList(nn.Linear(hidden_dim, width) for width in widths)

    def forward(self, visible_tokens, native_context, donor_context):
        features = self.trunk(torch.cat((visible_tokens, native_context, donor_context), -1))
        return tuple(head(features[:, index]) for index, head in enumerate(self.pathway_outputs))


def masked_pathway_mse(predictions, targets, masks, pathway_available):
    errors, counts = [], []
    for index, (prediction, target, mask) in enumerate(zip(predictions, targets, masks)):
        valid = mask & pathway_available[:, index:index + 1]
        count = valid.sum(-1)
        error = ((prediction.float() - target.float()).square() * valid).sum(-1) / count.clamp_min(1)
        errors.append(error)
        counts.append(count)
    return torch.stack(errors, -1), torch.stack(counts, -1)


def fixed_mask_mean(errors, counts):
    # Equal available pathway-patient weight, fixed by the observed mask only.
    valid = counts > 0
    return (errors * valid).sum() / valid.sum().clamp_min(1)
