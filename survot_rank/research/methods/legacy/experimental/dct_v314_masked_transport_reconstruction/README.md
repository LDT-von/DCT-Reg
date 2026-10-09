# DCT v3.14 — debugged experimental candidate

This version has no runtime import or inheritance from older model versions.
model.py uses its own TransportBackbone in backbone.py, plus stable logits-based
survival loss in losses.py. Only the common SNN_Block and WSI_Mlp are shared
outside this directory. Transport/reader code is adapted from this repository;
isolation is not a claim that those equations are newly invented.

## Candidate objective

    sum(patient NLL)/B
    + 0.10 * IPCW ranking
    + 0.05 * mean per-slot NLL
    + 0.02 * unit-content pair-distance hinge
    + ramp(epoch) * 0.10 * (0.5*self reconstruction + 0.5*OT-cross reconstruction)

Both NLL terms use the same censoring convention: c=1 is censored;
alpha_surv scales censored contributions by 1-alpha. Plain likelihood uses
alpha=0; the reference YAML explicitly retains the trainer's historical alpha=0.5.
IPCW is used for ranking, not multiplied a second time into per-slot censored NLL.
Do not divide the returned auxiliary loss by batch size again.

Persistent prototype queries route actual token values; they do not add an
identity embedding to output content. Attention logit scale compensates for
feature dimension. The anti-collapse loss is averaged over within-patient
slot pairs: with unit normalized, feature-centered content u, it penalizes
relu(1 - ||u_i-u_j|| / sqrt(0.2))**2. It does not demand divergent hazards.
Exact symmetry still has a zero subgradient: this is not a no-collapse theorem.

## Reconstruction modes

- masked (default): mask approximately 20% of pathway tokens before slot
  encoding; recompute OT and score only masked positions.
- full: unmasked self/cross reconstruction on all pathways, implemented locally.
- hybrid: average masked and full losses; preserve the same total 0.10 budget.
- off: no auxiliary reconstruction computation.

Rounding masks at least one position, leaving one visible when there are two or
more pathways. The target is a detached, dropout-free output of the current
omics encoder, not an EMA teacher. Auxiliary transport uniformly averages stages:
even a detached factual full-omics gate would be an unwanted information path.
Overlapping genes in different biological pathways remain shared information;
this is pathway-token completion, not proof of gene-level masking independence.

The factual survival branch always uses all available input modalities.
Missing modalities need placeholder tensors and explicit availability flags;
invalid unavailable values are ignored, invalid available values fail fast.
Missing-modality predictive quality is not validated.

## Reproducibility

The three auxiliary coefficients, mask ratio and reconstruction mode are explicit
CLI/YAML options. Checkpoints check the recipe and include training-reference
buffers and the transport annealing epoch. Older experimental checkpoints are
intentionally incompatible; do not silently load them with strict=False.

Use scripts/audit_v314.py for reproducible synthetic gradient/shape checks and
tests/test_dct_v314_masked_transport_reconstruction.py for regression tests.
The complete Chinese audit report is docs/DCT_V314_DEBUG_REPORT.md.
Real fold-level training and performance/interpretability validation remain pending.
