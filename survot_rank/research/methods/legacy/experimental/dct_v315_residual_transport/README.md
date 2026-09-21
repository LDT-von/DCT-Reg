# v3.15: NLL-only Residual Transport Interaction

Standalone experimental baseline. Directly inherits nn.Module; requires only
PyTorch and the standard library, with no older model or shared encoder imports.

- Small token encoders → one-shot attention slots (8+8 by default).
- Slot means → global survival MLP.
- Centered, variance-normalized slots → one cosine OT → residual bilinear moment.
- Final logits = global logits + bias-free linear interaction readout.
- Exactly one training loss: patient-level censored NLL, divided by B once.

Core: `v = sum_ij (T_ij - 1/(K_w*K_o)) * Rw_i * Ro_j`.
This is a centered cross-moment, not a new loss or a claim to invent covariance.
The independent-coupling control is exactly zero; pairwise logit contributions
sum to the interaction logits. This is not causal or additive risk attribution.

No rank, per-slot NLL, diversity penalty, reconstruction, recurrent slots,
event transformer, risk anchors, or epoch-dependent transport schedule.
Without extra losses there is still no theorem preventing collapsed encoders,
redundant queries, or an unused interaction branch. Diagnostics expose raw slot
cosine, slot standard deviation, attention overlap, OT error and interaction RMS.

Use configs/dct_v315_blca_uni.yaml (alpha_surv=0) and the standard training CLI.
The shared parser's generic alpha default remains 0.5, so use this YAML or specify
alpha explicitly. Compare `dct_v315_transport_mode=ot` against `independent` in
different result directories. Real-data performance is not yet established.

`explain_last_batch()` returns detached additive logit evidence for the last
forward, including its cached readout weights. It is not compatible with the
older risk-anchor / cost-intervention audit scripts.

Detailed rationale, limitations, diagnostics and commands: docs/DCT_V315_BASELINE.md.
