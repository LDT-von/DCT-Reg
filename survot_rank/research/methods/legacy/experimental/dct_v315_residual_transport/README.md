# v3.15: Residual Transport Interaction

Standalone experimental baseline. Directly inherits nn.Module; requires only
PyTorch and the standard library, with no older model or shared encoder imports.

- Small token encoders → one-shot attention slots (8+8 by default).
- Slot means → global survival MLP.
- Centered, variance-normalized slots → one cosine OT → residual bilinear moment.
- Final logits = global logits + bias-free linear interaction readout.
- Historical control: patient-level censored NLL, divided by B once.
- Ranked candidate: NLL plus `0.10 * RTI-directed IPCW rank`.

Core: `v = sum_ij (T_ij - 1/(K_w*K_o)) * Rw_i * Ro_j`.
This is a centered cross-moment, not a new loss or a claim to invent covariance.
The independent-coupling control is exactly zero; pairwise logit contributions
sum to the interaction logits. This is not causal or additive risk attribution.

The optional rank term selects fold-local comparable pairs using Uno-style
left-limit IPCW. It only selects pairs the detached global head has not already
ranked by the configured margin, then updates the RTI path through
`risk(global.detach() + interaction)`. This makes the transport residual learn
corrections instead of duplicating a generic global ranking objective. The
independent-coupling control has zero interaction and zero rank auxiliary.

There is no per-slot NLL, diversity penalty, reconstruction, recurrent slot,
event transformer, risk-anchor, or epoch-dependent transport schedule.

Use `configs/dct_v315_blca_uni2h.yaml` for the archived NLL-only control and
`configs/dct_v315_rti_rank_blca_uni2h.yaml` for the ranked candidate. Compare
the two recipes and `dct_v315_transport_mode=independent` with identical splits,
seeds and budgets. Real-data performance is not yet established.

`explain_last_batch()` returns detached additive logit evidence for the last
forward, including its cached readout weights. It is not compatible with the
older risk-anchor / cost-intervention audit scripts.

Detailed rationale, limitations, diagnostics and commands: docs/DCT_V315_BASELINE.md.
