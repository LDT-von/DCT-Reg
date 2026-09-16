# DCT v3.13 — Transport-Aware Omics Reconstruction

DCT v3.13 keeps the DCT v3.11 factual prediction path and replaces the
unimplemented v3.12 proposal with an executable reconstruction objective.

## Prediction path

1. Encode WSI patches and pathway-level omics inputs.
2. Form modality-specific slots and align them to shared semantic prototypes.
3. Build stage-specific cosine, Euclidean, and positive-dot costs.
4. Solve the factual Sinkhorn plans.
5. Encode the selected stage event tokens and predict discrete hazards.

## Training objective

```text
L = L_NLL
  + 0.10 L_IPCW-rank
  + 0.05 L_per-slot-NLL
  + 0.10 L_slot-diversity
  + lambda_rec(epoch) L_reconstruction
```

The reconstruction term is

```text
L_reconstruction = 0.5 L_omics-self + 0.5 L_transport-WSI-to-omics
```

`lambda_rec` is zero through epoch 2, ramps linearly to `0.10` over the next
five epochs, and remains at `0.10` afterwards.

The target is the stop-gradient encoded pathway-token tensor.  Each distance is
the mean of cosine distance and Smooth L1 after token-wise layer normalization.

## Transport-aware cross reconstruction

For each stage, the three factual geometry plans are averaged.  WSI slots are
transported into the Omics slot coordinates using the consensus plan and its
column mass.  The existing survival event gate, detached for this auxiliary
path, combines the stage-specific transported slots.  A shared pathway-query
decoder reconstructs pathway tokens from either Omics slots or transported WSI
slots.

The cross-reconstruction branch depends on paired WSI and Omics observations
because its factual transport plan uses both modalities.  It is a training-time
alignment regularizer and does not by itself establish WSI-only Omics
imputation at inference.
