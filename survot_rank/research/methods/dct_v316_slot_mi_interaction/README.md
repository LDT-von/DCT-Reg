# DCT v3.16: Slot Interaction Decomposition Candidate

This candidate tests whether patient-specific multimodal interaction routing can
improve discrete survival prediction. It combines independently initialized
WSI and Omics slots, bidirectional cross-attention, sparse Top-K slot decoders,
and four learned interaction channels.

The four channel labels are modelling hypotheses:

- `R`: shared/redundant interaction candidate
- `Up`: WSI-specific candidate
- `Ug`: Omics-specific candidate
- `S`: joint/synergistic candidate

They are not a formal partial-information decomposition. A channel profile is
a learned patient-local routing distribution and should only be interpreted as
R/Up/Ug/S evidence after matched ablations or intervention tests support those
semantics.

## Prediction and objective

The model emits four discrete-time hazard logits. The shared trainer supplies
the survival NLL, while the model adds censor-aware ranking and diagonal
in-batch contrastive alignment:

```text
L = L_NLL / B + 0.10 * L_IPCW-rank
    + lambda_MI * (L_R + L_Up + L_Ug + L_S)
```

The default `lambda_MI` is `0.01`. Each contrastive term aligns one channel
with the same patient's detached base hazard logits; other patients in the
mini-batch are negatives. Batch size one therefore has a zero contrastive
term. The learned profile also contributes a residual to the final logits, so
both the profile scorer and all four channels receive survival-loss gradients.

Training uses stochastic Gaussian slot initialization and Gumbel Top-K routing.
Evaluation uses posterior-mean slots and deterministic Top-K selection, making
repeat validation and explanations reproducible.

## Framework usage

```python
model = DCTV316SlotMIDecomposition(args, omic_input_dim=20)
model.configure_train_reference(train_times, train_censorship)
logits, auxiliary_loss = model(
    x_wsi=x_wsi,
    x_omics=x_omics,
    event_time=event_time,
    c=censorship,
)
```

`logits` has shape `[B, n_classes]`. The standard configuration is
`configs/dct_v316_slot_mi_decomposition.yaml` and the registered aliases are
`dct_v316` and `dct_v316_slot_mi`.

## Standalone block

```python
block = SlotBasedMIDecompositionBlock(
    dim=32,
    num_wsi_slots=8,
    num_omic_slots=4,
    hidden_dim=64,
    num_classes=4,
)
output, metrics = block(
    torch.randn(4, 10, 32),
    torch.randn(4, 5, 32),
)
assert output["logits"].shape == (4, 4)
assert output["profile"].shape == (4, 4)
```

This directory contains a research candidate. Passing structural tests does
not establish BLCA performance or biological validity; those require matched
five-fold experiments and an archived evaluation protocol.
