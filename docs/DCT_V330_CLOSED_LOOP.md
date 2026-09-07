# DCT v3.30: Closed-Loop Prognostic Transport

## Status and claim boundary

DCT v3.30 is an isolated research candidate. Its implementation and structural
tests can establish that the intended information path exists; they cannot
establish a performance gain, calibration improvement, or biological validity.
Those claims require the matched five-fold experiment below.

The candidate does not modify the frozen DCT v3.10 method or the original v3.2
TGSR study.

## Closed prediction path

The main prediction path is:

```text
encoded WSI/pathway tokens
        -> initial modality slots
        -> patient-specific cross-modal feedback plan
        -> plan-guided re-reading of the original tokens
        -> reaggregated modality slots
        -> stage-specific transport plans and event tokens
        -> survival logits and risk
```

The risk head receives only event tokens constructed from the reaggregated
slots and their final transport plans. Raw tokens and pre-feedback slots are not
concatenated into the risk head. The `baseline` arm still uses the same final
transport risk path, but skips feedback reaggregation so it remains an
architecture-matched control.

Survival labels are used only by the optional training-time prognostic
representation objective. They do not enter a patient's feedback cost,
marginals, plan, confidence features, or gate.

## Four modules

1. **Prognostic Slot Encoder** creates WSI and pathway slots from the encoded
   tokens.
2. **Confidence-aware OT** standardizes each patient's feedback cost, expresses
   epsilon relative to that cost scale, and mixes learned marginals with a
   uniform floor.
3. **OT-guided Reaggregation** turns the plan into cross-modal slot context,
   re-reads the original tokens, and applies patient/slot gates to the update.
4. **Transport-only Risk Head** forms stage event tokens from the reaggregated
   slots and predicts risk only from those tokens.

For each modality slot, the gate observes four quantities derived from the
current transport computation: marginal-weighted conditional plan uncertainty,
conditional slot entropy, transported cross-modal cosine consistency, and
marginal error. Joint-plan entropy remains a separate diagnostic; using it as
the gate's patient feature would confuse concentrated learned marginals with
confident cross-modal correspondence. The
gate is initialized to 0.25, matching the original fixed TGSR strength, but can
adapt independently for each patient and slot.

```text
new_slot = old_slot + gate(patient, slot) * (reread_slot - old_slot)
```

The learned row and column marginals are mixed with uniform marginals using a
fixed stability coefficient. This prevents the marginal scorer from assigning
arbitrarily concentrated mass at initialization. Epsilon is annealed as a
dimensionless ratio after per-patient cost standardization. Diagnostics report
the corresponding epsilon in the raw cost scale.

## Prognostic transport representation objective

The final arm learns low-risk and high-risk transport prototypes. Early
observed events are assigned to the high-risk prototype; sufficiently long
survival times are assigned to the low-risk prototype. The thresholds are
quantiles fitted from the training fold's observed event times. The contrast is
weighted by training-fold inverse censoring probabilities.

This objective supervises the event-token representation after feedback and
final transport. It does not require an anchor cost to have a prescribed time
trend and does not create a patient-specific anchor from that patient's label.

## Matched experiment

All arms use the same split, seed, optimizer, batch size, training budget, slot
encoder, and final risk architecture:

| Arm | Feedback | Confidence-aware plan/gate | Prognostic representation loss |
|---|---|---:|---:|
| A `baseline` | none | no | no |
| B `self_update` | within-modality reread | no | no |
| C `ot_feedback` | original fixed TGSR | no | no |
| D `confidence_gate` | adaptive OT reread | yes | no |
| E `prognostic_rank` | adaptive OT reread | yes | yes |

Arm D changes the feedback plan, gate, learned stable marginals, and the final
cost-scaled solver as one confidence-aware module. If D is promising, the next
mechanism audit must split these changes into solver-only, marginal-only, and
gate-only ablations before attributing the result specifically to the gate.

The queue contains 25 jobs: five arms by five BLCA folds. Each result directory
is fingerprinted by the configuration, source tree, Git revision, seed, fold,
and arm.

```powershell
python scripts/run_dct_v330_experiments.py --mode plan
python scripts/run_dct_v330_experiments.py --mode smoke
python scripts/run_dct_v330_experiments.py --mode run
```

## Required evaluation

Report fold-level and aggregate Harrell C-index, IPCW C-index, IBS, iAUC,
calibration, and cross-fold dispersion. Also archive feedback and final-plan
entropy, dependence total variation, cross-geometry plan total variation,
marginal error, effective epsilon, gate mean/range, and slot-change magnitude.

A C-index improvement alone is insufficient. The candidate should be rejected
or recalibrated if its ordering improves while IBS, iAUC, or calibration
consistently deteriorates. Plan entropy should be interpreted together with
dependence total variation and marginal error: a low-entropy plan is not by
itself evidence of meaningful cross-modal correspondence.
