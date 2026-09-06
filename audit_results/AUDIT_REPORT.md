# E4: Continuous Intervention Audit Report

**Model**: DCTV310DirectionalRegularizedTransport  
**Dataset**: BLCA (Bladder Cancer), Fold 0  
**Test Patients**: 76  
**Intervention Strengths (α)**: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
**Date**: 2026-09-05

---

## Executive Summary

This audit evaluates whether the DCT-Reg model's risk predictions respond consistently to embedding-space interventions toward learned prognostic anchors.

### Key Findings

⚠️ **Poor Direction Consistency Detected**

- **Low-Risk Direction**: 0.00% of patients show monotonic risk decrease (expected behavior)
- **High-Risk Direction**: 5.26% of patients show monotonic risk increase (expected behavior)
- **Mean Risk Change at α=1.0**:
  - Toward Low Risk: +0.000445 (should be negative)
  - Toward High Risk: -0.000756 (should be positive)

### Interpretation

The model shows **very poor intervention consistency**. When we perturb embeddings toward the "low-risk" anchor, risks do not decrease as expected. Similarly, perturbing toward "high-risk" anchors does not reliably increase risk predictions.

This suggests:

1. **Embedding-Anchor Mismatch**: The image-form anchors (shape `[2, 3, 8, 8]`) may not correspond to meaningful directions in the embedding space (dimension 256).

2. **Intervention Method Limitations**: The current intervention approach (flattening anchors and using them as direction vectors) may not be appropriate for this architecture.

3. **Decoupled Representations**: The embedding space learned by the encoder may be decoupled from the anchor-based transport regularization used during training.

---

## Detailed Results

### Monotonicity Analysis

For each patient and direction, we check if risk changes monotonically with intervention strength α:

| Direction | Expected Behavior | Observed Monotonicity | Status |
|-----------|-------------------|----------------------|--------|
| → Low Risk | Risk ↓ as α ↑ | 0.00% (0/76 patients) | ❌ Failed |
| → High Risk | Risk ↑ as α ↑ | 5.26% (4/76 patients) | ❌ Failed |

### Risk Change Magnitudes

At maximum intervention strength (α=1.0):

| Direction | Mean Δ Risk | Std Δ Risk | Expected Sign |
|-----------|-------------|------------|---------------|
| → Low Risk | +0.000445 | 0.000301 | Negative (↓) |
| → High Risk | -0.000756 | 0.000413 | Positive (↑) |

**Note**: The observed signs are opposite to expectations, and magnitudes are extremely small (~0.0004-0.0008), suggesting minimal impact.

---

## Visualizations

### 1. Individual Intervention Curves
**File**: `visualizations/intervention_curves.png`

Shows how risk predictions change with intervention strength for 12 sample patients. Ideal behavior would show:
- Blue lines (→ Low Risk) trending downward
- Red lines (→ High Risk) trending upward

**Observed**: Most curves are nearly flat or show opposite trends.

### 2. Risk Change Distributions
**File**: `visualizations/risk_change_distribution.png`

Histograms of risk changes at α=1.0 for both directions. 

**Expected**:
- Left panel (→ Low Risk): Distribution should be left of zero (negative changes)
- Right panel (→ High Risk): Distribution should be right of zero (positive changes)

**Observed**: Distributions are centered near zero with the wrong signs.

### 3. Aggregate Risk Trends
**File**: `visualizations/alpha_vs_risk_aggregate.png`

Average risk across all patients as a function of intervention strength.

**Expected**: Clear separation between blue (decreasing) and red (increasing) curves.

**Observed**: Both curves remain close to the original risk baseline.

---

## Technical Details

### Intervention Method

The audit uses the following intervention procedure:

1. **Extract Anchors**: From model's `risk_anchor_costs` parameter (shape `[n_bins, 2, 3, 8, 8]`)
   - Low-risk anchor: `risk_anchor_costs[:, 0]` averaged over bins
   - High-risk anchor: `risk_anchor_costs[:, 1]` averaged over bins

2. **Intervention in Embedding Space**:
   - Flatten anchor to create a direction vector
   - Pad/truncate to match embedding dimension (256)
   - Apply perturbation: `embedding' = embedding + α * scale * direction`

3. **Compute Risk**: Pass perturbed embedding through the model's `event_hazard` module

### Architectural Considerations

**Challenge**: The model uses image-form anchors (spatial 3×8×8 feature maps) but computes risk from 256-dimensional embeddings. There is no direct mapping between these spaces in the intervention code.

**Potential Issues**:
- The anchors may represent transport costs in WSI feature space, not embedding directions
- The embedding encoder may use different representations than the transport module
- The flattened anchor direction may not be semantically meaningful

---

## Recommendations

### 1. Investigate Alternative Intervention Strategies

Instead of manipulating embeddings directly, consider:

**Option A: WSI Feature-Space Interventions**
- Apply anchors directly in the WSI feature space (before encoding)
- This respects the spatial structure of the 3×8×8 anchor maps

**Option B: Learn Embedding-Space Directions**
- Train linear probes to map anchors to embedding space
- Use gradient-based methods to find directions that change predictions

**Option C: Use Model's Internal Transport**
- Leverage the model's OT transport mechanism to generate counterfactual inputs
- Apply transport plans between patient features and anchor distributions

### 2. Verify Anchor Semantics

Check if the learned anchors actually represent low/high risk:

```python
# Compute anchor statistics
low_anchor = model.risk_anchor_costs[:, 0].mean(0)
high_anchor = model.risk_anchor_costs[:, 1].mean(0)
anchor_distance = torch.norm(high_anchor - low_anchor)

# Compare with training data feature distributions
# Plot anchor activations vs. patient outcomes
```

### 3. Gradient-Based Intervention

Use gradient-based counterfactual generation:

```python
# Find minimal perturbation to achieve target risk
target_risk = low_risk_target
embedding.requires_grad = True
loss = (model.predict(embedding) - target_risk)**2
perturbation = -grad(loss, embedding)
```

### 4. Re-run with Alternative Models

Test the intervention audit on:
- Baseline models without transport regularization
- Models with explicit embedding-space anchors
- Compare consistency across architectures

---

## Conclusion

The current intervention audit reveals that **the model's risk predictions do not respond consistently to embedding perturbations toward learned anchors**. This indicates a fundamental mismatch between the intervention approach and the model's architecture.

The root cause is likely that:
1. Image-form anchors (3×8×8) are not directly applicable in embedding space (256-d)
2. The transport regularization operates in a different representation space than the final risk predictor

**Next Steps**:
1. Redesign the intervention method to respect the model's architecture
2. Implement WSI feature-space interventions
3. Validate anchor semantics against training data
4. Consider gradient-based or transport-based counterfactual generation

---

## Files Generated

- `blca_fold0_audit.pkl` (CSV): Raw intervention results (1,672 records)
- `blca_fold0_audit_summary.json`: Summary statistics
- `visualizations/intervention_curves.png`: Individual patient curves
- `visualizations/risk_change_distribution.png`: Risk change histograms  
- `visualizations/alpha_vs_risk_aggregate.png`: Aggregate risk trends
- `AUDIT_REPORT.md`: This report
