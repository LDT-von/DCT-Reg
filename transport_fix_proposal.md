# Transport Plan Fix: Revolutionary Ideas

## 🔍 Problem Identified

The analysis reveals a **critical issue**: Risk anchor costs show **very low temporal differentiation** for certain event-outcome combinations. This means:

- Event 0 Outcome 1: temporal std = 0.0048 (nearly identical across time)
- Event 1 Outcome 1: temporal std = 0.0161 (nearly identical across time)  
- Event 3 Outcome 0: temporal std = 0.0471 (very similar across time)

**Impact**: Transport plans cannot properly differentiate patient trajectories over time, leading to suboptimal risk stratification.

---

## 💡 Revolutionary Solution: Multi-Scale Temporal Transport with Adaptive Anchors

### Idea 1: **Temporal Contrast Learning for Anchors** ⭐⭐⭐⭐⭐

**Core Insight**: Risk anchors should be explicitly optimized to maximize temporal contrast while maintaining within-time spatial variation.

```python
class TemporalContrastiveAnchorLoss(nn.Module):
    """Force anchors to differentiate across time bins"""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, risk_anchor_costs):
        """
        risk_anchor_costs: [events, outcomes, time_bins, H, W]
        """
        losses = []
        
        for event_idx in range(risk_anchor_costs.shape[0]):
            for outcome_idx in range(risk_anchor_costs.shape[1]):
                # Get temporal embeddings: [time_bins, H*W]
                temporal_costs = risk_anchor_costs[event_idx, outcome_idx].flatten(1)
                
                # Compute temporal statistics
                time_means = temporal_costs.mean(dim=1)  # [time_bins]
                
                # Temporal contrast: adjacent time bins should be different
                # but far time bins should be even more different
                for t in range(len(time_means) - 1):
                    # Encourage monotonic trend (early time != late time)
                    temporal_diff = (time_means[t+1] - time_means[t]).abs()
                    
                    # Penalize if difference is too small
                    contrast_loss = torch.relu(0.1 - temporal_diff)
                    losses.append(contrast_loss)
                
                # Global temporal variance should be high
                temporal_variance = time_means.var()
                variance_loss = torch.relu(0.05 - temporal_variance)
                losses.append(variance_loss)
        
        return torch.stack(losses).mean()
```

**Why This Works**:
- Explicitly forces temporal differentiation
- Maintains spatial structure within each time bin
- Encourages meaningful progression patterns

---

### Idea 2: **Hierarchical Multi-Resolution Anchors** ⭐⭐⭐⭐⭐

**Core Insight**: Single-scale 8×8 anchors are too coarse. Use multi-resolution anchor pyramid.

```python
class MultiResolutionAnchorCosts(nn.Module):
    """Multi-scale transport anchors for fine-grained plans"""
    
    def __init__(self, n_events=4, n_outcomes=2, n_time_bins=3):
        super().__init__()
        
        # Multi-scale anchors
        self.anchor_coarse = nn.Parameter(torch.randn(n_events, n_outcomes, n_time_bins, 4, 4))   # Coarse
        self.anchor_medium = nn.Parameter(torch.randn(n_events, n_outcomes, n_time_bins, 8, 8))   # Medium
        self.anchor_fine = nn.Parameter(torch.randn(n_events, n_outcomes, n_time_bins, 16, 16))  # Fine
        
        # Learnable combination weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Initialize with temporal progression
        self._init_temporal_structure()
    
    def _init_temporal_structure(self):
        """Initialize with monotonic temporal structure"""
        with torch.no_grad():
            for anchor in [self.anchor_coarse, self.anchor_medium, self.anchor_fine]:
                for event_idx in range(anchor.shape[0]):
                    for outcome_idx in range(anchor.shape[1]):
                        # Create increasing temporal pattern
                        for t in range(anchor.shape[2]):
                            # Base value increases with time
                            base = 0.5 + t * 0.3
                            # Add spatial variation
                            anchor[event_idx, outcome_idx, t] = base + torch.randn_like(anchor[event_idx, outcome_idx, t]) * 0.2
    
    def forward(self, feature_maps, target_resolution=(8, 8)):
        """
        Compute multi-scale transport costs
        
        feature_maps: List of features at different scales
        Returns: Aggregated cost at target resolution
        """
        # Interpolate all anchors to target resolution
        w = F.softmax(self.scale_weights, dim=0)
        
        costs = []
        for scale_idx, anchor in enumerate([self.anchor_coarse, self.anchor_medium, self.anchor_fine]):
            # Interpolate to target resolution
            resized = F.interpolate(
                anchor.flatten(0, 2).unsqueeze(1),  # [events*outcomes*time, 1, H, W]
                size=target_resolution,
                mode='bilinear',
                align_corners=False
            )
            costs.append(resized * w[scale_idx])
        
        # Combine scales
        combined = torch.stack(costs).sum(dim=0)
        
        # Reshape back
        return combined.view(anchor.shape[0], anchor.shape[1], anchor.shape[2], *target_resolution)
```

**Why This Works**:
- Captures both coarse (regional) and fine-grained (cellular) transport patterns
- Multi-scale structure naturally creates richer cost landscapes
- Learnable combination allows model to emphasize relevant scales

---

### Idea 3: **Dynamically Conditioned Anchors** ⭐⭐⭐⭐⭐

**Core Insight**: Risk anchors should be patient-specific, not fixed across the entire dataset.

```python
class DynamicAnchorGenerator(nn.Module):
    """Generate patient-specific transport anchors"""
    
    def __init__(self, feature_dim=256, n_events=4, n_outcomes=2, n_time_bins=3, spatial_size=8):
        super().__init__()
        
        # Base anchors (learnable template)
        self.base_anchors = nn.Parameter(torch.randn(n_events, n_outcomes, n_time_bins, spatial_size, spatial_size))
        
        # Patient conditioning network
        self.condition_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Combined WSI + Omic features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_events * n_outcomes * n_time_bins),
            nn.Tanh()  # Output scale/shift factors
        )
        
        # Temporal progression network
        self.temporal_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_time_bins),
            nn.Softplus()  # Ensure positive temporal progression
        )
    
    def forward(self, patient_features):
        """
        patient_features: [batch, feature_dim * 2] (concatenated WSI + Omic)
        Returns: [batch, events, outcomes, time_bins, H, W]
        """
        batch_size = patient_features.shape[0]
        
        # Generate patient-specific modulation factors
        modulation = self.condition_net(patient_features)  # [batch, events*outcomes*time]
        modulation = modulation.view(batch_size, *self.base_anchors.shape[:3])  # [batch, events, outcomes, time]
        
        # Generate temporal progression factors
        temporal_factors = self.temporal_net(patient_features)  # [batch, time_bins]
        
        # Broadcast and apply
        dynamic_anchors = self.base_anchors.unsqueeze(0)  # [1, events, outcomes, time, H, W]
        
        # Apply patient-specific modulation
        modulation_expanded = modulation.unsqueeze(-1).unsqueeze(-1)  # [batch, events, outcomes, time, 1, 1]
        dynamic_anchors = dynamic_anchors * (1.0 + 0.3 * modulation_expanded)
        
        # Apply temporal progression
        temporal_expanded = temporal_factors.view(batch_size, 1, 1, -1, 1, 1)  # [batch, 1, 1, time, 1, 1]
        dynamic_anchors = dynamic_anchors * temporal_expanded
        
        return dynamic_anchors
```

**Why This Works**:
- Anchors adapt to each patient's specific characteristics
- Combines template knowledge (base anchors) with patient-specific information
- Temporal factors ensure proper time progression per patient
- Much more expressive than fixed anchors

---

### Idea 4: **Transport Plan Regularization with Entropy Control** ⭐⭐⭐⭐

**Core Insight**: Transport plans should be neither too uniform (collapsed) nor too chaotic (noisy).

```python
class TransportPlanRegularizer(nn.Module):
    """Regularize transport plans for optimal structure"""
    
    def __init__(self, target_entropy=2.0):
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(self, transport_plan, risk_anchor_costs):
        """
        transport_plan: [batch, events, time, H_feat, W_feat, H_anchor, W_anchor]
        risk_anchor_costs: [events, outcomes, time, H_anchor, W_anchor]
        """
        losses = {}
        
        # 1. Entropy regularization: plans should have moderate entropy
        plan_normalized = F.softmax(transport_plan.flatten(-2), dim=-1)
        entropy = -(plan_normalized * torch.log(plan_normalized + 1e-10)).sum(dim=-1).mean()
        
        entropy_loss = (entropy - self.target_entropy) ** 2
        losses['entropy'] = entropy_loss
        
        # 2. Temporal consistency: plans should evolve smoothly across time
        if transport_plan.shape[2] > 1:
            temporal_diff = (transport_plan[:, :, 1:] - transport_plan[:, :, :-1]).abs().mean()
            # Penalize too much or too little change
            temporal_loss = torch.relu(temporal_diff - 0.5) + torch.relu(0.1 - temporal_diff)
            losses['temporal_consistency'] = temporal_loss
        
        # 3. Spatial concentration: plans should focus on relevant regions
        # Compute effective support (how spread out the plan is)
        plan_flat = plan_normalized.flatten(-2)  # [batch, events, time, H*W*H*W]
        squared_plan = plan_flat ** 2
        effective_support = 1.0 / squared_plan.sum(dim=-1).mean()
        
        # We want moderate concentration (not too diffuse, not too peaked)
        target_support = plan_flat.shape[-1] * 0.2  # Use 20% of available mass
        concentration_loss = (effective_support - target_support) ** 2 / target_support
        losses['concentration'] = concentration_loss
        
        # 4. Anchor utilization: all anchor cells should be used
        anchor_usage = transport_plan.flatten(-2).max(dim=-1)[0].mean(dim=0)  # [events, time, H_feat, W_feat]
        unused_anchors = (anchor_usage < 0.01).float().mean()
        losses['anchor_utilization'] = unused_anchors
        
        return losses
```

**Why This Works**:
- Prevents mode collapse (all plans identical)
- Encourages structured, interpretable transport plans
- Ensures all learned anchors contribute meaningfully
- Balances exploration vs exploitation

---

### Idea 5: **Curriculum Learning for Transport Plans** ⭐⭐⭐⭐⭐

**Core Insight**: Start with simple transport, gradually increase complexity.

```python
class TransportCurriculumScheduler:
    """Progressive training schedule for transport learning"""
    
    def __init__(self, total_epochs=50):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Curriculum stages
        self.stages = [
            {'name': 'warmup', 'epochs': 10, 'focus': 'basic_anchors'},
            {'name': 'temporal', 'epochs': 15, 'focus': 'temporal_differentiation'},
            {'name': 'spatial', 'epochs': 15, 'focus': 'spatial_structure'},
            {'name': 'joint', 'epochs': 10, 'focus': 'full_transport'}
        ]
    
    def get_loss_weights(self, epoch):
        """Return loss weights for current curriculum stage"""
        self.current_epoch = epoch
        
        # Determine current stage
        cumsum = 0
        for stage in self.stages:
            cumsum += stage['epochs']
            if epoch < cumsum:
                return self._get_stage_weights(stage['focus'], epoch)
        
        return self._get_stage_weights('full_transport', epoch)
    
    def _get_stage_weights(self, focus, epoch):
        """Weights for different loss components"""
        weights = {
            'survival_loss': 1.0,
            'ranking_loss': 0.0,
            'temporal_contrast': 0.0,
            'transport_reg': 0.0,
            'anchor_lr_multiplier': 1.0
        }
        
        if focus == 'basic_anchors':
            # Stage 1: Learn basic anchor structure
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.1,
                'temporal_contrast': 0.0,
                'transport_reg': 0.0,
                'anchor_lr_multiplier': 5.0  # Higher LR for anchors
            })
        
        elif focus == 'temporal_differentiation':
            # Stage 2: Force temporal differentiation
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.3,
                'temporal_contrast': 2.0,  # HIGH weight
                'transport_reg': 0.1,
                'anchor_lr_multiplier': 3.0
            })
        
        elif focus == 'spatial_structure':
            # Stage 3: Refine spatial structure
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.5,
                'temporal_contrast': 1.0,
                'transport_reg': 0.5,  # Increase regularization
                'anchor_lr_multiplier': 1.0
            })
        
        elif focus == 'full_transport':
            # Stage 4: Joint optimization
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 1.0,
                'temporal_contrast': 0.5,
                'transport_reg': 0.3,
                'anchor_lr_multiplier': 0.5  # Lower LR for fine-tuning
            })
        
        return weights
    
    def get_anchor_lr(self, base_lr, epoch):
        """Get learning rate for anchor parameters"""
        weights = self.get_loss_weights(epoch)
        return base_lr * weights['anchor_lr_multiplier']
```

**Why This Works**:
- Prevents anchors from getting stuck in poor local minima
- Explicitly addresses temporal differentiation in dedicated stage
- Progressively refines transport structure
- Adaptive learning rates prevent catastrophic forgetting

---

## 🎯 Implementation Strategy

### Phase 1: Quick Wins (1-2 days)
1. ✅ Add **Temporal Contrast Loss** to existing model
2. ✅ Implement **Curriculum Scheduler**
3. ✅ Increase anchor learning rate in early epochs

### Phase 2: Core Improvements (3-5 days)
1. ✅ Implement **Multi-Resolution Anchors**
2. ✅ Add **Transport Plan Regularization**
3. ✅ Integrate all losses with curriculum

### Phase 3: Advanced Features (5-7 days)
1. ✅ Implement **Dynamic Anchor Generator**
2. ✅ Full integration and testing
3. ✅ Hyperparameter tuning

---

## 📊 Expected Results

### Before (Current State)
- Temporal differentiation std: **0.005-0.047** (too low)
- Transport plans: Nearly uniform across patients
- C-index: ~0.65

### After (With Fixes)
- Temporal differentiation std: **>0.15** (healthy variation)
- Transport plans: Patient-specific, interpretable
- C-index: **>0.75** (expected improvement)

### Key Metrics to Track
1. **Temporal variation**: std of time-bin means per event-outcome
2. **Spatial variation**: avg std within each time bin
3. **Plan diversity**: entropy of transport plans across patients
4. **Anchor utilization**: % of anchor cells actively used

---

## 💎 Why This is Revolutionary

1. **First principles**: Addresses root cause (lack of temporal differentiation) not symptoms
2. **Multi-scale**: Captures both coarse and fine patterns
3. **Adaptive**: Patient-specific anchors vs one-size-fits-all
4. **Principled regularization**: Prevents collapse while maintaining expressiveness
5. **Curriculum**: Systematic training that builds complexity progressively

This approach transforms transport learning from a **passive cost computation** to an **active, structured optimization** process.
