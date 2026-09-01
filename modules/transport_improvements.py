"""
Transport Plan Improvements: Revolutionary Components
Addresses critical temporal differentiation issues in risk anchor costs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class TemporalContrastiveAnchorLoss(nn.Module):
    """
    Force anchors to differentiate across time bins.
    
    Key insight: Risk anchors should explicitly maximize temporal contrast
    while maintaining within-time spatial variation.
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 min_temporal_diff: float = 0.1,
                 min_temporal_variance: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self.min_temporal_diff = min_temporal_diff
        self.min_temporal_variance = min_temporal_variance
    
    def forward(self, risk_anchor_costs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            risk_anchor_costs: [events, outcomes, time_bins, H, W]
        
        Returns:
            Scalar loss encouraging temporal differentiation
        """
        losses = []
        
        n_events, n_outcomes, n_time_bins = risk_anchor_costs.shape[:3]
        
        for event_idx in range(n_events):
            for outcome_idx in range(n_outcomes):
                # Get temporal costs for this event-outcome pair
                temporal_costs = risk_anchor_costs[event_idx, outcome_idx]  # [time_bins, H, W]
                
                # Compute mean cost per time bin
                time_means = temporal_costs.flatten(1).mean(dim=1)  # [time_bins]
                
                # 1. Adjacent time bins should be different
                if n_time_bins > 1:
                    for t in range(n_time_bins - 1):
                        temporal_diff = (time_means[t+1] - time_means[t]).abs()
                        
                        # Penalize if difference is too small
                        contrast_loss = F.relu(self.min_temporal_diff - temporal_diff)
                        losses.append(contrast_loss)
                
                # 2. Global temporal variance should be high
                if n_time_bins > 1:
                    temporal_variance = time_means.var()
                    variance_loss = F.relu(self.min_temporal_variance - temporal_variance)
                    losses.append(variance_loss)
                
                # 3. Monotonic trend encouragement (optional)
                # Early and late time costs should be consistently different
                if n_time_bins >= 3:
                    early_late_diff = (time_means[-1] - time_means[0]).abs()
                    monotonic_loss = F.relu(self.min_temporal_diff * 2 - early_late_diff)
                    losses.append(monotonic_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=risk_anchor_costs.device)
        
        return torch.stack(losses).mean()


class MultiResolutionAnchorCosts(nn.Module):
    """
    Multi-scale transport anchors for fine-grained transport plans.
    
    Key insight: Single-scale 8×8 anchors are too coarse.
    Use pyramid of resolutions for richer cost landscapes.
    """
    
    def __init__(self, 
                 n_events: int = 4,
                 n_outcomes: int = 2,
                 n_time_bins: int = 3,
                 resolutions: List[Tuple[int, int]] = [(4, 4), (8, 8), (16, 16)],
                 target_resolution: Tuple[int, int] = (8, 8)):
        super().__init__()
        
        self.n_events = n_events
        self.n_outcomes = n_outcomes
        self.n_time_bins = n_time_bins
        self.resolutions = resolutions
        self.target_resolution = target_resolution
        
        # Create multi-scale anchors
        self.anchors = nn.ParameterList([
            nn.Parameter(torch.randn(n_events, n_outcomes, n_time_bins, h, w))
            for h, w in resolutions
        ])
        
        # Learnable combination weights
        self.scale_weights = nn.Parameter(torch.ones(len(resolutions)) / len(resolutions))
        
        # Initialize with temporal structure
        self._init_temporal_structure()
    
    def _init_temporal_structure(self):
        """Initialize anchors with monotonic temporal progression"""
        with torch.no_grad():
            for anchor in self.anchors:
                for event_idx in range(self.n_events):
                    for outcome_idx in range(self.n_outcomes):
                        # Create increasing temporal pattern
                        for t in range(self.n_time_bins):
                            # Base value increases with time
                            base = 0.5 + t * 0.4
                            # Add spatial variation
                            spatial_noise = torch.randn_like(anchor[event_idx, outcome_idx, t]) * 0.25
                            anchor[event_idx, outcome_idx, t] = base + spatial_noise
    
    def forward(self) -> torch.Tensor:
        """
        Compute multi-scale aggregated transport costs.
        
        Returns:
            Combined cost at target resolution: [events, outcomes, time_bins, H, W]
        """
        # Normalize scale weights
        w = F.softmax(self.scale_weights, dim=0)
        
        costs = []
        for scale_idx, anchor in enumerate(self.anchors):
            # Reshape for interpolation
            batch_shape = anchor.shape[:3]
            flat_anchor = anchor.flatten(0, 2).unsqueeze(1)  # [events*outcomes*time, 1, H, W]
            
            # Interpolate to target resolution
            resized = F.interpolate(
                flat_anchor,
                size=self.target_resolution,
                mode='bilinear',
                align_corners=False
            )
            
            # Reshape back
            resized = resized.squeeze(1).view(*batch_shape, *self.target_resolution)
            
            # Apply scale weight
            costs.append(resized * w[scale_idx])
        
        # Combine all scales
        combined = torch.stack(costs).sum(dim=0)
        
        return combined
    
    def get_scale_weights(self) -> torch.Tensor:
        """Return normalized scale weights for analysis"""
        return F.softmax(self.scale_weights, dim=0)


class DynamicAnchorGenerator(nn.Module):
    """
    Generate patient-specific transport anchors.
    
    Key insight: Risk anchors should be conditioned on patient features,
    not fixed across the entire dataset.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 n_events: int = 4,
                 n_outcomes: int = 2,
                 n_time_bins: int = 3,
                 spatial_size: int = 8,
                 modulation_strength: float = 0.3):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_events = n_events
        self.n_outcomes = n_outcomes
        self.n_time_bins = n_time_bins
        self.spatial_size = spatial_size
        self.modulation_strength = modulation_strength
        
        # Base anchors (learnable template)
        self.base_anchors = nn.Parameter(
            torch.randn(n_events, n_outcomes, n_time_bins, spatial_size, spatial_size)
        )
        
        # Patient conditioning network
        self.condition_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Combined WSI + Omic features
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_events * n_outcomes * n_time_bins),
            nn.Tanh()  # Output modulation factors in [-1, 1]
        )
        
        # Temporal progression network
        self.temporal_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_time_bins),
            nn.Softplus()  # Ensure positive temporal factors
        )
        
        # Initialize base anchors with structure
        self._init_base_anchors()
    
    def _init_base_anchors(self):
        """Initialize base anchors with temporal progression"""
        with torch.no_grad():
            for event_idx in range(self.n_events):
                for outcome_idx in range(self.n_outcomes):
                    for t in range(self.n_time_bins):
                        base = 0.8 + t * 0.3
                        self.base_anchors[event_idx, outcome_idx, t] = \
                            base + torch.randn_like(self.base_anchors[event_idx, outcome_idx, t]) * 0.2
    
    def forward(self, patient_features: torch.Tensor) -> torch.Tensor:
        """
        Generate patient-specific anchors.
        
        Args:
            patient_features: [batch, feature_dim * 2] (concatenated WSI + Omic embeddings)
        
        Returns:
            Dynamic anchors: [batch, events, outcomes, time_bins, H, W]
        """
        batch_size = patient_features.shape[0]
        
        # Generate patient-specific modulation factors
        modulation = self.condition_net(patient_features)  # [batch, events*outcomes*time]
        modulation = modulation.view(batch_size, self.n_events, self.n_outcomes, self.n_time_bins)
        
        # Generate temporal progression factors
        temporal_factors = self.temporal_net(patient_features)  # [batch, time_bins]
        
        # Start with base anchors
        dynamic_anchors = self.base_anchors.unsqueeze(0)  # [1, events, outcomes, time, H, W]
        
        # Apply patient-specific modulation
        modulation_expanded = modulation.unsqueeze(-1).unsqueeze(-1)  # [batch, events, outcomes, time, 1, 1]
        dynamic_anchors = dynamic_anchors * (1.0 + self.modulation_strength * modulation_expanded)
        
        # Apply temporal progression factors
        temporal_expanded = temporal_factors.view(batch_size, 1, 1, -1, 1, 1)  # [batch, 1, 1, time, 1, 1]
        dynamic_anchors = dynamic_anchors * temporal_expanded
        
        return dynamic_anchors


class TransportPlanRegularizer(nn.Module):
    """
    Regularize transport plans for optimal structure.
    
    Key insight: Plans should be neither too uniform (collapsed)
    nor too chaotic (noisy). Enforce intermediate entropy and structure.
    """
    
    def __init__(self,
                 target_entropy: float = 2.0,
                 temporal_smooth_weight: float = 0.1,
                 concentration_weight: float = 0.1):
        super().__init__()
        self.target_entropy = target_entropy
        self.temporal_smooth_weight = temporal_smooth_weight
        self.concentration_weight = concentration_weight
    
    def forward(self, 
                transport_plan: torch.Tensor,
                return_details: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses for transport plans.
        
        Args:
            transport_plan: [batch, events, time, ...] Transport assignments
            return_details: Whether to return detailed loss breakdown
        
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Flatten spatial dimensions for analysis
        plan_shape = transport_plan.shape
        plan_flat = transport_plan.flatten(-2)  # [batch, events, time, spatial]
        
        # 1. Entropy regularization: moderate entropy
        plan_normalized = F.softmax(plan_flat, dim=-1)
        entropy = -(plan_normalized * torch.log(plan_normalized + 1e-10)).sum(dim=-1)
        entropy_mean = entropy.mean()
        
        entropy_loss = (entropy_mean - self.target_entropy) ** 2
        losses['entropy'] = entropy_loss
        
        # 2. Temporal consistency: smooth evolution
        if plan_shape[2] > 1:  # If multiple time bins
            temporal_diff = (transport_plan[:, :, 1:] - transport_plan[:, :, :-1]).pow(2).mean()
            
            # We want some change but not too chaotic
            # Penalize if change is too small (<0.05) or too large (>0.5)
            temporal_loss = F.relu(0.05 - temporal_diff) + F.relu(temporal_diff - 0.5)
            losses['temporal_consistency'] = temporal_loss * self.temporal_smooth_weight
        
        # 3. Concentration: plans should focus but not collapse
        squared_plan = plan_normalized ** 2
        effective_support = 1.0 / (squared_plan.sum(dim=-1).mean() + 1e-10)
        
        # Target: use 15-25% of available support
        target_support = plan_flat.shape[-1] * 0.2
        concentration_loss = ((effective_support - target_support) / target_support) ** 2
        losses['concentration'] = concentration_loss * self.concentration_weight
        
        # 4. Anchor utilization: all anchors should contribute
        anchor_max_usage = plan_flat.max(dim=0)[0]  # Max usage per anchor across batch
        unused_ratio = (anchor_max_usage < 0.01).float().mean()
        losses['anchor_utilization'] = unused_ratio
        
        if return_details:
            losses['_entropy_value'] = entropy_mean
            losses['_effective_support'] = effective_support
            losses['_unused_ratio'] = unused_ratio
        
        return losses


class TransportCurriculumScheduler:
    """
    Progressive training schedule for transport learning.
    
    Key insight: Start with simple transport, gradually increase complexity.
    Explicitly focus on temporal differentiation in dedicated stage.
    """
    
    def __init__(self, total_epochs: int = 50):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Define curriculum stages
        self.stages = [
            {
                'name': 'warmup',
                'epochs': 10,
                'focus': 'basic_anchors',
                'description': 'Learn basic anchor structure and survival prediction'
            },
            {
                'name': 'temporal',
                'epochs': 15,
                'focus': 'temporal_differentiation',
                'description': 'Force temporal contrast in anchors'
            },
            {
                'name': 'spatial',
                'epochs': 15,
                'focus': 'spatial_structure',
                'description': 'Refine spatial transport structure'
            },
            {
                'name': 'joint',
                'epochs': 10,
                'focus': 'full_transport',
                'description': 'Joint optimization of all components'
            }
        ]
    
    def get_current_stage(self, epoch: int) -> Dict:
        """Get current curriculum stage information"""
        cumsum = 0
        for stage in self.stages:
            cumsum += stage['epochs']
            if epoch < cumsum:
                return stage
        return self.stages[-1]
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Return loss weights for current curriculum stage.
        
        Args:
            epoch: Current training epoch
        
        Returns:
            Dictionary of loss component weights
        """
        self.current_epoch = epoch
        stage = self.get_current_stage(epoch)
        
        return self._get_stage_weights(stage['focus'], epoch)
    
    def _get_stage_weights(self, focus: str, epoch: int) -> Dict[str, float]:
        """Get weights for different loss components based on stage"""
        weights = {
            'survival_loss': 1.0,
            'ranking_loss': 0.0,
            'temporal_contrast': 0.0,
            'transport_reg_entropy': 0.0,
            'transport_reg_temporal': 0.0,
            'transport_reg_concentration': 0.0,
            'transport_reg_utilization': 0.0,
            'anchor_lr_multiplier': 1.0
        }
        
        if focus == 'basic_anchors':
            # Stage 1: Focus on survival and basic structure
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.1,
                'temporal_contrast': 0.0,
                'transport_reg_entropy': 0.05,
                'transport_reg_temporal': 0.0,
                'transport_reg_concentration': 0.0,
                'transport_reg_utilization': 0.1,
                'anchor_lr_multiplier': 5.0  # HIGH learning rate for anchors
            })
        
        elif focus == 'temporal_differentiation':
            # Stage 2: CRITICAL - force temporal differentiation
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.3,
                'temporal_contrast': 3.0,  # VERY HIGH weight
                'transport_reg_entropy': 0.1,
                'transport_reg_temporal': 0.5,  # Encourage smooth transitions
                'transport_reg_concentration': 0.05,
                'transport_reg_utilization': 0.2,
                'anchor_lr_multiplier': 3.0  # Still high LR for anchors
            })
        
        elif focus == 'spatial_structure':
            # Stage 3: Refine spatial structure while maintaining temporal
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 0.5,
                'temporal_contrast': 1.0,  # Maintain temporal contrast
                'transport_reg_entropy': 0.2,
                'transport_reg_temporal': 0.3,
                'transport_reg_concentration': 0.2,  # Focus on concentration
                'transport_reg_utilization': 0.3,
                'anchor_lr_multiplier': 1.0  # Normal LR
            })
        
        elif focus == 'full_transport':
            # Stage 4: Balanced joint optimization
            weights.update({
                'survival_loss': 1.0,
                'ranking_loss': 1.0,
                'temporal_contrast': 0.5,
                'transport_reg_entropy': 0.3,
                'transport_reg_temporal': 0.2,
                'transport_reg_concentration': 0.3,
                'transport_reg_utilization': 0.2,
                'anchor_lr_multiplier': 0.5  # Lower LR for fine-tuning
            })
        
        return weights
    
    def get_anchor_lr(self, base_lr: float, epoch: int) -> float:
        """Get learning rate specifically for anchor parameters"""
        weights = self.get_loss_weights(epoch)
        return base_lr * weights['anchor_lr_multiplier']
    
    def print_stage_info(self, epoch: int):
        """Print current stage information"""
        stage = self.get_current_stage(epoch)
        weights = self.get_loss_weights(epoch)
        
        print(f"\n{'='*60}")
        print(f"Curriculum Stage: {stage['name'].upper()} (Epoch {epoch})")
        print(f"Focus: {stage['description']}")
        print(f"Loss Weights:")
        for k, v in weights.items():
            if v > 0 and 'multiplier' not in k:
                print(f"  {k}: {v:.3f}")
        print(f"Anchor LR Multiplier: {weights['anchor_lr_multiplier']:.1f}x")
        print(f"{'='*60}\n")


# ============================================================================
# Utility Functions
# ============================================================================

def analyze_anchor_temporal_variation(risk_anchor_costs: torch.Tensor) -> Dict[str, float]:
    """
    Analyze temporal variation in risk anchors.
    
    Args:
        risk_anchor_costs: [events, outcomes, time_bins, H, W]
    
    Returns:
        Dictionary of variation statistics
    """
    stats = {}
    
    n_events, n_outcomes, n_time_bins = risk_anchor_costs.shape[:3]
    
    temporal_variations = []
    spatial_variations = []
    
    for event_idx in range(n_events):
        for outcome_idx in range(n_outcomes):
            costs = risk_anchor_costs[event_idx, outcome_idx]
            
            # Temporal variation
            time_means = costs.flatten(1).mean(dim=1)
            temporal_std = time_means.std().item()
            temporal_variations.append(temporal_std)
            
            # Spatial variation
            spatial_std = costs.std(dim=[1, 2]).mean().item()
            spatial_variations.append(spatial_std)
    
    stats['mean_temporal_variation'] = sum(temporal_variations) / len(temporal_variations)
    stats['min_temporal_variation'] = min(temporal_variations)
    stats['max_temporal_variation'] = max(temporal_variations)
    stats['mean_spatial_variation'] = sum(spatial_variations) / len(spatial_variations)
    
    return stats


def visualize_transport_anchors(risk_anchor_costs: torch.Tensor,
                                 save_path: Optional[str] = None):
    """
    Visualize risk anchor costs structure.
    
    Args:
        risk_anchor_costs: [events, outcomes, time_bins, H, W]
        save_path: Path to save visualization (if None, returns figure)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib required for visualization")
        return None
    
    costs_np = risk_anchor_costs.detach().cpu().numpy()
    n_events, n_outcomes, n_time_bins, H, W = costs_np.shape
    
    fig, axes = plt.subplots(n_events, n_outcomes * n_time_bins,
                            figsize=(4 * n_outcomes * n_time_bins, 4 * n_events))
    
    if n_events == 1:
        axes = axes.reshape(1, -1)
    
    for event_idx in range(n_events):
        for outcome_idx in range(n_outcomes):
            for time_idx in range(n_time_bins):
                col_idx = outcome_idx * n_time_bins + time_idx
                ax = axes[event_idx, col_idx]
                
                cost_map = costs_np[event_idx, outcome_idx, time_idx]
                im = ax.imshow(cost_map, cmap='viridis')
                ax.set_title(f'E{event_idx} O{outcome_idx} T{time_idx}')
                ax.axis('off')
                plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        return fig
