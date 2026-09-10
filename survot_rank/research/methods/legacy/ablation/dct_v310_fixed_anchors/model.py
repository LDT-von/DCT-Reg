"""DCT v3.10 with fixed pre-computed anchors.

Purpose: Validate that the directional transport mechanism works when provided
with high-quality anchors, proving the core idea is sound even if current
Slot Attention anchor extraction needs improvement.
"""

from __future__ import annotations

import torch
import pickle
from pathlib import Path

from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import (
    DCTV310DirectionalRegularizedTransport,
)


class DCTV310FixedAnchors(DCTV310DirectionalRegularizedTransport):
    """DCT v3.10 with frozen pre-computed risk anchors.
    
    This variant does NOT learn anchors via Slot Attention. Instead, it uses
    fixed anchors loaded from a pickle file. This isolates the directional
    transport mechanism from anchor quality issues.
    
    Usage:
        args.fixed_anchors_path = "results/ideal_anchors/blca_fold0.pkl"
    """
    
    def __init__(self, args, omic_input_dim=None, omic_names=None, pathway_names=None):
        # Get the fixed anchors path before parent init
        self.fixed_anchors_path = getattr(args, "fixed_anchors_path", None)
        if not self.fixed_anchors_path:
            raise ValueError(
                "DCTV310FixedAnchors requires args.fixed_anchors_path to be set"
            )
        
        # Initialize parent (this will create risk_anchor_costs buffer)
        super().__init__(args, omic_input_dim, omic_names, pathway_names)
        
        # Load and set the fixed anchors
        self._load_fixed_anchors()
        
        print(f"[DCTV310FixedAnchors] Loaded fixed anchors from {self.fixed_anchors_path}")
        print(f"  Anchor shape: {self.risk_anchor_costs.shape}")
        print(f"  Anchors are FROZEN (requires_grad=False)")
    
    def _load_fixed_anchors(self):
        """Load pre-computed anchors and freeze them."""
        anchor_path = Path(self.fixed_anchors_path)
        if not anchor_path.exists():
            raise FileNotFoundError(f"Fixed anchors file not found: {anchor_path}")
        
        with open(anchor_path, 'rb') as f:
            anchors_data = pickle.load(f)
        
        # Extract the risk_anchor_costs tensor
        if 'risk_anchor_costs' not in anchors_data:
            raise KeyError(
                f"Anchors file must contain 'risk_anchor_costs' key. "
                f"Found keys: {list(anchors_data.keys())}"
            )
        
        fixed_costs = torch.from_numpy(anchors_data['risk_anchor_costs'])
        
        # Verify shape matches the model's buffer
        if fixed_costs.shape != self.risk_anchor_costs.shape:
            raise ValueError(
                f"Fixed anchors shape mismatch: "
                f"expected {self.risk_anchor_costs.shape}, "
                f"got {fixed_costs.shape}"
            )
        
        # Copy fixed anchors into the buffer
        self.risk_anchor_costs.copy_(fixed_costs)
        
        # Mark all anchors as seen (so they are always used)
        self.risk_anchor_seen.fill_(True)
        
        # Make sure these buffers are truly frozen by converting to non-trainable buffers
        # (They already are buffers, but let's be explicit)
        self.risk_anchor_costs.requires_grad_(False)
        self.risk_anchor_seen.requires_grad_(False)
    
    def _update_risk_anchors(self, costs, low_weights, high_weights):
        """Override: Do NOT update anchors - they are fixed!"""
        # In the parent class, this method updates anchors based on training data.
        # We disable it completely to keep anchors frozen.
        pass
    
    def train(self, mode=True):
        """Override: Ensure anchors remain frozen even in train mode."""
        super().train(mode)
        # Redundant safety: ensure anchors never accidentally get gradients
        self.risk_anchor_costs.requires_grad_(False)
        self.risk_anchor_seen.requires_grad_(False)
        return self


# CONTINUED BELOW
