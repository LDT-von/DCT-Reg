"""Compatibility import for the archived intervention-consistency model.

The repository previously stored this path as an absolute Linux symlink, which
cannot be imported from a clean Windows checkout.  Keep the public import path
stable while the implementation remains in the legacy ablation package.
"""

from survot_rank.research.methods.legacy.ablation.dct_transport_intervention_consistency.model import (
    DCTTransportInterventionConsistency,
)

__all__ = ["DCTTransportInterventionConsistency"]
