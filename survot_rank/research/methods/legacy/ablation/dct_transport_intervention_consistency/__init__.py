"""DCT v3.8 Transport-Intervention Consistency [ARCHIVED - ABLATION PARENT].

LEGACY: This is the ablation parent class for structural losses.
NOT for paper use. Use DCTV310DirectionalRegularizedTransport instead.

Contains: direction loss, dose loss, reconfiguration loss (all disabled in paper method).
"""

from .model import DCTTransportInterventionConsistency

__all__ = ["DCTTransportInterventionConsistency"]
