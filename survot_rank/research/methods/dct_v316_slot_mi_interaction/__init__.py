# DCT v3.16 Slot-based MI Decomposition
#
# This package contains the standalone model components.
# To use with the full framework, import via:
#   from survot_rank.research.methods.dct_v316_slot_mi_interaction.dct_v316_model import DCTV316SlotMIDecomposition
#
# To use standalone components (no framework dependency):
#   from survot_rank.research.methods.dct_v316_slot_mi_interaction.model import (
#       MultiHeadSlotAttention,
#       MoESlotDecoder,
#       IterativeCrossAttention,
#       InteractionProfiler,
#       SlotBasedMIDecompositionBlock,
#   )

# Core components - available without framework dependency
from survot_rank.research.methods.dct_v316_slot_mi_interaction.model import (
    MultiHeadSlotAttention,
    MoESlotDecoder,
    IterativeCrossAttention,
    IterativeCrossAttTransformer,
    Transformer,
    InfoNCECritic,
    InteractionProfiler,
    SlotBasedMIDecompositionBlock,
    gumbel_topk_st,
    parallel_topk_st,
)

__all__ = [
    "MultiHeadSlotAttention",
    "MoESlotDecoder",
    "IterativeCrossAttention",
    "IterativeCrossAttTransformer",
    "Transformer",
    "InfoNCECritic",
    "InteractionProfiler",
    "SlotBasedMIDecompositionBlock",
    "gumbel_topk_st",
    "parallel_topk_st",
]
