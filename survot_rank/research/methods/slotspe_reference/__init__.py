"""SlotSPE reference adapter (read-only wrapper around third_party/SlotSPE).

This directory exists ONLY to expose the SlotSPE baseline under the unified
DCT trainer contract.  No code here mutates ``third_party/SlotSPE/``; all
original files remain read-only and untouched.

The adapter is responsible for:

* Loading ``third_party/SlotSPE/models/`` in an isolated importlib namespace
  so the SlotSPE implementation can never collide with DCT's runtime
  helpers (e.g. an ``omcis_encoder`` import shadowed by DCT utilities).
* Mapping DCT-style ``args`` to the SlotSPE model constructor.
* Selecting one of the two pre-specified recipes (``matched`` vs ``native``)
  from §2.2 of ``paper/V313_IMPLEMENTATION_PLAN.md``.  Both recipes share
  the same SlotSPE compute graph; only the optimizer, batch size, slot
  iterations, and ``alpha_surv`` differ.
* Providing ``forward(**kwargs) -> (logits, auxiliary)`` and the
  ``last_training_losses`` diagnostic dict that the DCT trainer consumes.
"""

from .model import RECIPE_SUMMARY, SlotSPEReference

__all__ = ["SlotSPEReference", "RECIPE_SUMMARY"]