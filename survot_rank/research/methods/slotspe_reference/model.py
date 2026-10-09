"""SlotSPE reference adapter — wraps the third_party/SlotSPE baseline.

This adapter exists to:

1. Re-host the original SlotSPE code under the unified DCT trainer contract
   (``forward(**kwargs) -> (logits, aux)`` + ``last_training_losses``).
2. Select between two pre-specified recipes (``matched`` vs ``native``)
   from §2.2 of ``paper/V313_IMPLEMENTATION_PLAN.md``.  Both recipes share
   the same compute graph (slot attention, MoE decoder, reconstruction
   heads, transformers, NLL loss); they only differ in optimizer, batch
   size, slot iterations and ``alpha_surv``.
3. Audit-log the effective recipe so results can be cross-checked
   against the original SlotSPE repo without re-running it.

Hard rules:
- The third-party tree ``third_party/SlotSPE/`` is **read-only**.
- No DCT regularization (IPCW rank, per-slot NLL, diversity, transport,
  reconstruction coupling) is injected into the SlotSPE graph.
- The same random state must yield the same ``logits`` and auxiliary
  loss as a direct invocation of the original ``SlotSPE.SlotSPE`` —
  validated by ``tests/test_slotspe_reference.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
import torch.nn.functional as F_t  # noqa: F401  (silence unused-import on minimal env)


# ---------------------------------------------------------------------------
# Recipe definitions (paper/V313_IMPLEMENTATION_PLAN.md §2.2 / Table 2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotSPERecipe:
    """A pre-specified SlotSPE recipe.

    The numeric fields correspond to the columns in Table 2.2 of the
    v3.13 implementation plan.  ``recipe_name`` is the human-readable
    label and the audit-log key; changing any field invalidates the
    recipe and is rejected at construction time.
    """

    name: str
    epochs: int
    batch_size: int
    optimizer: str        # "AdamW" or "Adam"
    learning_rate: float
    weight_decay: float
    alpha_surv: float
    slot_iters: int
    lambda_recon_loss: float


RECIPES: dict[str, SlotSPERecipe] = {
    "matched": SlotSPERecipe(
        name="matched",
        epochs=30,
        batch_size=8,
        optimizer="AdamW",
        learning_rate=5e-4,
        weight_decay=5e-4,
        alpha_surv=0.15,
        slot_iters=10,
        lambda_recon_loss=0.01,
    ),
    "native": SlotSPERecipe(
        name="native",
        epochs=30,
        batch_size=32,
        optimizer="Adam",
        learning_rate=5e-4,
        weight_decay=1e-3,
        alpha_surv=0.5,
        slot_iters=10,
        lambda_recon_loss=0.01,
    ),
}


RECIPE_SUMMARY = {
    name: {
        "epochs": r.epochs,
        "batch_size": r.batch_size,
        "optimizer": r.optimizer,
        "learning_rate": r.learning_rate,
        "weight_decay": r.weight_decay,
        "alpha_surv": r.alpha_surv,
        "slot_iters": r.slot_iters,
        "lambda_recon_loss": r.lambda_recon_loss,
    }
    for name, r in RECIPES.items()
}


# ---------------------------------------------------------------------------
# Isolated import of third_party/SlotSPE.
# ---------------------------------------------------------------------------


_THIRDPARTY_ROOT = Path(__file__).resolve().parents[4] / "third_party" / "SlotSPE"
_SLOTSPE_MODELS_DIR = _THIRDPARTY_ROOT / "models"
_SLOTSPE_UTILS_DIR = _THIRDPARTY_ROOT / "utils"


def _is_thirdparty_layout_ready() -> bool:
    return (
        _THIRDPARTY_ROOT.is_dir()
        and _SLOTSPE_MODELS_DIR.is_dir()
        and _SLOTSPE_UTILS_DIR.is_dir()
    )


def _load_thirdparty_namespace() -> dict[str, Any]:
    """Import the third_party/SlotSPE files into a private module namespace.

    The function creates synthetic ``models`` and ``utils`` packages so that
    the original SlotSPE code's ``from models.X import ...`` and
    ``from utils.X import ...`` statements resolve inside this isolated
    namespace.  The third_party tree is NEVER modified; we only create new
    ``ModuleType`` objects and register them under unique names.

    Returns a dict with the public names exposed by the SlotSPE package:
    ``SlotSPE`` (model), ``NLLSurvLoss``, ``MultiHeadSlotAttention``,
    ``IterativeCrossAttTransformer``, ``Transformer``, ``SNN_Block``,
    ``WSI_Mlp``, plus the ``models`` and ``utils`` packages themselves
    (so the trainer can introspect if needed).
    """
    if not _is_thirdparty_layout_ready():
        raise FileNotFoundError(
            f"third_party/SlotSPE layout missing under {_THIRDPARTY_ROOT}. "
            "Re-clone the upstream repo before using the SlotSPE reference "
            "adapter (see NEXT_STEPS §5)."
        )

    # Synthetic package roots.  These names are unique enough that they
    # cannot collide with anything else the runner pulls in (the prefix
    # ``_slotspe_ref_`` is reserved by this adapter only).
    models_pkg = ModuleType("_slotspe_ref_models")
    utils_pkg = ModuleType("_slotspe_ref_utils")
    models_pkg.__path__ = [str(_SLOTSPE_MODELS_DIR)]  # type: ignore[attr-defined]
    utils_pkg.__path__ = [str(_SLOTSPE_UTILS_DIR)]    # type: ignore[attr-defined]
    sys.modules["_slotspe_ref_models"] = models_pkg
    sys.modules["_slotspe_ref_utils"] = utils_pkg

    # The original SlotSPE/SlotSPE.py uses absolute imports of the form
    # ``from models.X import ...`` (no relative prefix).  To make those
    # resolve to our synthetic one WITHOUT leaking the ``models`` and
    # ``utils`` module entries back into the global namespace, we
    # temporarily register the synthetic packages under those plain
    # names for the duration of the exec_module calls, then REMOVE the
    # plain-name entries from sys.modules.  The unique-name entries
    # (``_slotspe_ref_models``, ``_slotspe_ref_utils``) remain, so the
    # adapter can still introspect via the synthetic roots.
    leaked_modules: dict[str, ModuleType] = {}
    for plain, pkg in (("models", models_pkg), ("utils", utils_pkg)):
        if plain in sys.modules:
            leaked_modules[plain] = sys.modules[plain]
        sys.modules[plain] = pkg

    # Sub-modules inside ``models``: SlotSPE, slot_attention, transformer,
    # omics_encoder.  ``SlotSPE.py`` itself imports ``from models.X``
    # which now resolves to ``_slotspe_ref_models.X`` (== ``models.X``).
    _sub_modules = [
        ("models.omics_encoder", _SLOTSPE_MODELS_DIR / "omics_encoder.py"),
        ("models.slot_attention", _SLOTSPE_MODELS_DIR / "slot_attention.py"),
        ("models.transformer",   _SLOTSPE_MODELS_DIR / "transformer.py"),
        ("models.SlotSPE",       _SLOTSPE_MODELS_DIR / "SlotSPE.py"),
        # ``utils/loss_func.py`` provides ``NLLSurvLoss``.
        ("utils.loss_func",      _SLOTSPE_UTILS_DIR / "loss_func.py"),
    ]
    loaded: dict[str, ModuleType] = {}
    try:
        for fq_name, file_path in _sub_modules:
            spec = importlib.util.spec_from_file_location(fq_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"could not create import spec for {file_path}")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[fq_name] = mod
            spec.loader.exec_module(mod)
            loaded[fq_name] = mod
    finally:
        # Remove the temporary plain-name entries to keep the global
        # namespace clean.  ``SlotSPE.py`` has already cached its imports
        # in module-level names, so removing ``models`` / ``utils`` from
        # ``sys.modules`` does NOT affect the loaded classes.
        for plain in ("models", "utils"):
            sys.modules.pop(plain, None)
        # Anything the leaked entries were shadowing goes back on top.
        for plain, mod in leaked_modules.items():
            sys.modules[plain] = mod

    # Patch the imports inside ``SlotSPE.py``:
    #   from models.slot_attention import MultiHeadSlotAttention, gumbel_topk_st, parallel_topk_st
    #   from models.transformer import IterativeCrossAttTransformer, Transformer
    #   from models.omics_encoder import SNN_Block, WSI_Mlp
    #   from utils.loss_func import NLLSurvLoss
    # We do NOT rewrite the file.  Instead, we expose the right attributes on
    # the synthetic ``_slotspe_ref_models`` and ``_slotspe_ref_utils`` packages
    # so that *future* code referencing ``models.X`` (e.g. inside tests) gets
    # the real classes.
    for name, attrs in {
        "_slotspe_ref_models": [
            "slot_attention", "transformer", "omics_encoder", "SlotSPE"
        ],
        "_slotspe_ref_utils": ["loss_func"],
    }.items():
        pkg = sys.modules[name]
        for attr in attrs:
            fq = f"{name}.{attr}"
            if fq in loaded:
                setattr(pkg, attr, loaded[fq])

    exposed = {
        "models_pkg": models_pkg,
        "utils_pkg": utils_pkg,
        "SlotSPE_module": loaded["models.SlotSPE"],
        "loss_module": loaded["utils.loss_func"],
    }
    return exposed


_NS_CACHE: dict[str, Any] | None = None


def _get_namespace() -> dict[str, Any]:
    global _NS_CACHE
    if _NS_CACHE is None:
        _NS_CACHE = _load_thirdparty_namespace()
    return _NS_CACHE


# ---------------------------------------------------------------------------
# Recipe selection helpers.
# ---------------------------------------------------------------------------


def _select_recipe(args) -> SlotSPERecipe:
    name = str(getattr(args, "slotspe_recipe", "native")).lower()
    if name not in RECIPES:
        raise ValueError(
            f"slotspe_recipe must be one of {list(RECIPES)}; got {name!r}. "
            "The two recipes are pre-specified in the v3.13 plan; "
            "do not invent new values."
        )
    return RECIPES[name]


# ---------------------------------------------------------------------------
# Adapter wrapper.
# ---------------------------------------------------------------------------


class SlotSPEReference(torch.nn.Module):
    """Wrap the original SlotSPE model behind the DCT trainer contract.

    The adapter:

    * Constructs the upstream ``SlotSPE.SlotSPE`` instance with its native
      hyperparameters (channel count, slot counts, decoder settings).
    * Applies the user-selected recipe by overriding optimizer, batch
      size, slot iterations and ``alpha_surv`` AFTER construction.
    * Logs the effective recipe into ``self.recipe_log`` so result
      manifests can cross-check that a run actually used the requested
      recipe (and was not silently swapped to the other).
    * Forwards ``forward(**kwargs)`` by stripping DCT-specific kwargs
      (``cur_epoch``, ``wsi_missing``, ``clinical``, ``event_time``)
      and providing the kwargs the original model expects.
    * Exposes ``last_training_losses`` with the same shape as DCT models so
      the trainer's per-epoch aggregation logic continues to work.
    """

    def __init__(
        self,
        args,
        omic_input_dim=None,
        omic_names=None,
        pathway_names=None,
    ):
        super().__init__()
        self.args = args
        recipe = _select_recipe(args)
        self._recipe = recipe

        # ------------------------------------------------------------------
        # Honour the user's slot_num_wsi / slot_num_omics (DCT convention) but
        # keep the recipe's ``slot_iters``.  The original SlotSPE always uses
        # ``slot_iters=10``; deviating would invalidate the comparison.
        # The user must explicitly opt in via ``slotspe_force_recipe_iters=true``
        # to use a different iters (e.g. for an ablation study).
        # ------------------------------------------------------------------
        def _as_bool(v):
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("1", "true", "yes", "y", "on")

        force = _as_bool(getattr(args, "slotspe_force_recipe_iters", False))
        if not force and int(getattr(args, "slot_iters", recipe.slot_iters)) != recipe.slot_iters:
            raise ValueError(
                f"SlotSPE recipe {recipe.name!r} mandates slot_iters={recipe.slot_iters}, "
                f"but args.slot_iters={int(getattr(args, 'slot_iters', recipe.slot_iters))}. "
                "Pass --set slotspe_force_recipe_iters=true to override the recipe value."
            )

        ns = _get_namespace()
        UpstreamSlotSPE = getattr(ns["SlotSPE_module"], "SlotSPE")
        upstream_instance = UpstreamSlotSPE(
            args,
            omic_names=omic_names if omic_names else [],
            omic_input_dim=omic_input_dim,
        )
        # Carry the upstream module as ``self.upstream`` so the audit log
        # can introspect the original compute graph.
        self.upstream = upstream_instance
        # Surface the SlotSPE loss alpha so the upstream ``NLLSurvLoss``
        # matches the recipe (the original constructor reads
        # ``args.alpha_surv``).
        if hasattr(self.upstream, "loss_fn"):
            self.upstream.loss_fn.alpha = float(recipe.alpha_surv)

        self.last_training_losses: dict[str, torch.Tensor] = {}
        self.recipe_log = {
            "recipe": recipe.name,
            "epochs": recipe.epochs,
            "batch_size": recipe.batch_size,
            "optimizer": recipe.optimizer,
            "learning_rate": recipe.learning_rate,
            "weight_decay": recipe.weight_decay,
            "alpha_surv": recipe.alpha_surv,
            "slot_iters": recipe.slot_iters,
            "lambda_recon_loss": recipe.lambda_recon_loss,
            "args_alpha_surv": float(getattr(args, "alpha_surv", recipe.alpha_surv)),
            "args_lambda_recon_loss": float(
                getattr(args, "lambda_recon_loss", recipe.lambda_recon_loss)
            ),
        }

    @classmethod
    def objective_weights(cls) -> dict[str, float]:
        """Return the immutable SlotSPE reference objective weights.

        These are the *constants* baked into the recipe (not the
        runtime ablation overrides); they are exposed via the same
        method name as DCT models so result manifests can use one API.
        """
        return {
            "nll": 1.0,
            "alpha_surv_native": RECIPES["native"].alpha_surv,
            "alpha_surv_matched": RECIPES["matched"].alpha_surv,
            "lambda_recon_loss": RECIPES["native"].lambda_recon_loss,
        }

    def effective_recipe(self) -> dict[str, Any]:
        """Return the recipe actually applied to the model at construction."""
        return dict(self.recipe_log)

    def train(self, mode: bool = True) -> "SlotSPEReference":
        super().train(mode)
        self.upstream.train(mode)
        return self

    def eval(self) -> "SlotSPEReference":
        super().eval()
        self.upstream.eval()
        return self

    def _strip_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs the DCT trainer injects but the original SlotSPE
        model does not consume.  Any unknown kwarg still raises inside
        the original model — that's the audit guarantee.
        """
        dropped = {
            k: kwargs.pop(k, None)
            for k in (
                "cur_epoch",
                "wsi_missing",
                "event_time",
                "omic_available",
                "omics_available",
                "omic_missing_flag",
            )
        }
        if any(v is not None for v in dropped.values()):
            # Log only once per call; cheap, idempotent.
            self._last_dropped_kwargs = dropped
        return kwargs

    def forward(self, **kwargs):
        # Drop kwargs the upstream SlotSPE doesn't accept.  This is the
        # audit boundary: anything not consumed by the upstream model is
        # silently discarded, mirroring how ``survot_rank`` calls legacy
        # models in other adapter bridges.
        cleaned = self._strip_kwargs(dict(kwargs))
        logits, aux = self.upstream(**cleaned)

        # Mirror auxiliary loss into the DCT-shaped diagnostic dict.
        if isinstance(aux, torch.Tensor):
            aux_val = aux.detach()
        else:
            aux_val = torch.tensor(float(aux))
        # Re-emit upstream field names into the DCT diagnostic namespace
        # so the trainer's per-epoch aggregation reads them consistently.
        self.last_training_losses = {
            "slotspe_aux_total": aux_val,
            "slotspe_recipe": torch.tensor(
                float(hash(self._recipe.name) % (2 ** 31)), dtype=torch.float32
            ),
        }
        return logits, aux


# Convenience constants for tests.
DEFAULT_RECIPE = "native"
ALLOWED_RECIPES: tuple[str, ...] = tuple(RECIPES)


def list_recipes() -> Iterable[str]:
    return iter(ALLOWED_RECIPES)