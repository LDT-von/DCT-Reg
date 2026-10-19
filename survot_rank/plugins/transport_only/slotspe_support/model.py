"""Load the original SlotSPE from an explicitly bound, read-only source tree.

No repair classes or attribution arms are included. Only state dictionaries
are serialized. Importing this module does not import torch or change paths.
The import-isolation helpers are reused from the previously checked delivery.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace


MODEL_SOURCE_FILES = (
    "models/SlotSPE.py",
    "models/slot_attention.py",
    "models/transformer.py",
    "models/omics_encoder.py",
    "utils/loss_func.py",
)
_SOURCE = None
_LOAD_LOCK = threading.RLock()
ATTRIBUTION_ARMS = ("A", "B", "C")
PROOF_ARMS = ("original", "eval_fix", "readout_fix", *ATTRIBUTION_ARMS)


def make_args(omic_sizes, encoding_dim=1024):
    """Keep the original diagnostic architecture and loss settings."""
    return SimpleNamespace(
        omic_sizes=list(omic_sizes), n_classes=4, encoding_dim=encoding_dim,
        wsi_projection_dim=256, rna_format="Pathways", slot_num_wsi=8,
        slot_num_omics=8, slot_iters=10, temperature=0.01, topk_ratio=0.25,
        top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=0.5,
        lambda_recon_loss=0.01,
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package(name, path):
    module = ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    return module


def _inside(path, root):
    try:
        Path(path).resolve().relative_to(root)
    except (ValueError, TypeError, OSError):
        return False
    return True


def _check_cached_source(root):
    """Refuse ambiguous absolute imports before exposing temporary aliases."""
    for name, module in tuple(sys.modules.items()):
        if not (name in ("models", "utils") or
                name.startswith(("models.", "utils."))):
            continue
        if module is None:
            raise RuntimeError(f"Conflicting import cache entry {name!r} is None")
        file = getattr(module, "__file__", None)
        paths = list(getattr(module, "__path__", ()))
        locations = [file] if file else paths
        if not locations or not all(_inside(p, root) for p in locations):
            raise RuntimeError(
                f"Conflicting cached module {name!r} from {file or paths!r}; "
                f"SlotSPE source is {str(root)!r}. Use a fresh process."
            )


def load_source(source_root):
    """Load one explicit source tree without lasting models/utils aliases.

    A process may use only one resolved source root.  Private module names keep
    source class identities available after temporary absolute-import aliases
    have been removed.  Existing foreign ``models`` or ``utils`` imports cause
    a failure rather than silently changing the baseline.
    """
    global _SOURCE
    root = Path(source_root).expanduser().resolve()
    with _LOAD_LOCK:
        if _SOURCE is not None:
            if root != _SOURCE.source_root:
                raise RuntimeError(
                    f"SlotSPE source is already locked to {_SOURCE.source_root}; "
                    f"cannot switch to {root} in the same process."
                )
            current = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}
            if current != _SOURCE.source_hashes:
                raise RuntimeError("SlotSPE model source changed after loading")
            _check_cached_source(root)
            return _SOURCE
        missing = [p for p in MODEL_SOURCE_FILES if not (root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete SlotSPE source {root}: {missing}")
        _check_cached_source(root)
        before_hashes = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}

        # Import dependencies before temporarily exposing generic package names.
        # Nothing here imports, installs, or invokes the transport plugin.
        import torch  # noqa: F401
        import einops  # noqa: F401

        prefix = "_slotspe_proof_source_" + hashlib.sha256(
            str(root).encode("utf-8")
        ).hexdigest()[:16]
        private_names = []
        saved_aliases = {
            name: module for name, module in tuple(sys.modules.items())
            if name in ("models", "utils") or
            name.startswith(("models.", "utils."))
        }
        previous_bytecode = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            private_root = _package(prefix, root)
            sys.modules[prefix] = private_root
            private_names.append(prefix)
            for name in ("models", "utils"):
                package = _package(f"{prefix}.{name}", root / name)
                sys.modules[package.__name__] = package
                private_names.append(package.__name__)
                setattr(private_root, name, package)
                sys.modules[name] = package

            modules = {}
            # Source SlotSPE.py imports these siblings by absolute names.
            for relative in (
                "utils/loss_func.py", "models/slot_attention.py",
                "models/transformer.py", "models/omics_encoder.py",
                "models/SlotSPE.py",
            ):
                alias = relative[:-3].replace("/", ".")
                name = f"{prefix}.{alias}"
                spec = importlib.util.spec_from_file_location(name, root / relative)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load SlotSPE module {root / relative}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                private_names.append(name)
                sys.modules[alias] = module
                package_name, child = alias.rsplit(".", 1)
                setattr(sys.modules[package_name], child, module)
                spec.loader.exec_module(module)
                if Path(module.__file__).resolve() != (root / relative).resolve():
                    raise RuntimeError(f"Unexpected loaded source for {alias}")
                modules[alias] = module
            after_hashes = {p: _digest(root / p) for p in MODEL_SOURCE_FILES}
            if after_hashes != before_hashes:
                raise RuntimeError("SlotSPE source changed while loading")
            slots = modules["models.slot_attention"]
            model = modules["models.SlotSPE"]
            _SOURCE = SimpleNamespace(
                source_root=root, source_hashes=before_hashes,
                module_paths={key: str(Path(value.__file__).resolve())
                              for key, value in modules.items()},
                SlotSPE=model.SlotSPE, MoESlotDecoder=model.MoESlotDecoder,
                MultiHeadSlotAttention=slots.MultiHeadSlotAttention,
                gumbel_topk_st=slots.gumbel_topk_st,
                parallel_topk_st=slots.parallel_topk_st,
                modules=modules,
            )
        except BaseException:
            for name in private_names:
                sys.modules.pop(name, None)
            raise
        finally:
            for name in tuple(sys.modules):
                if (name in ("models", "utils") or
                        name.startswith(("models.", "utils."))):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_aliases)
            sys.dont_write_bytecode = previous_bytecode
        return _SOURCE



def build_model(source_root, args, arm="original"):
    if arm != "original":
        raise ValueError("Transport comparison uses unrepaired SlotSPE only")
    return load_source(source_root).SlotSPE(args)
