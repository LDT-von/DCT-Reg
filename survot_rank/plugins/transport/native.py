"""Build runnable native baseline adapters from verified offline source snapshots.

The official projects share the top-level package name ``models``.  This loader
keeps one verified namespace per process and rejects collisions explicitly.
It never downloads code or imports a repository whose local source bytes fail
the pinned inventory.  MCAT's concat fusion works on CPU; its original bilinear
implementation retains the upstream CUDA requirement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import threading
import types
from typing import Any

import torch
from torch import nn

from .adapters import MCATTransportAdapter, SurvPathTransportAdapter
from .config import TransportPluginConfig
from .plugin import DCTTransportPlugin
from .reconstruction import TransportReconstructionHeads


_KNOWN = {
    "mcat": {"directory": "MCAT", "commit": "b9cca63be83c67de7f95308d54a58f80b78b0da1",
             "module": "models.model_coattn", "class": "MCAT_Surv"},
    "survpath": {"directory": "SurvPath", "commit": "3f73ddd6705ec67d643020c5bb04fb13f9f382cc",
                 "module": "models.model_SurvPath", "class": "SurvPath"},
}
_IMPORT_LOCK = threading.RLock()
# Anchors from the verified official commit trees.  A caller cannot make
# arbitrary Python trusted merely by editing both the file and its manifest.
_MODEL_BLOBS = {
    "mcat": {
        "models/model_coattn.py": "26c032ee74957fffa02f3f08453cce343496894e",
        "models/model_genomic.py": "c80c745759a157cb8c97116494872187bb19e289",
        "models/model_set_mil.py": "3f7171f36ed06222ba76795c8b84b6daa70faeb0",
        "models/model_utils.py": "5f78ad8b1590622579e10ddecdcd447ed347fdc7",
    },
    "survpath": {
        "models/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "models/layers/__init__.py": "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
        "models/layers/cross_attention.py": "b84048fe846e927b07ff78de4c36aa7368857fbc",
        "models/model_ABMIL.py": "134c15fbb963b739edb8a48f6b7fd5d3e8d7d21c",
        "models/model_DeepMISL.py": "50a687add90966dd782be2222243d357027dcaf9",
        "models/model_MCATPathways.py": "a4344add420abb3aa555151c17b088e8636db173",
        "models/model_MLPOmics.py": "675f36bd9a65e092635e3fd080d12283b470d55e",
        "models/model_MLPWSI.py": "0512e08bd2551c87e11b7d8cf3535709c129eea3",
        "models/model_MaskedOmics.py": "14b9bb04ae14865cf4e7db126a8d21597846b463",
        "models/model_SNNOmics.py": "4d76782d68442a9fcb058a731aecd3c4b8cc85ff",
        "models/model_SurvPath.py": "52c128162ce8383923a4718c3ccefdf411ecff38",
        "models/model_SurvPath_with_nystrom.py": "a505ea185774d1df41915ba54b94b8beccdbc4d6",
        "models/model_TMIL.py": "93cd99890a28762a093e589d2f395515ad5613a2",
        "models/model_utils.py": "2cf40a4ac2f1ab3591e5c985c26e0ae6dcc0e5be",
    },
}


def _default_source_root() -> Path:
    # native.py -> transport -> plugins -> survot_rank -> project root
    return Path(__file__).resolve().parents[3] / "third_party" / "transport_plugin_baselines"


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _verify_snapshot(source_root: Path, backbone_name: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = source_root / "SOURCE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"verified native source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "dct_transport_plugin_sources_v1":
        raise ValueError("unsupported native source manifest schema")
    repositories = manifest.get("repositories")
    if not isinstance(repositories, dict) or backbone_name not in repositories:
        raise ValueError(f"source manifest has no repository for {backbone_name}")
    entry = repositories[backbone_name]
    expected = _KNOWN[backbone_name]
    if entry.get("commit") != expected["commit"] or entry.get("directory") != expected["directory"]:
        raise ValueError(f"{backbone_name} requires the pinned official commit {expected['commit']}")
    repository = (source_root / expected["directory"]).resolve()
    if not repository.is_relative_to(source_root.resolve()):
        raise ValueError("native repository path escapes its source root")
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("native source inventory must be a nonempty file list")
    verified_paths: set[str] = set()
    hashes = {}
    for item in files:
        relative = PurePosixPath(item.get("path", ""))
        if not str(relative) or relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError("invalid native source inventory path")
        key = relative.as_posix()
        if key in verified_paths:
            raise ValueError(f"duplicate native source inventory path: {key}")
        verified_paths.add(key)
        local = (repository / Path(*relative.parts)).resolve()
        if not local.is_relative_to(repository) or not local.is_file():
            raise ValueError(f"native source is missing or outside the verified repository: {key}")
        payload = local.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
        if (len(payload) != item.get("bytes") or sha256 != item.get("sha256")
                or blob != item.get("git_blob_sha1")):
            raise ValueError(f"native source verification failed: {key}")
        if key.startswith("models/") and key.endswith(".py") and blob != _MODEL_BLOBS[backbone_name].get(key):
            raise ValueError(f"native Python is not anchored to the pinned official commit: {key}")
        hashes[key] = sha256
    actual_python = {path.relative_to(repository).as_posix()
                     for path in (repository / "models").rglob("*.py")}
    inventory_python = {path for path in verified_paths if path.startswith("models/") and path.endswith(".py")}
    if actual_python != inventory_python:
        raise ValueError("native models directory contains missing or unverified Python files")
    required = "models/model_coattn.py" if backbone_name == "mcat" else "models/model_SurvPath.py"
    if required not in inventory_python:
        raise ValueError(f"native source inventory is missing {required}")
    if (repository / "models.py").exists():
        raise ValueError("unverified top-level models.py conflicts with the native package")
    return repository, {
        "source_commit": expected["commit"],
        "repository_url": entry.get("repository_url"),
        "source_root": str(source_root.resolve()),
        "verified_file_sha256": hashes,
        "source_schema": manifest["schema"],
    }


def _check_models_namespace(repository: Path) -> None:
    package_root = (repository / "models").resolve()
    for name, module in list(sys.modules.items()):
        if name != "models" and not name.startswith("models."):
            continue
        if module is None:
            raise RuntimeError(f"foreign or incomplete {name!r} namespace is loaded; use a fresh process")
        filename = getattr(module, "__file__", None)
        if filename:
            if not Path(filename).resolve().is_relative_to(package_root):
                raise RuntimeError("another official/foreign models package is already loaded; use one backbone per process")
        else:
            locations = getattr(module, "__path__", ())
            if not locations:
                raise RuntimeError("a foreign models namespace is already loaded; use a fresh process")
            if any(not Path(path).resolve().is_relative_to(package_root) for path in locations):
                raise RuntimeError("another official/foreign models package is already loaded; use one backbone per process")


@contextmanager
def _mcat_linear_compatibility(backbone_name: str):
    from torch.nn.modules import linear
    temporary = backbone_name == "mcat" and not hasattr(linear, "_LinearWithBias")
    if temporary:
        linear._LinearWithBias = linear.NonDynamicallyQuantizableLinear
    try:
        yield
    finally:
        if temporary:
            del linear._LinearWithBias


def _load_native_class(repository: Path, backbone_name: str):
    with _IMPORT_LOCK:
        _check_models_namespace(repository)
        original_models = {name for name in sys.modules if name == "models" or name.startswith("models.")}
        sys.path.insert(0, str(repository))
        try:
            with _mcat_linear_compatibility(backbone_name):
                if "models" not in sys.modules:
                    package_root = repository / "models"
                    init_path = package_root / "__init__.py"
                    if init_path.is_file():
                        spec = importlib.util.spec_from_file_location(
                            "models", init_path, submodule_search_locations=[str(package_root)],
                        )
                        package = importlib.util.module_from_spec(spec)
                        sys.modules["models"] = package
                        spec.loader.exec_module(package)
                    else:
                        # MCAT has a namespace directory, not an __init__.py.
                        # Fix its search locations to avoid merging foreign packages.
                        package = types.ModuleType("models")
                        package.__package__ = "models"
                        package.__path__ = [str(package_root)]
                        package.__spec__ = importlib.machinery.ModuleSpec("models", loader=None, is_package=True)
                        package.__spec__.submodule_search_locations = package.__path__
                        sys.modules["models"] = package
                specification = _KNOWN[backbone_name]
                module = importlib.import_module(specification["module"])
                _check_models_namespace(repository)
                return getattr(module, specification["class"])
        except Exception:
            # Remove only namespace entries newly created by this failed load.
            for name in list(sys.modules):
                if (name == "models" or name.startswith("models.")) and name not in original_models:
                    del sys.modules[name]
            raise
        finally:
            sys.path.remove(str(repository))


def build_native_adapter(
    backbone_name: str,
    omic_sizes: Sequence[int],
    *,
    source_root: str | Path | None = None,
    wsi_input_dim: int = 1536,
    plugin_config: Mapping[str, Any] | TransportPluginConfig | None = None,
    reconstruction_config: Mapping[str, Any] | None = None,
    fusion: str = "concat",
    enabled: bool = True,
    num_classes: int = 4,
) -> MCATTransportAdapter | SurvPathTransportAdapter:
    """Create a random-initialized native backbone and a shape-preserving plugin.

    ``omic_sizes`` describes the raw native gene-group inputs. Plugin dimensions
    describe the native *encoded* tokens, which are 256 for these two backbones.
    Reconstruction is disabled by default; a dictionary enables training-only
    heads with optional coefficients and the separate/shared decoder setting.
    """
    name = backbone_name.lower() if isinstance(backbone_name, str) else ""
    if name not in _KNOWN:
        raise ValueError("supported native backbones are 'mcat' and 'survpath'")
    if not isinstance(omic_sizes, Sequence) or isinstance(omic_sizes, (str, bytes)) or not omic_sizes:
        raise ValueError("omic_sizes must be a nonempty sequence of positive gene-group widths")
    sizes = [_positive_int(width, "omic_sizes entry") for width in omic_sizes]
    if name == "mcat" and len(sizes) != 6:
        raise ValueError("native MCAT requires exactly six omic_sizes")
    _positive_int(wsi_input_dim, "wsi_input_dim")
    _positive_int(num_classes, "num_classes")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be bool")
    if fusion not in {"concat", "bilinear"} or (name == "survpath" and fusion != "concat"):
        raise ValueError("MCAT supports concat/bilinear; SurvPath uses its native concat head")

    if plugin_config is None:
        config = TransportPluginConfig(wsi_dim=256, omics_dim=256)
    elif isinstance(plugin_config, TransportPluginConfig):
        config = plugin_config
    elif isinstance(plugin_config, Mapping):
        config = TransportPluginConfig.from_dict({"wsi_dim": 256, "omics_dim": 256, **dict(plugin_config)})
    else:
        raise TypeError("plugin_config must be a mapping or TransportPluginConfig")
    if config.wsi_dim != 256 or config.omics_dim != 256:
        raise ValueError("native MCAT/SurvPath encoded feature dimensions are both 256")
    reconstruction = None
    values = None
    if reconstruction_config is not None:
        if not isinstance(reconstruction_config, Mapping):
            raise TypeError("reconstruction_config must be a mapping or None")
        values = {"dim": config.dim, "num_omics_tokens": len(sizes), "heads": config.heads,
                  **dict(reconstruction_config)}
        if values["dim"] != config.dim or values["num_omics_tokens"] < len(sizes):
            raise ValueError("reconstruction dimensions/token capacity must match the plugin and native pathways")

    root = Path(source_root) if source_root is not None else _default_source_root()
    repository, provenance = _verify_snapshot(root.resolve(), name)
    native_class = _load_native_class(repository, name)
    if name == "mcat":
        backbone = native_class(omic_sizes=sizes, fusion=fusion, n_classes=num_classes)
        original = backbone.wsi_net[0]
        if not isinstance(original, nn.Linear) or original.out_features != 256:
            raise RuntimeError("pinned MCAT encoder no longer exposes its expected 256-dimensional linear input")
        old_width = original.in_features
        if wsi_input_dim != old_width:
            backbone.wsi_net[0] = nn.Linear(wsi_input_dim, original.out_features, bias=original.bias is not None)
        adapter_class = MCATTransportAdapter
        adaptation = {"mechanism": "native_instance_input_linear", "from": old_width,
                      "to": wsi_input_dim, "applied": old_width != wsi_input_dim}
    else:
        backbone = native_class(omic_sizes=sizes, wsi_embedding_dim=wsi_input_dim, num_classes=num_classes)
        adapter_class = SurvPathTransportAdapter
        adaptation = {"mechanism": "native_constructor_argument", "from": 1024,
                      "to": wsi_input_dim, "applied": wsi_input_dim != 1024}
    # Every comparison arm starts from the same backbone/core/bridge under a
    # common seed. Auxiliary layouts must neither shift their initialization
    # nor change the caller's subsequent dropout/batch random sequence.
    plugin = DCTTransportPlugin(config).set_enabled(enabled)
    if values is not None:
        with torch.random.fork_rng(devices=[]):
            if "wsi_head_enabled" in values:
                # Saved configurations fix physical checkpoint layout too.
                reconstruction = TransportReconstructionHeads.from_config(values)
            else:
                reconstruction = TransportReconstructionHeads(**values)
        plugin.reconstruction = reconstruction
    adapter = adapter_class(backbone, plugin)
    adapter.source_provenance = {
        **provenance, "backbone_name": name, "host_class": native_class.__name__,
        "adapter_class": adapter_class.__name__, "input_dim_adaptation": adaptation,
        "initialization": "random; no pretrained checkpoint loaded",
    }
    adapter.native_config = {
        "backbone_name": name, "omic_sizes": sizes, "wsi_input_dim": wsi_input_dim,
        "fusion": fusion, "enabled": enabled, "num_classes": num_classes,
        "plugin_config": config.to_dict(),
        "reconstruction_config": reconstruction.configuration() if reconstruction is not None else None,
    }
    return adapter
