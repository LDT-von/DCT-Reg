"""Synthetic interface checks against pinned *real* MCAT and SurvPath sources.

Each native import runs in an isolated process because both official projects
use the global ``models`` package name.  No patient data, checkpoints, training
runs, network requests, or upstream source mutations are involved.  Missing
official source snapshots fail with an actionable message rather than passing
a fake backbone check.  MCAT's native bilinear fusion needs CUDA.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from survot_rank.plugins.transport.adapters import MCATTransportAdapter, SurvPathTransportAdapter


PINNED_COMMITS = {
    "MCAT": "b9cca63be83c67de7f95308d54a58f80b78b0da1",
    "SurvPath": "3f73ddd6705ec67d643020c5bb04fb13f9f382cc",
}


def _source_root() -> Path:
    configured = os.environ.get("E047_OFFICIAL_SOURCE_ROOT")
    if configured:
        candidate = Path(configured)
        if not (candidate / "SOURCE_MANIFEST.json").is_file():
            raise AssertionError(f"E047_OFFICIAL_SOURCE_ROOT lacks SOURCE_MANIFEST.json: {candidate}")
        return candidate
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "third_party" / "transport_plugin_baselines"
        if (candidate / "SOURCE_MANIFEST.json").is_file():
            return candidate
        candidate = ancestor / "third_party" / "E047_official_code_20261011_v1"
        if (candidate / "SOURCE_MANIFEST.json").is_file():
            return candidate
    raise AssertionError(
        "Pinned official MCAT/SurvPath sources are required. Set "
        "E047_OFFICIAL_SOURCE_ROOT to the offline verified source snapshot; "
        "this test never downloads code or substitutes a fake native model."
    )


def _verified_native_import(method: str, root: Path):
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("schema") == "dct_transport_plugin_sources_v1":
        inventory = manifest["repositories"][method.lower()]
    else:
        inventory = json.loads((root / "inventories" / f"{method}.json").read_text(encoding="utf-8"))
    assert inventory["commit"] == PINNED_COMMITS[method]
    for entry in inventory["files"]:
        if entry["path"].endswith(".py"):
            payload = (root / method / entry["path"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["sha256"], entry["path"]
    sys.path.insert(0, str(root / method))
    if method == "MCAT":
        # Compatibility alias confined to this test process, never upstream.
        from torch.nn.modules import linear
        if not hasattr(linear, "_LinearWithBias"):
            linear._LinearWithBias = linear.NonDynamicallyQuantizableLinear
        return importlib.import_module("models.model_coattn").MCAT_Surv
    return importlib.import_module("models.model_SurvPath").SurvPath


def _assert_native_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_native_equal(actual[key], expected[key])
    else:
        assert isinstance(actual, tuple) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_native_equal(left, right)


class _AuxiliaryProbe(nn.Module):
    """Check wrapper routing separately from the real plugin's algorithm."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))
        self.received = None

    def forward(self, wsi, omics, **kwargs):
        self.received = kwargs
        aux = self.weight.square() * (wsi.square().mean() + omics.square().mean())
        return SimpleNamespace(wsi=wsi, omics=omics, aux_loss=aux, aux_losses={"probe": aux}, transport=None)


def _native_case(method: str, fusion: str, pathway_count: int, root: Path) -> None:
    torch.manual_seed(32)
    torch.set_num_threads(2)
    native_class = _verified_native_import(method, root)
    device = torch.device("cuda" if method == "MCAT" and fusion == "bilinear" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The pinned native MCAT bilinear implementation requires CUDA")
    sizes = [5 + index for index in range(pathway_count)]
    if method == "MCAT":
        backbone = native_class(omic_sizes=sizes, fusion=fusion, dropout=0).to(device)
        kwargs = {"x_path": torch.randn(9, 1024, device=device)}
        adapter_class = MCATTransportAdapter
    else:
        backbone = native_class(omic_sizes=sizes, wsi_embedding_dim=24, dropout=0).to(device)
        kwargs = {"x_path": torch.randn(1, 9, 24, device=device), "return_attn": True}
        adapter_class = SurvPathTransportAdapter
    kwargs.update({f"x_omic{index}": torch.randn(width, device=device)
                   for index, width in enumerate(sizes, start=1)})

    from survot_rank.plugins.transport import DCTTransportPlugin, TransportPluginConfig
    from survot_rank.plugins.transport.reconstruction import TransportReconstructionHeads
    config = TransportPluginConfig(
        wsi_dim=256, omics_dim=256, dim=32, wsi_slots=2, omics_slots=3,
        stages=2, heads=4, slot_iters=2, dropout=0,
        sinkhorn_iters=20, projection_iters=1000, projection_tol=1e-4,
    )
    reconstruction = TransportReconstructionHeads(dim=32, num_omics_tokens=pathway_count, heads=4)
    plugin = DCTTransportPlugin(config, reconstruction=reconstruction).to(device)
    adapter = adapter_class(backbone, plugin).to(device).eval()
    assert adapter.backbone is backbone
    assert not backbone._forward_hooks and not backbone._forward_pre_hooks
    with torch.no_grad():
        expected = backbone(**kwargs)
        plugin.enabled = False
        def forbidden(*_args, **_kwargs):
            raise AssertionError("disabled plugin called transport or reconstruction")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(plugin.core, "forward", forbidden)
            patch.setattr(plugin.reconstruction, "forward", forbidden)
            actual, aux = adapter.forward_with_aux(**kwargs)
        _assert_native_equal(actual, expected)
        assert aux.ndim == 0 and aux.item() == 0
        plugin.enabled = True
        # The zero residual gains initialize an inserted plugin to identity.
        _assert_native_equal(adapter(**kwargs), expected)
        plugin.wsi_gain.fill_(0.2)
        plugin.omics_gain.fill_(0.2)
        changed = adapter(**kwargs)
        first_native = expected[0] if isinstance(expected, tuple) else expected
        first_changed = changed[0] if isinstance(changed, tuple) else changed
        assert first_changed.shape == first_native.shape
        assert torch.isfinite(first_changed).all()
        assert not torch.equal(first_changed, first_native), "enabled transport did not reach native prediction"
        output = adapter.last_plugin_output
        assert output.wsi.shape == (1, 9, 256)
        assert output.omics.shape == (1, pathway_count, 256)

    # The scalar returned to the trainer reaches actual native encoding weights.
    adapter.train()
    adapter.zero_grad(set_to_none=True)
    native_output, auxiliary = adapter.forward_with_aux(**kwargs, cur_epoch=7, plugin_compute_aux=True)
    assert auxiliary.requires_grad and auxiliary.ndim == 0 and auxiliary.item() > 0
    auxiliary.backward()
    encoder = backbone.wsi_net if method == "MCAT" else backbone.wsi_projection_net
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in backbone.sig_networks.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in plugin.reconstruction.parameters())

    # Keep native non-attention SurvPath output, and per-call disabling behavior.
    adapter.eval()
    if method == "SurvPath":
        no_attention = dict(kwargs, return_attn=False)
        with torch.no_grad():
            actual = adapter(**no_attention, plugin_enabled=False)
            _assert_native_equal(actual, backbone(**no_attention))
            assert actual.shape == (1, 4)

    # Explicit input errors precede opaque native tensor failures.
    bad = dict(kwargs)
    bad["x_omic1"] = torch.randn(sizes[0] + 1, device=device)
    with pytest.raises(ValueError, match="native encoder expects"):
        adapter(**bad)
    bad = dict(kwargs)
    bad[f"x_omic{pathway_count + 1}"] = torch.randn(4, device=device)
    with pytest.raises(ValueError, match="expected exactly"):
        adapter(**bad)
    bad = dict(kwargs)
    bad["x_path"] = kwargs["x_path"][..., :2]
    with pytest.raises(ValueError, match="native encoder expects"):
        adapter(**bad)
    # The original downstream attention/pooling is not padding-aware.
    for mask_name, count in (("plugin_wsi_mask", 9), ("plugin_omics_mask", pathway_count)):
        partial_mask = torch.ones(1, count, dtype=torch.bool, device=device)
        partial_mask[0, -1] = False
        with pytest.raises(ValueError, match="mask-aware"):
            adapter(**dict(kwargs, **{mask_name: partial_mask}))
    if method == "SurvPath":
        bad = dict(kwargs, x_path=kwargs["x_path"].expand(2, -1, -1))
        with pytest.raises(ValueError, match="batch size 1"):
            adapter(**bad)

    # Native state is registered once; train/eval and state reload propagate.
    state = adapter.state_dict()
    assert any(key.startswith("backbone.") for key in state)
    assert any(key.startswith("plugin.") for key in state)
    adapter.load_state_dict(state, strict=True)
    adapter.train()
    assert backbone.training and plugin.training
    adapter.eval()
    assert not backbone.training and not plugin.training

    # Metadata is not forwarded to the generic feature module.
    probe = _AuxiliaryProbe().to(device)
    probed = adapter_class(backbone, probe).to(device).eval()
    with torch.no_grad():
        _assert_native_equal(probed(**kwargs, patient_metadata="ignored", cur_epoch=11), backbone(**kwargs))
    assert probe.received == {"wsi_mask": None, "omics_mask": None, "epoch": 11,
                              "compute_aux": None, "enabled": None}

    class WrongShapeProbe(_AuxiliaryProbe):
        def forward(self, wsi, omics, **kwargs):
            output = super().forward(wsi, omics, **kwargs)
            output.wsi = output.wsi[:, :-1]
            return output
    wrong_shape = adapter_class(backbone, WrongShapeProbe().to(device)).eval()
    with pytest.raises(ValueError, match="preserve both native encoded feature shapes"):
        wrong_shape(**kwargs)


@pytest.mark.parametrize("method,fusion,pathway_count", [
    ("MCAT", "concat", 6),
    ("MCAT", "bilinear", 6),
    ("SurvPath", "concat", 3),
    ("SurvPath", "concat", 7),
])
def test_real_native_transport_adapter(method, fusion, pathway_count):
    root = _source_root()
    env = dict(os.environ)
    project = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--native-case", method,
         fusion, str(pathway_count), str(root)],
        env=env, cwd=project, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=90,
    )
    assert result.returncode == 0, f"{method} native integration failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("adapter_class", [MCATTransportAdapter, SurvPathTransportAdapter])
def test_adapter_rejects_incompatible_backbone(adapter_class):
    with pytest.raises(TypeError, match="native backbone must expose"):
        adapter_class(nn.Identity(), _AuxiliaryProbe())


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--native-case":
    sys.stdout.reconfigure(encoding="utf-8")
    _native_case(sys.argv[2], sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]))
    print(f"PASS: pinned {sys.argv[2]} {sys.argv[3]} with {sys.argv[4]} pathways")
