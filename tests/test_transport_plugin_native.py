"""Offline native-construction checks; forwards use only small synthetic tensors."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest
import torch

from survot_rank.plugins.transport.native import build_native_adapter


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "transport_plugin_baselines"


@pytest.fixture
def copied_source(tmp_path):
    assert (SOURCE_ROOT / "SOURCE_MANIFEST.json").is_file(), "offline native source subset is required"
    destination = tmp_path / "sources"
    shutil.copytree(SOURCE_ROOT, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def _manifest(root):
    return json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))


def _write_manifest(root, manifest):
    (root / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def test_factory_rejects_unsupported_backbone():
    with pytest.raises(ValueError, match="supported native backbones"):
        build_native_adapter("untrusted_repository", [5] * 6)


@pytest.mark.parametrize("sizes", [[], [5] * 5, [True] * 6, [0] * 6])
def test_factory_rejects_invalid_mcat_signature_sizes(sizes):
    with pytest.raises(ValueError):
        build_native_adapter("mcat", sizes)


def test_factory_rejects_wrong_encoded_dimensions():
    with pytest.raises(ValueError, match="encoded feature dimensions"):
        build_native_adapter("mcat", [5] * 6, plugin_config={"wsi_dim": 1536})


def test_factory_rejects_wrong_reconstruction_dimensions():
    with pytest.raises(ValueError, match="reconstruction dimensions"):
        build_native_adapter("survpath", [5] * 3, plugin_config={"dim": 32},
                             reconstruction_config={"dim": 16})


def test_factory_rejects_absent_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="source manifest is missing"):
        build_native_adapter("mcat", [5] * 6, source_root=tmp_path)


def test_factory_rejects_unpinned_commit(copied_source):
    manifest = _manifest(copied_source)
    manifest["repositories"]["mcat"]["commit"] = "0" * 40
    _write_manifest(copied_source, manifest)
    with pytest.raises(ValueError, match="pinned official commit"):
        build_native_adapter("mcat", [5] * 6, source_root=copied_source)


def test_factory_rejects_tampered_source(copied_source):
    source = copied_source / "MCAT" / "models" / "model_coattn.py"
    source.write_bytes(source.read_bytes() + b"\n# modified local source\n")
    with pytest.raises(ValueError, match="source verification failed"):
        build_native_adapter("mcat", [5] * 6, source_root=copied_source)


def test_forged_inventory_does_not_make_python_trusted(copied_source):
    source = copied_source / "MCAT" / "models" / "model_coattn.py"
    payload = source.read_bytes() + b"\nraise RuntimeError('untrusted source must never execute')\n"
    source.write_bytes(payload)
    manifest = _manifest(copied_source)
    for item in manifest["repositories"]["mcat"]["files"]:
        if item["path"] == "models/model_coattn.py":
            item.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
                        git_blob_sha1=hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest())
    _write_manifest(copied_source, manifest)
    with pytest.raises(ValueError, match="not anchored to the pinned official commit"):
        build_native_adapter("mcat", [5] * 6, source_root=copied_source)


def test_factory_rejects_unlisted_python(copied_source):
    (copied_source / "MCAT" / "models" / "unlisted.py").write_text("raise RuntimeError('not imported')\n")
    with pytest.raises(ValueError, match="unverified Python files"):
        build_native_adapter("mcat", [5] * 6, source_root=copied_source)


def test_factory_rejects_foreign_models_namespace(monkeypatch):
    foreign = types.ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)
    with pytest.raises(RuntimeError, match="foreign models namespace"):
        build_native_adapter("mcat", [5] * 6, source_root=SOURCE_ROOT)


def _native_factory_case(name: str):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    count = 6 if name == "mcat" else 3
    sizes = [5 + index for index in range(count)]
    plugin_config = {"dim": 32, "heads": 4, "wsi_slots": 2, "omics_slots": 3,
                     "stages": 2, "slot_iters": 2, "dropout": 0.0,
                     "sinkhorn_iters": 20, "projection_iters": 1000}
    from torch.nn.modules import linear
    original_alias = getattr(linear, "_LinearWithBias", None)
    recipe = {"backbone_name": name, "omic_sizes": sizes, "wsi_input_dim": 16,
              "plugin_config": plugin_config, "reconstruction_config": {"mode": "separate"}}
    adapter = build_native_adapter(**recipe)
    assert getattr(linear, "_LinearWithBias", None) is original_alias, "temporary compatibility alias leaked"
    assert adapter.source_provenance["input_dim_adaptation"]["to"] == 16
    assert adapter.source_provenance["source_commit"]
    json.dumps(adapter.source_provenance)
    json.dumps(adapter.native_config)
    path = torch.randn(7, 16) if name == "mcat" else torch.randn(1, 7, 16)
    kwargs = {"x_path": path, **{f"x_omic{index}": torch.randn(width)
                                for index, width in enumerate(sizes, start=1)}}
    if name == "survpath":
        kwargs["return_attn"] = False
    adapter.train()
    result, aux = adapter.forward_with_aux(**kwargs, cur_epoch=7, plugin_compute_aux=True)
    prediction = result[0] if isinstance(result, tuple) else result
    assert prediction.shape == (1, 4) and torch.isfinite(prediction).all()
    assert aux.item() > 0
    (prediction.square().mean() + aux).backward()
    assert adapter.plugin.wsi_gain.grad is not None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in adapter.plugin.reconstruction.parameters())
    adapter.eval()
    with torch.no_grad():
        expected = adapter(**kwargs)
    buffer = io.BytesIO()
    torch.save(adapter.state_dict(), buffer)
    buffer.seek(0)
    # Reload construction from the JSON-serializable recipe, not hidden defaults.
    saved_recipe = json.loads(json.dumps(adapter.native_config))
    restored = build_native_adapter(**saved_recipe).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(**kwargs)
    if isinstance(actual, tuple):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    default_width = build_native_adapter(name, sizes, plugin_config=plugin_config)
    encoder = default_width.backbone.wsi_net if name == "mcat" else default_width.backbone.wsi_projection_net
    assert encoder[0].in_features == 1536
    assert default_width.plugin.reconstruction is None
    other = "survpath" if name == "mcat" else "mcat"
    with pytest.raises(RuntimeError, match="one backbone per process"):
        build_native_adapter(other, [5] * (3 if other == "survpath" else 6), plugin_config=plugin_config)


def _native_initialization_case(name: str):
    """Compare initial parameters and caller RNG, never noisy train predictions."""
    torch.set_num_threads(2)
    count = 6 if name == "mcat" else 3
    sizes = [5 + index for index in range(count)]
    plugin_config = {"dim": 32, "heads": 4, "wsi_slots": 2, "omics_slots": 3,
                     "stages": 2, "slot_iters": 2, "dropout": 0.0,
                     "sinkhorn_iters": 20, "projection_iters": 1000}
    arms = []
    for reconstruction in (None, {"mode": "separate"}, {"mode": "shared"}):
        torch.manual_seed(8123)
        adapter = build_native_adapter(name, sizes, wsi_input_dim=16,
                                       plugin_config=plugin_config,
                                       reconstruction_config=reconstruction)
        arms.append((adapter, torch.random.get_rng_state().clone()))
    base, base_rng = arms[0]
    for adapter, rng in arms[1:]:
        assert torch.equal(rng, base_rng), "auxiliary layout changed post-factory RNG"
        baseline = base.backbone.state_dict()
        candidate = adapter.backbone.state_dict()
        assert candidate.keys() == baseline.keys()
        for key in baseline:
            torch.testing.assert_close(candidate[key], baseline[key], rtol=0, atol=0)
        baseline = {key: value for key, value in base.plugin.state_dict().items()
                    if not key.startswith("reconstruction.")}
        candidate = {key: value for key, value in adapter.plugin.state_dict().items()
                     if not key.startswith("reconstruction.")}
        assert candidate.keys() == baseline.keys()
        for key in baseline:
            if isinstance(baseline[key], torch.Tensor):
                torch.testing.assert_close(candidate[key], baseline[key], rtol=0, atol=0)
            else:
                assert candidate[key] == baseline[key]


@pytest.mark.parametrize("name", ["mcat", "survpath"])
@pytest.mark.parametrize("case", ["--native-factory", "--native-initialization"])
def test_real_native_factory_forward_backward_restore_and_initialization(name, case):
    env = dict(os.environ)
    project = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = project + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, name],
        env=env, cwd=project, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=90,
    )
    assert result.returncode == 0, f"{name} factory failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in {"--native-factory", "--native-initialization"}:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "--native-factory":
        _native_factory_case(sys.argv[2])
    else:
        _native_initialization_case(sys.argv[2])
    print(f"PASS: verified {sys.argv[2]} native factory")
