"""Launcher checks use a tiny fake trainer and never patient data."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_transport_only import _validate_native_selection, make_plan


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_slotspe_source(tmp_path):
    source = tmp_path / "upstream"
    (source / "models").mkdir(parents=True)
    (source / "models/__init__.py").write_text("", encoding="utf-8")
    (source / "models/SlotSPE.py").write_text('''
import torch
from torch import nn
class Fusion(nn.Module):
    def forward(self, w, o):
        return torch.cat((w.mean(1), o.mean(1)), -1)
class SlotSPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsi_projection_dim = 8
        self.cross_attention = Fusion()
        self.to_logits = nn.Linear(16, 1)
    def forward(self, w, o):
        return self.to_logits(self.cross_attention(w, o))
''', encoding="utf-8")
    (source / "survival.py").write_text('''
import torch
from models.SlotSPE import SlotSPE
torch.set_num_threads(1)
torch.manual_seed(13)
model = SlotSPE()
optimizer = torch.optim.Adam(model.parameters())
loss = model(torch.randn(2, 4, 8), torch.randn(2, 3, 8)).square().mean()
loss.backward()
if hasattr(model, 'transport_mechanism') and model.transport_mechanism.enabled:
    assert model.transport_mechanism.wsi_gain.grad is not None
''', encoding="utf-8")
    return source


def test_metadata_plan_retains_native_arguments_and_rejects_old_loss_fields(fake_slotspe_source):
    args = ["--bag_loss", "nll_surv", "--lambda_recon_loss", ".01", "--seed", "3"]
    plan = make_plan("slotspe", "learned", fake_slotspe_source, args)
    assert plan["native_args"] == args
    assert plan["dct_auxiliary_losses"] == []
    assert plan["loss_owner"] == "native_host_only"
    with pytest.raises(TypeError):
        make_plan("slotspe", "learned", fake_slotspe_source, args, {"reconstruction_weight": .1})
    with pytest.raises(ValueError, match="fixed epsilon"):
        make_plan("slotspe", "learned", fake_slotspe_source, args, {"eps_start": .1})


def test_unsupported_native_variants_fail_before_invoking_any_trainer():
    for name, argv in [("mmp", ["--model_mm_type=coattn_mot"]),
                       ("ld_cvae", ["--model_type", "survpath"]), ("tta", [])]:
        with pytest.raises(ValueError):
            _validate_native_selection(name, argv)
    _validate_native_selection("tta", ["--modality_type", "multi", "--fusion_type=coattn"])


@pytest.mark.parametrize("arm", ["native", "bypass", "independent", "learned"])
def test_fake_native_trainer_receives_only_feature_transport(fake_slotspe_source, tmp_path, arm):
    record = tmp_path / (arm + ".json")
    cmd = [sys.executable, str(ROOT / "scripts/run_transport_only.py"), "--host", "slotspe",
           "--source-root", str(fake_slotspe_source), "--arm", arm, "--record", str(record)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads(record.read_text(encoding="utf-8"))
    assert saved["status"] == "native_entry_completed"
    assert saved["loss_owner"] == "native_host_only" and saved["dct_auxiliary_losses"] == []
    assert len(saved["attachments"]) == (0 if arm == "native" else 1)
    # Existing records are not replaced by rerunning the same command.
    before = record.read_bytes()
    repeated = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert repeated.returncode != 0 and record.read_bytes() == before
