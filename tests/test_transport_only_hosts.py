"""Host interface contracts and separately labelled native synthetic checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from survot_rank.plugins.transport_only.hosts import HOST_ORDER, attach_transport


class PairFusion(nn.Module):
    def forward(self, w, o, *trailing):
        return w, o, trailing


class QKVFusion(nn.Module):
    def forward(self, query, key, value, **kwargs):
        return query, key, value, kwargs


def fixture_host(name):
    host = nn.Module()
    host.anchor = nn.Linear(8, 1)
    host.wsi_projection_dim = host.path_proj_dim = 8
    host.num_pathways = 3
    host.numOfproto = 4
    host.modality = "coattn"
    host.use_image = host.use_omics = host.label_use_coattn = True
    host.shared_prototypes, host.shared_proto_num, host.image_proto_num = 1, 4, 4
    host._label_num_path_tokens = 3
    host.cross_attention, host.PID = PairFusion(), PairFusion()
    host.identity, host.label_coattn = nn.Identity(), nn.Identity()
    host.coattn = QKVFusion()
    return host


@pytest.mark.parametrize("name", HOST_ORDER)
def test_host_interface_fixture_preserves_shapes_and_other_arguments(name):
    torch.set_num_threads(1)
    host = fixture_host(name)
    before = set(host.state_dict())
    rng = torch.random.get_rng_state().clone()
    attachment = attach_transport(host, name,
        {"slot_budget": 4, "stages": 1, "geometries": ["cosine"], "eps": .2, "eps_start": .2})
    assert torch.equal(rng, torch.random.get_rng_state())
    added = set(host.state_dict()) - before
    assert added and all(key.startswith("transport_mechanism.") for key in added)
    optimizer = torch.optim.Adam(host.parameters())
    assert id(attachment.mechanism.wsi_gain) in {id(p) for group in optimizer.param_groups for p in group["params"]}
    w, o = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
    marker = object()
    if name in {"slotspe", "pibd"}:
        target = host.cross_attention if name == "slotspe" else host.PID
        updated_w, updated_o, trailing = target(w, o, marker)
        assert trailing == (marker,)
    elif name == "ld_cvae":
        sequence_w, sequence_o = w.transpose(0, 1), o.transpose(0, 1)
        query, key, value, trailing = host.coattn(sequence_o, sequence_w, sequence_w, marker=marker)
        updated_w, updated_o = key.transpose(0, 1), query.transpose(0, 1)
        assert torch.equal(key, value) and trailing["marker"] is marker
    else:
        target = host.label_coattn if name == "tta" else host.identity
        updated = target(torch.cat((o, w), dim=1))
        updated_o, updated_w = updated[:, :3], updated[:, 3:]
    assert updated_w.shape == w.shape and updated_o.shape == o.shape
    assert not torch.equal(updated_w, w)
    assert not torch.equal(updated_o, o)
    (updated_w.square().mean() + updated_o.square().mean()).backward()
    assert attachment.mechanism.stage_pair_cost[1].weight.grad.norm() > 0
    attachment.set_epoch(2).set_enabled(False)
    if name in {"slotspe", "pibd"}:
        assert target(w, o)[0] is w
    elif name == "ld_cvae":
        assert host.coattn(sequence_o, sequence_w, sequence_w)[0] is sequence_o
    else:
        tokens = torch.cat((o, w), dim=1)
        assert target(tokens) is tokens
    attachment.remove()
    assert set(host.state_dict()) == before


def test_mmp_ot_arm_and_duplicate_attachment_are_rejected():
    host = fixture_host("mmp")
    host.mot = nn.Identity()
    with pytest.raises(ValueError, match="already contains OT"):
        attach_transport(host, "mmp")
    host = fixture_host("slotspe")
    attach_transport(host, "slotspe")
    with pytest.raises(ValueError, match="already attached"):
        attach_transport(host, "slotspe")


def native_case(name):
    """Separate process keeps official models/utils package names independent."""
    torch.set_num_threads(1)
    torch.manual_seed(13)
    source_root = Path(os.environ.get("DCT_TRANSPORT_ONLY_SOURCE_ROOT", ROOT))
    sources = {
        "slotspe": source_root / "third_party/SlotSPE",
        "survpath": source_root / "third_party/E047_official_code_20261011_v1/SurvPath",
        "pibd": source_root / "third_party/E047_official_code_20261011_v1/PIBD",
        "mmp": source_root / "third_party/E047_official_code_20261011_v1/MMP/src",
        "ld_cvae": source_root / "third_party/E047_official_code_20261011_v1/LD-CVAE",
        "tta": source_root / "third_party/three_paper_sources_20261013/nearest_prior_art/TTA/TTA/src",
    }
    if not sources[name].is_dir():
        print(f"NATIVE_SKIP: source snapshot missing at {sources[name]}; set DCT_TRANSPORT_ONLY_SOURCE_ROOT")
        return
    sys.path.insert(0, str(sources[name]))
    # Optional, isolated copy of the version pinned by MMP/env.yaml. Never
    # change the user's Python installation or silently alter native trainers.
    isolated_deps = source_root / ".review/transport-only-deps-20261018"
    if name == "mmp" and isolated_deps.is_dir():
        sys.path.insert(0, str(isolated_deps))
    sizes = [5, 6, 7]
    if name == "pibd" and not torch.cuda.is_available():
        print("NATIVE_SKIP: upstream PIBD allocates .cuda() tensors; no CUDA available")
        return
    try:
        if name == "slotspe":
            from models.SlotSPE import SlotSPE
            args = SimpleNamespace(omic_sizes=sizes, n_classes=4, encoding_dim=12,
                wsi_projection_dim=256, rna_format="Pathways", slot_num_wsi=4,
                slot_num_omics=4, slot_iters=2, temperature=.5, topk_ratio=.5,
                top_k_method="parallel_topk_st", bag_loss="nll_surv", alpha_surv=.5,
                lambda_recon_loss=.01)
            model = SlotSPE(args)
            inputs = {"x_wsi": torch.randn(2, 11, 12), "omic_missing": False,
                      "y": torch.tensor([1, 2]), "c": torch.tensor([0., 1.]),
                      **{f"x_omic{i+1}": torch.randn(2, s) for i, s in enumerate(sizes)}}
            call = lambda: model(**inputs)
            from utils.loss_func import NLLSurvLoss
            criterion = NLLSurvLoss(alpha=.5)
            objective = lambda output: criterion(output[0], inputs["y"], None, inputs["c"]) / 2 + output[1]
            unpack = lambda output: output[0]
        elif name == "survpath":
            from models.model_SurvPath import SurvPath
            model = SurvPath(omic_sizes=sizes, wsi_embedding_dim=12)
            inputs = {"x_path": torch.randn(1, 11, 12), "return_attn": False,
                      **{f"x_omic{i+1}": torch.randn(s) for i, s in enumerate(sizes)}}
            call = lambda: model(**inputs)
            # Native SurvPath criterion uses logits, matching its own trainer.
            from utils.loss_func import NLLSurvLoss
            criterion = NLLSurvLoss(alpha=.5)
            objective = lambda output: criterion(h=output, y=torch.tensor([1]), t=None, c=torch.tensor([0.]))
            unpack = lambda output: output
        elif name == "mmp":
            from mil_models.model_multimodal import coattn
            model = coattn(omic_sizes=sizes, histo_in_dim=12, histo_model="PANTHER",
                           numOfproto=4, modality="coattn", append_embed="none")
            call = lambda: model.forward_no_loss(torch.randn(2, 4, 25), [torch.randn(2, s) for s in sizes])
            from utils.losses import NLLSurvLoss
            criterion = NLLSurvLoss(alpha=.5)
            objective = lambda output: criterion(output["logits"], torch.tensor([[1], [2]]), torch.tensor([[0.], [1.]]))["loss"]
            unpack = lambda output: output["logits"]
        elif name == "tta":
            from mil_models.model_otmil_label import OTMIL_Label
            model = OTMIL_Label(histo_in_dim=12, hidden_dim=32, num_classes=4,
                use_image=True, use_omics=True, omic_sizes=sizes, image_proto_num=4,
                omics_proto_num=4, shared_prototypes=1, shared_proto_num=4,
                shared_proto_dim=32, joint_ot_single_path=1, fusion_type="coattn",
                enable_modality_refine=1, modref_weight=.1, modref_apply_to_tokens=1)
            w, o = torch.randn(2, 11, 12), [torch.randn(2, s) for s in sizes]
            call = lambda: model.forward_no_loss(w, o)
            from utils.losses import NLLSurvLoss
            criterion = NLLSurvLoss(alpha=.5)
            objective = lambda output: criterion(output["logits"], torch.tensor([[1], [2]]), torch.tensor([[0.], [1.]]))["loss"]
            unpack = lambda output: output["logits"]
        elif name == "ld_cvae":
            # Import compatibility required by the original MCAT attention code.
            import torch.nn.modules.linear as linear
            if not hasattr(linear, "_LinearWithBias"):
                linear._LinearWithBias = linear.NonDynamicallyQuantizableLinear
            from models.model_coattn import Robust_MCAT
            if not torch.cuda.is_available():
                print("NATIVE_SKIP: LD-CVAE generator has upstream CUDA allocations; requires CUDA")
                return
            model = Robust_MCAT(generator=True, use_condition=True, wsi_encoding_dim=12,
                                omic_sizes=[5] * 6).cuda()
            w, o = torch.randn(11, 12).cuda(), [torch.randn(5).cuda() for _ in range(6)]
            inputs = {"x_path": w, "omic_missing": False, "train": True, "stage": "warmup",
                      "label": torch.tensor([1]).cuda(), "c": torch.tensor([0.]).cuda(),
                      **{f"x_omic{i+1}": x for i, x in enumerate(o)}}
            call = lambda: model(**inputs)
            unpack = lambda output: output[0]
            from utils.utils import NLLSurvLoss
            criterion = NLLSurvLoss(alpha=.5)
            objective = lambda output: criterion(hazards=output[0], S=output[1], Y=inputs["label"], c=inputs["c"])
        else:
            from models.model_PIBD import PIBD
            args = SimpleNamespace(omic_sizes=sizes, n_classes=4, encoding_dim=12,
                wsi_projection_dim=32, omics_format="pathways", ratio_wsi=.01,
                ratio_omics=1., sample_num=2, seed=13, num_patches=512, alpha=.1, beta=.01)
            model = PIBD(args).cuda()
            inputs = {"x_wsi": torch.randn(2, 512, 12).cuda(), "return_attn": False,
                      "y": torch.tensor([1, 2]).cuda(), "c": torch.tensor([0., 1.]).cuda(),
                      **{f"x_omic{i+1}": torch.randn(2, s).cuda() for i, s in enumerate(sizes)}}
            call = lambda: model(**inputs)
            objective = lambda output: model.loss_surv(output[0], inputs["y"], None, inputs["c"]) / 2
            unpack = lambda output: output[0]
    except ModuleNotFoundError as error:
        print(f"NATIVE_SKIP: missing upstream dependency {error.name}")
        return
    model.train()
    state = torch.random.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state_all() if name in {"pibd", "ld_cvae"} else None
    baseline = call()
    attachment = attach_transport(model, name, {"stages": 2, "eps": .2, "eps_start": .2}, enabled=False)
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    disabled = call()
    assert torch.equal(unpack(baseline), unpack(disabled)), "disabled transport changed native prediction"
    if name == "slotspe":
        assert torch.equal(baseline[1], disabled[1]), "disabled transport changed native auxiliary loss"
    attachment.set_enabled(True)
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    output = call()
    assert not torch.equal(unpack(output), unpack(baseline)), "enabled transport did not reach the native head"
    objective(output).backward()
    grad = model.transport_mechanism.stage_pair_cost[1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    assert attachment.mechanism.last_diagnostics["paired_patients"] > 0
    if name in {"slotspe", "mmp", "tta"}:
        assert not any(attachment.mechanism.last_diagnostics["wsi_compressed"])
        assert not any(attachment.mechanism.last_diagnostics["omics_compressed"])
    print("NATIVE_PASS: " + json.dumps({"host": name, "shape": list(unpack(output).shape),
        "transport_gradient_norm": float(grad.norm()), "patient_data_used": False}))


@pytest.mark.parametrize("name", HOST_ORDER)
def test_native_source_synthetic_forward_backward(name):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), name],
        cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    if "NATIVE_SKIP:" in result.stdout:
        pytest.skip(result.stdout.strip())
    assert "NATIVE_PASS:" in result.stdout, result.stdout


if __name__ == "__main__":
    native_case(sys.argv[1])
