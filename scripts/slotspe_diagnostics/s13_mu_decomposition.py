"""Is the high inter-slot cosine caused by the shared slots_mu component?

slots_mu has shape (1, 1, dim): all eight slots share ONE centre vector and
differ only by the Gaussian noise term.  If the trained slot vectors retain a
large component along that shared mu, then the mutual cosine of the slots is
explained by the shared centre, not by the input data.

Decompose each final slot vector into
    parallel  = projection onto the unit mu direction
    residual  = the remainder
and report the fraction of squared norm in the parallel part, plus the
cosine predicted from the parallel parts alone.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path("/data1/DCT-Reg"); sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.patient import _batch

RUN = REPO / "results/slotspe_proof/slotspe_full_blca_5fold_20261020"
DEV = "cuda:0"; SEED = 20261022; BATCH = 8

man = json.loads((RUN/"prepared/manifest.json").read_text())
out = {}
for fold in range(5):
    fd = man["folds"][str(fold)]
    args = make_args(man["omic_sizes"], encoding_dim=man["encoding_dim"]); args.rna_format="Pathways"
    m = build_model(str(REPO/"third_party/SlotSPE"), args, "original").to(DEV)
    ck = torch.load(RUN/f"pairs/fold_{fold}/seed_3/original/last.pt", map_location=DEV, weights_only=False)
    m.load_state_dict(ck.get("state_dict", ck), strict=True); m.eval()
    for key, mod, inp_fn in (("wsi", m.slot_attention_wsi, lambda b: m.wsi_mlp(b["x_wsi"])),
                             ("omic", m.slot_attention_omic, None)):
        mu = mod.slots_mu.detach().reshape(-1)          # (D,)
        mu_hat = F.normalize(mu, dim=0)
        fr, par_cos, res_cos = [], [], []
        for st in range(0, len(fd["val"]), BATCH):
            ids = fd["val"][st:st+BATCH]
            b = _batch(str(RUN), man, fd, ids, DEV, torch)
            with torch.no_grad():
                torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
                if key=="wsi": x = inp_fn(b)
                else: x = torch.stack([m.sig_networks[i](b[f"x_omic{i+1}"]) for i in range(m.num_pathways)], dim=1)
                s = mod(x)                                # (B,S,D)
                par = s @ mu_hat                          # (B,S)
                res = s - par.unsqueeze(-1) * mu_hat
                frac = (par**2) / (s**2).sum(-1)
                fr.append(frac.cpu().numpy())
                pn = F.normalize(par, dim=-1); rn = F.normalize(res, dim=-1)
                par_cos.append((pn.unsqueeze(2)*pn.unsqueeze(1)).cpu().numpy())
                res_cos.append(torch.einsum("bsd,btd->bst", rn, rn).cpu().numpy())
        fr = np.concatenate(fr); pc = np.concatenate(par_cos); rc = np.concatenate(res_cos)
        off = ~np.eye(8, dtype=bool)
        out.setdefault(key, []).append({
            "fold": fold,
            "frac_norm_along_shared_mu": float(fr.mean()),
            "parallel_part_cos_offdiag": float(pc[:, off].mean()),
            "residual_part_cos_offdiag": float(rc[:, off].mean()),
        })
print(json.dumps(out, indent=1))
Path(REPO/"docs/research/slotspe_slot_content_20261022/mu_decomposition.json").write_text(json.dumps(out, indent=1))
