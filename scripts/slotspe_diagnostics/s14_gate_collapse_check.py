"""Why do 38/379 omic patients have a near-zero decoder gate?

Hypothesis: with temperature=0.01, softmax(keep_score/0.01) is extremely
peaked.  The kept slots are chosen by Gumbel-noised scores, so the kept set
can be disjoint from the slots that carry the raw softmax mass.  When the
kept slots hold negligible mass, the normalisation

    gate = softmax(keep_score/T) * hard_keep
    gate = gate / (gate.sum() + 1e-8)

divides by the 1e-8 epsilon instead of the true mass, and the gate collapses
to ~0 -- the omic branch then contributes nothing to the prediction.

Read-only.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

REPO = Path("/data1/DCT-Reg"); sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from survot_rank.research.slotspe_proof.model import build_model, make_args
from survot_rank.research.slotspe_proof.patient import _batch

RUN = REPO/"results/slotspe_proof/slotspe_full_blca_5fold_20261020"
DEV="cuda:0"; SEED=20261022; BATCH=8
man = json.loads((RUN/"prepared/manifest.json").read_text())
rows=[]
for fold in range(5):
    fd=man["folds"][str(fold)]
    args=make_args(man["omic_sizes"], encoding_dim=man["encoding_dim"]); args.rna_format="Pathways"
    m=build_model(str(REPO/"third_party/SlotSPE"),args,"original").to(DEV)
    ck=torch.load(RUN/f"pairs/fold_{fold}/seed_3/original/last.pt",map_location=DEV,weights_only=False)
    m.load_state_dict(ck.get("state_dict",ck),strict=True); m.eval()
    for st in range(0,len(fd["val"]),BATCH):
        ids=fd["val"][st:st+BATCH]
        b=_batch(str(RUN),man,fd,ids,DEV,torch)
        with torch.no_grad():
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            xo=torch.stack([m.sig_networks[i](b[f"x_omic{i+1}"]) for i in range(m.num_pathways)],dim=1)
            so=m.slot_attention_omic(xo)
            dec=m.slot_decoder_omic
            mapped=dec.map(so)
            scores=dec.pred_keep_slot(mapped).squeeze(-1)
            _,gate,keep=dec(so)
            raw=F.softmax(scores/dec.temperature,dim=-1)
            mass=(raw*keep).sum(-1)                    # softmax mass carried by KEPT slots
            # which slots would the raw (un-noised) top-2 be?
            raw_top2=raw.topk(2,dim=-1).indices
            kept=keep>0.5
            overlap=[]
            for i in range(kept.shape[0]):
                ov=len(set(raw_top2[i].tolist()) & set(kept[i].nonzero().flatten().tolist()))
                overlap.append(ov)
            for i,pid in enumerate(ids):
                rows.append({"fold":fold,"patient_id":pid,
                             "gate_sum":float(gate[i].sum()),
                             "kept_mass":float(mass[i]),
                             "raw_top2_overlap_with_kept":overlap[i],
                             "scores_kept_min":float(scores[i][kept[i]].min()) if kept[i].any() else None,
                             "scores_all_max":float(scores[i].max())})
Path(REPO/"docs/research/slotspe_slot_content_20261022/gate_collapse_check.json").write_text(json.dumps(rows))
gs=np.array([r["gate_sum"] for r in rows]); km=np.array([r["kept_mass"] for r in rows])
ov=np.array([r["raw_top2_overlap_with_kept"] for r in rows])
print("n:",len(rows))
print(f"gate_sum: <1e-6: {(gs<1e-6).sum()}  <0.5: {(gs<0.5).sum()}  ==1: {(gs>0.999).sum()}")
print(f"kept softmax mass: min={km.min():.3e} med={np.median(km):.4f} max={km.max():.4f}")
print(f"raw_top2 overlap with kept: mean={ov.mean():.3f}  ==0: {(ov==0).sum()}  ==2: {(ov==2).sum()}")
bad=gs<0.5
print(f"\npatients with gate_sum<0.5 (n={bad.sum()}):")
print(f"   their kept_mass: max={km[bad].max():.3e} med={np.median(km[bad]):.3e}")
print(f"   their raw_top2 overlap: mean={ov[bad].mean():.3f}  ==0: {(ov[bad]==0).sum()}")
ok=~bad
print(f"patients with gate_sum>=0.5 (n={ok.sum()}): kept_mass med={np.median(km[ok]):.4f}, overlap mean={ov[ok].mean():.3f}")
