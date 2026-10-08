"""Synthetic regressions for diagnostic aggregation, never patient experiments."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from scripts.summarize_v313_exp6 import collect_groups, summarize_group, make_summary_figures, check_paper_reference
from survot_rank.evidence.additional import RECON_MODES, RETRIEVAL_VERSION, RETRIEVAL_CENTER, summarize_patches
from survot_rank.evidence.additional_plots import read_exports
from survot_rank.evidence.manifest import cindex, save_json, sha256


def synthetic_folds(root, folds=range(5), *, version=RETRIEVAL_VERSION):
    for fold in folds:
        n, repeats, pathways = 3+fold, 2+fold, 2
        ids=np.asarray([f"synthetic_f{fold}_p{i}" for i in range(n)])
        risk=np.arange(n,0,-1,dtype=float)/n
        time=np.arange(1,n+1,dtype=float)
        shuffled=np.stack([np.roll(risk,1+r%(n-1)) for r in range(repeats)],axis=1)
        arrays=dict(case_ids=ids,risk=risk,time=time,censor=np.zeros(n),fractions=np.asarray([0.,.5]),
                    pathway_names=np.asarray(["synthetic_A","synthetic_B"]),retrieval_training_mean=np.zeros((2,4)),
                    pairing_shuffled_risk=shuffled,
                    donor_case_ids=np.stack([np.roll(ids,1+r%(n-1)) for r in range(repeats)]),
                    unpadded_risk=risk,removed_counts=np.tile([0,2],(n,1)),real_patches=np.full(n,4),
                    attention_relative_range=np.full(n,.2),attention_score_std=np.full(n,.02))
        deletion=np.repeat(risk[:,None,None,None],2,axis=1)
        deletion=np.repeat(np.repeat(deletion,repeats,axis=2),3,axis=3)
        deletion[:,1,:,0]=risk[::-1,None]
        deletion[:,1,:,2]=shuffled
        budget=np.repeat(risk[:,None,None],2,axis=1)
        budget=np.repeat(budget,repeats,axis=2)
        budget[:,1,:]=shuffled
        arrays.update(deletion_risk=deletion,budget_risk=budget)
        reconstruction={}
        for i,k in enumerate(RECON_MODES):
            error=np.full((n,pathways),.2+i*.04+fold*.01)
            rank=np.full(n,n if k=="train_mean" else 1,dtype=int)
            arrays["reconstruction_"+k+"_error"]=error
            arrays["retrieval_"+k+"_rank"]=rank
            reconstruction[k]=dict(mean_error=float(error.mean()),top1=float((rank==1).mean()),mrr=float((1/rank).mean()),chance_top1=1/n)
        arrays["reconstruction_shuffled_all_error"]=np.repeat(arrays["reconstruction_shuffled_error"][:,None,:],repeats,axis=1)
        arrays["retrieval_shuffled_all_rank"]=np.repeat(arrays["retrieval_shuffled_rank"][:,None],repeats,axis=1)
        record=dict(schema_version=1,run=dict(id=f"synthetic_f{fold}",cancer="blca",arm="exp6",seed=3,fold=fold,
                    protocol="legacy_val",source_commit="synthetic_only"),hashes={},export_commit="synthetic_only",
                    coefficients=dict(self=.05,cross=.05),repeats=repeats,experiment="all",reconstruction=reconstruction,
                    pairing=dict(native_cindex=cindex(time,arrays["censor"],risk),
                        shuffled_cindex=[cindex(time,arrays["censor"],shuffled[:,r]) for r in range(repeats)],
                        mean_abs_delta=float(np.abs(shuffled-risk[:,None]).mean())))
        if version is not None:
            record.update(retrieval_metric_version=version,retrieval_center_definition=RETRIEVAL_CENTER)
        record["patches"]=summarize_patches(arrays)
        directory=Path(root)/f"fold_{fold}";directory.mkdir(parents=True)
        np.savez_compressed(directory/"additional.npz",**arrays)
        record["array_sha256"]=sha256(directory/"additional.npz")
        save_json(directory/"additional.json",record)
    return Path(root)


def change_arrays(root, fold, change):
    directory=root/f"fold_{fold}"
    path=directory/"additional.npz"
    with np.load(path,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
    record=json.loads((directory/"additional.json").read_text(encoding="utf-8"))
    change(arrays)
    record["patches"]=summarize_patches(arrays)
    np.savez_compressed(path,**arrays)
    record["array_sha256"]=sha256(path)
    save_json(directory/"additional.json",record)


def test_five_fold_metrics_use_real_counts_and_pairing_cindex(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    group=next(iter(collect_groups([root]).values()))
    result=summarize_group(group)
    assert result["patients"]==[3,4,5,6,7]
    assert result["pairing"]["repeats_by_fold"]==[2,3,4,5,6]
    assert result["chance_by_fold"]==[1/3,1/4,1/5,1/6,1/7]
    assert result["pairing"]["native_cindex_by_fold"]==[1.]*5
    assert result["reconstruction"]["native"]["error_by_fold"]!=[1.]*5
    assert result["patches"]["retained_fractions"]==[1.,.5]


def test_corrupt_arrays_and_native_pairing_summary_rejected(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    meta=root/"fold_0/additional.json"
    r=json.loads(meta.read_text(encoding="utf-8"));r["pairing"]["native_cindex"]=.123
    save_json(meta,r)
    with pytest.raises(ValueError,match="Pairing summary"):
        collect_groups([root])
    # Input corruption must be checked before the summary gets written.
    arrays=root/"fold_0/additional.npz";arrays.write_bytes(arrays.read_bytes()+b"corrupt")
    with pytest.raises(ValueError,match="hash mismatch"):
        make_summary_figures([root],tmp_path/"never_written")
    assert not (tmp_path/"never_written").exists()


def test_missing_folds_and_duplicate_roots_rejected(tmp_path):
    missing=synthetic_folds(tmp_path/"missing",range(4))
    with pytest.raises(ValueError,match="exactly folds"):
        collect_groups([missing])
    full=synthetic_folds(tmp_path/"complete")
    with pytest.raises(ValueError,match="Duplicate experiment"):
        collect_groups([full,full])


def test_pathway_order_cannot_be_silently_averaged(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    change_arrays(root,1,lambda a:a.update(pathway_names=a["pathway_names"][::-1]))
    group=next(iter(collect_groups([root]).values()))
    with pytest.raises(ValueError,match="Pathway order"):
        summarize_group(group)


def test_budget_fraction_order_must_match_between_folds(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    change_arrays(root,2,lambda a:a.update(fractions=np.asarray([0.,.25])))
    group=next(iter(collect_groups([root]).values()))
    with pytest.raises(ValueError,match="fractions/order"):
        summarize_group(group)


def test_legacy_retrieval_blocked_but_unaffected_diagnostics_reusable(tmp_path):
    root=synthetic_folds(tmp_path/"exports",version=None)
    group=next(iter(collect_groups([root]).values()))
    with pytest.raises(ValueError,match="Retrieval v1"):
        summarize_group(group)
    result=summarize_group(group,("pairing","pathway_advantage","patch_deletion","patch_budget"))
    assert "top1_by_fold" not in result["reconstruction"]["native"]
    with pytest.raises(ValueError,match="Legacy retrieval center"):
        read_exports(root,require_corrected_retrieval=True)


def test_paper_reference_flags_wrong_full_checkpoint_or_patients(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    group=next(iter(collect_groups([root]).values()))
    path=tmp_path/"reference.json"
    save_json(path,{"models":[{"id":"dct_v313","folds":{"BLCA":[1.]*5}}]})
    check_paper_reference(group,path)
    save_json(path,{"models":[{"id":"dct_v313","folds":{"BLCA":[.7]*5}}]})
    with pytest.raises(ValueError,match="Paper Full mismatch"):
        check_paper_reference(group,path)


def test_all_five_summary_figures_and_provenance(tmp_path):
    root=synthetic_folds(tmp_path/"exports")
    out=tmp_path/"figures"
    report=make_summary_figures([root],out)
    assert len(report["figures"])==10
    assert len(list(out.glob("*.png")))==5
    assert len(list(out.glob("*.pdf")))==5
    for item in report["figures"]:assert sha256(out/item["path"])==item["sha256"]
    assert len(report["groups"][0]["sources"])==5
    with pytest.raises(ValueError,match="fresh directory"):
        make_summary_figures([root],out)
