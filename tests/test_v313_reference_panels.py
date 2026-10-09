"""Synthetic figure/provenance checks, never model performance experiments."""
import copy
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pytest
import torch
from survot_rank.evidence.reference_panels import recorded_ablation, ablation_table, render_ablation, validate_tradeoff, render_tradeoff, text_sha
from survot_rank.evidence.manifest import save_json, sha256
from survot_rank.evidence.slotspe_style import pathway_percentiles
from survot_rank.evidence.v313 import profile_model
from scripts.plot_v313_slotspe_style import plot_pathway_case_panel
ROOT=Path(__file__).resolve().parents[1]


def test_recorded_table_preserves_all_folds_and_missing_cohorts():
    data=recorded_ablation(ROOT)
    assert len(data['variants'])==9
    full=data['variants'][6]
    assert sum(len(v) for v in full['folds'].values())==50
    assert all('\\' not in s['path'] for s in data['sources'])
    compact=ablation_table(data,['BLCA','KIRC'],dispersion=True)
    assert compact['complete']
    assert compact['styles'][6,0]=='first' and compact['styles'][6,1]=='first'
    assert compact['styles'][3,0]=='second' and compact['styles'][0,1]=='second'
    assert compact['text'][4][0].endswith('(0.025)')
    coverage=ablation_table(data,data['cancers'])
    assert not coverage['complete']
    assert coverage['values'][0][0] is None
    assert coverage['values'][6][-1]==pytest.approx(.702956)
    assert not any(col==0 or col==10 for row,col in coverage['styles'])
    assert all(coverage['values'][i][-1] is None for i in range(9) if i!=6)


def test_missing_or_invalid_ablation_folds_cannot_be_ranked():
    data=recorded_ablation(ROOT)
    data['variants'][0]['folds']['BLCA']=[.5]*4
    with pytest.raises(ValueError,match='five finite'):ablation_table(data,['BLCA'])


def test_pathway_percentiles_preserve_ties_without_false_contrast():
    raw=np.array([[.1,.1,.3,.5],[.25,.25,.25,.25]])
    percentile=pathway_percentiles(raw)
    np.testing.assert_allclose(percentile[0],[.25,.25,.625,.875])
    np.testing.assert_allclose(percentile[1],.5)
    np.testing.assert_array_equal(raw,[[.1,.1,.3,.5],[.25,.25,.25,.25]])


def synthetic_case():
    n=329
    raw=np.random.default_rng(3).uniform(.1,1,(8,n));raw/=raw.sum(1,keepdims=True)
    names=np.array([f'Synthetic_pathway_{i}_with_long_named_biological_process_for_layout_QA' for i in range(n)])
    return dict(case_id='SYNTHETIC_QA',arrays=dict(attention_omic=raw[None],risk=np.array([-1.])),
                names=names,meta={'run':{'id':'synthetic_only'}},source='synthetic_only',time=1.,censor=0)


def test_combined_panel_records_raw_values_and_each_real_slot(tmp_path):
    case=synthetic_case()
    plot_pathway_case_panel(case,tmp_path,scale='percentile')
    record=json.loads((tmp_path/'pathway_case_panel_percentile.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(record['raw_attention'],case['arrays']['attention_omic'][0])
    assert len(record['panels'])==8 and len(record['pathway_order'])==329
    for p in record['panels']:
        assert len(p['pathways'])==6
        for row in p['pathways']:
            i=list(case['names']).index(row['pathway'])
            assert row['raw_attention']==case['arrays']['attention_omic'][0,p['slot'],i]
    assert (tmp_path/'pathway_case_panel_percentile.png').stat().st_size>1000


def synthetic_tradeoff(root):
    data={'schema_version':1,'context':{'endpoint':'DSS','encoder':'UNI2-h','protocol':'legacy_val','seed':3,'cancers':['BLCA']},'models':[]}
    for model in range(2):
        records=[]
        for fold in range(5):
            directory=root/f'm{model}_f{fold}';directory.mkdir()
            ids=[f'SYNTHETIC_f{fold}_p{i}' for i in range(4)]
            preds={cid:{'risk':float(4-i if model==0 else i+1),'time':float(i+1),'censor':0} for i,cid in enumerate(ids)}
            path=directory/'pred.pkl'
            with path.open('wb') as f:pickle.dump(preds,f)
            split=directory/'split.csv'
            train=[f'SYNTHETIC_f{f}_p{i}' for f in range(5) if f!=fold for i in range(4)]
            split.write_text('train,val\n'+'\n'.join(f'{cid},{ids[i] if i<len(ids) else str()}' for i,cid in enumerate(train)),encoding='utf-8')
            profile=dict(device='cuda:0',hardware='SYNTHETIC_GPU',torch_version='synthetic',cuda_version='synthetic',cudnn_version=1,
                precision='float32',batch_size=1,patches=2048,feature_dim=1536,repeats=30,warmup=5,forward_mode='eval_no_grad',
                cudnn_benchmark=False,cudnn_deterministic=True,matmul_allow_tf32=False,benchmark_case_ids=[ids[0]],wsi_input_sha256=hashlib.sha256(str(fold).encode()).hexdigest(),
                peak_allocated_mb=400+model*200+fold,latency_ms_median=10+model*5+fold)
            meta=dict(run=dict(cancer='blca',fold=fold,seed=3,protocol='legacy_val',split_csv=str(split.resolve())),profile=profile,
                      hashes=dict(predictions=sha256(path),split_csv=sha256(split),checkpoint='synthetic_checkpoint'))
            prof=directory/'export.json';save_json(prof,meta)
            records.append(dict(cancer='BLCA',fold=fold,predictions=str(path.relative_to(root)),predictions_sha256=sha256(path),
                                profile_json=str(prof.relative_to(root)),profile_sha256_lf=text_sha(prof)))
        data['models'].append(dict(id=f'synthetic_model_{model}',label=f'SYNTHETIC_QA method {model}',ours=model==0,measurements=records))
    return data


def test_tradeoff_recomputes_score_and_rejects_unmatched_costs(tmp_path):
    data=synthetic_tradeoff(tmp_path)
    report=validate_tradeoff(data,tmp_path)
    assert [p['cindex'] for p in report['points']]==[1.,0.]
    assert report['points'][0]['memory_mib']==402
    record=data['models'][1]['measurements'][0]
    path=tmp_path/record['profile_json'];meta=json.loads(path.read_text(encoding='utf-8'))
    meta['profile']['patches']=4096;save_json(path,meta);record['profile_sha256_lf']=text_sha(path)
    with pytest.raises(ValueError,match='settings differ'):validate_tradeoff(data,tmp_path)


def test_tradeoff_blocks_missing_folds_source_corruption_and_patient_input_change(tmp_path):
    data=synthetic_tradeoff(tmp_path)
    short=copy.deepcopy(data);short['models'][1]['measurements'].pop()
    with pytest.raises(ValueError,match='complete five folds'):validate_tradeoff(short,tmp_path)
    record=data['models'][1]['measurements'][0]
    path=tmp_path/record['profile_json'];meta=json.loads(path.read_text(encoding='utf-8'))
    meta['profile']['wsi_input_sha256']='a'*64;save_json(path,meta);record['profile_sha256_lf']=text_sha(path)
    with pytest.raises(ValueError,match='WSI inputs differ'):validate_tradeoff(data,tmp_path)
    path.write_text(path.read_text(encoding='utf-8')+' ',encoding='utf-8')
    with pytest.raises(ValueError,match='hash mismatch'):validate_tradeoff(data,tmp_path)


def test_profile_metadata_is_measured_from_actual_synthetic_payload():
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.linear=torch.nn.Linear(2,1)
        def forward(self,x_wsi):return self.linear(x_wsi).mean(1)
    model=Tiny();payload={'x_wsi':torch.arange(10,dtype=torch.float32).reshape(1,5,2)}
    with pytest.raises(ValueError,match='eval mode'):profile_model(model,payload,repeats=2,warmup=1)
    model.eval();a=profile_model(model,payload,repeats=2,warmup=1)
    assert a['feature_dim']==2 and a['patches']==5 and a['forward_mode']=='eval_no_grad'
    assert a['peak_allocated_mb'] is None and a['latency_ms_median']>0
    b=profile_model(model,{'x_wsi':payload['x_wsi']+1},repeats=2,warmup=1)
    assert a['wsi_input_sha256']!=b['wsi_input_sha256']


def test_plot_outputs_have_hashes_and_never_overwrite(tmp_path):
    data=recorded_ablation(ROOT)
    report=render_ablation(data,tmp_path/'ablation')
    assert b'\r' not in (tmp_path/'ablation/ablation_tables.json').read_bytes()
    for table in report:
        assert b'\r' not in (tmp_path/'ablation'/f"{table['name']}.svg").read_bytes()
        for ext,digest in table['files'].items():assert sha256(tmp_path/'ablation'/f"{table['name']}.{ext}")==digest
    with pytest.raises(ValueError,match='fresh empty'):render_ablation(data,tmp_path/'ablation')
    fixture=tmp_path/'fixtures';fixture.mkdir()
    data=synthetic_tradeoff(fixture)
    report=render_tradeoff(data,fixture,tmp_path/'cost')
    assert len(report['points'])==2
    for ext,digest in report['files'].items():assert sha256(tmp_path/'cost'/f'efficiency_tradeoff.{ext}')==digest
