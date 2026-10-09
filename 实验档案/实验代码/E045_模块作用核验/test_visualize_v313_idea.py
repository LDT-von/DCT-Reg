import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).with_name('visualize_v313_idea.py')
spec=importlib.util.spec_from_file_location('v313_idea_visualization',path)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def fixture():
    risk=np.array([3.,2.,1.]);ids=['a','b','c'];time=np.array([1.,2.,3.]);censor=np.zeros(3)
    saved=dict(case_ids=ids,risk=risk.copy(),time=time.copy(),censor=censor.copy())
    a=dict(case_ids=np.array(ids),risk=risk.copy(),time=time.copy(),censor=censor.copy(),alphas=v.ALPHAS.copy(),
        sweep_risk=np.stack([risk,risk+1,risk+2,risk+3,risk+4],axis=1),
        sweep_row_residual=np.zeros((3,5)),sweep_col_residual=np.zeros((3,5)),
        pathway_names=np.array(['p0','p1','p2','p3']),cross_pathway_error=np.ones((3,5,4))*.1,
        attention_omic=np.ones((3,2,4))/4)
    for modality in ('wsi','omic'):
        for name,value in (('mean_cosine',.8),('mean_distance',.2),('hazard_variance',.01)):a[f'{modality}_{name}']=np.full(3,value)
    return {'sweep':[{'cindex':1.} for _ in range(5)]},a,saved

def test_unchanged_ranking_can_have_changed_patient_risk():
    meta,a,saved=fixture();stats=v.validate_arrays(meta,a,saved)
    assert stats['delta_cindex']==[0]*5
    assert stats['mean_abs_risk_change_over_factual_iqr']==[0,1,2,3,4]
    assert np.allclose(stats['omics_normalized_attention_entropy'],1)

def test_factual_prediction_mismatch_is_not_accepted():
    meta,a,saved=fixture();a['risk'][0]+=1
    with pytest.raises(ValueError,match='Factual replay'):v.validate_arrays(meta,a,saved)

def test_plan_marginal_change_is_rejected():
    meta,a,saved=fixture();a['sweep_col_residual'][1,2]=.002
    with pytest.raises(ValueError,match='marginals'):v.validate_arrays(meta,a,saved)

def test_plot_cannot_hide_inconsistent_stored_score():
    meta,a,saved=fixture();meta['sweep'][4]['cindex']=.5
    with pytest.raises(ValueError,match='Stored sweep scores'):v.validate_arrays(meta,a,saved)

def test_unscaled_attention_cannot_be_given_entropy_label():
    meta,a,saved=fixture();a['attention_omic']*=2
    with pytest.raises(ValueError,match='normalized'):v.validate_arrays(meta,a,saved)

def test_zero_risk_iqr_is_reported_as_undefined():
    meta,a,saved=fixture();a['risk'][:]=1;saved['risk'][:]=1;a['sweep_risk'][:]=1
    meta['sweep']=[{'cindex':.5} for _ in range(5)]
    with pytest.raises(ValueError,match='IQR is zero'):v.validate_arrays(meta,a,saved)

def test_saved_training_and_negative_evidence_are_both_retained():
    report=v.recorded(v.ROOT)
    assert set(report['gains'])=={'BLCA','KIRC'}
    assert [r['positive_folds'] for r in report['gains']['BLCA'].values()]==[3,3]
    assert [r['positive_folds'] for r in report['gains']['KIRC'].values()]==[4,5]
    assert all(row['delta_cindex']==[0]*5 for row in report['historical_sweeps']['BLCA'])
    assert sum(any(abs(x)>1e-12 for x in row['delta_cindex']) for row in report['historical_sweeps']['KIRC'])==1
    assert len(report['sources'])==13
