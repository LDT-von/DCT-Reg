"""Inspect saved v313 exports only. Never imports or runs a model."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from survot_rank.evidence.manifest import audit, cindex, predictions, sha256
ALPHAS = np.array([0., .25, .5, .75, 1.])


def pair_states(time, censor, risk, tol=1e-8):
    time, censor, risk = [np.asarray(x, dtype=float) for x in (time, censor, risk)]
    if any(x.ndim != 1 for x in (time, censor, risk)) or not (len(time)==len(censor)==len(risk)):
        raise ValueError('Outcome and risk vectors must have equal lengths')
    if not len(time) or not all(np.isfinite(x).all() for x in (time,censor,risk)) or not np.isin(censor,[0,1]).all():
        raise ValueError('Invalid or nonfinite outcomes/risks')
    if not np.isfinite(tol) or tol < 0:
        raise ValueError('Invalid tie tolerance')
    comparable=((time[None,:]>time[:,None]) | ((time[None,:]==time[:,None]) & (censor[None,:]==1))) & (censor[:,None]==0)
    i,j=np.nonzero(comparable)
    if not len(i): raise ValueError('No comparable survival pairs')
    delta=risk[i]-risk[j]
    states=np.where(delta>tol,1,np.where(delta < -tol,-1,0))
    concordant=int((states==1).sum()); discordant=int((states==-1).sum()); ties=int((states==0).sum())
    counts={'concordant':concordant,'discordant':discordant,'tied_risk':ties,'tied_time_event_censor':int((time[i]==time[j]).sum()),'comparable_pairs':len(i), 'cindex':float((concordant+.5*ties)/len(i))}
    if abs(counts['cindex']-cindex(time,censor,risk,tied_tol=tol))>1e-14:
        raise ValueError('Pair counts disagree with repository C-index')
    return states,counts


def difference_stats(delta):
    delta=np.asarray(delta,dtype=float)
    if not delta.size or not np.isfinite(delta).all(): raise ValueError('Invalid difference array')
    flat=delta.reshape(delta.shape[0],-1)
    return {'mean_signed':float(flat.mean()),'mean_absolute':float(np.abs(flat).mean()),'max_absolute':float(np.abs(flat).max()),'median_patient_l2':float(np.median(np.linalg.norm(flat,axis=1)))}


def plan_diagnostics(plans, alphas=ALPHAS, rows=None, cols=None):
    p=np.asarray(plans,dtype=float)
    if p.ndim!=5 or min(p.shape)==0 or not np.isfinite(p).all() or (p<0).any():
        raise ValueError('plans must be finite nonnegative [patient,stage,geometry,wsi,omics]')
    mass=p.sum(axis=(-2,-1),keepdims=True)
    if (mass<=0).any(): raise ValueError('Zero transport mass')
    a=p.sum(axis=-1,keepdims=True); b=p.sum(axis=-2,keepdims=True)
    independent=a*b/mass
    if (rows is None)!=(cols is None): raise ValueError('Both target marginal arrays are required together')
    if rows is not None:
        rows=np.asarray(rows,dtype=float);cols=np.asarray(cols,dtype=float)
        if rows.shape!=(p.shape[0],p.shape[1],p.shape[3]) or cols.shape!=(p.shape[0],p.shape[1],p.shape[4]):
            raise ValueError('Target marginal shape mismatch')
        if not all(np.isfinite(x).all() for x in (rows,cols)) or (rows<0).any() or (cols<0).any():
            raise ValueError('Invalid target marginals')
    out=[]
    for alpha in alphas:
        mixed=(1-alpha)*p+alpha*independent; d=mixed-p
        normalized_l1=np.abs(d).sum(axis=(-2,-1))/mass[...,0,0]
        record={'alpha':float(alpha),'max_element_difference':float(np.abs(d).max()),'mean_normalized_plan_l1':float(normalized_l1.mean()),'max_normalized_plan_l1':float(normalized_l1.max()),'mean_normalized_plan_l1_by_stage_geometry':normalized_l1.mean(axis=0).tolist(),'max_row_change_from_factual':float(np.abs(mixed.sum(-1)-p.sum(-1)).max()),'max_col_change_from_factual':float(np.abs(mixed.sum(-2)-p.sum(-2)).max()),'max_mass_change':float(np.abs(mixed.sum(axis=(-2,-1))-mass[...,0,0]).max())}
        if rows is not None:
            record['max_row_residual_to_solver_target']=float(np.abs(mixed.sum(-1)-rows[:,:,None,:]).max())
            record['max_col_residual_to_solver_target']=float(np.abs(mixed.sum(-2)-cols[:,:,None,:]).max())
        out.append(record)
    return {'kind':'mathematically_reconstructed_from_saved_factual_plans_NOT_runtime_capture','alphas':out}



def captured_plan_diagnostics(factual, captured):
    p=np.asarray(factual,dtype=float);v=np.asarray(captured,dtype=float)
    if v.shape!=(p.shape[0],5,*p.shape[1:]) or not np.isfinite(v).all() or (v<0).any():
        raise ValueError('sweep_plans must be finite nonnegative [patient,alpha,stage,geometry,wsi,omics]')
    mass=p.sum(axis=(-2,-1),keepdims=True)
    if (mass<=0).any():raise ValueError('Zero factual transport mass')
    independent=p.sum(-1,keepdims=True)*p.sum(-2,keepdims=True)/mass
    records=[]
    for k,alpha in enumerate(ALPHAS):
        actual=v[:,k];expected=(1-alpha)*p+alpha*independent;delta=actual-p
        records.append({'alpha':float(alpha),'max_difference_from_expected_formula':float(np.abs(actual-expected).max()),
                        'matches_expected_formula_with_float_tolerance':bool(np.allclose(actual,expected,rtol=1e-5,atol=1e-6)),
                        'max_element_difference_from_factual':float(np.abs(delta).max()),
                        'mean_normalized_plan_l1_from_factual':float((np.abs(delta).sum(axis=(-2,-1))/mass[...,0,0]).mean()),
                        'max_row_change_from_factual':float(np.abs(actual.sum(-1)-p.sum(-1)).max()),
                        'max_col_change_from_factual':float(np.abs(actual.sum(-2)-p.sum(-2)).max()),
                        'max_mass_change':float(np.abs(actual.sum(axis=(-2,-1))-mass[...,0,0]).max())})
    return {'kind':'runtime_arrays_supplied_by_exporter_requires_exporter_provenance','alphas':records}


def analyze_arrays(data):
    required={'case_ids','time','censor','risk','sweep_risk','alphas'}
    if not required<=set(data):raise ValueError(f'Missing fields: {sorted(required-set(data))}')
    ids=np.asarray(data['case_ids']);time=np.asarray(data['time'],dtype=float);censor=np.asarray(data['censor'],dtype=float);risk=np.asarray(data['risk'],dtype=float);sweep=np.asarray(data['sweep_risk'],dtype=float)
    n=len(ids)
    if ids.ndim!=1 or len(set(map(str,ids)))!=n or not n:raise ValueError('Missing or duplicated patient IDs')
    if any(x.shape!=(n,) for x in (time,censor,risk)) or sweep.shape!=(n,5):raise ValueError('Patient vector shape mismatch')
    if np.asarray(data['alphas']).shape!=(5,) or not np.allclose(data['alphas'],ALPHAS,rtol=0,atol=1e-12):raise ValueError('Require alpha=0,.25,.5,.75,1')
    if not np.isfinite(sweep).all() or not np.allclose(sweep[:,0],risk,rtol=1e-5,atol=1e-6):raise ValueError('alpha=0 differs from factual risk or nonfinite sweep')
    baseline=sweep[:,0]; states0,counts0=pair_states(time,censor,baseline)
    patient_rows=[];out=[]
    all_i,all_j=np.triu_indices(n,k=1)
    order=lambda v:np.where(v[all_i]-v[all_j]>1e-8,1,np.where(v[all_i]-v[all_j]<-1e-8,-1,0))
    orders0=order(baseline)
    for k,alpha in enumerate(ALPHAS):
        delta=sweep[:,k]-baseline;st,counts=pair_states(time,censor,sweep[:,k]);abs_delta=np.abs(delta)
        transition={f'{a}_to_{b}':int(((states0==a)&(st==b)).sum()) for a in (-1,0,1) for b in (-1,0,1)}
        row={'alpha':float(alpha),**counts,'delta_cindex_from_alpha0':counts['cindex']-counts0['cindex'],'risk_mean_signed_change':float(delta.mean()),'risk_mean_absolute_change':float(abs_delta.mean()),'risk_median_absolute_change':float(np.median(abs_delta)),'risk_p95_absolute_change':float(np.quantile(abs_delta,.95)),'risk_max_absolute_change':float(abs_delta.max()),'risk_max_change_after_removing_common_offset':float(np.abs(delta-delta.mean()).max()),'patients_changed_exact':int((delta!=0).sum()),'patients_changed_above_tie_tolerance':int((abs_delta>1e-8).sum()),'comparable_pair_state_changes':int((st!=states0).sum()),'all_patient_pair_order_changes_with_tie_tolerance':int((order(sweep[:,k])!=orders0).sum()),'pair_state_transition_counts_minus1_discordant_0_tied_1_concordant':transition}
        out.append(row)
        for i,cid in enumerate(ids):patient_rows.append({'case_id':str(cid),'alpha':float(alpha),'risk_alpha0':float(baseline[i]),'risk_alpha':float(sweep[i,k]),'signed_change':float(delta[i]),'absolute_change':float(abs_delta[i]),'time':float(time[i]),'censor':int(censor[i])})
    plan_report=None
    if 'plans' in data:
        if np.asarray(data['plans']).shape[0]!=n:raise ValueError('Plan patient dimension mismatch')
        plan_report=plan_diagnostics(data['plans'],rows=data.get('rows'),cols=data.get('cols'))
    captured_plans=None
    if 'sweep_plans' in data:
        if 'plans' not in data:raise ValueError('sweep_plans requires factual plans')
        captured_plans=captured_plan_diagnostics(data['plans'],data['sweep_plans'])
    layers={};missing=[]
    for field in ['sweep_stage_gate','sweep_logits','sweep_fusion_events','sweep_encoded_event_tokens']:
        if field not in data:missing.append(field);continue
        v=np.asarray(data[field],dtype=float)
        if v.ndim<3 or v.shape[:2]!=(n,5) or not np.isfinite(v).all():raise ValueError(f'Invalid {field}')
        layers[field]=[{'alpha':float(a),**difference_stats(v[:,k]-v[:,0])} for k,a in enumerate(ALPHAS)]
    reconstruction=None
    if 'cross_pathway_error' in data:
        err=np.asarray(data['cross_pathway_error'],dtype=float)
        if err.ndim!=3 or err.shape[:2]!=(n,5) or not err.shape[2] or not np.isfinite(err).all():raise ValueError('Invalid pathway error array')
        reconstruction=[{'alpha':float(a),'mean_error':float(err[:,k].mean()),'delta_mean_error_from_alpha0':float((err[:,k]-err[:,0]).mean())} for k,a in enumerate(ALPHAS)]
    return {'n':n,'alpha0_minus_factual_max_abs':float(np.abs(baseline-risk).max()),'risk_pair_response':out,'plan_change':plan_report,'captured_plan_change':captured_plans,'layer_response':layers,'missing_runtime_layers':missing,'reconstruction':reconstruction,'interpretation':'Frozen-checkpoint diagnostic only; no automatic cause, training contribution, or monotonicity conclusion.'},patient_rows



def effective_config(run):
    try:
        from survot_rank.config import apply_overrides, flatten_config, load_config
    except ModuleNotFoundError as exc:
        if exc.name!='yaml':raise
        # Flat JSON is also valid YAML. Permit this dependency-free test/input format only.
        try:flat=json.loads(Path(run['config']).read_text(encoding='utf-8'))
        except json.JSONDecodeError as err:
            raise ValueError('YAML config requires PyYAML in the existing research environment') from err
        if not isinstance(flat,dict) or any(isinstance(v,dict) for v in flat.values()) or run.get('overrides'):
            raise ValueError('Sectioned config/overrides require PyYAML')
        return flat
    return flatten_config(apply_overrides(load_config(run['config']),run.get('overrides',[])))


def read_export(directory):
    directory=Path(directory).resolve(); ep=directory/'export.json';npz=directory/'patients.npz'
    meta=json.loads(ep.read_text(encoding='utf-8'));run=meta['run']
    if meta.get('schema_version')!=1 or not meta.get('export_commit'):raise ValueError('Missing export schema/source commit')
    if run.get('arm')!='exp6' or run.get('cancer') not in ['blca','kirc'] or run.get('seed')!=3 or run.get('protocol')!='legacy_val':raise ValueError('Require BLCA/KIRC Exp6 Full seed3 legacy_val')
    if not run.get('checkpoint'):raise ValueError('Fixed-checkpoint diagnosis requires checkpoint provenance')
    flat=effective_config(run)
    if flat.get('survot_method')!='dct_v313_transport_reconstruction' or flat.get('rna_format')!='Pathways':
        raise ValueError('Effective config must be v3.13 transport reconstruction with Pathways')
    # Validate existing trusted researcher artifacts; no model/checkpoint loading.
    verified=audit({'runs':[run]})
    if not verified['passed']:raise ValueError('; '.join(verified['errors']))
    for key,value in verified['runs'][0]['hashes'].items():
        if meta.get('hashes',{}).get(key)!=value:raise ValueError(f'Source hash changed: {key}')
    if meta.get('best_epoch')!=verified['runs'][0]['best_epoch']:raise ValueError('Export best epoch differs from source curve')
    with np.load(npz,allow_pickle=False) as source:data={k:source[k] for k in source.files}
    result,patients=analyze_arrays(data);saved=predictions(run['predictions']);ids=[str(x) for x in data['case_ids']]
    if set(ids)!=set(saved['case_ids']):raise ValueError('Export patient IDs differ from saved best predictions')
    index={cid:i for i,cid in enumerate(saved['case_ids'])};positions=[index[cid] for cid in ids]
    for key in ['risk','time','censor']:
        atol=2e-5 if key=='risk' else 1e-5
        if not np.allclose(data[key],saved[key][positions],rtol=1e-5,atol=atol):raise ValueError(f'Export {key} differs from saved patient data')
    claimed=meta.get('sweep',[])
    if len(claimed)!=5:raise ValueError('Export sweep must have all five alphas')
    for actual,claim in zip(result['risk_pair_response'],claimed):
        if abs(actual['alpha']-float(claim['alpha']))>1e-12 or abs(actual['cindex']-float(claim['cindex']))>1e-12:raise ValueError('Computed alpha/C-index differs from export metadata')
    result.update(run=run,export_commit=meta['export_commit'],source_hashes=verified['runs'][0]['hashes'],export_json_sha256=sha256(ep),patients_npz_sha256=sha256(npz),best_epoch=meta['best_epoch'])
    return result,patients


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--require-ten-folds',action='store_true');args=parser.parse_args(argv)
    spec=json.loads(args.input.read_text(encoding='utf-8'))
    if spec.get('schema_version')!=1 or not spec.get('exports'):raise ValueError('Require schema_version=1 and exports list')
    reports=[];patient_rows=[];identities=set();ids_by_cancer={'blca':set(),'kirc':set()}
    for item in spec['exports']:
        directory=(args.input.resolve().parent/str(item)).resolve();result,rows=read_export(directory);run=result['run'];identity=(run['cancer'],int(run['fold']))
        if identity in identities:raise ValueError('Duplicate fold identity')
        identities.add(identity);ids={row['case_id'] for row in rows}
        if ids & ids_by_cancer[run['cancer']]:raise ValueError('Validation patients overlap across folds')
        ids_by_cancer[run['cancer']].update(ids);reports.append(result)
        patient_rows.extend({'cancer':run['cancer'],'fold':run['fold'],**row} for row in rows)
    expected={(c,f) for c in ['blca','kirc'] for f in range(5)}
    if not identities<=expected:raise ValueError('Unexpected fold identity')
    if args.require_ten_folds and identities!=expected:raise ValueError(f'Missing folds: {sorted(expected-identities)}')
    all_sources=audit({'runs':[r['run'] for r in reports]})
    if not all_sources['passed']:raise ValueError('; '.join(all_sources['errors']))
    output=args.output.resolve()
    if output.exists():raise ValueError(f'Refuse to overwrite any existing output path: {output}')
    output.mkdir(parents=True)
    summary={'schema_version':1,'model_jobs_launched':False,'training_launched':False,'complete_ten_folds':identities==expected,'missing_folds':[list(x) for x in sorted(expected-identities)],'input_sha256':sha256(args.input),'auditor_sha256':sha256(__file__),'reports':reports}
    (output/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (output/'patient_risk_changes.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(patient_rows[0]));writer.writeheader();writer.writerows(patient_rows)
    print(f"Audited {len(reports)} existing exports; complete_ten_folds={summary['complete_ten_folds']}; output={output}")


if __name__=='__main__':main()
