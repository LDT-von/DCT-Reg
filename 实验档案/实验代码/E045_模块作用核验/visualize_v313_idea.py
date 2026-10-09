"""v3.13 module-effect figures from existing records or audited frozen exports.
No model, checkpoint inference or training is executed by this script.
"""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from survot_rank.evidence.reference_panels import recorded_ablation, text_sha
from survot_rank.evidence.manifest import audit, predictions, split_ids, cindex, sha256, revision

ALPHAS=np.array([0,.25,.5,.75,1.])

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise ValueError('Use a new empty output directory')
    path.mkdir(parents=True,exist_ok=True)
    return path

def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    return plt

def save(fig,out,name):
    files={}
    for ext in ('png','pdf','svg'):
        p=out/f'{name}.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight',facecolor='white')
        if ext=='svg':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
        files[ext]=sha256(p)
    plotting().close(fig)
    return files

def recorded(root):
    root=Path(root);record=recorded_ablation(root)
    variants={v['id']:v for v in record['variants']};gains={};sweeps={};sources=record['sources'].copy()
    for cancer in ('BLCA','KIRC'):
        full=np.array(variants['exp6']['folds'][cancer]);gains[cancer]={}
        for control in ('direct','independent'):
            values=np.array(variants[control]['folds'][cancer])
            gains[cancer][control]={'full':full.tolist(),'control':values.tolist(),'delta':(full-values).tolist(),
                'mean_delta':float((full-values).mean()),'positive_folds':int((full>values).sum())}
        folds=[]
        for fold in range(5):
            p=root/f'实验档案/表格图片/E010_运输计划干预/原始材料/fig3_sweep_{cancer.lower()}_fold{fold}.json'
            row=read_json(p);a=np.asarray(row['alphas']);scores=np.asarray(row['cindex'])
            if row['cancer']!=cancer.lower() or row['fold']!=fold or not np.array_equal(a,ALPHAS) or scores.shape!=(5,) or not np.isfinite(scores).all():raise ValueError('Invalid historical sweep identity')
            if not np.isclose(scores[0],full[fold],atol=5e-5,rtol=0):raise ValueError('Historical sweep baseline differs from recorded Full')
            folds.append({'fold':fold,'n':row['n_samples'],'cindex':scores.tolist(),'delta_cindex':(scores-scores[0]).tolist()})
            sources.append({'path':p.relative_to(root).as_posix(),'sha256_lf':text_sha(p)})
        sweeps[cancer]=folds
    return {'schema_version':1,'source_commit':revision(root),'sources':sources,'gains':gains,'historical_sweeps':sweeps,
        'note':'Reanalysis of existing seed-3 best-validation records, not a new training experiment. Full appendix values are rounded to 4 decimals. Historical sweep has no patient-level replay alignment proof. Direct keeps prediction OT; Independent changes both prediction and reconstruction. No significance stars.'}

def render_recorded(root,output):
    report=recorded(root);out=fresh(output);plt=plotting()
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.3),sharey=True)
    extent=max(abs(v) for g in report['gains'].values() for r in g.values() for v in r['delta'])*1.25
    for ax,(cancer,records) in zip(axes,report['gains'].items()):
        ax.axhline(0,color='black',lw=.8)
        for shift,(control,color) in zip((-.08,.08),(('direct','#0072B2'),('independent','#D55E00'))):
            values=records[control]['delta'];ax.plot(np.arange(5)+shift,values,'o-',color=color,lw=1,
                label=f"Full - {control.title()} (mean {np.mean(values):+.4f})")
        ax.set(xticks=range(5),xlabel='Matched fold',title=cancer,ylim=(-extent,extent));ax.grid(axis='y',ls='--',alpha=.3);ax.legend(fontsize=8,loc='best')
    axes[0].set_ylabel('Full minus trained control: validation C-index')
    fig.suptitle('Training-control evidence for the v3.13 idea',fontweight='bold')
    fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.015,'Same named cohorts / seed 3; historical records. Positive values favor Full. Final patient/config matching remains pending.',ha='center',fontsize=8)
    report['training_figure']=save(fig,out,'training_module_gains')
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.3),sharey=True)
    extent=max(0.001,max(abs(v) for f in report['historical_sweeps'].values() for row in f for v in row['delta_cindex'])*1.25)
    for ax,(cancer,folds) in zip(axes,report['historical_sweeps'].items()):
        for row in folds:ax.plot(ALPHAS,row['delta_cindex'],'o-',lw=1,label=f"Fold {row['fold']}")
        ax.axhline(0,color='black',lw=.7);ax.set(title=cancer,xlabel='Fraction of plan replaced by marginal product',xticks=ALPHAS,ylim=(-extent,extent));ax.grid(ls='--',alpha=.3);ax.legend(fontsize=7)
    axes[0].set_ylabel('C-index change from factual plan')
    fig.suptitle('Frozen-model ranking response: retained negative evidence',fontweight='bold')
    fig.tight_layout(rect=(0,.13,1,.94));fig.text(.5,.015,'0: factual learned plan; 1: same-marginal independent plan. Flat C-index does not establish unchanged patient risk or reconstruction.',ha='center',fontsize=8)
    report['sweep_figure']=save(fig,out,'frozen_plan_ranking_response')
    write_json(out/'evidence.json',report)
    with (out/'paired_fold_gains.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['cancer','control','fold','full_cindex','control_cindex','full_minus_control'])
        for cancer,records in report['gains'].items():
            for control,row in records.items():
                for fold,(a,b,c) in enumerate(zip(row['full'],row['control'],row['delta'])):w.writerow([cancer,control,fold,a,b,c])
    return report

def validate_arrays(meta,a,saved):
    ids=a['case_ids'].astype(str).tolist();n=len(ids)
    if n<2 or len(set(ids))!=n or set(ids)!=set(saved['case_ids']):raise ValueError('Patient identities differ')
    if not np.array_equal(a['alphas'],ALPHAS):raise ValueError('Require the five preregistered plan strengths')
    lookup={cid:i for i,cid in enumerate(saved['case_ids'])};order=[lookup[cid] for cid in ids]
    for key in ('risk','time','censor'):
        if a[key].shape!=(n,) or not np.allclose(a[key],saved[key][order],rtol=1e-5,atol=2e-5):raise ValueError('Factual replay/outcomes differ')
    for key in ('sweep_risk','sweep_row_residual','sweep_col_residual'):
        if a[key].shape!=(n,5) or not np.isfinite(a[key]).all():raise ValueError('Invalid sweep array')
    if not np.allclose(a['sweep_risk'][:,0],a['risk'],rtol=1e-5,atol=1e-6):raise ValueError('alpha=0 must reproduce the factual predictor')
    if any(np.max(a[key])>1e-4 or np.min(a[key])<0 for key in ('sweep_row_residual','sweep_col_residual')):raise ValueError('Plan intervention changed factual marginals')
    error=a['cross_pathway_error'];attention=a['attention_omic']
    if error.ndim!=3 or error.shape[:2]!=(n,5) or error.shape[2]!=len(a['pathway_names']) or not np.isfinite(error).all() or (error<0).any():raise ValueError('Invalid reconstruction errors')
    if attention.ndim!=3 or attention.shape[0]!=n or attention.shape[2]!=error.shape[2] or not np.isfinite(attention).all() or (attention<0).any() or not np.allclose(attention.sum(-1),1,atol=3e-5):raise ValueError('Attention is not a normalized patient/slot/pathway matrix')
    for modality in ('wsi','omic'):
        for metric in ('mean_cosine','mean_distance','hazard_variance'):
            v=a[f'{modality}_{metric}']
            if v.shape!=(n,) or not np.isfinite(v).all():raise ValueError('Invalid slot diagnosis')
            if metric=='mean_cosine' and ((v < -1.00001)|(v>1.00001)).any():raise ValueError('Invalid cosine range')
            if metric!='mean_cosine' and (v<0).any():raise ValueError('Negative slot distance/variance')
    score=[cindex(a['time'],a['censor'],a['sweep_risk'][:,j]) for j in range(5)]
    if len(meta['sweep'])!=5 or not np.allclose(score,[x['cindex'] for x in meta['sweep']],atol=1e-9,rtol=0):raise ValueError('Stored sweep scores do not match patient arrays')
    scale=float(np.subtract(*np.percentile(a['risk'],[75,25])))
    if scale<=1e-12:raise ValueError('Factual risk IQR is zero; normalized risk response undefined')
    p=attention.clip(min=1e-30);entropy=-(p*np.log(p)).sum(-1)/np.log(attention.shape[-1])
    return {'n':n,'delta_cindex':(np.array(score)-score[0]).tolist(),
        'mean_abs_risk_change_over_factual_iqr':(np.abs(a['sweep_risk']-a['risk'][:,None]).mean(0)/scale).tolist(),
        'mean_cross_error_change':(error.mean((0,2))-error[:,0].mean()).tolist(),
        'maximum_marginal_residual':float(max(a['sweep_row_residual'].max(),a['sweep_col_residual'].max())),
        'omics_normalized_attention_entropy':entropy.mean(1).tolist()}

def load_replays(input_path):
    spec=read_json(input_path)
    if spec.get('schema_version')!=1 or not spec.get('exports'):raise ValueError('Expected an explicit exports list')
    rows=[];seen=set();patients={};universes={};names={}
    for name in spec['exports']:
        d=(Path(input_path).parent/name).resolve();meta=read_json(d/'export.json');run=meta['run']
        cancer=run['cancer'].upper();key=(cancer,run['fold'])
        if run['arm']!='exp6' or run['seed']!=3 or run['protocol']!='legacy_val' or cancer not in ('BLCA','KIRC') or key in seen:raise ValueError('Require distinct BLCA/KIRC Full seed-3 folds')
        seen.add(key);check=audit({'runs':[run]})
        if not check['passed'] or meta['hashes']!=check['runs'][0]['hashes']:raise ValueError('Raw-source audit/hash mismatch')
        saved=predictions(run['predictions'])
        with np.load(d/'patients.npz',allow_pickle=False) as data:a={k:data[k] for k in data.files}
        stats=validate_arrays(meta,a,saved);split=split_ids(run['split_csv']);ids=set(a['case_ids'].astype(str))
        if ids!=set(split['val']) or patients.setdefault(cancer,set())&ids:raise ValueError('Validation fold mismatch/overlap')
        patients[cancer].update(ids);u=set(split['train'])|ids
        if cancer in universes and universes[cancer]!=u:raise ValueError('Cohort universe differs across folds')
        universes[cancer]=u;pn=a['pathway_names'].astype(str).tolist()
        if len(set(pn))!=len(pn) or (cancer in names and names[cancer]!=pn):raise ValueError('Pathway identities/order differ')
        names[cancer]=pn
        rows.append({'cancer':cancer,'fold':run['fold'],'arrays':a,'stats':stats,'source':{'export':str(d/'export.json'),'export_sha256_lf':text_sha(d/'export.json'),'patients':str(d/'patients.npz'),'patients_sha256':sha256(d/'patients.npz'),'run':run,'hashes':meta['hashes']}})
    for cancer in patients:
        if {fold for c,fold in seen if c==cancer}!=set(range(5)) or patients[cancer]!=universes[cancer]:raise ValueError('Require a complete nonoverlapping five-fold cohort')
    return sorted(rows,key=lambda r:(r['cancer'],r['fold']))

def render_replays(input_path,output):
    rows=load_replays(input_path);out=fresh(output);plt=plotting();artifacts=[]
    for cancer in sorted({r['cancer'] for r in rows}):
        folds=[r for r in rows if r['cancer']==cancer]
        fig,axes=plt.subplots(1,3,figsize=(13.8,4.3))
        for ax,key,title in zip(axes,('delta_cindex','mean_abs_risk_change_over_factual_iqr','mean_cross_error_change'),('Ranking response','Patient risk response','Conditional cross-reconstruction response')):
            for row in folds:ax.plot(ALPHAS,row['stats'][key],'o-',lw=1,label=f"Fold {row['fold']}")
            ax.axhline(0,color='black',lw=.7);ax.set(title=title,xlabel='Plan replacement fraction',xticks=ALPHAS);ax.grid(ls='--',alpha=.3)
        for ax,label in zip(axes,('Change in validation C-index','Mean absolute risk change / factual risk IQR','Error change (modified minus factual)')):ax.set_ylabel(label)
        axes[0].legend(fontsize=7);fig.suptitle(cancer+' | frozen Full plan intervention',fontweight='bold')
        fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.015,'Same patients / checkpoint; all plan geometries and gates replayed. Paired omics remains observed: latent conditional reconstruction, not WSI-only imputation.',ha='center',fontsize=8)
        files=save(fig,out,f'{cancer.lower()}_plan_mechanism_response');artifacts.append({'cancer':cancer,'kind':'plan_response','files':files})
        fig,axes=plt.subplots(1,3,figsize=(13.8,4.3))
        for modality,shift,color in (('wsi',-.12,'#0072B2'),('omic',.12,'#D55E00')):
            for row in folds:
                for ax,metric in zip(axes[:2],('mean_cosine','hazard_variance')):
                    vals=row['arrays'][f'{modality}_{metric}'];x=row['fold']+shift
                    ax.scatter(np.full(len(vals),x),vals,s=6,alpha=.18,color=color)
                    ax.plot(x,np.median(vals),'_',ms=16,color=color,label=modality.upper() if row['fold']==0 else None)
        for row in folds:
            vals=row['stats']['omics_normalized_attention_entropy'];axes[2].scatter(np.full(len(vals),row['fold']),vals,s=6,alpha=.2,color='#009E73');axes[2].plot(row['fold'],np.median(vals),'_',ms=16,color='black')
        for ax,title,label in zip(axes,('Slot representation separation','Slot hazard separation','Effective omics attention'),('Mean off-diagonal slot cosine','Across-slot hazard variance','Normalized pathway entropy (1 = uniform)')):
            ax.set(title=title,ylabel=label,xlabel='Validation fold',xticks=range(5));ax.grid(axis='y',ls='--',alpha=.3)
        axes[0].legend(fontsize=8);fig.suptitle(cancer+' | patient-level slot diagnostics',fontweight='bold')
        fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.015,'Each dot is one validation patient; line marks its fold median. Slot indices are not matched across folds. These diagnostics do not establish biological slot identity.',ha='center',fontsize=8)
        files=save(fig,out,f'{cancer.lower()}_slot_function_diagnostics');artifacts.append({'cancer':cancer,'kind':'slot_diagnostics','files':files})
    write_json(out/'replay_review.json',{'schema_version':1,'source_commit':revision(ROOT),'folds':[{'cancer':r['cancer'],'fold':r['fold'],'stats':r['stats'],'source':r['source']} for r in rows],
        'artifacts':artifacts,'new_model_runs_by_plotter':0,'note':'Audited frozen-export arrays, not new training; no biological causal claim. Risk responses normalized within each checkpoint. Patient NPZ hashes recorded on read; exporter did not originally pin aggregate NPZ hash in export.json.'})

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('recorded');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('replay');p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.command=='recorded':render_recorded(ROOT,args.output)
        else:render_replays(args.input,args.output)
    except (ValueError,KeyError,OSError) as e:ap.exit(1,f'[proof figures] {e}\nNo model was executed.\n')

if __name__=='__main__':main()
