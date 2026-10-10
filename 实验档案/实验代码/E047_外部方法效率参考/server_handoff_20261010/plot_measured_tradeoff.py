"""Read real benchmark JSONs; never estimates or fills missing GPU costs.
This renders reported BLCA means versus newly measured standardized native costs.
It does NOT represent a matched performance evaluation. Matched mode uses the
existing survot_rank.evidence.reference_panels.validate_tradeoff pipeline instead.
"""
import argparse,json,hashlib,csv,math,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def collect(spec_path,profiles):
    s=json.loads(Path(spec_path).read_text(encoding='utf-8-sig')); expected=sha(spec_path)
    require(s['comparison_mode']=='reported_BLCA_score_vs_measured_standardized_cost','Wrong comparison mode')
    require(s['cancer']=='BLCA' and s['endpoint']=='DSS','BLCA DSS only')
    score_file=Path(spec_path).parent/s['score_snapshot_file']
    require(sha(score_file)==s['score_snapshot_sha256'],'Frozen score file changed')
    with score_file.open(encoding='utf-8-sig') as handle:
        frozen={r['id']:r for r in csv.DictReader(handle)}
    require(set(frozen)=={m['id'] for m in s['methods']},'Method list differs from frozen snapshot')
    for m in s['methods']:
        require(float(frozen[m['id']]['mean'])==m['reported_blca_mean'],'BLCA score changed')
    require(len(s['methods'])==9 and len({m['id'] for m in s['methods']})==9,'Expected all nine planned methods; do not cherry pick')
    require(sum(m['ours'] for m in s['methods'])==1,'Exactly one DCT point')
    require(len(s['cases'])==5 and {c['fold'] for c in s['cases']}==set(range(5)),'Five fixed real BLCA cases required')
    require(s['n_replicates']==3 and s['warmup']>=5 and s['repeats']>=30,'Incomplete repeat protocol')
    env_keys=['hardware','device_uuid','torch_version','cuda_version','cudnn_version','precision','matmul_allow_tf32','cudnn_allow_tf32','cudnn_benchmark','cudnn_deterministic','forward_mode']
    common=None; seen=set(); points=[];sources=[]
    records=[]
    for p in sorted(Path(profiles).glob('*.json')):
        record=json.loads(p.read_text(encoding='utf-8'));record['_path']=str(p); records.append(record)
    require(len(records)==len(s['methods'])*s['n_replicates'],'Require all 27 method/replicate JSONs in a dedicated profile directory')
    for m in s['methods']:
        mem=[];lat=[];kinds=set();adapters=set()
        rr=[r for r in records if r['method_id']==m['id']]
        require(len(rr)==s['n_replicates'],'Missing method '+m['id'])
        for r in rr:
            key=(r['method_id'],r['replicate']);require(key not in seen and 0<=r['replicate']<s['n_replicates'],'Duplicate replicate');seen.add(key)
            require(r['spec_sha256']==expected,'Spec hash mismatch')
            env=tuple(r[k] for k in env_keys)
            if common is None:common=env
            require(env==common,'Hardware or numerical settings differ')
            require(r['precision']=='float32' and r['forward_mode']=='eval_no_grad_native_full','Unapproved inference scope')
            meta=r['metadata'];require(meta['source_commit']==m['source_commit'] and meta['native_full_prediction_forward'],'Source/native scope mismatch')
            require(meta['weights_kind'] in ('trained_checkpoint','random_init'),'Invalid weight provenance')
            kinds.add(meta['weights_kind']);adapters.add(r['adapter_sha256'])
            cases=r['cases'];require(len(cases)==5 and {c['fold'] for c in cases}==set(range(5)),'Five cases required')
            for c in cases:
                wanted=next(x for x in s['cases'] if x['fold']==c['fold'])
                for k in ('case_id','common_wsi_sha256','common_omics_sha256'):require(c[k]==wanted[k] and bool(c[k]),'Patient/input mismatch')
                require(c['input_audit']['same_wsi_values'] and c['input_audit']['same_raw_omics'],'Input mapping failed')
                values=c['latency_samples_ms'];require(len(values)==s['repeats'],'Missing raw timings')
                require(all(math.isfinite(x) and x>0 for x in values),'Invalid raw timings')
                require(abs(statistics.median(values)-c['latency_ms_median'])<1e-9,'Timing summary mismatch')
                peaks=c['peak_allocated_samples_mib']
                require(len(peaks)==s['repeats'] and all(math.isfinite(x) and x>0 for x in peaks),'Missing raw memory peaks')
                require(max(peaks)==c['peak_allocated_mib'],'Peak memory summary mismatch')
                require(0<=c['baseline_allocated_mib']<=c['peak_allocated_mib'],'Baseline exceeds total allocated peak')
            mem.append(statistics.mean(c['peak_allocated_mib'] for c in cases))
            lat.append(statistics.mean(c['latency_ms_median'] for c in cases))
            sources.append({'path':r['_path'],'sha256':sha(r['_path'])})
        require(len(kinds)==1 and len(adapters)==1,'Weight kind or adapter differs across replicates')
        require(0<=m['reported_blca_mean']<=1,'Invalid BLCA score')
        points.append({**m,'memory_mib':statistics.mean(mem),'runtime_sec':statistics.mean(lat)/1000,
                       'memory_process_sd_mib':statistics.stdev(mem),'runtime_process_sd_sec':statistics.stdev(lat)/1000,'weights_kind':next(iter(kinds))})
    return s,points,sources,dict(zip(env_keys,common))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--profiles',required=True);ap.add_argument('--output',required=True);ap.add_argument('--validate-only',action='store_true')
    a=ap.parse_args();spec,points,sources,env=collect(a.spec,a.profiles)
    if a.validate_only:print('PASS: all 27 measured profiles, fixed inputs and common environment');return
    out=Path(a.output);require(not out.exists() or not any(out.iterdir()),'Use a fresh figure output directory');out.mkdir(parents=True,exist_ok=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(14,5.6));colors=['#8678E1','#62B57F','#B977BF','#ED7980','#4A9BAA','#AC975C','#4787BA','#D19D75'];markers=['o','s','D','^','P','v','X','h']
    external=0;handles=[];labels=[]
    for p in points:
        ours=p['ours'];color='#F39C12' if ours else colors[external];marker='*' if ours else markers[external]
        if not ours:external+=1
        label=f"{p['label']} ({p['year']})"+(' (Ours)' if ours else '')
        for index,(ax,key) in enumerate(zip(axes,('memory_mib','runtime_sec'))):
            handle=ax.scatter(p[key],p['reported_blca_mean'],s=260 if ours else 75,color=color,marker=marker,edgecolors='white',linewidths=.7,zorder=4)
            offset=(5,10) if ours else [(5,-13),(-10,10),(5,10),(-8,-13)][(external-1)%4]
            ax.annotate(p['label'],(p[key],p['reported_blca_mean']),xytext=offset,textcoords='offset points',fontsize=8,fontweight='bold' if ours else 'normal')
            if index==0:handles.append(handle);labels.append(label)
    for ax,title,xlabel in zip(axes,('A  BLCA: performance vs inference memory','B  BLCA: performance vs inference runtime'),('Peak GPU allocated memory (MiB) ↓','Synchronized wall-clock time (sec / patient) ↓')):
        ax.set_title(title,pad=12);ax.set_xlabel(xlabel);ax.set_ylabel('Reported BLCA mean C-index ↑');ax.grid(ls='--',alpha=.35);ax.set_axisbelow(True);ax.margins(x=.18,y=.25);ax.xaxis.set_major_locator(MaxNLocator(6))
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.08),ncol=5,frameon=False,fontsize=8)
    fig.text(.06,.025,'Y: reported BLCA DSS means; cohorts, splits and encoders differ. X: new standardized native-forward measurements.',fontsize=8)
    fig.text(.06,.002,'Preloaded UNI2-h features, 2048 patches, batch 1, FP32. Cost weights/provenance are recorded; this is not a matched accuracy benchmark.',fontsize=8)
    fig.subplots_adjust(left=.07,right=.98,top=.89,bottom=.27,wspace=.25)
    for ext in ('png','pdf','svg'):fig.savefig(out/f'blca_external_memory_runtime.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    fields=['id','label','year','reported_blca_mean','score_encoder','memory_mib','runtime_sec','memory_process_sd_mib','runtime_process_sd_sec','weights_kind','source_commit']
    with (out/'blca_external_memory_runtime.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(points)
    report={'comparison_mode':spec['comparison_mode'],'spec_sha256':sha(a.spec),'environment':env,'points':points,'profiles':sources,'same_condition_performance_claim':False}
    report['output_sha256']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'blca_external_memory_runtime.audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(out/'blca_external_memory_runtime.png'))
if __name__=='__main__':main()
