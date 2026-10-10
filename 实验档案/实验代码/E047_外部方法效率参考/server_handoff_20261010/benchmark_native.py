"""Server-only GPU forward benchmark. No training or survival-score evaluation.
Adapter: build(spec, method) -> {'model': CPU nn.Module, 'cases': CPU inputs,
 'metadata': native source/weights/tokenization provenance}. See adapter_contract.md.
Each invocation runs exactly one method and one independent process replicate.
"""
import argparse, importlib.util, json, hashlib, time, statistics, math, subprocess
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(d,f,ensure_ascii=False,indent=2)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--adapter',required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--replicate',required=True,type=int);ap.add_argument('--output',required=True)
    a=ap.parse_args(); spec=json.loads(Path(a.spec).read_text(encoding='utf-8-sig'))
    assert spec['cancer']=='BLCA' and spec['precision']=='float32'
    assert spec['batch_size']==1 and spec['patches']==2048 and spec['feature_dim']==1536
    assert spec['warmup']>=5 and spec['repeats']>=30 and spec['no_training'] is True
    assert 0<=a.replicate<spec['n_replicates']
    assert len(spec['cases'])==5 and {c['fold'] for c in spec['cases']}==set(range(5))
    for c in spec['cases']:
        assert c['case_id'] and len(c['common_wsi_sha256'] or '')==64 and len(c['common_omics_sha256'] or '')==64, 'Fill audited real cases before running'
    method=next(m for m in spec['methods'] if m['id']==a.method)
    assert method['source_commit'] and len(method['source_commit'])==40
    assert not Path(a.output).exists(), 'Use a fresh output file'
    import torch
    assert torch.cuda.is_available(), 'GPU measurements required; CPU cannot supply a GPU memory point'
    device=torch.device(spec['device']);torch.cuda.set_device(device)
    assert torch.cuda.memory_allocated(device)==0, 'Start a fresh process with no resident models'
    torch.manual_seed(spec['seed']);torch.cuda.manual_seed_all(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    module_spec=importlib.util.spec_from_file_location('native_adapter',a.adapter)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    built=module.build(spec,method);model=built['model'];cases=built['cases'];meta=built['metadata']
    assert all(p.device.type=='cpu' for p in model.parameters()), 'Adapter must build on CPU'
    assert meta['source_commit']==method['source_commit'] and meta['native_full_prediction_forward'] is True
    actual_commit=subprocess.check_output(['git','-C',meta['repo_root'],'rev-parse','HEAD'],text=True).strip()
    assert actual_commit==method['source_commit'], 'Official/local Git revision mismatch'
    assert meta['source_files'], 'Record native code snapshots, including dimension patches'
    for item in meta['source_files']:
        assert sha(item['path'])==item['sha256'], 'Native code snapshot changed'
    assert meta['native_tokenization'] and meta['input_dimension_adaptation'] is not None
    assert meta['weights_kind'] in ('trained_checkpoint','random_init')
    if meta['weights_kind']=='trained_checkpoint':
        assert sha(meta['checkpoint_path'])==meta['checkpoint_sha256']
        assert meta['strict_checkpoint_load'] is True
    else:
        assert spec['allow_random_init_for_cost_only'] and not meta.get('checkpoint_sha256')
    assert len(cases)==5 and {c['fold'] for c in cases}==set(range(5))
    def tensor_sha(t):
        t=t.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.view(torch.uint8).numpy().tobytes()).hexdigest()
    def tensors(obj):
        if isinstance(obj,torch.Tensor):return [obj]
        if isinstance(obj,dict):return sum((tensors(v) for v in obj.values()),[])
        if isinstance(obj,(tuple,list)):return sum((tensors(v) for v in obj),[])
        return []
    def move(obj):
        if isinstance(obj,torch.Tensor):return obj.to(device)
        if isinstance(obj,dict):return {k:move(v) for k,v in obj.items()}
        if isinstance(obj,list):return [move(v) for v in obj]
        if isinstance(obj,tuple):return tuple(move(v) for v in obj)
        return obj
    assert all(t.device.type=='cpu' for c in cases for t in tensors(c['kwargs'])), 'Preload CPU inputs only'
    model=model.to(device).eval()
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.is_floating_point())
    records=[]
    for case in sorted(cases,key=lambda c:c['fold']):
        wanted=next(c for c in spec['cases'] if c['fold']==case['fold'])
        assert wanted['case_id']==case['case_id']
        raw=case['raw_wsi'];assert list(raw.shape)==[1,2048,1536] and raw.dtype==torch.float32
        assert tensor_sha(raw)==wanted['common_wsi_sha256']
        assert sha(case['raw_omics_path'])==wanted['common_omics_sha256']
        assert case['input_audit']['same_wsi_values'] and case['input_audit']['same_raw_omics']
        kwargs=move(case['kwargs']); assert all(t.dtype==torch.float32 for t in tensors(kwargs) if t.is_floating_point())
        with torch.no_grad():
            for _ in range(spec['warmup']):
                result=model(**kwargs);del result
            torch.cuda.synchronize(device)
            baseline=torch.cuda.memory_allocated(device)/2**20
            samples=[];peaks=[];reserved=[]
            for _ in range(spec['repeats']):
                torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                start=time.perf_counter();result=model(**kwargs);torch.cuda.synchronize(device)
                samples.append((time.perf_counter()-start)*1000)
                peaks.append(torch.cuda.max_memory_allocated(device)/2**20)
                reserved.append(torch.cuda.max_memory_reserved(device)/2**20)
                del result
            result=model(**kwargs)
            vals=tensors(result)
            assert vals and all(torch.isfinite(t).all().item() for t in vals if t.is_floating_point()), 'Invalid native prediction outputs'
            shapes=[list(t.shape) for t in vals];del result
        assert all(math.isfinite(x) and x>0 for x in samples+peaks)
        records.append({'fold':case['fold'],'case_id':case['case_id'],'common_wsi_sha256':wanted['common_wsi_sha256'],
            'common_omics_sha256':wanted['common_omics_sha256'],'input_audit':case['input_audit'],
            'latency_samples_ms':samples,'latency_ms_median':statistics.median(samples),
            'latency_ms_p95':sorted(samples)[math.ceil(.95*len(samples))-1],
            'peak_allocated_samples_mib':peaks,'peak_reserved_samples_mib':reserved,'peak_allocated_mib':max(peaks),'peak_reserved_mib':max(reserved),'baseline_allocated_mib':baseline,'output_shapes':shapes})
        del kwargs;torch.cuda.empty_cache()
    prop=torch.cuda.get_device_properties(device)
    record={'schema_version':1,'method_id':a.method,'replicate':a.replicate,'spec_sha256':sha(a.spec),'adapter_sha256':sha(a.adapter),
        'hardware':torch.cuda.get_device_name(device),'device_uuid':str(getattr(prop,'uuid','unavailable')),
        'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),
        'precision':'float32','matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,
        'forward_mode':'eval_no_grad_native_full','metadata':meta,'cases':records}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);dump(a.output,record)
    print(json.dumps({'method':a.method,'replicate':a.replicate,'cases':len(records),'output':a.output}))
if __name__=='__main__':main()
