"""Train NEW three-decoder version only; default prints metadata/commands, not training."""
from pathlib import Path
import argparse,copy,hashlib,importlib.util,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
BASE_RUNNER=ROOT/'实验档案/实验代码/E046_WSI自重建消融/run_wsi_reconstruction_ablation.py'
spec=importlib.util.spec_from_file_location('v313_existing_reconstruction_runner',BASE_RUNNER)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
CANCERS=['brca','coadread','ucec','hnsc','skcm','stad']
def parser():
    p=base.parser();p.description=__doc__
    p.set_defaults(base_config=ROOT/'configs/dct_v313_full_wsi_three_decoders.yaml',cancers=CANCERS,arms=['wsi_additive'],protocol='legacy_val',wsi_weight=0.05,folds=list(range(5)),seeds=[3],which_splits='5fold_uni2h')
    for a in p._actions:
        if a.dest=='protocol':a.choices=['legacy_val']
        if a.dest=='arms':a.choices=['wsi_additive'];a.help='Internal driver alias; only new three_decoders tasks are created.'
    return p

def build_plan(args):
    if args.cancers!=CANCERS or args.folds!=list(range(5)) or args.seeds!=[3]:
        raise ValueError('This delivery freezes the six requested cancers, folds 0..4, seed 3.')
    if args.protocol!='legacy_val' or args.which_splits!='5fold_uni2h' or args.wsi_weight!=0.05:
        raise ValueError('Historical comparison requires legacy_val / 5fold_uni2h, WSI weight 0.05.')
    config=base.yaml.safe_load(args.base_config.read_text(encoding='utf-8-sig'))
    flat={}
    for k,v in config.items():flat.update(v if isinstance(v,dict) else {k:v})
    expected={'max_epochs':30,'batch_size':8,'seed':3,'num_patches':2048,'encoding_dim':1536,'label_col':'survival_months_dss','dct_v313_lambda_reconstruction':0.1,'dct_v313_reconstruction_self_fraction':0.5,'dct_v313_reconstruction_cross_fraction':0.5,'dct_v313_reconstruction_ramp_start':2,'dct_v313_reconstruction_ramp_epochs':5}
    for k,v in expected.items():
        if flat.get(k)!=v:raise ValueError(f'Frozen recipe mismatch: {k} must be {v!r}')
    model=ROOT/'survot_rank/research/methods/legacy/experimental/dct_v313_transport_reconstruction/model.py'
    if 'dct_v313_omics_decoder_mode' not in model.read_text(encoding='utf-8-sig'):
        raise ValueError('Source lacks independent decoder support; update source before planning.')
    plan=base.build_plan(args)
    for t in plan['tasks']:
        t['arm']='three_decoders'
        t['output']=str(args.results_root.resolve()/'legacy_val'/'three_decoders'/t['cancer']/f"fold{t['fold']}"/'seed3')
        o=t['overrides'];o.update(results_dir=t['output'],specific_simple=f"v313_three_decoders_six_{t['cancer']}",dct_v313_revision='repaired',dct_v313_omics_decoder_mode='separate',dct_v313_lambda_wsi_reconstruction=0.05,dct_v313_wsi_reconstruction_budget='additive')
        t['command']=[sys.executable,'-m','survot_rank.cli','train','--config',str(args.base_config.resolve())]
        for k,v in o.items():t['command']+=['--set',f'{k}={v}']
    plan.update(experiment='v313_six_cohort_version_comparison',baseline_training_requested=False,comparison_kind='historical_version_comparison_not_isolated_decoder_causality',reported_metric='five-fold mean of maximum val_cindex, legacy_val development evidence',baseline_file=str(HERE/'baseline_frozen.json'),baseline_sha256=base.sha256(HERE/'baseline_frozen.json'),baseline_server_reaudit_required=True)
    for p in [Path(__file__),HERE/'baseline_frozen.json']:
        plan['source_sha256'][p.resolve().relative_to(ROOT).as_posix()]=base.sha256(p)
    diff=subprocess.check_output(['git','diff','HEAD','--','survot_rank','configs'],cwd=ROOT)
    plan['training_source_diff_sha256']=hashlib.sha256(diff).hexdigest()
    plan['training_source_has_uncommitted_diff']=bool(diff)
    return plan

def main(argv=None):
    a=parser().parse_args(argv);p=build_plan(a)
    if a.execute:
        # Existing driver checks source/config/split hashes and refuses nonempty results roots.
        base.execute_plan(p)
    else:print(json.dumps(p,ensure_ascii=False,indent=2))
    return p
if __name__=='__main__':main()
