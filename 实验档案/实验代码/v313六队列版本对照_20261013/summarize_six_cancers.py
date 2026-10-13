"""Read only the completed NEW-version epoch CSVs; do not train or infer."""
from pathlib import Path
import argparse,csv,json,statistics,hashlib
HERE=Path(__file__).resolve().parent
CANCERS=['BRCA','COADREAD','UCEC','HNSC','SKCM','STAD']
def main():
    a=argparse.ArgumentParser();a.add_argument('--results-root',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);args=a.parse_args()
    plan=json.loads((args.results_root/'plan.json').read_text(encoding='utf-8'))
    if plan.get('protocol')!='legacy_val' or plan.get('baseline_training_requested') is not False or len(plan['tasks'])!=30:
        raise ValueError('Expected new-version-only 30-task legacy_val plan.')
    baseline_path=Path(plan['baseline_file'])
    if hashlib.sha256(baseline_path.read_bytes()).hexdigest()!=plan['baseline_sha256']:raise ValueError('Frozen baseline changed')
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'));observed={};evidence=[]
    for t in plan['tasks']:
        cancer=t['cancer'].upper();fold=t['fold']
        if t['arm']!='three_decoders' or t['seed']!=3 or (cancer,fold) in observed:raise ValueError('Unexpected or duplicate task')
        curves=list(Path(t['output']).rglob(f'epoch_curve_fold{fold}.csv'))
        if len(curves)!=1:raise ValueError(f'{cancer} fold{fold}: require exactly one epoch curve')
        for name in [f'model_best_s{fold}.pth',f'split_{fold}_results_final.pkl']:
            if not (curves[0].parent/name).is_file():raise ValueError(f'{cancer} fold{fold}: missing checkpoint/prediction export {name}')
        rows=list(csv.DictReader(curves[0].open(encoding='utf-8-sig',newline='')))
        epochs=[int(x['epoch']) for x in rows]
        if sorted(epochs)!=list(range(30)):raise ValueError(f'{cancer} fold{fold}: frozen 30-epoch run not complete')
        best=max(rows,key=lambda x:float(x['val_cindex']));score=float(best['val_cindex'])
        if not 0<=score<=1:raise ValueError('Invalid C-index')
        observed[cancer,fold]=score;evidence.append({'cancer':cancer,'fold':fold,'best_epoch':int(best['epoch']),'cindex':score,'csv':str(curves[0]),'csv_sha256':hashlib.sha256(curves[0].read_bytes()).hexdigest(),'split_sha256':t['split_sha256']})
    summary=[];paired=[]
    for cancer in CANCERS:
        old=baseline['rows'][cancer]['folds'];new=[observed[cancer,f] for f in range(5)];d=[n-o for n,o in zip(new,old)]
        summary.append({'cancer':cancer,'original_mean':statistics.mean(old),'new_mean':statistics.mean(new),'delta':statistics.mean(d),'original_sample_sd':statistics.stdev(old),'new_sample_sd':statistics.stdev(new),'positive_folds':sum(v>0 for v in d),'protocol':'legacy_val','causal_decoder_claim':False})
        paired += [{'cancer':cancer,'fold':f,'original':o,'new':n,'delta':n-o} for f,(o,n) in enumerate(zip(old,new))]
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,rows in [('六队列版本均值对照.csv',summary),('逐折版本对照.csv',paired)]:
        with (args.output_dir/name).open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.output_dir/'来源核验.json').write_text(json.dumps({'evidence':evidence,'metric':'legacy_val validation peaks; not outer test','historical_baseline_server_identity_reaudit_required':True,'paired_fold_difference_requires_patient_split_identity_confirmation':True},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
