"""Launch one new process per method/replicate. Default is plan-only."""
import argparse,json,random,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--adapters',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--execute',action='store_true')
    a=ap.parse_args();s=json.loads(Path(a.spec).read_text(encoding='utf-8-sig'));target=Path(a.output_dir)
    schedule=[]
    for rep in range(s['n_replicates']):
        methods=s['methods'].copy();random.Random(s['seed']+rep).shuffle(methods)
        for m in methods:
            adapter=Path(a.adapters)/(m['id']+'.py');output=target/'profiles'/(m['id']+f'_r{rep}.json')
            cmd=[sys.executable,str(Path(__file__).with_name('benchmark_native.py')),'--spec',str(Path(a.spec).resolve()),'--adapter',str(adapter.resolve()),'--method',m['id'],'--replicate',str(rep),'--output',str(output.resolve())]
            schedule.append({'method':m['id'],'replicate':rep,'command':cmd,'adapter':str(adapter)})
    if not a.execute:
        print(json.dumps({'plan_only':True,'processes':len(schedule),'schedule':schedule},indent=2));return
    assert all(c['case_id'] and c['common_wsi_sha256'] and c['common_omics_sha256'] for c in s['cases']), 'Fill real cases first'
    assert all(Path(x['adapter']).is_file() for x in schedule), 'Implement all nine native adapters first'
    assert not target.exists(), 'Use a fresh results directory; never overwrite'
    (target/'logs').mkdir(parents=True)
    (target/'launch_schedule.json').write_text(json.dumps(schedule,indent=2),encoding='utf-8')
    for task in schedule:
        name=task['method']+f"_r{task['replicate']}"
        with (target/'logs'/(name+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(task['command'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed all processes; run validation and plotting next.')
if __name__=='__main__':main()
