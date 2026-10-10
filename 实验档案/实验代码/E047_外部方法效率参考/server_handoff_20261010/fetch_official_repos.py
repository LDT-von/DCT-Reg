"""Download eight pinned official repositories without importing/running them."""
import argparse,json,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--destination',required=True);a=p.parse_args()
    s=json.loads(Path(a.spec).read_text(encoding='utf-8-sig'));dst=Path(a.destination);dst.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for m in s['methods']:
        if m['ours']:continue
        target=dst/m['id'];assert not target.exists(), 'Never overwrite an existing repository'
        subprocess.run(['git','init',str(target)],check=True)
        subprocess.run(['git','-C',str(target),'remote','add','origin',m['repo_url']+'.git'],check=True)
        subprocess.run(['git','-C',str(target),'fetch','--depth','1','origin',m['source_commit']],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach','FETCH_HEAD'],check=True)
        actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip();assert actual==m['source_commit']
        manifest.append({'method_id':m['id'],'repo':m['repo_url'],'commit':actual,'path':str(target)})
    with (dst/'download_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
if __name__=='__main__':main()
